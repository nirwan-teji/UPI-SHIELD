from fastapi import APIRouter, Body
import torch
import asyncio
from app.core.model_loader import model_loader
from app.core.groq_client import GroqClient
from app.core.response_formatter import ResponseFormatter
from app.core.groq_utils import parse_groq_response

router = APIRouter(tags=["Payment Scam"])

async def get_groq_analysis(text: str) -> dict:
    """Analyze payment request using Groq LLM"""
    groq_client = GroqClient()
    
    messages = [{
        "role": "user",
        "content": f"""Analyze this payment request:
        {text}
        
        Respond ONLY with valid JSON:
        {{
            "label": "SCAM"|"LEGITIMATE",
            "risk_score": 0-100,
            "explanation": "Detailed analysis"
        }}
        No markdown, no extra text."""
    }]
    
    try:
        response = await groq_client.generate_response(messages)
        return parse_groq_response(response)
    except Exception as e:
        print(f"Groq analysis failed: {str(e)}")
        return {"error": str(e)}

@router.post("/predict/payment")
async def detect_payment_scam(text: str = Body(..., embed=True)):
    try:
        def model_predict():
            model = model_loader.get_model('payment')
            vocab = model_loader.get_vocab()
            tokens = text.lower().split()
            encoded = [vocab.get(token, vocab['<UNK>']) for token in tokens][:100]
            padded = encoded + [vocab['<PAD>']] * (100 - len(encoded))
            inputs = torch.tensor([padded], dtype=torch.long)
            with torch.no_grad():
                outputs = model(inputs)
                return torch.sigmoid(outputs).item()
        
        model_prob = await asyncio.to_thread(model_predict)
        is_scam_model = model_prob > 0.5
        model_risk = int(model_prob * 100)

        groq_data = await get_groq_analysis(text)
        
        if "error" not in groq_data:
            is_scam_groq = groq_data.get("label", "").upper() == "SCAM"
            final_risk = model_risk if (is_scam_model == is_scam_groq) else groq_data.get("risk_score", 0)
            return ResponseFormatter.format_response({
                "is_scam": is_scam_groq if not (is_scam_model == is_scam_groq) else is_scam_model,
                "label": "SCAM" if is_scam_model else "LEGITIMATE",
                "risk_score": final_risk,
                "scam_type": "Payment Scam",
                "explanation": groq_data.get("explanation", "Analysis unavailable"),
                "model_verified": (is_scam_model == is_scam_groq)
            })
        
        return ResponseFormatter.format_response({
            "is_scam": is_scam_model,
            "label": "SCAM" if is_scam_model else "LEGITIMATE",
            "risk_score": model_risk,
            "scam_type": "Payment Scam",
            "explanation": "Model analysis: " + ("Suspicious payment request" if is_scam_model else "Legitimate request"),
            "model_verified": False
        })
        
    except Exception as e:
        return ResponseFormatter.format_error_response(str(e))
