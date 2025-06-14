from fastapi import APIRouter, Body
import torch
import asyncio
from app.core.model_loader import model_loader
from app.core.groq_client import GroqClient
from app.core.response_formatter import ResponseFormatter
from app.core.groq_utils import parse_groq_response

router = APIRouter(tags=["Investment Scam"])

async def get_groq_analysis(text: str) -> dict:
    """Analyze investment offer using Groq LLM"""
    groq_client = GroqClient()
    
    messages = [{
        "role": "user",
        "content": f"""Analyze this investment offer:
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

@router.post("/predict/investment")
async def detect_investment_scam(text: str = Body(..., embed=True)):
    try:
        def model_predict():
            model = model_loader.get_model('investment')
            tokenizer = model_loader.get_tokenizer('investment')
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                return torch.argmax(probs).item(), probs[0].tolist()
        
        model_pred, model_probs = await asyncio.to_thread(model_predict)
        is_scam_model = bool(model_pred == 1)
        model_risk = int(model_probs[model_pred] * 100)

        groq_data = await get_groq_analysis(text)
        
        if "error" not in groq_data:
            is_scam_groq = groq_data.get("label", "").upper() == "SCAM"
            final_risk = model_risk if (is_scam_model == is_scam_groq) else groq_data.get("risk_score", 0)
            return ResponseFormatter.format_response({
                "is_scam": is_scam_groq if not (is_scam_model == is_scam_groq) else is_scam_model,
                "label": "SCAM" if is_scam_model else "LEGITIMATE",
                "risk_score": final_risk,
                "scam_type": "Investment Scam",
                "explanation": groq_data.get("explanation", "Analysis unavailable"),
                "model_verified": (is_scam_model == is_scam_groq)
            })
        
        return ResponseFormatter.format_response({
            "is_scam": is_scam_model,
            "label": "SCAM" if is_scam_model else "LEGITIMATE",
            "risk_score": model_risk,
            "scam_type": "Investment Scam",
            "explanation": "Model analysis: " + ("Unrealistic returns" if is_scam_model else "Legitimate offer"),
            "model_verified": False
        })
        
    except Exception as e:
        return ResponseFormatter.format_error_response(str(e))
