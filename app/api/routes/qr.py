from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np
import base64
import asyncio
from app.core.model_loader import model_loader
from app.core.groq_client import GroqClient
from app.core.response_formatter import ResponseFormatter
from app.core.groq_utils import parse_groq_response

router = APIRouter(tags=["QR Scam"])

async def analyze_qr_image(file: UploadFile) -> dict:
    """Analyze QR code image using Groq vision model"""
    groq_client = GroqClient()
    image_data = await file.read()
    base64_image = base64.b64encode(image_data).decode('utf-8')

    # OpenCV QR code detection
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    detector = cv2.QRCodeDetector()
    qr_content, points, _ = detector.detectAndDecode(img)
    if not qr_content:
        qr_content = "Unable to decode QR code"

    try:
        response = await groq_client.generate_response(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""Analyze this QR code:
Decoded content: {qr_content}
Check for malicious patterns and domain reputation.
Respond in JSON format:
{{
    "label": "SCAM"|"LEGITIMATE",
    "risk_score": 0-100,
    "explanation": "Detailed analysis"
}}
""" },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{file.content_type};base64,{base64_image}"
                        }
                    }
                ]
            }],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=1024
        )
        return parse_groq_response(response)
    except Exception as e:
        print(f"Groq vision analysis failed: {str(e)}")
        return {"error": str(e)}

@router.post("/predict/qr")
async def detect_qr_scam(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # OpenCV QR code detection
        detector = cv2.QRCodeDetector()
        qr_content, points, _ = detector.detectAndDecode(img)
        if not qr_content:
            return ResponseFormatter.format_error_response("No QR code found in image")

        # YOLO model prediction (if you have a visual model for scam QR)
        def model_predict():
            model = model_loader.get_model('qr')
            results = model(img)
            return any(box.cls.item() == 1 for r in results for box in r.boxes)
        is_scam_model = await asyncio.to_thread(model_predict)
        model_risk = 100 if is_scam_model else 0

        # Groq vision analysis
        file.file.seek(0)  # Reset file pointer for Groq
        groq_data = await analyze_qr_image(file)

        if "error" not in groq_data:
            is_scam_groq = groq_data.get("label", "").upper() == "SCAM"
            groq_risk = groq_data.get("risk_score", 0)
            explanation = groq_data.get("explanation", "QR code analysis unavailable")
            return ResponseFormatter.format_response({
                "is_scam": is_scam_model or is_scam_groq,
                "label": "SCAM" if (is_scam_model or is_scam_groq) else "LEGITIMATE",
                "risk_score": max(model_risk, groq_risk),
                "scam_type": "qr_scam",
                "explanation": explanation,
                "qr_content": qr_content,
                "model_verified": (is_scam_model == is_scam_groq),
                "image_analysis": groq_data.get("analysis", "")
            })

        return ResponseFormatter.format_response({
            "is_scam": is_scam_model,
            "label": "SCAM" if is_scam_model else "LEGITIMATE",
            "risk_score": model_risk,
            "scam_type": "qr_scam",
            "explanation": "Model analysis: " + ("Malicious QR code detected" if is_scam_model else "Legitimate QR code"),
            "qr_content": qr_content,
            "model_verified": False
        })

    except Exception as e:
        return ResponseFormatter.format_error_response(str(e))
