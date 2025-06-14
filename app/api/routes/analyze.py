import json
from fastapi import APIRouter, Form, UploadFile, File, HTTPException
import httpx
import time
import base64
import cv2
import numpy as np
from typing import Optional
from app.core.groq_client import GroqClient
from app.core.response_formatter import ResponseFormatter

router = APIRouter(tags=["Unified Analysis"])

async def classify_scam_type(text: str) -> str:
    """Classify scam type using Groq LLM"""
    groq_client = GroqClient()
    messages = [
        {
            "role": "system",
            "content": """Classify text into exactly one category:
            - url_scam (phishing links)
            - qr_scam (malicious QR codes)
            - payment_request (direct money requests)
            - investment_scam (unrealistic returns)
            Return ONLY the lowercase category name."""
        },
        {
            "role": "user",
            "content": f"TEXT: {text}"
        }
    ]
    try:
        response = await groq_client.generate_response(messages)
        return response.strip().lower()
    except Exception as e:
        raise HTTPException(500, f"Classification failed: {str(e)}")

async def is_qr_code(file: UploadFile) -> bool:
    """Detect if the uploaded image contains a QR code using OpenCV."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        detector = cv2.QRCodeDetector()
        qr_content, _, _ = detector.detectAndDecode(img)
        file.file.seek(0)  # Reset file pointer for future reads
        return bool(qr_content)
    except Exception as e:
        print(f"QR detection error: {str(e)}")
        file.file.seek(0)
        return False

async def analyze_image_content(file: UploadFile) -> dict:
    """Analyze image using Groq vision model"""
    groq_client = GroqClient()
    image_data = await file.read()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    response = await groq_client.generate_response(
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image for scam indicators"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{file.content_type};base64,{base64_image}"
                    }
                }
            ]
        }]
    )
    try:
        return json.loads(response.strip('` \n'))
    except Exception as e:
        print(f"Groq image JSON parse error: {str(e)}")
        return {"error": "Failed to parse Groq response"}

@router.post("/analyze")
async def unified_analysis(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    start_time = time.time()
    try:
        # Handle image upload
        if file and file.filename and file.size > 0:
            # Check for QR code
            if await is_qr_code(file):
                file.file.seek(0)  # Reset file pointer for QR route
                async with httpx.AsyncClient(timeout=60) as client:
                    response = await client.post(
                        "http://localhost:8000/predict/qr",
                        files={"file": (file.filename, await file.read())}
                    )
                qr_result = response.json()
                return ResponseFormatter.format_response(qr_result)
            else:
                # General image scam analysis
                file.file.seek(0)
                groq_result = await analyze_image_content(file)
                return ResponseFormatter.format_response({
                    "is_scam": groq_result.get("label", "").upper() == "SCAM",
                    "label": groq_result.get("label", "UNKNOWN"),
                    "risk_score": groq_result.get("risk_score", 0),
                    "scam_type": "image_scam",
                    "explanation": groq_result.get("explanation", "Image analysis failed"),
                    "model_verified": False
                })

        # Handle text input
        if not text or not text.strip():
            return ResponseFormatter.format_error_response("No input provided")

        # Classify scam type
        scam_type = await classify_scam_type(text)
        endpoints = {
            "url_scam": "/predict/url",
            "payment_request": "/predict/payment",
            "investment_scam": "/predict/investment"
        }
        if scam_type not in endpoints:
            return ResponseFormatter.format_error_response(f"Unsupported scam type: {scam_type}")

        # Forward request to specialized endpoint
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"http://localhost:8000{endpoints[scam_type]}",
                json={"text": text}
            )
        return response.json()

    except httpx.ReadTimeout:
        return ResponseFormatter.format_error_response("Analysis timed out (60s limit)")
    except HTTPException as he:
        return ResponseFormatter.format_error_response(str(he.detail))
    except Exception as e:
        return ResponseFormatter.format_error_response(str(e))
    finally:
        print(f"⏱️ Total processing time: {time.time() - start_time:.2f}s")
