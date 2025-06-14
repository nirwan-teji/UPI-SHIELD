class ResponseFormatter:
    @staticmethod
    def format_response(data: dict):
        return {
            "is_scam": data.get("is_scam", False),
            "risk_score": int(data.get("risk_score", 0)),
            "scam_type": data.get("scam_type") or data.get("type_of_scam") or data.get("label", "Unknown"),
            "label": data.get("label", "LEGITIMATE"),
            "explanation": data.get("explanation", ""),
            "model_verified": data.get("model_verified", False)
        }

    @staticmethod
    def format_error_response(error_message: str) -> dict:
        return {
            "is_scam": False,
            "risk_score": 0,
            "scam_type": "Unknown",
            "label": "LEGITIMATE",
            "explanation": f"Error: {error_message}",
            "model_verified": False
        }
