from app.models.url_detector import URLScamDetector
from app.models.qr_detector import QRScamDetector
from app.models.payment_detector import PaymentScamDetector
from app.models.investment_detector import InvestmentScamDetector
from app.config import settings

class ConsensusEngine:
    def __init__(self):
        self.detectors = {
            "url_scam": URLScamDetector(),
            "qr_scam": QRScamDetector(),
            "payment_request": PaymentScamDetector(),
            "investment_scam": InvestmentScamDetector()
        }
    
    def get_ml_prediction(self, scam_type: str, query: str) -> dict:
        """Get prediction from appropriate ML model"""
        if scam_type not in self.detectors:
            return None
        
        detector = self.detectors[scam_type]
        return detector.predict(query)
    
    def create_consensus(self, llama_result: dict, ml_result: dict, query: str) -> dict:
        """Create consensus between LLaMA and ML model predictions"""
        
        if not ml_result:
            return {
                **llama_result,
                "source": "llama_fallback"
            }
        
        llama_prediction = llama_result.get("label", "SUSPICIOUS")
        ml_prediction = ml_result.get("prediction", "SUSPICIOUS")
        
        # Check if predictions agree
        llama_is_scam = llama_prediction == "SCAM"
        ml_is_scam = ml_prediction == "SCAM"
        
        if llama_is_scam == ml_is_scam:
            # Predictions agree - use ML risk score with LLaMA explanation
            risk_score = ml_result.get("risk_score", llama_result.get("risk_score", 50))
            
            # Determine label based on risk score
            if risk_score >= settings.SCAM_THRESHOLD:
                label = "SCAM"
            elif risk_score >= settings.SUSPICIOUS_THRESHOLD:
                label = "SUSPICIOUS"
            else:
                label = "LEGITIMATE"
            
            return {
                "type_of_scam": llama_result.get("type_of_scam"),
                "risk_score": risk_score,
                "label": label,
                "explanation": llama_result.get("explanation"),
                "source": "ml_consensus"
            }
        else:
            # Predictions disagree - use LLaMA result
            return {
                **llama_result,
                "source": "llama_fallback"
            }
