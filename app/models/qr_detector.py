from app.models.base_model import BaseScamDetector
import numpy as np

class QRScamDetector(BaseScamDetector):
    def __init__(self):
        super().__init__("ml_models/trained_models/qr_scam_model.pkl")
    
    def preprocess(self, input_data: str) -> np.ndarray:
        """Preprocess text for QR scam detection"""
        qr_keywords = ['qr code', 'scan', 'qr', 'barcode']
        suspicious_contexts = ['payment', 'verify', 'download app', 'free gift']
        
        features = {
            'mentions_qr': any(keyword in input_data.lower() for keyword in qr_keywords),
            'suspicious_context': any(context in input_data.lower() for context in suspicious_contexts),
            'urgency_words': any(word in input_data.lower() for word in ['urgent', 'expires', 'limited']),
            'text_length': len(input_data)
        }
        
        return np.array([list(features.values())])
    
    def predict(self, input_data: str) -> dict:
        """Predict if QR code request is scam"""
        if not self.model:
            qr_mentioned = 'qr' in input_data.lower() or 'scan' in input_data.lower()
            suspicious_words = ['free', 'prize', 'winner', 'urgent']
            has_suspicious = any(word in input_data.lower() for word in suspicious_words)
            
            is_scam = qr_mentioned and has_suspicious
            risk_score = 75 if is_scam else 25
            
            return {
                "prediction": "SCAM" if is_scam else "LEGITIMATE",
                "risk_score": risk_score,
                "confidence": 0.6
            }
        
        features = self.preprocess(input_data)
        prediction = self.model.predict(features)[0]
        prediction_proba = self.model.predict_proba(features)[0]
        
        is_scam = prediction == 1
        risk_score = self.calculate_risk_score(prediction_proba, is_scam)
        
        return {
            "prediction": "SCAM" if is_scam else "LEGITIMATE",
            "risk_score": risk_score,
            "confidence": max(prediction_proba)
        }
