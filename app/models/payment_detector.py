from app.models.base_model import BaseScamDetector
import numpy as np

class PaymentScamDetector(BaseScamDetector):
    def __init__(self):
        super().__init__("ml_models/trained_models/payment_model.pkl")
    
    def preprocess(self, input_data: str) -> np.ndarray:
        """Preprocess text for payment scam detection"""
        payment_keywords = ['send money', 'transfer', 'payment', 'pay', 'upi', 'bank']
        urgency_keywords = ['urgent', 'emergency', 'help', 'immediately']
        
        features = {
            'mentions_payment': any(keyword in input_data.lower() for keyword in payment_keywords),
            'urgency_present': any(keyword in input_data.lower() for keyword in urgency_keywords),
            'personal_request': any(word in input_data.lower() for word in ['family', 'friend', 'relative']),
            'amount_mentioned': any(char.isdigit() for char in input_data)
        }
        
        return np.array([list(features.values())])
    
    def predict(self, input_data: str) -> dict:
        """Predict if payment request is scam"""
        if not self.model:
            payment_words = ['money', 'payment', 'transfer', 'send']
            urgency_words = ['urgent', 'emergency', 'help']
            
            has_payment = any(word in input_data.lower() for word in payment_words)
            has_urgency = any(word in input_data.lower() for word in urgency_words)
            
            is_scam = has_payment and has_urgency
            risk_score = 85 if is_scam else 15
            
            return {
                "prediction": "SCAM" if is_scam else "LEGITIMATE",
                "risk_score": risk_score,
                "confidence": 0.8
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
