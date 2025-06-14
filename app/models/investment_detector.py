from app.models.base_model import BaseScamDetector
import numpy as np

class InvestmentScamDetector(BaseScamDetector):
    def __init__(self):
        super().__init__("ml_models/trained_models/investment_model.pkl")
    
    def preprocess(self, input_data: str) -> np.ndarray:
        """Preprocess text for investment scam detection"""
        investment_keywords = ['invest', 'profit', 'returns', 'trading', 'crypto', 'forex']
        unrealistic_promises = ['guaranteed', '100%', 'risk-free', 'double', 'triple']
        
        features = {
            'mentions_investment': any(keyword in input_data.lower() for keyword in investment_keywords),
            'unrealistic_promises': any(promise in input_data.lower() for promise in unrealistic_promises),
            'high_returns': any(word in input_data for word in ['%', 'percent', 'times', 'x']),
            'urgency': any(word in input_data.lower() for word in ['limited', 'expires', 'today only'])
        }
        
        return np.array([list(features.values())])
    
    def predict(self, input_data: str) -> dict:
        """Predict if investment offer is scam"""
        if not self.model:
            investment_words = ['invest', 'profit', 'returns', 'trading']
            scam_indicators = ['guaranteed', 'risk-free', '100%', 'double your money']
            
            has_investment = any(word in input_data.lower() for word in investment_words)
            has_scam_indicators = any(indicator in input_data.lower() for indicator in scam_indicators)
            
            is_scam = has_investment and has_scam_indicators
            risk_score = 90 if is_scam else 10
            
            return {
                "prediction": "SCAM" if is_scam else "LEGITIMATE",
                "risk_score": risk_score,
                "confidence": 0.9
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
