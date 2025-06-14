from app.models.base_model import BaseScamDetector
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np

class URLScamDetector(BaseScamDetector):
    def __init__(self):
        super().__init__("ml_models/trained_models/url_scam_model.pkl")
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    def preprocess(self, input_data: str) -> np.ndarray:
        """Preprocess text for URL scam detection"""
        # Extract URLs and suspicious patterns
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, input_data)
        
        # Feature engineering
        features = {
            'has_url': len(urls) > 0,
            'url_count': len(urls),
            'has_suspicious_words': any(word in input_data.lower() for word in 
                ['click here', 'urgent', 'verify', 'suspended', 'limited time']),
            'text_length': len(input_data)
        }
        
        # Convert to vector (simplified for demo)
        return np.array([list(features.values())])
    
    def predict(self, input_data: str) -> dict:
        """Predict if URL is scam"""
        if not self.model:
            # Fallback prediction based on keywords
            suspicious_keywords = ['click here', 'verify account', 'suspended', 'urgent action']
            is_scam = any(keyword in input_data.lower() for keyword in suspicious_keywords)
            risk_score = 80 if is_scam else 20
            
            return {
                "prediction": "SCAM" if is_scam else "LEGITIMATE",
                "risk_score": risk_score,
                "confidence": 0.7
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
