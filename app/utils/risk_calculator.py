from typing import Dict, List
import math

class RiskCalculator:
    def __init__(self):
        # Keyword weights for different scam types
        self.scam_keywords = {
            'url_scam': {
                'high_risk': ['click here', 'verify now', 'suspended', 'urgent action', 'limited time'],
                'medium_risk': ['click', 'verify', 'update', 'confirm', 'secure'],
                'low_risk': ['link', 'website', 'page']
            },
            'qr_scam': {
                'high_risk': ['scan qr', 'free gift', 'winner', 'prize'],
                'medium_risk': ['qr code', 'scan', 'download app'],
                'low_risk': ['barcode', 'code']
            },
            'payment_request': {
                'high_risk': ['send money', 'urgent help', 'emergency', 'transfer now'],
                'medium_risk': ['payment', 'transfer', 'money', 'help'],
                'low_risk': ['pay', 'amount', 'cost']
            },
            'investment_scam': {
                'high_risk': ['guaranteed profit', '100% returns', 'risk-free', 'double money'],
                'medium_risk': ['investment', 'profit', 'returns', 'trading'],
                'low_risk': ['invest', 'earn', 'income']
            }
        }
    
    def calculate_keyword_risk(self, text: str, scam_type: str) -> int:
        """Calculate risk based on keyword presence"""
        if scam_type not in self.scam_keywords:
            return 0
        
        text_lower = text.lower()
        risk_score = 0
        
        keywords = self.scam_keywords[scam_type]
        
        # High risk keywords
        for keyword in keywords.get('high_risk', []):
            if keyword in text_lower:
                risk_score += 25
        
        # Medium risk keywords
        for keyword in keywords.get('medium_risk', []):
            if keyword in text_lower:
                risk_score += 15
        
        # Low risk keywords
        for keyword in keywords.get('low_risk', []):
            if keyword in text_lower:
                risk_score += 5
        
        return min(risk_score, 100)
    
    def calculate_urgency_risk(self, text: str) -> int:
        """Calculate risk based on urgency indicators"""
        urgency_words = [
            'urgent', 'immediately', 'now', 'asap', 'emergency', 
            'expires', 'limited time', 'act fast', 'hurry'
        ]
        
        text_lower = text.lower()
        urgency_count = sum(1 for word in urgency_words if word in text_lower)
        
        return min(urgency_count * 15, 50)
    
    def calculate_structure_risk(self, text: str) -> int:
        """Calculate risk based on text structure"""
        risk_score = 0
        
        # Too many exclamation marks
        exclamation_count = text.count('!')
        if exclamation_count > 3:
            risk_score += 20
        elif exclamation_count > 1:
            risk_score += 10
        
        # All caps words
        words = text.split()
        caps_ratio = sum(1 for word in words if word.isupper() and len(word) > 2) / len(words) if words else 0
        if caps_ratio > 0.3:
            risk_score += 25
        elif caps_ratio > 0.1:
            risk_score += 15
        
        # Suspicious patterns
        if any(pattern in text.lower() for pattern in ['click here', 'act now', 'limited offer']):
            risk_score += 20
        
        return min(risk_score, 40)
    
    def calculate_combined_risk(self, text: str, scam_type: str, ml_confidence: float = 0.5) -> int:
        """Calculate combined risk score from multiple factors"""
        keyword_risk = self.calculate_keyword_risk(text, scam_type)
        urgency_risk = self.calculate_urgency_risk(text)
        structure_risk = self.calculate_structure_risk(text)
        
        # Weighted combination
        base_risk = (keyword_risk * 0.4 + urgency_risk * 0.3 + structure_risk * 0.3)
        
        # Adjust based on ML model confidence
        ml_adjustment = (ml_confidence - 0.5) * 40  # -20 to +20 adjustment
        
        final_risk = base_risk + ml_adjustment
        
        return max(0, min(100, int(final_risk)))
    
    def get_risk_label(self, risk_score: int) -> str:
        """Convert risk score to label"""
        if risk_score >= 70:
            return "SCAM"
        elif risk_score >= 40:
            return "SUSPICIOUS"
        else:
            return "LEGITIMATE"

# Global instance
risk_calculator = RiskCalculator()
