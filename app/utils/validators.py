import re
from typing import Optional
from urllib.parse import urlparse

class InputValidator:
    def __init__(self):
        self.max_text_length = 5000
        self.min_text_length = 5
    
    def validate_query_text(self, text: str) -> tuple[bool, Optional[str]]:
        """Validate input query text"""
        if not text or not text.strip():
            return False, "Query text cannot be empty"
        
        if len(text) < self.min_text_length:
            return False, f"Query text must be at least {self.min_text_length} characters"
        
        if len(text) > self.max_text_length:
            return False, f"Query text cannot exceed {self.max_text_length} characters"
        
        # Check for potentially malicious content
        if self._contains_malicious_patterns(text):
            return False, "Query contains potentially malicious content"
        
        return True, None
    
    def _contains_malicious_patterns(self, text: str) -> bool:
        """Check for basic malicious patterns"""
        malicious_patterns = [
            r'<script.*?>.*?</script>',  # Script tags
            r'javascript:',              # JavaScript URLs
            r'data:text/html',          # Data URLs
            r'vbscript:',               # VBScript
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL) for pattern in malicious_patterns)
    
    def validate_url(self, url: str) -> tuple[bool, Optional[str]]:
        """Validate URL format"""
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                return False, "Invalid URL format"
            
            if result.scheme not in ['http', 'https']:
                return False, "URL must use HTTP or HTTPS protocol"
            
            return True, None
        except Exception:
            return False, "Invalid URL format"
    
    def validate_scam_type(self, scam_type: str) -> tuple[bool, Optional[str]]:
        """Validate scam type"""
        valid_types = ['url_scam', 'qr_scam', 'payment_request', 'investment_scam']
        
        if scam_type not in valid_types:
            return False, f"Invalid scam type. Must be one of: {', '.join(valid_types)}"
        
        return True, None
    
    def validate_risk_score(self, risk_score: int) -> tuple[bool, Optional[str]]:
        """Validate risk score"""
        if not isinstance(risk_score, int):
            return False, "Risk score must be an integer"
        
        if not 0 <= risk_score <= 100:
            return False, "Risk score must be between 0 and 100"
        
        return True, None

# Global instance
input_validator = InputValidator()
