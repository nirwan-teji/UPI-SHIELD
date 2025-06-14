import re
import string
from typing import List, Dict
import unicodedata

class TextPreprocessor:
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
        }
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if not text:
            return ""
        
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, text)
    
    def extract_phone_numbers(self, text: str) -> List[str]:
        """Extract phone numbers from text"""
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        return re.findall(phone_pattern, text)
    
    def extract_email_addresses(self, text: str) -> List[str]:
        """Extract email addresses from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(email_pattern, text)
    
    def count_suspicious_keywords(self, text: str, keywords: List[str]) -> int:
        """Count occurrences of suspicious keywords"""
        text_lower = text.lower()
        return sum(1 for keyword in keywords if keyword.lower() in text_lower)
    
    def get_text_features(self, text: str) -> Dict[str, any]:
        """Extract various text features for ML models"""
        cleaned_text = self.clean_text(text)
        
        return {
            'length': len(text),
            'word_count': len(text.split()),
            'char_count': len(cleaned_text),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
            'punctuation_ratio': sum(1 for c in text if c in string.punctuation) / len(text) if text else 0,
            'url_count': len(self.extract_urls(text)),
            'phone_count': len(self.extract_phone_numbers(text)),
            'email_count': len(self.extract_email_addresses(text)),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'has_currency_symbols': any(symbol in text for symbol in ['$', '₹', '€', '£', '¥'])
        }

# Global instance
text_preprocessor = TextPreprocessor()
