from fastapi import HTTPException, Depends
from app.config import settings
import time
from typing import Optional

class RateLimiter:
    def __init__(self):
        self.requests = {}
    
    def check_rate_limit(self, client_ip: str, max_requests: int = 10, window: int = 60):
        """Simple rate limiting"""
        current_time = time.time()
        
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        # Remove old requests outside the window
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip] 
            if current_time - req_time < window
        ]
        
        if len(self.requests[client_ip]) >= max_requests:
            raise HTTPException(
                status_code=429, 
                detail="Too many requests. Please try again later."
            )
        
        self.requests[client_ip].append(current_time)

rate_limiter = RateLimiter()

def get_rate_limiter():
    return rate_limiter

def validate_groq_api_key():
    """Validate that Groq API key is configured"""
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your_groq_api_key_here":
        raise HTTPException(
            status_code=500,
            detail="Groq API key not configured. Please set GROQ_API_KEY in .env file."
        )
    return True
