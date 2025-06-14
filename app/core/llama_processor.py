import json
import asyncio
from groq import Groq
from app.config import settings

class LlamaProcessor:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        
    async def analyze_scam(self, query: str) -> dict:
        prompt = f"""
        Analyze the following text for potential scam indicators and respond with ONLY a valid JSON object:

        Text: "{query}"

        Categorize into one of these scam types:
        - url_scam: involves clicking phishing links
        - qr_scam: involves scanning malicious QR codes  
        - payment_request: text requests asking for money
        - investment_scam: unrealistic investment promises

        Respond with this exact JSON structure:
        {{
            "type_of_scam": "one_of_four_types",
            "risk_score": 85,
            "label": "SCAM",
            "explanation": "detailed_reasoning_and_protective_measures"
        }}

        Rules:
        - SCAM: risk_score > 70
        - SUSPICIOUS: risk_score 40-70  
        - LEGITIMATE: risk_score < 40
        """
        
        try:
            completion = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            response_text = completion.choices[0].message.content.strip()
            
            # Simple JSON parsing without complex string handling
            return json.loads(response_text)
            
        except Exception as e:
            # Fallback response
            return {
                "type_of_scam": "url_scam",
                "risk_score": 50,
                "label": "SUSPICIOUS",
                "explanation": "Error in analysis. Please verify manually."
            }
