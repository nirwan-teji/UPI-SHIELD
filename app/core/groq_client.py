from groq import Groq
from app.config import settings
from typing import List, Dict, Any

# app/core/groq_client.py
class GroqClient:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    async def generate_response(
        self,
        messages: List[Dict[str, str]],  # Must be List[Dict]
        model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        max_tokens: int = 1024
    ) -> str:
        try:
            completion = self.client.chat.completions.create(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=0.7
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Groq API error: {str(e)}")
