from pydantic import BaseModel
from typing import Literal

class ScamDetectionResponse(BaseModel):
    type_of_scam: Literal["url_scam", "qr_scam", "payment_request", "investment_scam"]
    risk_score: int
    label: Literal["SCAM", "SUSPICIOUS", "LEGITIMATE"]
    explanation: str
    source: Literal["llama", "ml_consensus", "llama_fallback"] = "llama"
