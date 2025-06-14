from pydantic import BaseModel
from typing import Optional

class ScamDetectionRequest(BaseModel):
    query: str
    
class SpecializedDetectionRequest(BaseModel):
    query: str
    llama_prediction: dict
