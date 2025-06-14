from abc import ABC, abstractmethod
import joblib
import numpy as np
from typing import Dict, Any
import os

class BaseScamDetector(ABC):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained ML model"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            print(f"Warning: Model file {self.model_path} not found")
    
    @abstractmethod
    def preprocess(self, input_data: str) -> np.ndarray:
        """Each model implements its own preprocessing"""
        pass
    
    @abstractmethod
    def predict(self, input_data: str) -> Dict[str, Any]:
        """Each model implements its own prediction logic"""
        pass
    
    def get_prediction_confidence(self, prediction_proba):
        """Common method to calculate confidence scores"""
        if hasattr(prediction_proba, '__iter__'):
            return max(prediction_proba) * 100
        return prediction_proba * 100
    
    def calculate_risk_score(self, prediction_proba, is_scam: bool) -> int:
        """Convert prediction probability to risk score"""
        confidence = self.get_prediction_confidence(prediction_proba)
        if is_scam:
            return min(100, int(50 + confidence/2))
        else:
            return max(0, int(50 - confidence/2))
