# app/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Existing Settings class
class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    SCAM_THRESHOLD: int = 70
    SUSPICIOUS_THRESHOLD: int = 40
    SCAM_TYPES = {
        "url_scam": "URL/Phishing Link Scam",
        "qr_scam": "QR Code Scam", 
        "payment_request": "Payment Request Scam",
        "investment_scam": "Investment Scam"
    }

# New ModelConfig class

class ModelConfig:
    BASE_DIR = Path(__file__).resolve().parent.parent
    URL_MODEL_PATH = BASE_DIR / "ml_models/trained_models/finbert_url/finbert_url_model.bin"
    URL_TOKENIZER_PATH = BASE_DIR / "ml_models/trained_models/finbert_url"
    INVESTMENT_MODEL_PATH = BASE_DIR / "ml_models/trained_models/finbert_investment/finbert_investment_model.bin"
    INVESTMENT_TOKENIZER_PATH = BASE_DIR / "ml_models/trained_models/finbert_investment"
    QR_MODEL_PATH = BASE_DIR / "ml_models/trained_models/yolo_qr_subset/best.pt"
    PAYMENT_MODEL_PATH = BASE_DIR / "ml_models/trained_models/bilstm_payment_scam_classifier.pt"
    VOCAB_PATH = BASE_DIR / "ml_models/trained_models/bilstm_vocab.pkl"

# Instantiate both configurations
settings = Settings()
model_config = ModelConfig()
