import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.config import model_config
import pickle

class ModelLoader:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.vocab = None

    def load_all(self):
        try:
            # --- FinBERT URL Model ---
            print("🔍 Loading FinBERT URL model...")
            self.models['url'] = AutoModelForSequenceClassification.from_pretrained(
                str(model_config.URL_MODEL_PATH.parent),
                ignore_mismatched_sizes=True,
                num_labels=2
            )
            self.tokenizers['url'] = AutoTokenizer.from_pretrained(
                str(model_config.URL_TOKENIZER_PATH)
            )

            # --- FinBERT Investment Model ---
            print("🔍 Loading FinBERT Investment model...")
            self.models['investment'] = AutoModelForSequenceClassification.from_pretrained(
                str(model_config.INVESTMENT_MODEL_PATH.parent),
                ignore_mismatched_sizes=True,
                num_labels=2
            )
            self.tokenizers['investment'] = AutoTokenizer.from_pretrained(
                str(model_config.INVESTMENT_TOKENIZER_PATH)
            )

            # --- BiLSTM Payment Model ---
            print("🔍 Loading BiLSTM Payment model...")
            self.models['payment'] = torch.load(
                str(model_config.PAYMENT_MODEL_PATH),
                map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )

            # Load BiLSTM vocab
            print("🔍 Loading BiLSTM vocab...")
            with open(model_config.VOCAB_PATH, "rb") as f:
                self.vocab = pickle.load(f)

            print("✅ All models (except YOLO) loaded successfully!")
            
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}\n"
                               "💡 Troubleshooting Tips:\n"
                               "1. Verify all model files exist in specified paths\n"
                               "2. Check FinBERT config.json matches your tokenizer\n"
                               "3. If retraining, use model.resize_token_embeddings()")

    def get_model(self, model_type: str):
        if model_type not in self.models:
            if model_type == 'qr':
                print("🐢 Lazy loading YOLO QR model...")
                from ultralytics import YOLO
                self.models['qr'] = YOLO(str(model_config.QR_MODEL_PATH))
            else:
                raise ValueError(f"Model '{model_type}' not found and is not supported for lazy loading.")
        return self.models[model_type]

    def get_tokenizer(self, model_type: str):
        if model_type not in self.tokenizers:
            raise ValueError(f"Tokenizer for '{model_type}' not found. Available: {list(self.tokenizers.keys())}")
        return self.tokenizers[model_type]

    def get_vocab(self):
        if self.vocab is None:
            raise ValueError("BiLSTM vocabulary not loaded. Check VOCAB_PATH in config.")
        return self.vocab

# ✅ Global instance
model_loader = ModelLoader()
