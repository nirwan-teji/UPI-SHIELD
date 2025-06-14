import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# 1. Configuration for URL scam
SEED = 42
BATCH_SIZE = 8
MAX_LENGTH = 256
LR = 2e-5
NUM_EPOCHS = 1
MODEL_NAME = "yiyanghkust/finbert-pretrain"
MODEL_SAVE_DIR = "ml_models/trained_models/finbert_url"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 2. Seeding for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 3. Dataset class
class ScamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 4. Model class
class FinBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes=2, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

# 5. Load URL scam data
df = pd.read_csv("ml_models/data/datasets/url_scam_data.csv")  # Changed dataset
texts = df['text'].values  # Fixed typo (was ttexts)
labels = df['label'].values

# 6. Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, random_state=SEED, stratify=labels
)

# 7. Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = FinBERTClassifier(MODEL_NAME, num_classes=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 8. Create DataLoaders
train_dataset = ScamDataset(X_train, y_train, tokenizer, MAX_LENGTH)
val_dataset = ScamDataset(X_val, y_val, tokenizer, MAX_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 9. Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
criterion = nn.CrossEntropyLoss()

# 10. Training loop
best_f1 = 0
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
    
    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f'Validation Accuracy: {accuracy:.4f} | F1-score: {f1:.4f}')
    
    if f1 > best_f1:
        best_f1 = f1
        # Save model and tokenizer
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'finbert_url_model.bin'))
        tokenizer.save_pretrained(MODEL_SAVE_DIR)
        print(f'Best model saved to {MODEL_SAVE_DIR} (F1: {best_f1:.4f})')

# 11. Prediction function
def predict_text(text, model, tokenizer, device):
    model.eval()
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    
    label = 'Scam' if pred == 1 else 'Legitimate'
    return label, confidence

# 12. Test prediction for URL scam
example_text = "Urgent: Your PayPal account needs verification! Click now: http://paypal-security-update.fake"
label, confidence = predict_text(example_text, model, tokenizer, device)
print(f'Prediction: {label} (Confidence: {confidence:.2f})')
