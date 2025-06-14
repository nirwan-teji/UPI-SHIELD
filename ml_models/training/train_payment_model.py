import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle
import numpy as np
import re
import collections
import os
from pathlib import Path
import pickle

# 1. Load and preprocess payment scam dataset from project structure
data_path = "ml_models/data/datasets/payment_data.csv"
df = pd.read_csv(data_path)
df = df.drop_duplicates(subset=['text'])  # Remove duplicate messages

# Use existing label and text columns
texts = df['text'].astype(str).tolist()
labels = df['label'].astype(int).tolist()
texts, labels = shuffle(texts, labels, random_state=42)

# 2. Enhanced tokenizer for payment scams
def tokenize(text):
    text = text.lower()
    # Keep common payment-related symbols
    text = re.sub(r'[^a-z0-9₹$\@\-\+\:\/\s]', '', text)
    return text.split()

all_tokens = [token for text in texts for token in tokenize(text)]
vocab = ['<PAD>', '<UNK>'] + sorted(set(all_tokens))
word2idx = {word: idx for idx, word in enumerate(vocab)}
MODEL_SAVE_DIR = Path("ml_models/trained_models")
with open(MODEL_SAVE_DIR / "bilstm_vocab.pkl", "wb") as f:
    pickle.dump(word2idx, f)
print("💾 Saved vocabulary to bilstm_vocab.pkl")

def encode(text):
    return [word2idx.get(token, word2idx['<UNK>']) for token in tokenize(text)]

max_len = max(len(tokenize(text)) for text in texts)
def pad_sequence(seq, max_len):
    return seq + [word2idx['<PAD>']] * (max_len - len(seq))

encoded_texts = [pad_sequence(encode(text), max_len) for text in texts]

# 3. Stratified Split with No Overlap
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(encoded_texts, labels):
    X_train = [encoded_texts[i] for i in train_index]
    X_test = [encoded_texts[i] for i in test_index]
    y_train = [labels[i] for i in train_index]
    y_test = [labels[i] for i in test_index]

# 4. Check for overlap between train and test sets
train_set = set(tuple(x) for x in X_train)
test_set = set(tuple(x) for x in X_test)
overlap = train_set & test_set
print(f"Overlap between train and test sets: {len(overlap)}")

# 5. Print class distribution
print("Class distribution in full set:", collections.Counter(labels))
print("Class distribution in train set:", collections.Counter(y_train))
print("Class distribution in test set:", collections.Counter(y_test))

# 6. Prepare DataLoader
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float32))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 7. BiLSTM Model with Dropout (unchanged)
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=word2idx['<PAD>'])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = torch.cat((lstm_out[:, -1, :self.lstm.hidden_size], lstm_out[:, 0, self.lstm.hidden_size:]), dim=1)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits.squeeze(1)

# 8. Training Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = len(vocab)
embed_dim = 50
hidden_dim = 64
num_layers = 1

# Compute class weights for imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

model = BiLSTMClassifier(vocab_size, embed_dim, hidden_dim, num_layers).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 9. Training Loop
epochs = 10
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

# 10. Evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        preds = torch.sigmoid(logits).cpu().numpy() > 0.5
        all_preds.extend(preds.astype(int))
        all_labels.extend(y_batch.numpy().astype(int))

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
print(f"\nTest Accuracy: {acc:.4f}")
print(f"Test F1-score: {f1:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# 11. Save Model in project directory
model_save_path = "ml_models/trained_models/bilstm_payment_scam_classifier.pt"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved as {model_save_path}")

# 12. Payment scam inference example
def predict(text):
    encoded = pad_sequence(encode(text), max_len)
    tensor = torch.tensor([encoded], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).item()
        pred = int(prob > 0.5)
        # Adjust probability for class 0
        if pred == 0:
            prob = 1 - prob
    return pred, prob

sample_text = "Urgent! Send ₹5000 to UPI ID 1234567890@upi immediately for account verification."
pred, prob = predict(sample_text)
print(f"\nSample inference:")
print(f"Text: {sample_text}")
print(f"Prediction: {pred} (probability: {prob:.4f})")
