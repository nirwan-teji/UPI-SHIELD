import torch
import torch.nn as nn

class BiLSTMPaymentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=50, hidden_dim=64, num_layers=1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        # Take last and first time step of bidirectional LSTM
        out = torch.cat(
            (lstm_out[:, -1, :self.lstm.hidden_size], lstm_out[:, 0, self.lstm.hidden_size:]),
            dim=1
        )
        out = self.dropout(out)
        logits = self.fc(out)
        return logits.squeeze(1)
