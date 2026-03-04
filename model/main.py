from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import torch
import torch.nn as nn
import numpy as np
from pythainlp.tokenize import word_tokenize

app = FastAPI(title="Fake News Detection API - BiLSTM")

# เปิด CORS เพื่อให้ Frontend เรียกได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define BiLSTM Model Class
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=True, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        out = self.relu(self.fc1(hidden))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        return out

# โหลดโมเดล
print("🔧 Loading BiLSTM model...")
device = torch.device('cpu')

word2idx = joblib.load("word2idx.pkl")
model_config = joblib.load("model_config.pkl")
MAX_LEN = model_config['max_len']

model = BiLSTMClassifier(
    vocab_size=len(word2idx),
    embedding_dim=128,
    hidden_dim=64,
    num_layers=2,
    dropout=0.3
).to(device)

model.load_state_dict(torch.load('lstm_model.pth', map_location=device, weights_only=True))
model.eval()

print("✅ Model loaded successfully!")

def text_to_sequence(tokens, word2idx, max_len):
    seq = [word2idx.get(word, 1) for word in tokens]
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq = seq + [0] * (max_len - len(seq))
    return seq

class News(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {
        "message": "Fake News Detection API - BiLSTM",
        "model": "Bidirectional LSTM with PyTorch",
        "accuracy": "98.90%",
        "status": "running",
        "endpoints": {
            "/predict": "POST - ตรวจสอบข่าวปลอม"
        }
    }

@app.post("/predict")
def predict(news: News):
    tokens = word_tokenize(news.text, engine='newmm')
    seq = np.array([text_to_sequence(tokens, word2idx, MAX_LEN)])
    seq_tensor = torch.LongTensor(seq).to(device)
    
    with torch.no_grad():
        pred = model(seq_tensor).item()
    
    # คำนวณ confidence และ label
    if pred > 0.5:
        confidence_percent = round(pred * 100, 1)
    else:
        confidence_percent = round((1 - pred) * 100, 1)
    
    decision_score = round((pred - 0.5) * 2, 3)
    
    return {
        "confidence": confidence_percent,
        "decision_score": decision_score
    }
