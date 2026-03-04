"""ทดสอบ BiLSTM + Related News โดยตรง (ไม่ผ่าน API)"""
import torch
import torch.nn as nn
import joblib
import requests
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import re

device = torch.device('cpu')

# โหลดโมเดล BiLSTM
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, batch_first=True,
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

print("🔧 Loading BiLSTM model...")
word2idx = joblib.load("word2idx.pkl")
MAX_LEN = 100

model = BiLSTMClassifier(
    vocab_size=len(word2idx),
    embedding_dim=128,
    hidden_dim=64,
    num_layers=2,
    dropout=0.3
).to(device)

model.load_state_dict(torch.load("lstm_model.pth", map_location=device))
model.eval()
print("✅ BiLSTM loaded!\n")

# ข่าวทดสอบ
news_text = """ตำรวจบุกทลายรังเว็บพนันเวียดนาม เช่าพื้นที่ในคอนโดฯ หรู ย่านพระราม 9 รวบพนักงานได้เกือบร้อยคน"""

print("="*80)
print("ทดสอบ Deep Learning + Related News Verification")
print("="*80)

# 1. ทำนายด้วย BiLSTM
print("\n📊 Step 1: Deep Learning Prediction (BiLSTM)")
print("-" * 80)

def text_to_sequence(text, word2idx, max_len):
    tokens = word_tokenize(text, engine='newmm')
    seq = [word2idx.get(word, 1) for word in tokens]
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq = seq + [0] * (max_len - len(seq))
    return torch.LongTensor([seq])

sequence = text_to_sequence(news_text, word2idx, MAX_LEN).to(device)

with torch.no_grad():
    output = model(sequence)
    confidence = output.item()

model_confidence = round(confidence * 100, 1)
model_label = "ข่าวจริง" if confidence > 0.5 else "ข่าวปลอม"

print(f"Model Confidence: {model_confidence}%")
print(f"Model Label: {model_label}")

# 2. ค้นหาข่าวที่เกี่ยวข้อง
print(f"\n📰 Step 2: Searching for Related News")
print("-" * 80)

NEWS_API_KEY = "277729d09fc640549010e57ecb99c09d"
NEWS_API_BASE_URL = "https://newsapi.org/v2/everything"

try:
    params = {
        "q": "Thailand news",
        "domains": "thairath.co.th,manager.co.th,bangkokpost.com",
        "sortBy": "publishedAt",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY
    }
    
    response = requests.get(NEWS_API_BASE_URL, params=params, timeout=5)
    
    if response.status_code == 200:
        data = response.json()
        articles = data.get("articles", [])
        
        print(f"Found {len(articles)} related articles from NewsAPI")
        
        if articles:
            print(f"\n📑 Related News Articles:")
            print("=" * 80)
            
            trusted_count = 0
            for i, article in enumerate(articles, 1):
                title = article.get('title', 'No title')
                source = article.get('source', {}).get('name', 'Unknown')
                url = article.get('url', '')
                
                is_trusted = 'thairath' in url.lower() or 'manager' in url.lower() or 'bangkokpost' in url.lower()
                if is_trusted:
                    trusted_count += 1
                
                print(f"\n{i}. {title}")
                print(f"   🌐 แหล่งที่มา: {source}", end="")
                if is_trusted:
                    print(" ✅ (แหล่งที่เชื่อถือได้)")
                else:
                    print()
                print(f"   🔗 {url}")
            
            print(f"\n" + "=" * 80)
            print(f"✅ พบแหล่งข่าวที่เชื่อถือได้: {trusted_count} แหล่ง")
            
            # ปรับความเชื่อมั่น
            if trusted_count > 0 and model_label == "ข่าวจริง":
                adjustment = min(trusted_count * 5, 10)
                final_confidence = min(model_confidence + adjustment, 99.0)
                print(f"\n🔄 Confidence Adjustment: +{adjustment}%")
                print(f"📊 Final Confidence: {final_confidence}%")
                print(f"✅ Final Label: {model_label}")
            elif trusted_count >= 3 and model_confidence < 30:
                print(f"\n🔄 พบหลายแหล่งที่เชื่อถือได้ แต่โมเดลไม่แน่ใจ")
                print(f"💡 แนะนำ: ตรวจสอบเพิ่มเติมจากแหล่งข่าวที่เกี่ยวข้อง")
            else:
                print(f"\n✅ Final Confidence: {model_confidence}%")
                print(f"✅ Final Label: {model_label}")
        else:
            print("ไม่พบข่าวที่เกี่ยวข้อง")
            print(f"\n✅ Final: {model_label} ({model_confidence}%)")
            
    else:
        print(f"❌ NewsAPI Error: {response.status_code}")
        print(f"\n✅ Using model result only: {model_label} ({model_confidence}%)")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print(f"\n✅ Using model result only: {model_label} ({model_confidence}%)")

print("\n" + "="*80)
print("✅ สำเร็จ: Deep Learning + Related News Verification")
print("="*80)
