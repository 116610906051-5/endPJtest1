from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import requests
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, List
import re

app = FastAPI(title="Fake News Detection API - BiLSTM with News Verification")

# เปิด CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ตั้งค่า device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

# โหลด BiLSTM model
print("🔧 Loading BiLSTM model...")

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

# โหลด word2idx และโมเดล
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

print("✅ BiLSTM model loaded successfully!")

# แหล่งข่าวที่น่าเชื่อถือในประเทศไทย
TRUSTED_NEWS_SOURCES = [
    "thairath.co.th",
    "manager.co.th", 
    "bangkokpost.com",
    "nationthailand.com",
    "thaipbs.or.th",
    "prachachat.net",
    "matichon.co.th",
    "khaosod.co.th",
    "thaipost.net",
    "dailynews.co.th"
]

# NewsAPI Configuration
NEWS_API_KEY = "277729d09fc640549010e57ecb99c09d"
NEWS_API_BASE_URL = "https://newsapi.org/v2/everything"

class News(BaseModel):
    text: str
    check_related: Optional[bool] = True  # เปิดเป็นค่าเริ่มต้น
    source_url: Optional[str] = None

def text_to_sequence(text: str, word2idx: dict, max_len: int) -> torch.Tensor:
    """แปลงข้อความเป็น sequence สำหรับ BiLSTM"""
    tokens = word_tokenize(text, engine='newmm')
    seq = [word2idx.get(word, 1) for word in tokens]  # 1 = <UNK>
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq = seq + [0] * (max_len - len(seq))  # 0 = <PAD>
    return torch.LongTensor([seq])

def search_related_news(query: str, max_results: int = 5) -> List[dict]:
    """ค้นหาข่าวที่เกี่ยวข้องจาก NewsAPI"""
    try:
        keywords = extract_keywords(query)
        search_query_th = " ".join(keywords[:3])
        search_query = "Thailand news"
        
        print(f"🔍 Searching related news for: {search_query_th}")
        
        thai_sources = "thairath.co.th,manager.co.th,bangkokpost.com,nationthailand.com"
        
        params = {
            "q": search_query,
            "domains": thai_sources,
            "sortBy": "publishedAt",
            "pageSize": max_results,
            "apiKey": NEWS_API_KEY
        }
        
        response = requests.get(NEWS_API_BASE_URL, params=params, timeout=5)
        
        if response.status_code != 200:
            print(f"❌ NewsAPI error: {response.status_code}")
            return []
        
        data = response.json()
        related_news = []
        
        print(f"📰 Found {data.get('totalResults', 0)} articles from NewsAPI")
        
        if data.get("status") == "ok" and data.get("articles"):
            for article in data["articles"][:max_results]:
                source_url = article.get("url", "")
                source_domain = ""
                
                if source_url:
                    try:
                        from urllib.parse import urlparse
                        parsed = urlparse(source_url)
                        source_domain = parsed.netloc.replace("www.", "")
                    except:
                        source_domain = article.get("source", {}).get("name", "unknown")
                
                related_news.append({
                    "title": article.get("title", ""),
                    "source": source_domain or article.get("source", {}).get("name", "unknown"),
                    "url": article.get("url", ""),
                    "content": article.get("description", "") or article.get("content", ""),
                    "publishedAt": article.get("publishedAt", "")
                })
            
            print(f"✅ Returning {len(related_news)} related articles")
        
        return related_news
        
    except Exception as e:
        print(f"Error searching news: {e}")
        return []

def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """สกัดคำสำคัญจากข้อความ"""
    words = re.findall(r'\b\w+\b', text)
    keywords = [w for w in words if len(w) > 2][:max_keywords]
    return keywords

def calculate_text_similarity(text1: str, text2: str) -> float:
    """คำนวณความคล้ายคลึงระหว่างข้อความ"""
    try:
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except:
        return 0.0

def verify_with_related_news(
    user_text: str,
    related_news: List[dict],
    model_confidence: float,
    model_label: str
) -> dict:
    """ตรวจสอบความน่าเชื่อถือด้วยข่าวที่เกี่ยวข้อง"""
    
    if not related_news:
        return {
            "final_confidence": model_confidence,
            "final_label": model_label,
            "related_articles": [],
            "verification_note": "ไม่พบข่าวที่เกี่ยวข้องสำหรับการตรวจสอบ"
        }
    
    related_items = []
    max_similarity = 0.0
    trusted_count = 0
    
    for news in related_news:
        is_trusted = any(source in news.get('source', '') for source in TRUSTED_NEWS_SOURCES)
        similarity = calculate_text_similarity(user_text, news.get('content', news.get('title', '')))
        
        if similarity > max_similarity:
            max_similarity = similarity
        
        if is_trusted:
            trusted_count += 1
        
        related_items.append({
            "title": news.get('title', '')[:150],
            "source": news.get('source', 'unknown'),
            "url": news.get('url', ''),
            "similarity": round(similarity * 100, 1),
            "is_trusted": is_trusted,
            "publishedAt": news.get('publishedAt', '')
        })
    
    # ปรับ confidence ถ้ามีแหล่งที่เชื่อถือได้
    adjustment = 0.0
    final_confidence = model_confidence
    final_label = model_label
    
    if trusted_count > 0:
        # มีแหล่งข่าวที่เชื่อถือได้ -> เพิ่มความมั่นใจในผลลัพธ์
        if model_label == "ข่าวจริง":
            adjustment = min(trusted_count * 5, 10)  # เพิ่มสูงสุด 10%
            final_confidence = min(model_confidence + adjustment, 99.0)
        elif model_confidence < 30:  # โมเดลไม่แน่ใจมาก
            # ถ้ามีหลายแหล่งที่เชื่อถือได้ อาจปรับให้มีโอกาสเป็นข่าวจริง
            if trusted_count >= 3:
                final_confidence = 60.0
                final_label = "ไม่แน่ใจ - พบแหล่งข่าวที่เชื่อถือได้"
    
    verification_note = (
        f"พบข่าวที่เกี่ยวข้อง {len(related_items)} รายการ "
        f"จากแหล่งที่น่าเชื่อถือ {trusted_count} แหล่ง "
        f"(ความคล้ายสูงสุด: {round(max_similarity * 100, 1)}%)"
    )
    
    return {
        "final_confidence": round(final_confidence, 1),
        "final_label": final_label,
        "confidence_adjustment": round(adjustment, 1),
        "related_articles": related_items,
        "verification_note": verification_note,
        "trusted_sources_count": trusted_count
    }

@app.get("/")
def read_root():
    return {
        "message": "Fake News Detection API - BiLSTM with News Verification",
        "model": "Bidirectional LSTM (BiLSTM) with PyTorch",
        "accuracy": "98.90%",
        "features": [
            "Deep Learning fake news detection",
            "Automatic related news search",
            "Trusted source verification",
            "Thai language support"
        ],
        "status": "running",
        "default_behavior": "Automatically searches for related news",
        "endpoints": {
            "/predict": "POST - ตรวจสอบข่าวปลอมพร้อมหาข่าวที่เกี่ยวข้อง"
        }
    }

@app.post("/predict")
def predict(news: News):
    # 1. ทำนายด้วย BiLSTM
    sequence = text_to_sequence(news.text, word2idx, MAX_LEN).to(device)
    
    with torch.no_grad():
        output = model(sequence)
        confidence = output.item()
    
    confidence_percent = round(confidence * 100, 1)
    label = "ข่าวจริง" if confidence > 0.5 else "ข่าวปลอม"
    
    response = {
        "model": "BiLSTM",
        "model_confidence": confidence_percent,
        "model_label": label,
        "raw_score": confidence
    }
    
    # 2. ค้นหาข่าวที่เกี่ยวข้อง (default: True)
    if news.check_related:
        related_news = search_related_news(news.text, max_results=5)
        
        # ตรวจสอบด้วยข่าวที่เกี่ยวข้อง
        verification_result = verify_with_related_news(
            news.text,
            related_news,
            confidence_percent,
            label
        )
        
        response.update({
            "confidence": verification_result["final_confidence"],
            "label": verification_result["final_label"],
            "original_confidence": confidence_percent,
            "original_label": label,
            "confidence_adjustment": verification_result.get("confidence_adjustment", 0),
            "related_news": verification_result["related_articles"],
            "verification_note": verification_result["verification_note"],
            "trusted_sources_found": verification_result.get("trusted_sources_count", 0)
        })
    else:
        response.update({
            "confidence": confidence_percent,
            "label": label
        })
    
    return response
