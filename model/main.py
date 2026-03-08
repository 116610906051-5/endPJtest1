from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import math
import requests
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, List
import re

# Version: 1.0.1 - BiLSTM with SearchAPI and Thai keyword support
app = FastAPI(title="Fake News Detection API - BiLSTM with SearchAPI")

# เปิด CORS เพื่อให้ Frontend เรียกได้
# Updated: Added check_related flag support for SearchAPI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ตั้งค่า device (CPU-only)
device = torch.device('cpu')
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

# โหลด word2idx
word2idx = joblib.load("word2idx.pkl")
MAX_LEN = 100

# สร้างและโหลด model
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
print(f"📊 Model: BiLSTM (Bidirectional LSTM)")
print(f"📊 Accuracy: 98.90%")

def text_to_sequence(text: str, word2idx: dict, max_len: int) -> torch.Tensor:
    """แปลงข้อความเป็น sequence สำหรับ BiLSTM"""
    tokens = word_tokenize(text, engine='newmm')
    seq = [word2idx.get(word, 1) for word in tokens]  # 1 = <UNK>
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq = seq + [0] * (max_len - len(seq))  # 0 = <PAD>
    return torch.LongTensor([seq]).to(device)

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

# SearchAPI Configuration (ใช้แทน NewsAPI - มีข่าวไทยเยอะกว่า)
SEARCHAPI_KEY = "jkiyja5rtFwfdQYjXh5nLLjC"
SEARCHAPI_URL = "https://www.searchapi.io/api/v1/search"

class News(BaseModel):
    text: str
    check_related: Optional[bool] = True
    source_url: Optional[str] = None  # URL ของข่าว (ถ้ามี)

TRUSTED_SOURCES = {
    "thairath.co.th",
    "manager.co.th", 
    "bangkokpost.com",
    "nationthailand.com",
    "thaipbs.or.th",
    "matichon.co.th",
    "khaosod.co.th"
}

def verify_url(url: str) -> dict:
    """ตรวจสอบว่า URL มาจากแหล่งที่น่าเชื่อถือและเข้าถึงได้"""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        
        # เช็คว่ามาจากแหล่งที่น่าเชื่อถือหรือไม่
        is_trusted = domain in TRUSTED_SOURCES
        
        # ลองเข้าถึง URL (HEAD request เพื่อประหยัด bandwidth)
        response = requests.head(url, timeout=5, allow_redirects=True)
        is_accessible = response.status_code == 200
        
        return {
            "domain": domain,
            "is_trusted": is_trusted,
            "is_accessible": is_accessible,
            "verified": is_trusted and is_accessible
        }
    except Exception as e:
        print(f"❌ Error verifying URL: {e}")
        return {
            "domain": "",
            "is_trusted": False,
            "is_accessible": False,
            "verified": False
        }

class RelatedNews(BaseModel):
    title: str
    source: str
    url: str
    similarity: float
    is_trusted: bool

def search_related_news(query: str, max_results: int = 5) -> List[dict]:
    """ค้นหาข่าวที่เกี่ยวข้องจาก SearchAPI (รองรับข่าวไทย)"""
    try:
        print(f"\n🔍 Starting search_related_news...")
        print(f"📝 Input query: {query}")
        
        # สกัด keywords สำหรับค้นหา
        keywords = extract_keywords(query)
        search_query = " ".join(keywords[:3])  # ใช้ 3 keywords แรก
        
        print(f"🔍 Searching SearchAPI for: {search_query}")
        
        # เรียก SearchAPI
        params = {
            "engine": "google_news",
            "q": search_query,
            "gl": "th",  # ประเทศไทย
            "hl": "th",  # ภาษาไทย
            "num": max_results * 2,  # ขอมากกว่าเพื่อกรอง
            "api_key": SEARCHAPI_KEY
        }
        
        response = requests.get(SEARCHAPI_URL, params=params, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ SearchAPI error: {response.status_code}")
            return []
        
        data = response.json()
        related_news = []
        
        # SearchAPI คืนผลลัพธ์ใน data['news_results']
        if data.get("news_results"):
            print(f"📰 Found {len(data['news_results'])} results from SearchAPI")
            
            for article in data["news_results"][:max_results]:
                # ดึง domain จาก link
                source_url = article.get("link", "")
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
                    "url": source_url,
                    "content": article.get("snippet", "")
                })
            
            print(f"✅ Processed {len(related_news)} related articles")
        else:
            print(f"⚠️ No news_results found in SearchAPI response")
        
        return related_news
        
    except requests.Timeout:
        print("⏱️ SearchAPI timeout")
        return []
    except Exception as e:
        print(f"❌ Error searching news: {e}")
        return []

def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """สกัดคำสำคัญจากข้อความ (รองรับภาษาไทย)"""
    # ใช้ pythainlp สำหรับภาษาไทย
    words = word_tokenize(text, engine='newmm')
    # กรองคำที่ยาวกว่า 2 ตัวอักษร และไม่ใช่เครื่องหมาย
    keywords = [w for w in words if len(w) > 2 and not w.isspace()][:max_keywords]
    print(f"🔑 Extracted keywords: {keywords}")
    return keywords

def calculate_text_similarity(text1: str, text2: str) -> float:
    """คำนวณความคล้ายคลึงระหว่างข้อความ 2 ข้อความ"""
    try:
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except:
        return 0.0

def adjust_confidence_with_related_news(
    original_confidence: float,
    related_news: List[dict],
    user_text: str
) -> tuple:
    """ปรับ confidence ตามข่าวที่เกี่ยวข้อง"""
    
    if not related_news:
        return original_confidence, []
    
    related_items = []
    max_similarity = 0.0
    trusted_count = 0
    
    for news in related_news:
        # ตรวจสอบว่าเป็นแหล่งที่น่าเชื่อถือหรือไม่
        is_trusted = any(source in news.get('source', '') for source in TRUSTED_NEWS_SOURCES)
        
        # คำนวณความคล้ายคลึง
        similarity = calculate_text_similarity(user_text, news.get('content', news.get('title', '')))
        
        if similarity > max_similarity:
            max_similarity = similarity
        
        if is_trusted:
            trusted_count += 1
        
        related_items.append({
            "title": news.get('title', '')[:100],
            "source": news.get('source', 'unknown'),
            "url": news.get('url', ''),
            "similarity": round(similarity * 100, 1),
            "is_trusted": is_trusted
        })
    
    # ปรับ confidence
    # ถ้ามีข่าวจากแหล่งที่น่าเชื่อถือและคล้ายคลึงกัน -> เพิ่ม confidence
    adjustment = 0.0
    
    if trusted_count > 0 and max_similarity > 0.3:
        # เพิ่ม confidence ตามจำนวนแหล่งที่เชื่อถือและความคล้ายคลึง
        adjustment = min(trusted_count * max_similarity * 10, 15)  # เพิ่มสูงสุด 15%
    
    adjusted_confidence = min(original_confidence + adjustment, 99.9)
    
    return adjusted_confidence, related_items

@app.get("/")
def read_root():
    return {
        "message": "Fake News Detection API - Stacking Ensemble",
        "model": "Stacking (XGBoost + Random Forest + SVM + Logistic Regression)",
        "accuracy": "85.19%",
        "features": [
            "Fake news detection",
            "Related news verification",
            "Trusted source checking"
        ],
        "status": "running",
        "endpoints": {
            "/predict": "POST - ตรวจสอบข่าวปลอม (with optional related news check)"
        }
    }

@app.post("/predict")
def predict(news: News):
    print(f"\n{'='*50}")
    print(f"🆕 New prediction request")
    print(f"📝 Text length: {len(news.text)} chars")
    print(f"✅ check_related flag: {news.check_related}")
    print(f"{'='*50}\n")
    
    # ตรวจสอบ URL ถ้ามี
    url_verification = None
    url_override = False
    
    if news.source_url:
        url_verification = verify_url(news.source_url)
        print(f"🔗 URL Verification: {url_verification}")
        
        # ถ้า URL ยืนยันได้ว่ามาจากแหล่งที่น่าเชื่อถือและเข้าถึงได้
        if url_verification.get("verified"):
            url_override = True
            print(f"✅ Verified URL from trusted source: {url_verification.get('domain')}")
    
    # ทำนายด้วย BiLSTM
    sequence = text_to_sequence(news.text, word2idx, MAX_LEN)
    
    with torch.no_grad():
        output = model(sequence)
        probability = output.item()
    
    # แปลงเป็นความเชื่อมั่นและ label
    confidence_percent = round(probability * 100, 1)
    decision_score = round(probability * 2 - 1, 3)  # แปลงจาก [0,1] เป็น [-1,1]
    label = "ข่าวจริง" if probability > 0.5 else "ข่าวปลอม"
    
    response = {
        "confidence": confidence_percent,
        "decision_score": decision_score,
        "label": label,
        "raw_score": probability
    }
    
    # ถ้ามี URL ที่ยืนยันได้จากแหล่งที่เชื่อถือ -> ปรับผลลัพธ์
    if url_override and url_verification:
        original_confidence = confidence_percent
        original_label = label
        
        # ปรับเป็นข่าวจริงด้วยความเชื่อมั่นสูง
        response["confidence"] = 95.0
        response["label"] = "ข่าวจริง"
        response["original_confidence"] = original_confidence
        response["original_label"] = original_label
        response["url_verification"] = url_verification
        response["override_reason"] = f"ยืนยันจาก URL แหล่งที่เชื่อถือได้: {url_verification['domain']}"
        print(f"🔄 Override: {original_label} ({original_confidence}%) -> ข่าวจริง (95.0%) ด้วย URL verification")
    
    # ถ้าต้องการตรวจสอบข่าวที่เกี่ยวข้อง
    if news.check_related:
        print(f"\n✅ check_related is True, searching for related news...")
        related_news = search_related_news(news.text)
        print(f"📊 Search returned {len(related_news)} articles")
        
        if related_news:
            adjusted_confidence, related_items = adjust_confidence_with_related_news(
                confidence_percent,
                related_news,
                news.text
            )
            
            response["confidence"] = round(adjusted_confidence, 1)
            response["original_confidence"] = confidence_percent
            response["confidence_adjustment"] = round(adjusted_confidence - confidence_percent, 1)
            response["related_news"] = related_items
            response["verification_note"] = (
                f"พบข่าวที่เกี่ยวข้อง {len(related_items)} รายการ "
                f"จากแหล่งที่น่าเชื่อถือ {sum(1 for r in related_items if r['is_trusted'])} แหล่ง"
            )
            print(f"✅ Added {len(related_items)} related news items to response")
        else:
            print(f"⚠️ No related news found")
    else:
        print(f"⚠️ check_related is False, skipping news search")
    
    return response
