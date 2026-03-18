from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import joblib
import math
import json
import requests
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, List
import re
import os
from pathlib import Path
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from itertools import groupby

# Version: 2.0.0 - WangchanBERTa (fine-tuned) with BiLSTM fallback
app = FastAPI(title="Fake News Detection API - WangchanBERTa with SearchAPI")

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
BASE_DIR = Path(__file__).resolve().parent


def resolve_asset_file(filename: str) -> Path:
    """Resolve model artifact path from common deployment locations."""
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "model" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    # fallback to BASE_DIR for clear error path
    return BASE_DIR / filename


def load_cnn_calibration(model_dir: Path) -> dict:
    """Load CNN calibration config from cnn_calibration.json with safe defaults."""
    calibration_path = resolve_asset_file("cnn_calibration.json")
    default_temperature = float(os.getenv("CNN_TEMPERATURE", "2.2"))
    default_threshold = float(os.getenv("CNN_DECISION_THRESHOLD", "0.5"))

    if not calibration_path.exists():
        print(f"ℹ️ Calibration file not found, using defaults: {calibration_path}")
        return {
            "temperature": default_temperature,
            "threshold": default_threshold,
            "source": "defaults",
        }

    try:
        with open(calibration_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        temperature = float(data.get("temperature", default_temperature))
        threshold = float(data.get("threshold", default_threshold))

        # Clamp to safe ranges
        temperature = min(max(temperature, 0.3), 10.0)
        threshold = min(max(threshold, 0.1), 0.9)

        print(
            f"✅ Loaded calibration from {calibration_path.name}: "
            f"temperature={temperature:.4f}, threshold={threshold:.4f}"
        )
        return {
            "temperature": temperature,
            "threshold": threshold,
            "source": str(calibration_path),
        }
    except Exception as e:
        print(f"⚠️ Failed to load calibration file: {e}")
        return {
            "temperature": default_temperature,
            "threshold": default_threshold,
            "source": "defaults_after_error",
        }

print("🔧 Loading prediction model...")

class CNNClassifier(nn.Module):
    """CNN-based text classifier for fake news detection."""
    def __init__(self, vocab_size, embedding_dim, num_filters=128, filter_sizes=None, dropout=0.4):
        super(CNNClassifier, self).__init__()
        if filter_sizes is None:
            filter_sizes = [2, 3, 4, 5]

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 1D Convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, 
                     out_channels=num_filters, 
                     kernel_size=fs) 
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = embedded.transpose(1, 2)  # (batch_size, embedding_dim, seq_len) for Conv1d
        
        # Apply multiple convolutional filters
        conv_outputs = []
        for conv in self.convs:
            # Apply convolution and ReLU activation
            conv_out = self.relu(conv(embedded))  # (batch_size, num_filters, seq_len - filter_size + 1)
            # Apply max pooling over time dimension
            pooled = torch.max(conv_out, dim=2)[0]  # (batch_size, num_filters)
            conv_outputs.append(pooled)
        
        # Concatenate all pooled outputs
        cat = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)
        
        # Fully connected layers (return logits)
        cat = self.dropout(cat)
        out = self.relu(self.fc1(cat))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class LegacyBiLSTMClassifier(nn.Module):
    """Legacy BiLSTM architecture kept for robust fallback on deployment."""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, num_layers=2, dropout=0.3):
        super(LegacyBiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        out = self.relu(self.fc1(hidden))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def load_cnn_model_compatible(model_dir: Path, vocab_size: int) -> tuple:
    """
    Load CNN weights with architecture compatibility.
    Try latest training architecture first, then legacy one.
    """
    model_path = resolve_asset_file("cnn_model.pth")

    # Latest training profile (train_cnn.py)
    latest_cfg = {
        "embedding_dim": int(os.getenv("CNN_EMBEDDING_DIM", "128")),
        "num_filters": int(os.getenv("CNN_NUM_FILTERS", "128")),
        "filter_sizes": [2, 3, 4, 5],
        "dropout": float(os.getenv("CNN_DROPOUT", "0.4")),
    }

    # Legacy profile for backward compatibility
    legacy_cfg = {
        "embedding_dim": 128,
        "num_filters": 100,
        "filter_sizes": [3, 4, 5],
        "dropout": 0.3,
    }

    candidate_cfgs = [latest_cfg, legacy_cfg]
    last_error = None

    for idx, cfg in enumerate(candidate_cfgs, start=1):
        try:
            model = CNNClassifier(vocab_size=vocab_size, **cfg).to(device)
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            profile = "latest" if idx == 1 else "legacy"
            print(f"✅ Loaded CNN weights with {profile} profile: {cfg}")
            return model, cfg
        except Exception as e:
            last_error = e
            print(f"⚠️ CNN load profile #{idx} failed: {e}")

    raise RuntimeError(f"Unable to load cnn_model.pth with compatible profiles: {last_error}")


class CNNPredictor:
    """CNN-based predictor for fake news detection (primary model)."""
    def __init__(self, model_dir: Path):
        self.calibration = load_cnn_calibration(model_dir)
        self.word2idx = joblib.load(resolve_asset_file("word2idx.pkl"))
        self.max_len = int(os.getenv("CNN_MAX_LEN", "200"))

        self.model, self.model_config = load_cnn_model_compatible(
            model_dir=model_dir,
            vocab_size=len(self.word2idx),
        )
        
        # Temperature + threshold from calibration file
        self.temperature = self.calibration["temperature"]
        self.decision_threshold = self.calibration["threshold"]
        print(
            f"✅ CNN model loaded with temperature={self.temperature} "
            f"and threshold={self.decision_threshold}"
        )

    def _text_to_sequence(self, text: str) -> torch.Tensor:
        tokens = word_tokenize(text, engine='newmm')
        seq = [self.word2idx.get(word, 1) for word in tokens]  # 1 = <UNK>
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        else:
            seq = seq + [0] * (self.max_len - len(seq))  # 0 = <PAD>
        return torch.LongTensor([seq]).to(device)

    def predict_proba(self, text: str) -> float:
        sequence = self._text_to_sequence(text)
        with torch.no_grad():
            logit = self.model(sequence).squeeze(1).item()
            scaled_logit = logit / max(self.temperature, 1e-6)
            probability = 1 / (1 + math.exp(-scaled_logit))
            
        return float(probability)


class BiLSTMPredictor:
    """BiLSTM predictor (fallback when CNN model is unavailable)."""
    def __init__(self, model_dir: Path):
        self.calibration = load_cnn_calibration(model_dir)
        self.word2idx = joblib.load(resolve_asset_file("word2idx.pkl"))
        self.max_len = int(os.getenv("CNN_MAX_LEN", "200"))
        self.backend = "cnn"
        
        # Try to load CNN model, fall back to random if not available
        try:
            self.model, self.model_config = load_cnn_model_compatible(
                model_dir=model_dir,
                vocab_size=len(self.word2idx),
            )
            print("✅ CNN fallback model loaded")
        except Exception as e:
            print(f"⚠️ CNN model not found, trying legacy BiLSTM: {e}")
            lstm_path = resolve_asset_file("lstm_model.pth")
            try:
                self.model = LegacyBiLSTMClassifier(vocab_size=len(self.word2idx)).to(device)
                self.model.load_state_dict(
                    torch.load(lstm_path, map_location=device, weights_only=True)
                )
                self.model.eval()
                self.backend = "legacy_bilstm"
                print(f"✅ Loaded legacy BiLSTM fallback: {lstm_path}")
            except Exception as e2:
                print(f"⚠️ Legacy BiLSTM fallback failed, using untrained safety model: {e2}")
                self.model = CNNClassifier(vocab_size=len(self.word2idx), embedding_dim=128).to(device)
                self.model.eval()
                self.backend = "untrained_cnn"
        
        self.temperature = self.calibration["temperature"]
        self.decision_threshold = self.calibration["threshold"]

    def _text_to_sequence(self, text: str) -> torch.Tensor:
        tokens = word_tokenize(text, engine='newmm')
        seq = [self.word2idx.get(word, 1) for word in tokens]
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        else:
            seq = seq + [0] * (self.max_len - len(seq))
        return torch.LongTensor([seq]).to(device)

    def predict_proba(self, text: str) -> float:
        sequence = self._text_to_sequence(text)
        with torch.no_grad():
            logit = self.model(sequence).squeeze(1).item()
            scaled_logit = logit / max(self.temperature, 1e-6)
            probability = 1 / (1 + math.exp(-scaled_logit))
        return float(probability)


class WangchanBERTaPredictor:
    """Fine-tuned WangchanBERTa predictor for Thai fake news classification."""
    def __init__(self, model_dir: Path):
        self.max_len = int(os.getenv("WANGCHANBERTA_MAX_LEN", "128"))
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
        self.model.eval()

    def predict_proba(self, text: str) -> float:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_len
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[0]
            # class 1 = ข่าวจริง, class 0 = ข่าวปลอม
            return float(probs[1].item())


def load_predictor():
    """
    Load predictor with priority: CNN → WangchanBERTa → CNN fallback
    """
    preferred_model = os.getenv("PREFERRED_MODEL", "cnn").lower()
    wangchanberta_dir = Path(os.getenv("WANGCHANBERTA_MODEL_DIR", str(BASE_DIR / "wangchanberta_model")))
    cnn_model_path = resolve_asset_file("cnn_model.pth")

    # ✅ Try CNN first
    if preferred_model in ["cnn", "primary"] and cnn_model_path.exists():
        try:
            print(f"🔧 Loading CNN (primary) model...")
            predictor = CNNPredictor(BASE_DIR)
            print("✅ CNN model loaded successfully!")
            return "CNN (primary)", predictor
        except Exception as e:
            print(f"⚠️ Failed to load CNN model: {e}")
            print("↩️ Falling back to next model...")

    # ✅ Try WangchanBERTa second
    if preferred_model in ["wangchanberta", "bert"] and wangchanberta_dir.exists():
        try:
            print(f"🔧 Loading WangchanBERTa model from: {wangchanberta_dir}")
            predictor = WangchanBERTaPredictor(wangchanberta_dir)
            print("✅ WangchanBERTa model loaded successfully!")
            return "WangchanBERTa (fine-tuned)", predictor
        except Exception as e:
            print(f"⚠️ Failed to load WangchanBERTa model: {e}")
            print("↩️ Falling back to CNN fallback...")

    # ✅ CNN fallback (last resort)
    print("🔧 Loading CNN fallback model...")
    try:
        predictor = CNNPredictor(BASE_DIR)
        print("✅ CNN fallback model loaded!")
        return "CNN (fallback)", predictor
    except Exception as e:
        print(f"⚠️ CNN fallback failed: {e}")
        # Use BiLSTM predictor which can fallback to real legacy BiLSTM weights
        predictor = BiLSTMPredictor(BASE_DIR)
        if getattr(predictor, "backend", "") == "legacy_bilstm":
            print("✅ Using legacy BiLSTM fallback")
            return "BiLSTM (legacy fallback)", predictor
        print("✅ Using emergency fallback predictor")
        return "CNN (emergency fallback)", predictor

ACTIVE_MODEL_NAME, predictor = load_predictor()
print(f"📊 Active model: {ACTIVE_MODEL_NAME}")

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

# Google Custom Search API Configuration
# ควรกำหนดผ่าน Environment Variables บน Railway/Server เท่านั้น
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")
GOOGLE_CSE_URL = "https://www.googleapis.com/customsearch/v1"

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
    """ค้นหาข่าวที่เกี่ยวข้องจาก Google Custom Search API (รองรับภาษาไทย)."""
    try:
        print(f"\n🔍 Starting search_related_news (Google API)...")

        if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
            print("⚠️ Missing GOOGLE_API_KEY or GOOGLE_CSE_ID; skipping related-news search")
            return []

        # สกัด keywords สำหรับค้นหาให้เหมาะกับข่าว
        keywords = extract_keywords(query, max_keywords=8)
        search_query = " ".join(keywords[:6]).strip() or query[:120]

        print(f"🔍 Google search query: {search_query}")

        # Google Custom Search จำกัด num <= 10
        num_results = max(1, min(max_results * 2, 10))
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "q": search_query,
            "num": num_results,
            "hl": "th",
            "gl": "th",
            "safe": "off",
        }

        response = requests.get(GOOGLE_CSE_URL, params=params, timeout=10)

        if response.status_code != 200:
            print(f"❌ Google API error: {response.status_code}")
            print(f"📄 Response: {response.text[:500]}")
            return []

        data = response.json()

        if data.get("error"):
            print(f"❌ Google API returned error: {data['error']}")
            return []

        print(f"📦 Google response keys: {list(data.keys())}")
        related_news = []

        # Google CSE ผลลัพธ์อยู่ใน data['items']
        items = data.get("items", [])
        if items:
            print(f"📰 Found {len(items)} results from Google CSE")

            for article in items[:max_results]:
                source_url = article.get("link", "")
                source_domain = article.get("displayLink", "")

                if source_url and not source_domain:
                    try:
                        from urllib.parse import urlparse
                        source_domain = urlparse(source_url).netloc.replace("www.", "")
                    except Exception:
                        source_domain = "unknown"

                related_news.append({
                    "title": article.get("title", ""),
                    "source": source_domain or "unknown",
                    "url": source_url,
                    "content": article.get("snippet", ""),
                })

            print(f"✅ Processed {len(related_news)} related articles")
        else:
            print("⚠️ No related items found in Google response")

        return related_news

    except requests.Timeout:
        print("⏱️ Google API timeout")
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

def calculate_uncertainty(probability: float) -> dict:
    """
    คำนวณระดับความไม่แน่นอน (uncertainty)
    ถ้า probability ใกล้ 0.5 = ไม่แน่นอนมาก
    ถ้า probability ใกล้ 0 หรือ 1 = แน่นอนมาก
    """
    distance_from_neutral = abs(probability - 0.5)
    uncertainty_score = 1.0 - (distance_from_neutral * 2)  # 0-1, where 1 = most uncertain
    
    if uncertainty_score > 0.7:
        confidence_level = "ต่ำ - โมเดลไม่แน่ใจ"
    elif uncertainty_score > 0.4:
        confidence_level = "ปานกลาง - โมเดลมีข้อสงสัย"
    else:
        confidence_level = "สูง - โมเดลค่อนข้างแน่ใจ"
    
    return {
        "uncertainty_score": round(uncertainty_score, 3),
        "confidence_level": confidence_level,
        "recommendation": "💡 ควรตรวจสอบข่าวที่เกี่ยวข้อง" if uncertainty_score > 0.4 else "✅ ตรวจสอบแล้ว"
    }

def validate_text_quality(text: str) -> dict:
    """
    ตรวจสอบคุณภาพของข้อความ
    ป้องกันข้อความมั่วๆที่ไม่มีความหมาย
    """
    # ✅ รายการคำหยาบคายและคำไร้ความหมาย (Thai language) - ขยายเพิ่มเติม
    PROFANITY_WORDS = {
        "เย็ด", "แม่ม", "แม่", "ไอสัส", "ไอสัด", "สัตว์", "หมาป่า", 
        "ห่วยเหลือเกิน", "เหี้ยแหก", "หมาขี้ไก่", "คนไม่รู้", "ฉ้อฉล", 
        "เสือก", "งง", "ยังไงวะ", "ไอสัสน้อย", "หาทำไม", "ขี้นอก", 
        "จบหลวง", "ปิดปาก", "ลิ้นยาว", "หนังสือห่วย", "คนห่วย",
        "โง่", "ตับ", "โดนัล", "ธรรม", "ไอ"
    }
    
    # ✅ คำแปลกประหลาด/ไม่มีความหมาย
    GIBBERISH_WORDS = {
        "กาก", "ชิบ", "หาย", "เล่น", "ล่ะ", "อะไรยะ",
        "นั่นนี่", "โน่นนี่", "เฟืองเดือย", "วุ่นวาย", "หมา"
    }
    
    if not text or len(text.strip()) == 0:
        return {
            "is_valid": False,
            "reason": "ข้อความว่างเปล่า",
            "quality_score": 0.0,
            "should_skip_api": True
        }
    
    # ตรวจสอบความยาวข้อความ (ต้องมีอย่างน้อย 20 ตัวอักษร)
    if len(text.strip()) < 20:
        return {
            "is_valid": False,
            "reason": "ข้อความสั้นเกินไป (ต้องมีอย่างน้อย 20 ตัวอักษร)",
            "quality_score": 0.1,
            "should_skip_api": True
        }
    
    # แยกคำและกรองเฉพาะคำที่มีความหมาย
    tokens = word_tokenize(text, engine='newmm')
    meaningful_words = [w for w in tokens if len(w) > 1 and not w.isspace() and not all(c in '่้๊๋์-็ำฺ' for c in w)]
    
    # ✅ ตรวจสอบคำหยาบคาย
    profanity_count = sum(1 for word in meaningful_words if word in PROFANITY_WORDS)
    gibberish_count = sum(1 for word in meaningful_words if word in GIBBERISH_WORDS)
    
    total_bad_words = profanity_count + gibberish_count
    bad_word_ratio = total_bad_words / len(meaningful_words) if meaningful_words else 0
    
    # ✅ เข้มงวดมากขึ้น: ถ้ามีคำหยาบคายหรือไร้ความหมาย > 20% → ปฏิเสธ
    if bad_word_ratio > 0.2:
        return {
            "is_valid": False,
            "reason": f"ข้อความมีคำไม่สมควร/หยาบคาย {profanity_count} คำ, ไร้ความหมาย {gibberish_count} คำ ({bad_word_ratio*100:.1f}%) - เกินเกณฑ์ยอมรับ 20%",
            "quality_score": max(0.0, 1.0 - bad_word_ratio),
            "should_skip_api": True
        }
    
    # ตรวจสอบจำนวนคำที่มีความหมาย (ต้องมี >= 3 คำ)
    if len(meaningful_words) < 3:
        return {
            "is_valid": False,
            "reason": f"คำที่มีความหมายน้อยเกินไป (พบ {len(meaningful_words)} คำ, ต้อง >= 3)",
            "quality_score": 0.2,
            "should_skip_api": True
        }
    
    # ตรวจสอบอัตราส่วน whitespace/special chars กับตัวอักษรปกติ
    special_char_count = sum(1 for c in text if c in '!@#$%^&*()_+-=[]{}|;:,.<>?/~`' or not c.isalnum())
    normal_char_count = sum(1 for c in text if c.isalnum() or c.isspace())
    
    if normal_char_count == 0:
        return {
            "is_valid": False,
            "reason": "ข้อความประกอบด้วยตัวอักษรพิเศษเกินไป",
            "quality_score": 0.15,
            "should_skip_api": True
        }
    
    special_ratio = special_char_count / len(text) if len(text) > 0 else 0
    
    if special_ratio > 0.5:
        return {
            "is_valid": False,
            "reason": f"ข้อความมีตัวอักษรพิเศษมากเกินไป ({special_ratio*100:.1f}%)",
            "quality_score": 0.25,
            "should_skip_api": True
        }
    
    # ตรวจสอบว่ามีตัวอักษรติดต่อกันซ้ำๆ (เช่น "aaaa" หรือ "ววววว")
    max_repeat = max([len(list(group)) for char, group in groupby(text)] if text else [1])
    if max_repeat > 10:
        return {
            "is_valid": False,
            "reason": f"ข้อความมีตัวอักษรซ้ำเกินไป (ตัวซ้ำสูงสุด {max_repeat})",
            "quality_score": 0.2,
            "should_skip_api": True
        }
    
    # คำนวณคะแนนคุณภาพ
    quality_score = min(1.0, len(meaningful_words) / 10.0 * (1 - special_ratio * 0.5))
    
    if quality_score < 0.4:
        return {
            "is_valid": False,
            "reason": f"ข้อความคุณภาพต่ำ (คะแนน {quality_score:.2f})",
            "quality_score": quality_score,
            "should_skip_api": True
        }
    
    return {
        "is_valid": True,
        "reason": "ข้อความมีคุณภาพดี",
        "quality_score": min(1.0, quality_score),
        "should_skip_api": False,
        "meaningful_words": len(meaningful_words)
    }

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
    """
    ปรับ confidence ตามผลค้นหาข่าวที่เกี่ยวข้อง

    กติกาใหม่:
    - ไม่เพิ่มความน่าเชื่อถือจากการค้นหา (เพิ่มไม่ได้)
    - ถ้าไม่เจอข่าวที่เกี่ยวข้อง/ใกล้เคียง -> ลดความน่าเชื่อถือ
    """
    
    if not related_news:
        adjustment = -10.0
        adjusted_confidence = max(original_confidence + adjustment, 5.0)
        return adjusted_confidence, [], adjustment, "no_related_news"
    
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
    
    # ปรับ confidence (ไม่เพิ่มคะแนน)
    # - คล้ายกันมากและมีแหล่งน่าเชื่อถือ: คงเดิม
    # - คล้ายกันน้อย: ลดเล็กน้อย
    # - แทบไม่คล้าย: ลดมากขึ้น
    if max_similarity >= 0.45 and trusted_count > 0:
        adjustment = 0.0
        reason = "strong_related_match"
    elif max_similarity >= 0.25:
        adjustment = -3.0
        reason = "weak_related_match"
    else:
        adjustment = -8.0
        reason = "unrelated_search_results"

    adjusted_confidence = max(min(original_confidence + adjustment, 85.0), 5.0)
    return adjusted_confidence, related_items, adjustment, reason

@app.get("/")
def read_root():
    return {
        "message": "Fake News Detection API - Stacking Ensemble",
        "model": ACTIVE_MODEL_NAME,
        "accuracy": "98.90%" if "BiLSTM" in ACTIVE_MODEL_NAME else "(ใช้ค่าจากผล fine-tune ล่าสุด)",
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
    
    # ✅ ตรวจสอบคุณภาพข้อความก่อน
    text_quality = validate_text_quality(news.text)
    print(f"📊 Text quality: {text_quality}")
    
    if not text_quality["is_valid"]:
        return {
            "error": True,
            "label": "ข้อความไม่สมควร",
            "confidence": 0.0,
            "reason": text_quality["reason"],
            "quality_score": text_quality["quality_score"],
            "recommendation": "⚠️ กรุณาป้อนข้อความที่มีความหมายและมีความยาวเพียงพอ",
            "model": ACTIVE_MODEL_NAME
        }
    
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
    
    # ทำนายด้วยโมเดลที่ active อยู่
    probability = predictor.predict_proba(news.text)
    decision_threshold = float(getattr(predictor, "decision_threshold", 0.5))
    decision_threshold = min(max(decision_threshold, 0.1), 0.9)
    
    # แปลงเป็นความเชื่อมั่นและ label
    # Calibrate probability ให้ยืดหยุ่นขึ้นและลด overconfidence
    # ดึงค่าปลายสุดเข้าหา 0.5 เล็กน้อย (shrink toward neutral)
    shrink_factor = float(os.getenv("PREDICTION_SHRINK", "0.72"))
    calibrated_probability = 0.5 + (probability - 0.5) * shrink_factor
    capped_probability = min(max(calibrated_probability, 0.10), 0.90)
    
    # ✅ ลดความเชื่อมั่นถ้ามีคำไม่เหมาะสม
    if text_quality["quality_score"] < 1.0:
        confidence_penalty = 1.0 - text_quality["quality_score"]
        # ✅ เพิ่ม penalty เป็น 70% แทน 50% ให้เข้มงวดมากขึ้น
        capped_probability = capped_probability * (1.0 - confidence_penalty * 0.7)  # ลดความเชื่อมั่นขึ้นถึง 70%
        print(f"⚠️ Confidence reduced due to text quality issues: {confidence_penalty*100:.1f}% penalty applied")
    
    confidence_percent = round(capped_probability * 100, 1)
    decision_score = round(capped_probability - decision_threshold, 3)
    label = "ข่าวจริง" if capped_probability >= decision_threshold else "ข่าวปลอม"
    
    response = {
        "confidence": confidence_percent,
        "decision_score": decision_score,
        "label": label,
        "raw_score": probability,
        "threshold": round(decision_threshold, 4),
        "temperature": round(float(getattr(predictor, "temperature", 1.0)), 4),
        "model": ACTIVE_MODEL_NAME,
        "uncertainty": calculate_uncertainty(capped_probability),
        "quality_adjusted": text_quality["quality_score"] < 1.0
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
    
    # ถ้าต้องการตรวจสอบข่าวที่เกี่ยวข้อง (แต่ถ้าข้อความมีคุณภาพต่ำ ห้ามค้นหา)
    if news.check_related and text_quality["is_valid"] and not url_override:
        print(f"\n✅ check_related is True, searching for related news...")
        related_news = search_related_news(news.text)
        print(f"📊 Search returned {len(related_news)} articles")

        adjusted_confidence, related_items, adjustment, reason = adjust_confidence_with_related_news(
            confidence_percent,
            related_news,
            news.text
        )

        response["confidence"] = round(adjusted_confidence, 1)
        response["original_confidence"] = confidence_percent
        response["confidence_adjustment"] = round(adjustment, 1)
        response["related_news"] = related_items
        response["related_news_reason"] = reason

        if related_items:
            response["verification_note"] = (
                f"พบข่าวที่เกี่ยวข้อง {len(related_items)} รายการ "
                f"จากแหล่งที่น่าเชื่อถือ {sum(1 for r in related_items if r['is_trusted'])} แหล่ง "
                f"(ไม่มีการเพิ่มความน่าเชื่อถือจากผลค้นหา)"
            )
            print(f"✅ Added {len(related_items)} related news items to response")
        else:
            response["verification_note"] = "ไม่พบข่าวที่เกี่ยวข้องหรือใกล้เคียง จึงลดความน่าเชื่อถือลง"
            print(f"⚠️ No related news found, confidence reduced")

        # ปรับ label ตาม threshold หลังผ่าน related-news logic
        response["label"] = "ข่าวจริง" if (response["confidence"] / 100.0) >= decision_threshold else "ข่าวปลอม"
    elif news.check_related and url_override:
        print(f"ℹ️ URL verified: skip related-news confidence adjustment")
        response["search_skipped"] = True
        response["search_skip_reason"] = "ยืนยันด้วย URL จากแหล่งน่าเชื่อถือแล้ว จึงไม่ปรับคะแนนจากผลค้นหา"
    elif news.check_related and not text_quality["is_valid"]:
        print(f"⚠️ Skipping API search - text quality too low")
        response["search_skipped"] = True
        response["search_skip_reason"] = "ข้อความมีคุณภาพต่ำ จึงไม่ค้นหาข่าวเพิ่มเติม"
    else:
        print(f"⚠️ check_related is False, skipping news search")
    
    # เพิ่มข้อมูลคุณภาพข้อความในการตอบกลับ
    response["text_quality"] = {
        "quality_score": round(text_quality["quality_score"], 3),
        "is_valid": text_quality["is_valid"],
        "meaningful_words": text_quality.get("meaningful_words", 0)
    }
    
    return response
