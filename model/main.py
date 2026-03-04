from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import math
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, List
import re

app = FastAPI(title="Fake News Detection API - Stacking Ensemble")

# เปิด CORS เพื่อให้ Frontend เรียกได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลดโมเดล Stacking Ensemble
print("🔧 Loading Stacking Ensemble model...")
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf.pkl")
print("✅ Model loaded successfully!")

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

# NewsAPI Configuration (เก่า - ไม่มีข่าวไทย)
# NEWS_API_KEY = "277729d09fc640549010e57ecb99c09d"
# NEWS_API_BASE_URL = "https://newsapi.org/v2/everything"

# SearchAPI Configuration (ใช้แทน NewsAPI - มีข่าวไทยเยอะกว่า)
SEARCHAPI_KEY = "jkiyja5rtFwfdQYjXh5nLLjC"
SEARCHAPI_URL = "https://www.searchapi.io/api/v1/search"

class News(BaseModel):
    text: str
    check_related: Optional[bool] = True  # เปิดเป็นค่าเริ่มต้น
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
    """ค้นหาข่าวที่เกี่ยวข้องจาก SearchAPI (Google Search)"""
    try:
        # ตรวจสอบความยาวข้อความก่อน
        if len(query.strip()) < 10:
            print(f"⚠️ Text too short, skipping")
            return []
        
        # สกัด keywords
        keywords = extract_keywords(query)
        if len(keywords) < 2:
            print(f"⚠️ Not enough keywords")
            return []
        
        search_query = " ".join(keywords[:3])
        print(f"🔍 SearchAPI: {search_query}")
        
        # ค้นหาจากแหล่งข่าวไทย
        thai_sites = ["thairath.co.th", "manager.co.th", "khaosod.co.th"]
        all_results = []
        
        for site in thai_sites:
            params = {
                "engine": "google",
                "q": f"{search_query} site:{site}",
                "api_key": SEARCHAPI_KEY
            }
            
            try:
                response = requests.get(SEARCHAPI_URL, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    organic = data.get('organic_results', [])
                    
                    print(f"  📰 {site}: {len(organic)} articles")
                    
                    for result in organic[:3]:  # เอา 3 ข่าวแรกจากแต่ละเว็บ
                        all_results.append({
                            "title": result.get('title', ''),
                            "source": site,
                            "url": result.get('link', ''),
                            "content": result.get('snippet', '') or result.get('title', ''),
                            "publishedAt": ""
                        })
                        
                elif response.status_code == 429:
                    print(f"  ⚠️ SearchAPI rate limit reached")
                    break
                    
            except Exception as e:
                print(f"  ⚠️ Error with {site}: {e}")
                continue
        
        print(f"✅ Collected {len(all_results)} articles")
        return all_results[:10]  # ส่งไปเช็ค similarity
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return []
        
        if data.get("status") == "ok" and data.get("articles"):
            for article in data["articles"][:max_results]:
                # ดึง domain จาก URL
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
        
    except requests.Timeout:
        print("NewsAPI timeout")
        return []
    except Exception as e:
        print(f"Error searching news: {e}")
        return []

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """สกัดคำสำคัญจากข้อความภาษาไทย"""
    try:
        from pythainlp.tokenize import word_tokenize
        from pythainlp.corpus import thai_stopwords
        
        # Tokenize ภาษาไทย
        words = word_tokenize(text, engine='newmm')
        
        # โหลด stopwords
        stopwords = thai_stopwords()
        
        # กรองคำที่ไม่ใช่ stopwords และยาวกว่า 1 ตัวอักษร
        keywords = [w for w in words if w not in stopwords and len(w) > 1 and not w.isspace()]
        
        # เอาเฉพาะคำที่มีความหมาย (กรองเครื่องหมายออก)
        meaningful_keywords = [w for w in keywords if any(c.isalnum() for c in w)]
        
        return meaningful_keywords[:max_keywords]
    except Exception as e:
        print(f"⚠️ Keyword extraction error: {e}, using simple method")
        # Fallback: ใช้ regex เบื้องต้น
        words = re.findall(r'\b\w+\b', text)
        keywords = [w for w in words if len(w) > 2][:max_keywords]
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
        
        # กรองข่าวที่ similarity ต่ำมาก (< 3%) ออก - ใช้ threshold ต่ำเพื่อให้โอกาส
        if similarity < 0.03:
            print(f"⚠️ Skipping low similarity: {news.get('title', '')[:40]}... ({round(similarity * 100, 1)}%)")
            continue
        
        print(f"✅ Relevant article: {news.get('title', '')[:40]}... ({round(similarity * 100, 1)}%)")
        
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
    
    # ปรับ confidence
    # ถ้ามีข่าวจากแหล่งที่น่าเชื่อถือและคล้ายคลึงกัน -> เพิ่ม confidence
    adjustment = 0.0
    
    # ถ้าไม่มีข่าวที่เกี่ยวข้องหลังกรอง ให้ return ว่าง
    if len(related_items) == 0:
        print(f"⚠️ All articles filtered out due to low similarity")
        return original_confidence, []
    
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
    
    X = vectorizer.transform([news.text])
    
    # ตรวจสอบว่าโมเดลมี decision_function หรือไม่
    if hasattr(model, 'decision_function'):
        # สำหรับ LinearSVC หรือโมเดลที่มี decision_function
        decision = model.decision_function(X)[0]
        confidence = 1 / (1 + math.exp(-decision))
        confidence_percent = round(confidence * 100, 1)
        decision_score = round(decision, 3)
        label = "ข่าวจริง" if confidence > 0.5 else "ข่าวปลอม"
    elif hasattr(model, 'predict_proba'):
        # สำหรับ Ensemble models และโมเดลอื่นๆ ที่มี predict_proba
        proba = model.predict_proba(X)[0]
        confidence_percent = round(proba[1] * 100, 1)
        decision_score = round(proba[1] - proba[0], 3)
        label = "ข่าวจริง" if proba[1] > 0.5 else "ข่าวปลอม"
    else:
        # fallback สำหรับโมเดลที่ไม่มีทั้งสองแบบ
        prediction = model.predict(X)[0]
        confidence_percent = 100.0 if prediction == 1 else 0.0
        decision_score = 1.0 if prediction == 1 else -1.0
        label = "ข่าวจริง" if prediction == 1 else "ข่าวปลอม"
    
    response = {
        "confidence": confidence_percent,
        "decision_score": decision_score,
        "label": label,
        "raw_score": confidence_percent / 100.0
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
        related_news = search_related_news(news.text)
        
        if related_news and len(related_news) > 0:
            adjusted_confidence, related_items = adjust_confidence_with_related_news(
                confidence_percent,
                related_news,
                news.text
            )
            
            if len(related_items) > 0:
                response["confidence"] = round(adjusted_confidence, 1)
                response["original_confidence"] = confidence_percent
                response["confidence_adjustment"] = round(adjusted_confidence - confidence_percent, 1)
                response["related_news"] = related_items
                response["verification_note"] = (
                    f"พบข่าวที่เกี่ยวข้อง {len(related_items)} รายการ "
                    f"จากแหล่งที่น่าเชื่อถือ {sum(1 for r in related_items if r['is_trusted'])} แหล่ง"
                )
            else:
                # ไม่มีข่าวที่ similarity ผ่าน
                response["related_news"] = []
                response["verification_note"] = "ไม่พบข่าวที่เกี่ยวข้องในช่วงเวลานี้ (NewsAPI Free มีข้อจำกัดข่าวไทย)"
        else:
            # ไม่มีข่าวเลยจาก NewsAPI
            response["related_news"] = []
            response["verification_note"] = "ไม่สามารถตรวจสอบข่าวที่เกี่ยวข้องได้ในขณะนี้ (NewsAPI Free ไม่มีข่าวไทย)"
    
    return response
