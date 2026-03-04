"""ทดสอบโหลดโมเดลและทดสอบ URL verification โดยตรง"""
import joblib
import math
import requests
from urllib.parse import urlparse

# โหลดโมเดล
print("🔧 Loading model...")
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf.pkl")
print("✅ Model loaded!\n")

# ข่าวที่ผู้ใช้ให้มา
news_text = """ตำรวจบุกทลายรังเว็บพนันเวียดนาม เช่าพื้นที่ในคอนโดฯ หรู ย่านพระราม 9 รวบพนักงานได้เกือบร้อยคน ยึดคอมฯ-มือถืออีกเกือบ 500 เครื่อง แต่ยังไม่มีใครยอมรับเป็นหัวหน้าขบวนการ"""

source_url = "https://www.thairath.co.th/news/crime/2917937"

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
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        
        # เช็คว่ามาจากแหล่งที่น่าเชื่อถือหรือไม่
        is_trusted = domain in TRUSTED_SOURCES
        
        # ลองเข้าถึง URL
        response = requests.head(url, timeout=5, allow_redirects=True)
        is_accessible = response.status_code == 200
        
        return {
            "domain": domain,
            "is_trusted": is_trusted,
            "is_accessible": is_accessible,
            "verified": is_trusted and is_accessible
        }
    except Exception as e:
        return {
            "domain": "",
            "is_trusted": False,
            "is_accessible": False,
            "verified": False,
            "error": str(e)
        }

print("="*80)
print("ทดสอบ URL Verification")
print("="*80)

# ตรวจสอบ URL
print(f"\n🔗 Testing URL: {source_url}")
url_result = verify_url(source_url)
print(f"   Domain: {url_result['domain']}")
print(f"   Is Trusted: {url_result['is_trusted']}")
print(f"   Is Accessible: {url_result['is_accessible']}")
print(f"   Verified: {url_result['verified']}")

if 'error' in url_result:
    print(f"   Error: {url_result['error']}")

# ทำนายด้วยโมเดล
print(f"\n📊 Model Prediction:")
X = vectorizer.transform([news_text])

if hasattr(model, 'predict_proba'):
    proba = model.predict_proba(X)[0]
    confidence_percent = round(proba[1] * 100, 1)
    label = "ข่าวจริง" if proba[1] > 0.5 else "ข่าวปลอม"
    
    print(f"   Original Label: {label}")
    print(f"   Original Confidence: {confidence_percent}%")
    
    # ปรับด้วย URL verification
    if url_result['verified']:
        print(f"\n✅ URL Verified! Overriding to:")
        print(f"   Final Label: ข่าวจริง")
        print(f"   Final Confidence: 95.0%")
        print(f"   Reason: ยืนยันจาก URL แหล่งที่เชื่อถือได้: {url_result['domain']}")

print("\n" + "="*80)
