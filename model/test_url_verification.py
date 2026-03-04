"""ทดสอบการตรวจสอบข่าวจาก URL"""
import requests
import json

API_URL = "http://localhost:8000/predict"

# ข่าวที่ผู้ใช้ให้มา พร้อม URL
news_text = """ตำรวจบุกทลายรังเว็บพนันเวียดนาม เช่าพื้นที่ในคอนโดฯ หรู ย่านพระราม 9 รวบพนักงานได้เกือบร้อยคน ยึดคอมฯ-มือถืออีกเกือบ 500 เครื่อง แต่ยังไม่มีใครยอมรับเป็นหัวหน้าขบวนการ

เมื่อเวลา 15.00 น. วันที่ 4 มี.ค. พล.ต.ต.วสันต์ เตชะอัครเกษม รองผบช.น. ร่วมกับ ตำรวจ สน.มักกะสัน ตำรวจตรวจคนเข้าเมือง และตำรวจไซเบอร์ นำหมายเข้าตรวจค้นสำนักงานแห่งหนึ่ง"""

source_url = "https://www.thairath.co.th/news/crime/2917937"

print("="*80)
print("ทดสอบ URL Verification Feature")
print("="*80)

# Test 1: ไม่มี URL (ผลลัพธ์เดิม - ข่าวปลอม 10.6%)
print("\n1️⃣ Test without URL:")
response = requests.post(API_URL, json={
    "text": news_text,
    "check_related": False
})

if response.status_code == 200:
    result = response.json()
    print(f"   Label: {result['label']}")
    print(f"   Confidence: {result['confidence']}%")
else:
    print(f"   Error: {response.status_code}")

# Test 2: มี URL จาก Thairath (ควรได้ข่าวจริง 95%)
print("\n2️⃣ Test with verified URL from Thairath:")
response = requests.post(API_URL, json={
    "text": news_text,
    "check_related": False,
    "source_url": source_url
})

if response.status_code == 200:
    result = response.json()
    print(f"   Label: {result['label']}")
    print(f"   Confidence: {result['confidence']}%")
    
    if 'override_reason' in result:
        print(f"   ✅ Override: {result['override_reason']}")
    if 'original_label' in result:
        print(f"   Original: {result['original_label']} ({result['original_confidence']}%)")
    if 'url_verification' in result:
        print(f"   URL Verified: {result['url_verification']}")
else:
    print(f"   Error: {response.status_code}")

# Test 3: URL ปลอม (ไม่น่าเชื่อถือ)
print("\n3️⃣ Test with untrusted URL:")
response = requests.post(API_URL, json={
    "text": news_text,
    "check_related": False,
    "source_url": "https://fake-news-site.com/article123"
})

if response.status_code == 200:
    result = response.json()
    print(f"   Label: {result['label']}")
    print(f"   Confidence: {result['confidence']}%")
    if 'override_reason' in result:
        print(f"   Override: {result['override_reason']}")
    else:
        print(f"   No override (URL not trusted)")
else:
    print(f"   Error: {response.status_code}")

print("\n" + "="*80)
