"""ทดสอบ BiLSTM API พร้อม Related News"""
import requests
import json

API_URL = "http://localhost:8002/predict"

# ข่าวที่ผู้ใช้ให้มา
news_text = """ตำรวจบุกทลายรังเว็บพนันเวียดนาม เช่าพื้นที่ในคอนโดฯ หรู ย่านพระราม 9 รวบพนักงานได้เกือบร้อยคน ยึดคอมฯ-มือถืออีกเกือบ 500 เครื่อง แต่ยังไม่มีใครยอมรับเป็นหัวหน้าขบวนการ

เมื่อเวลา 15.00 น. วันที่ 4 มี.ค. พล.ต.ต.วสันต์ เตชะอัครเกษม รองผบช.น. ร่วมกับ ตำรวจ สน.มักกะสัน ตำรวจตรวจคนเข้าเมือง และตำรวจไซเบอร์ นำหมายเข้าตรวจค้นสำนักงานแห่งหนึ่ง"""

print("="*80)
print("ทดสอบ BiLSTM API with Related News Verification")
print("="*80)

# ส่งข้อมูลไปยัง API (default: check_related=True)
print("\n📤 Sending news to API for verification...\n")

response = requests.post(API_URL, json={
    "text": news_text
})

if response.status_code == 200:
    result = response.json()
    
    print("🤖 Model Prediction:")
    print(f"   Model: {result.get('model', 'N/A')}")
    print(f"   Model Label: {result.get('model_label', 'N/A')}")
    print(f"   Model Confidence: {result.get('model_confidence', 'N/A')}%")
    
    print("\n🔍 Verification Result:")
    print(f"   Final Label: {result.get('label', 'N/A')}")
    print(f"   Final Confidence: {result.get('confidence', 'N/A')}%")
    
    if 'confidence_adjustment' in result:
        print(f"   Adjustment: +{result['confidence_adjustment']}%")
    
    if 'verification_note' in result:
        print(f"\n📝 {result['verification_note']}")
    
    if 'related_news' in result and result['related_news']:
        print(f"\n📰 Related News ({len(result['related_news'])} articles):")
        for i, article in enumerate(result['related_news'], 1):
            print(f"\n   {i}. {article['title'][:80]}...")
            print(f"      Source: {article['source']} {'✓ (Trusted)' if article['is_trusted'] else ''}")
            print(f"      Similarity: {article['similarity']}%")
            print(f"      URL: {article['url'][:60]}...")
    
    print("\n" + "="*80)
    print("✅ Test completed successfully!")
    
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)
