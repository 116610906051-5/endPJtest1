"""ทดสอบ BiLSTM API พร้อม Related News"""
import requests
import json
import time

# รอ server พร้อม
time.sleep(3)

API_URL = "http://localhost:8003/predict"

# ข่าวที่ผู้ใช้ให้มา
news_text = """ตำรวจบุกทลายรังเว็บพนันเวียดนาม เช่าพื้นที่ในคอนโดฯ หรู ย่านพระราม 9 รวบพนักงานได้เกือบร้อยคน ยึดคอมฯ-มือถืออีกเกือบ 500 เครื่อง แต่ยังไม่มีใครยอมรับเป็นหัวหน้าขบวนการ

เมื่อเวลา 15.00 น. วันที่ 4 มี.ค. พล.ต.ต.วสันต์ เตชะอัครเกษม รองผบช.น. ร่วมกับ ตำรวจ สน.มักกะสัน ตำรวจตรวจคนเข้าเมือง และตำรวจไซเบอร์ นำหมายเข้าตรวจค้นสำนักงานแห่งหนึ่ง"""

print("="*80)
print("ทดสอบ BiLSTM API with Automatic Related News Verification")
print("="*80)

# ส่งข้อมูลไปยัง API (ระบบจะหาข่าวเกี่ยวข้องอัตโนมัติ)
print("\n📤 Sending news to Deep Learning API...\n")

try:
    response = requests.post(API_URL, json={
        "text": news_text
    }, timeout=10)

    if response.status_code == 200:
        result = response.json()
        
        print("🤖 Deep Learning Model (BiLSTM):")
        print(f"   Model: {result.get('model', 'N/A')}")
        print(f"   Model Label: {result.get('model_label', 'N/A')}")
        print(f"   Model Confidence: {result.get('model_confidence', 'N/A')}%")
        
        print("\n🔍 Verification with Related News:")
        print(f"   Final Label: {result.get('label', 'N/A')}")
        print(f"   Final Confidence: {result.get('confidence', 'N/A')}%")
        
        if 'confidence_adjustment' in result and result['confidence_adjustment'] != 0:
            print(f"   Confidence Adjustment: +{result['confidence_adjustment']}%")
        
        if 'trusted_sources_found' in result:
            print(f"   Trusted Sources Found: {result['trusted_sources_found']}")
        
        if 'verification_note' in result:
            print(f"\n📝 Verification: {result['verification_note']}")
        
        if 'related_news' in result and result['related_news']:
            print(f"\n📰 แหล่งข่าวที่เกี่ยวข้อง ({len(result['related_news'])} รายการ):")
            print("-" * 80)
            for i, article in enumerate(result['related_news'], 1):
                print(f"\n   {i}. {article['title']}")
                print(f"      🌐 แหล่งที่มา: {article['source']}", end="")
                if article['is_trusted']:
                    print(" ✅ (แหล่งที่เชื่อถือได้)")
                else:
                    print()
                print(f"      📊 ความคล้าย: {article['similarity']}%")
                print(f"      🔗 {article['url']}")
                if 'publishedAt' in article and article['publishedAt']:
                    print(f"      📅 เผยแพร่: {article['publishedAt']}")
        
        print("\n" + "="*80)
        print("✅ Deep Learning + News Verification สำเร็จ!")
        
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to API server. Please make sure server is running on port 8003")
except Exception as e:
    print(f"❌ Error: {e}")
