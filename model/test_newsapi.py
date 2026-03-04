import requests
import json

# ทดสอบกับ NewsAPI จริง
API_URL = "http://127.0.0.1:8000/predict"

print("=" * 80)
print("ทดสอบ Related News Verification ด้วย NewsAPI จริง")
print("=" * 80)

test_cases = [
    {
        "name": "ข่าวการเมือง",
        "text": "นายกรัฐมนตรีเปิดเผยนโยบายเศรษฐกิจใหม่ในการประชุมคณะรัฐมนตรี",
        "check_related": True
    },
    {
        "name": "ข่าวปลอมเกี่ยวกับธนาคาร",
        "text": "ธนาคารแห่งประเทศไทยแจกเงินให้ประชาชน 10,000 บาท ผ่านแอพพลิเคชั่น",
        "check_related": True
    },
    {
        "name": "ข่าวรัฐบาล",
        "text": "กระทรวงสาธารณสุขประกาศมาตรการป้องกันโรคระบาดใหม่",
        "check_related": True
    }
]

for i, test in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"Test {i}: {test['name']}")
    print(f"{'='*80}")
    print(f"Text: {test['text']}")
    print()
    
    try:
        response = requests.post(API_URL, json=test, timeout=10)
        result = response.json()
        
        print(f"🎯 Label: {result.get('label', 'N/A')}")
        print(f"📊 Confidence: {result.get('confidence', 0)}%")
        
        if 'original_confidence' in result:
            print(f"   Original: {result['original_confidence']}%")
            print(f"   Adjustment: {result['confidence_adjustment']:+.1f}%")
        
        if 'related_news' in result and result['related_news']:
            print(f"\n📰 Related News: {len(result['related_news'])} รายการ")
            for idx, news in enumerate(result['related_news'][:3], 1):
                print(f"\n   {idx}. {news['title'][:80]}...")
                print(f"      Source: {news['source']}")
                print(f"      Trusted: {'✅' if news['is_trusted'] else '❌'}")
                print(f"      Similarity: {news['similarity']}%")
        
        if 'verification_note' in result:
            print(f"\n💡 {result['verification_note']}")
            
    except requests.Timeout:
        print("⏱️ Request timeout - API ใช้เวลานานเกินไป")
    except Exception as e:
        print(f"❌ Error: {e}")

print(f"\n{'='*80}")
print("✅ Testing completed!")
print("=" * 80)
