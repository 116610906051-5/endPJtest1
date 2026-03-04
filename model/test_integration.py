from main import app, predict, News
import json

print("=" * 80)
print("ทดสอบ NewsAPI Integration")
print("=" * 80)

# Test 1: ไม่ check related
print("\n1️⃣ Test without related news:")
result1 = predict(News(text="นายกรัฐมนตรีเปิดเผยนโยบายเศรษฐกิจใหม่", check_related=False))
print(json.dumps(result1, ensure_ascii=False, indent=2))

# Test 2: check related
print("\n2️⃣ Test with related news:")
result2 = predict(News(text="นายกรัฐมนตรีเปิดเผยนโยบายเศรษฐกิจใหม่", check_related=True))
print(json.dumps(result2, ensure_ascii=False, indent=2))

if 'related_news' in result2:
    print(f"\n✅ พบข่าวที่เกี่ยวข้อง {len(result2['related_news'])} รายการ")
    for i, news in enumerate(result2['related_news'], 1):
        print(f"\n{i}. {news['title'][:60]}...")
        print(f"   Source: {news['source']}")
        print(f"   Trusted: {news['is_trusted']}")
        print(f"   URL: {news['url'][:50]}...")
else:
    print("\n⚠️ ไม่พบข่าวที่เกี่ยวข้อง")
