import requests

# ทดสอบ API โดยไม่ตรวจสอบข่าวที่เกี่ยวข้อง
print("=== Test 1: Basic prediction ===")
response1 = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"text": "กรมพัฒนาธุรกิจการค้าอนุญาตใบทะเบียนพาณิชย์รายบุคคล ประกอบธุรกิจเงินกู้นอกระบบแบบออนไลน์"}
)
print(response1.json())

print("\n=== Test 2: With related news check ===")
response2 = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "text": "นายกรัฐมนตรีเปิดเผยนโยบายเศรษฐกิจใหม่ในการประชุมคณะรัฐมนตรี",
        "check_related": True
    }
)
result = response2.json()
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']}%")
if 'original_confidence' in result:
    print(f"Original Confidence: {result['original_confidence']}%")
    print(f"Adjustment: +{result['confidence_adjustment']}%")
if 'related_news' in result:
    print(f"\nRelated News: {len(result['related_news'])} items")
    for news in result['related_news']:
        print(f"  - {news['title'][:50]}... (similarity: {news['similarity']}%)")
