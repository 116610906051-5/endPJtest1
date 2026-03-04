import requests
import json

url = "http://127.0.0.1:8000/predict"

test_cases = [
    ("ข่าวปลอม", "กรมพัฒนาธุรกิจการค้าอนุญาตใบทะเบียนพาณิชย์รายบุคคล ประกอบธุรกิจเงินกู้นอกระบบแบบออนไลน์"),
    ("ข่าวจริง", "นายกรัฐมนตรีเปิดเผยนโยบายเศรษฐกิจใหม่ในการประชุมคณะรัฐมนตรี"),
]

print("Testing BiLSTM API:")
print("=" * 80)

for label, text in test_cases:
    try:
        response = requests.post(url, json={"text": text})
        result = response.json()
        print(f"\n✓ {label}: {text[:50]}...")
        print(f"  Confidence: {result['confidence']}%")
        print(f"  Decision Score: {result['decision_score']}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
