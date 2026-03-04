from main import app
import json

# สร้าง test client
from fastapi.testclient import TestClient
client = TestClient(app)

print("=== Test 1: Basic prediction ===")
response1 = client.post(
    "/predict",
    json={"text": "กรมพัฒนาธุรกิจการค้าอนุญาตใบทะเบียนพาณิชย์รายบุคคล ประกอบธุรกิจเงินกู้นอกระบบแบบออนไลน์"}
)
print(json.dumps(response1.json(), ensure_ascii=False, indent=2))

print("\n=== Test 2: With related news check ===")
response2 = client.post(
    "/predict",
    json={
        "text": "นายกรัฐมนตรีเปิดเผยนโยบายเศรษฐกิจใหม่ในการประชุมคณะรัฐมนตรี",
        "check_related": True
    }
)
result = response2.json()
print(json.dumps(result, ensure_ascii=False, indent=2))

print("\n=== Test 3: Fake news with related check ===")
response3 = client.post(
    "/predict",
    json={
        "text": "ธนาคารแห่งประเทศไทยแจกเงินให้ประชาชน 10,000 บาท ผ่านแอพพลิเคชั่น",
        "check_related": True
    }
)
result3 = response3.json()
print(json.dumps(result3, ensure_ascii=False, indent=2))
