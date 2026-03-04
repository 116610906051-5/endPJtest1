# Related News Verification Feature

## 🎯 ฟีเจอร์ใหม่: ตรวจสอบข่าวที่เกี่ยวข้อง

ระบบสามารถตรวจสอบความน่าเชื่อถือของข่าวโดยเปรียบเทียบกับข่าวจากแหล่งที่น่าเชื่อถือ

### 📋 API Endpoint

**POST /predict**

```json
{
  "text": "ข้อความข่าวที่ต้องการตรวจสอบ",
  "check_related": true  // เพิ่ม parameter นี้เพื่อเปิดใช้งาน
}
```

### 📊 Response Format

#### แบบปกติ (check_related: false)
```json
{
  "confidence": 12.0,
  "decision_score": -1.992,
  "label": "ข่าวปลอม",
  "raw_score": 0.12
}
```

#### แบบตรวจสอบข่าวที่เกี่ยวข้อง (check_related: true)
```json
{
  "confidence": 22.0,              // ความน่าเชื่อถือที่ปรับแล้ว
  "decision_score": -1.992,
  "label": "ข่าวปลอม",
  "raw_score": 0.12,
  "original_confidence": 12.0,     // ความน่าเชื่อถือเดิม
  "confidence_adjustment": 10.0,   // การปรับเพิ่ม/ลด
  "related_news": [                // ข่าวที่เกี่ยวข้อง
    {
      "title": "หัวข้อข่าว",
      "source": "thaipbs.or.th",
      "url": "https://...",
      "similarity": 85.5,           // ความคล้ายคลึง (%)
      "is_trusted": true            // เป็นแหล่งที่น่าเชื่อถือ
    }
  ],
  "verification_note": "พบข่าวที่เกี่ยวข้อง 1 รายการ จากแหล่งที่น่าเชื่อถือ 1 แหล่ง"
}
```

## 🔍 วิธีการทำงาน

### 1. **ค้นหาข่าวที่เกี่ยวข้อง**
   - สกัด keywords จากข้อความที่ต้องการตรวจสอบ
   - ค้นหาข่าวที่เกี่ยวข้องจากเว็บข่าวที่น่าเชื่อถือ

### 2. **คำนวณความคล้ายคลึง**
   - ใช้ TF-IDF + Cosine Similarity
   - เปรียบเทียบเนื้อหาข่าวกับแหล่งที่น่าเชื่อถือ

### 3. **ปรับความน่าเชื่อถือ**
   - ถ้าพบข่าวคล้ายกันจากแหล่งที่น่าเชื่อถือ → **เพิ่ม confidence**
   - การปรับเพิ่มสูงสุด: **+15%**
   - สูตร: `adjustment = min(trusted_count × similarity × 10, 15)`

## 🌐 แหล่งข่าวที่น่าเชื่อถือ

1. thairath.co.th
2. manager.co.th
3. bangkokpost.com
4. nationthailand.com
5. thaipbs.or.th
6. prachachat.net
7. matichon.co.th
8. khaosod.co.th
9. thaipost.net
10. dailynews.co.th

## 💡 ตัวอย่างการใช้งาน

### Python
```python
import requests

response = requests.post(
    "https://your-api.railway.app/predict",
    json={
        "text": "นายกรัฐมนตรีเปิดเผยนโยบายเศรษฐกิจใหม่",
        "check_related": True
    }
)

result = response.json()
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']}%")
if 'confidence_adjustment' in result:
    print(f"Adjusted: +{result['confidence_adjustment']}%")
```

### JavaScript/TypeScript
```javascript
const response = await fetch('https://your-api.railway.app/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: 'นายกรัฐมนตรีเปิดเผยนโยบายเศรษฐกิจใหม่',
    check_related: true
  })
});

const result = await response.json();
console.log(`Label: ${result.label}`);
console.log(`Confidence: ${result.confidence}%`);
```

## ⚠️ หมายเหตุ

1. **Demo Mode**: ปัจจุบันเป็น simulated search - ในการใช้งานจริงควรต่อ News API
2. **API Key**: ต้องมี API key สำหรับ Google News API หรือ News API
3. **Rate Limit**: ควรจำกัดจำนวน request เพื่อประหยัด API quota
4. **Cache**: ควร cache ผลลัพธ์เพื่อลด API calls

## 🚀 การพัฒนาต่อ

- [ ] เชื่อมต่อ News API จริง (newsapi.org)
- [ ] เพิ่ม RSS feeds reader
- [ ] Cache ผลลัพธ์ใน Redis
- [ ] รองรับหลายภาษา
- [ ] Real-time fact checking
