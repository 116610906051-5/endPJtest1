# News Verification System
ระบบตรวจสอบข่าวปลอมด้วย Machine Learning (SVM)

## โครงสร้างโปรเจค

```
endpj/
├── client/          # Frontend (React + TypeScript + Vite)
├── model/           # Backend (FastAPI + SVM Model)
└── README.md
```

## การติดตั้งและรันโปรเจค

### 1. Backend (Model)

```bash
# เข้าไปในโฟลเดอร์ model
cd model

# ติดตั้ง Python packages
pip install -r requirements.txt

# (ถ้ายังไม่มีโมเดล) เทรนโมเดลก่อน
python train_svm.py

# รัน FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend จะรันที่: http://localhost:8000

### 2. Frontend (Client)

เปิด Terminal ใหม่:

```bash
# เข้าไปในโฟลเดอร์ client
cd client

# ติดตั้ง dependencies
npm install

# รัน development server
npm run dev
```

Frontend จะรันที่: http://localhost:5173

## การใช้งาน

1. เปิดเบราว์เซอร์ไปที่ http://localhost:5173
2. พิมพ์ข้อความข่าวที่ต้องการตรวจสอบ
3. กดปุ่ม "วิเคราะห์"
4. ระบบจะแสดงผลว่าเป็นข่าวจริงหรือข่าวปลอม

## API Endpoint

### POST /predict
ตรวจสอบข่าวปลอม

**Request:**
```json
{
  "text": "ข้อความข่าวที่ต้องการตรวจสอบ"
}
```

**Response:**
```json
{
  "result": 1  // 1 = ข่าวจริง, 0 = ข่าวปลอม
}
```

## เทคโนโลยีที่ใช้

### Backend
- **FastAPI**: Web framework สำหรับ Python
- **scikit-learn**: Machine Learning library (SVM)
- **TF-IDF**: เทคนิคแปลงข้อความเป็นตัวเลข
- **joblib**: บันทึกและโหลดโมเดล

### Frontend
- **React**: UI library
- **TypeScript**: Type-safe JavaScript
- **Vite**: Build tool และ dev server
- **CSS**: สำหรับ styling

## หมายเหตุ

- ต้องรัน Backend ก่อน Frontend จึงจะใช้งานได้
- Backend ต้องมีไฟล์ `svm_model.pkl` และ `tfidf.pkl` ก่อนรัน
- ถ้ายังไม่มีโมเดล ให้รัน `python train_svm.py` ก่อน
