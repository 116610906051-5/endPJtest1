# Fake News Detection API - Deployment Guide

## 🚀 Deploy Backend บน Render

### ขั้นตอนที่ 1: เตรียม Repository
```bash
cd model
git add .
git commit -m "Add Render deployment files"
git push
```

### ขั้นตอนที่ 2: Deploy บน Render
1. ไปที่ https://render.com/ และ Sign up/Login
2. คลิก **New +** → **Web Service**
3. เชื่อมต่อกับ GitHub repository ของคุณ
4. ตั้งค่า:
   - **Name**: fake-news-api (หรือชื่อที่ต้องการ)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Root Directory**: `model`
5. คลิก **Create Web Service**
6. รอ deploy เสร็จ (5-10 นาที)
7. คัดลอก URL ที่ได้ เช่น `https://fake-news-api.onrender.com`

### ขั้นตอนที่ 3: ทดสอบ API
เปิดเบราว์เซอร์ไปที่: `https://your-app-name.onrender.com/`

ควรเห็น:
```json
{
  "message": "Fake News Detection API",
  "status": "running"
}
```

---

## 🎨 Deploy Frontend บน Vercel/Netlify

### ขั้นตอนที่ 1: สร้างไฟล์ .env.local
```bash
cd client
echo "VITE_API_URL=https://your-render-app.onrender.com" > .env.local
```

### ขั้นตอนที่ 2: Deploy บน Vercel
1. ไปที่ https://vercel.com/ และ Login
2. คลิก **Add New** → **Project**
3. Import repository จาก GitHub
4. ตั้งค่า:
   - **Root Directory**: `client`
   - **Framework Preset**: Vite
   - **Environment Variables**: 
     - Key: `VITE_API_URL`
     - Value: `https://your-render-app.onrender.com`
5. คลิก **Deploy**
6. รอ deploy เสร็จ (1-2 นาที)

---

## 📝 หมายเหตุสำคัญ

### Backend (Render)
- ⚠️ Free tier จะ sleep หลังไม่มีการใช้งาน 15 นาที
- ⚠️ ต้องมีไฟล์ `svm_model.pkl` และ `tfidf.pkl` ใน repository
- ⚠️ ไฟล์ model ต้องไม่เกิน 500MB

### Frontend (Vercel)
- ✅ ต้องตั้ง environment variable `VITE_API_URL`
- ✅ Build command: `npm run build`
- ✅ Output directory: `dist`

---

## 🔧 Local Development

### Backend
```bash
cd model
uvicorn main:app --reload
# API: http://localhost:8000
```

### Frontend (ใช้ local API)
```bash
cd client
# ไม่ต้องตั้ง VITE_API_URL (จะใช้ localhost:8000 อัตโนมัติ)
npm run dev
# App: http://localhost:5173
```

### Frontend (ใช้ production API)
```bash
cd client
# สร้างไฟล์ .env.local
echo "VITE_API_URL=https://your-render-app.onrender.com" > .env.local
npm run dev
```
