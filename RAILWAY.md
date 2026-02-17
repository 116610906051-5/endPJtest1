# Railway Deployment Guide

## 🚂 Deploy บน Railway.app

### ขั้นตอนที่ 1: เตรียม Repository
```bash
git add .
git commit -m "Configure for Railway deployment"
git push
```

### ขั้นตอนที่ 2: Deploy บน Railway
1. ไปที่ https://railway.app/ และ Login ด้วย GitHub
2. คลิก **New Project**
3. เลือก **Deploy from GitHub repo**
4. เลือก repository `endPJtest1`
5. Railway จะ detect Python project อัตโนมัติ

### ขั้นตอนที่ 3: ตั้งค่า Service
1. คลิกที่ service ที่สร้าง
2. ไปที่ **Settings**:
   - **Root Directory**: `model`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. ไปที่ **Variables** → คลิก **Generate Domain** เพื่อสร้าง public URL
4. รอ deploy เสร็จ (3-5 นาที)

### ขั้นตอนที่ 4: ทดสอบ API
URL จะเป็นแบบนี้: `https://your-app.up.railway.app`

ทดสอบ:
```bash
curl https://your-app.up.railway.app/
```

---

## 🎨 Deploy Frontend บน Vercel

### ตั้งค่า Environment Variable:
```bash
cd client
# สร้างไฟล์ .env.local
echo "VITE_API_URL=https://your-app.up.railway.app" > .env.local
```

### Deploy:
1. ไปที่ https://vercel.com/
2. Import repository
3. ตั้งค่า:
   - **Root Directory**: `client`
   - **Environment Variable**: 
     - `VITE_API_URL` = `https://your-app.up.railway.app`
4. Deploy

---

## ✅ ข้อดี Railway vs Render

| Feature | Railway | Render |
|---------|---------|--------|
| Build Speed | ⚡ เร็วกว่า | 🐌 ช้ากว่า |
| Python 3.11+ | ✅ รองรับดี | ⚠️ มีปัญหา |
| Free Tier | $5 credit/month | 750 hours/month |
| Auto Deploy | ✅ | ✅ |
| Custom Domain | ✅ Free | ✅ Free |

---

## 📝 Local Development

```bash
# Backend
cd model
pip install -r requirements.txt
uvicorn main:app --reload
# API: http://localhost:8000

# Frontend
cd client
npm install
npm run dev
# App: http://localhost:5173
```

---

## 🔧 Troubleshooting

### ถ้า deploy ล้มเหลว:
1. ตรวจสอบ logs ใน Railway dashboard
2. ตรวจสอบว่ามีไฟล์ `svm_model.pkl` และ `tfidf.pkl` ใน repository
3. ตรวจสอบ Root Directory ตั้งเป็น `model`

### ถ้า API ไม่ตอบสนอง:
1. เช็ค health endpoint: `https://your-app.up.railway.app/`
2. ดู logs ว่ามี error อะไร
3. ตรวจสอบ PORT variable
