from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Fake News Detection API")

# เปิด CORS เพื่อให้ Frontend เรียกได้
# สำหรับ production ให้รองรับทุก origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production: รองรับทุก domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลดโมเดล
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

class News(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {
        "message": "Fake News Detection API",
        "status": "running",
        "endpoints": {
            "/predict": "POST - ตรวจสอบข่าวปลอม"
        }
    }

@app.post("/predict")
def predict(news: News):
    X = vectorizer.transform([news.text])
    
    # ใช้ decision_function เพื่อหา confidence score
    decision = model.decision_function(X)[0]
    
    # แปลง decision score เป็น confidence percentage (0-100%)
    # decision > 0 = ข่าวจริง, decision < 0 = ข่าวปลอม
    # ใช้ sigmoid-like transformation
    import math
    confidence = 1 / (1 + math.exp(-decision))  # sigmoid
    confidence_percent = round(confidence * 100, 1)
    
    return {
        "confidence": confidence_percent,
        "decision_score": round(decision, 3)
    }
