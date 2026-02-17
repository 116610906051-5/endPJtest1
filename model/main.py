from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import math

app = FastAPI(title="Fake News Detection API")

# เปิด CORS เพื่อให้ Frontend เรียกได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production อนุญาตทุก domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลดโมเดล (ไฟล์ต้องอยู่ในโฟลเดอร์ model)
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
    decision = model.decision_function(X)[0]

    # sigmoid แปลงเป็น %
    confidence = 1 / (1 + math.exp(-decision))
    confidence_percent = round(confidence * 100, 1)

    return {
        "confidence": confidence_percent,
        "decision_score": round(decision, 3)
    }
