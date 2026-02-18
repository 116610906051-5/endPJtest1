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
    
    # ตรวจสอบว่าโมเดลมี decision_function หรือไม่
    if hasattr(model, 'decision_function'):
        # สำหรับ LinearSVC หรือโมเดลที่มี decision_function
        decision = model.decision_function(X)[0]
        confidence = 1 / (1 + math.exp(-decision))
        confidence_percent = round(confidence * 100, 1)
        decision_score = round(decision, 3)
    elif hasattr(model, 'predict_proba'):
        # สำหรับ Ensemble models และโมเดลอื่นๆ ที่มี predict_proba
        proba = model.predict_proba(X)[0]
        confidence_percent = round(proba[1] * 100, 1)  # probability ของ class 1 (ข่าวจริง)
        decision_score = round(proba[1] - proba[0], 3)  # ส่งกลับความแตกต่างระหว่าง class
    else:
        # fallback สำหรับโมเดลที่ไม่มีทั้งสองแบบ
        prediction = model.predict(X)[0]
        confidence_percent = 100.0 if prediction == 1 else 0.0
        decision_score = 1.0 if prediction == 1 else -1.0

    return {
        "confidence": confidence_percent,
        "decision_score": decision_score
    }
