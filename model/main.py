from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

# เปิด CORS เพื่อให้ Frontend เรียกได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

class News(BaseModel):
    text: str

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
