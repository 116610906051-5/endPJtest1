from main import *
import pandas as pd

# โหลดดาต้าจริง
df = pd.read_csv("../dataset.csv")
df = df.dropna()

# ทดสอบกับข้อความที่อยู่ในดาต้าเซต
samples = [
    ("ข่าวปลอม", df[df['Verification_Status']=='ข่าวปลอม']['Title'].iloc[0]),
    ("ข่าวปลอม", df[df['Verification_Status']=='ข่าวปลอม']['Title'].iloc[1]),
    ("ข่าวจริง", df[df['Verification_Status']=='ข่าวจริง']['Title'].iloc[0]),
    ("ข่าวจริง", df[df['Verification_Status']=='ข่าวจริง']['Title'].iloc[1]),
]

print("=" * 80)
print("ทดสอบโมเดลกับข้อความจากดาต้าเซต")
print("=" * 80)

for true_label, text in samples:
    result = predict(News(text=text))
    pred_label = result['label']
    confidence = result['confidence']
    raw_score = result['raw_score']
    
    status = "✅" if pred_label == true_label else "❌"
    print(f"\n{status} True: {true_label} | Pred: {pred_label} ({confidence}%)")
    print(f"   Text: {text[:60]}...")
    print(f"   Raw score: {raw_score:.4f}")
