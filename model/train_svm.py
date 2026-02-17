import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib

# 1. โหลดข้อมูล
df = pd.read_csv("../dataset.csv")

# ลบแถวที่มีค่า NaN
df = df.dropna()

# ใช้คอลัมน์ Title เป็น text และแปลง Verification_Status เป็น label (0/1)
texts = df["Title"]
labels = df["Verification_Status"].map({"ข่าวปลอม": 0, "ข่าวจริง": 1})

print(f"จำนวนข้อมูลทั้งหมด: {len(df)} รายการ")
print(f"ข่าวปลอม: {(labels == 0).sum()} รายการ")
print(f"ข่าวจริง: {(labels == 1).sum()} รายการ")

# 2. แบ่ง train / test
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# 3. แปลงข้อความเป็นตัวเลข (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. เทรน SVM
model = LinearSVC()
model.fit(X_train_vec, y_train)

# 5. ประเมินผล
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# 6. เซฟโมเดล
joblib.dump(model, "svm_model.pkl")
joblib.dump(vectorizer, "tfidf.pkl")

print("✅ เทรนเสร็จ และบันทึกโมเดลแล้ว")
