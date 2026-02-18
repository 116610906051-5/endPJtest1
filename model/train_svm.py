import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

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

print("\n" + "="*60)
print("🔍 เปรียบเทียบโมเดลแต่ละตัว")
print("="*60)

# 4. เทรนและทดสอบโมเดลแต่ละตัว
models = {
    "LinearSVC": LinearSVC(random_state=42, max_iter=2000),
    "XGBoost": XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.1, verbosity=0),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
}

model_scores = {}
for name, model in models.items():
    print(f"\n📊 {name}:")
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    model_scores[name] = accuracy
    print(f"Accuracy: {accuracy:.4f}")

# 5. Ensemble Model - Voting Classifier (Soft Voting)
print("\n" + "="*60)
print("🎯 Ensemble Model - Voting Classifier")
print("="*60)

voting_model = VotingClassifier(
    estimators=[
        ('svm', LinearSVC(random_state=42, max_iter=2000)),
        ('xgb', XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.1, verbosity=0)),
        ('rf', RandomForestClassifier(random_state=42, n_estimators=100)),
        ('lr', LogisticRegression(random_state=42, max_iter=1000))
    ],
    voting='hard'  # hard voting เพราะ LinearSVC ไม่มี predict_proba
)

voting_model.fit(X_train_vec, y_train)
y_pred_voting = voting_model.predict(X_test_vec)
voting_accuracy = accuracy_score(y_test, y_pred_voting)
print(f"Voting Ensemble Accuracy: {voting_accuracy:.4f}")

# 6. Ensemble Model - Stacking Classifier
print("\n" + "="*60)
print("🚀 Ensemble Model - Stacking Classifier")
print("="*60)

stacking_model = StackingClassifier(
    estimators=[
        ('xgb', XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.1, verbosity=0)),
        ('rf', RandomForestClassifier(random_state=42, n_estimators=100)),
        ('svm', LinearSVC(random_state=42, max_iter=2000))
    ],
    final_estimator=LogisticRegression(random_state=42, max_iter=1000)
)

stacking_model.fit(X_train_vec, y_train)
y_pred_stacking = stacking_model.predict(X_test_vec)
stacking_accuracy = accuracy_score(y_test, y_pred_stacking)
print(f"Stacking Ensemble Accuracy: {stacking_accuracy:.4f}")

# 7. สรุปผลลัพธ์
print("\n" + "="*60)
print("📈 สรุปผลลัพธ์ทั้งหมด")
print("="*60)
for name, score in model_scores.items():
    print(f"{name:25s}: {score:.4f}")
print(f"{'Voting Ensemble':25s}: {voting_accuracy:.4f}")
print(f"{'Stacking Ensemble':25s}: {stacking_accuracy:.4f}")

# 8. เลือกโมเดลที่ดีที่สุด
best_model_name = max(
    [*model_scores.items(), ('Voting Ensemble', voting_accuracy), ('Stacking Ensemble', stacking_accuracy)],
    key=lambda x: x[1]
)[0]
print(f"\n🏆 โมเดลที่ดีที่สุด: {best_model_name} ({max(voting_accuracy, stacking_accuracy, *model_scores.values()):.4f})")

# 9. Classification Report สำหรับโมเดลที่ดีที่สุด
print("\n" + "="*60)
print(f"📊 Classification Report - {best_model_name}")
print("="*60)
if best_model_name == "Voting Ensemble":
    print(classification_report(y_test, y_pred_voting, target_names=['ข่าวปลอม', 'ข่าวจริง']))
    final_model = voting_model
elif best_model_name == "Stacking Ensemble":
    print(classification_report(y_test, y_pred_stacking, target_names=['ข่าวปลอม', 'ข่าวจริง']))
    final_model = stacking_model
else:
    best_single_model = models[best_model_name]
    y_pred_best = best_single_model.predict(X_test_vec)
    print(classification_report(y_test, y_pred_best, target_names=['ข่าวปลอม', 'ข่าวจริง']))
    final_model = best_single_model

# 10. เซฟโมเดลที่ดีที่สุด (ใช้ Stacking เป็นค่าเริ่มต้น)
joblib.dump(stacking_model, "svm_model.pkl")  # เปลี่ยนชื่อเป็น ensemble_model.pkl ถ้าต้องการ
joblib.dump(vectorizer, "tfidf.pkl")

print("\n✅ เทรนเสร็จ และบันทึกโมเดลแล้ว (Stacking Ensemble)")
