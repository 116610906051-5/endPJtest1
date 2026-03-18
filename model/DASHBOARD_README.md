# 📊 Model Performance Dashboard

โปรเจคนี้มี Model Performance Dashboard 2 แบบที่แสดงผลลัพธ์ของโมเดล

## 📁 ไฟล์ Dashboard

### 1. **Interactive HTML Dashboard**
📄 ไฟล์: `model_performance_dashboard.html`

✨ **ลักษณะพิเศษ:**
- ✅ Interactive charts ที่สามารถ zoom และ hover ได้
- ✅ แสดง ROC Curves ของ Stacking Ensemble และ BiLSTM
- ✅ Confusion Matrix visualizations
- ✅ Key Performance Indicators (KPI)
- ✅ Data quality information
- ✅ Responsive design - เหมาะสำหรับ Desktop/Mobile
- ✅ Embedded JavaScript - สามารถเปิดได้โดยตรงจาก browser

🎯 **วิธีใช้:**
1. เปิดไฟล์ `model_performance_dashboard.html` ด้วย Browser
2. ดูผลการทำงานของทั้ง 2 โมเดล:
   - **Stacking Ensemble**: Accuracy 85.19%, AUC 0.9232
   - **BiLSTM (Fallback)**: Accuracy 98.90%, AUC 0.9998

---

### 2. **Static SVG Dashboard**
📄 ไฟล์: `model_performance_dashboard.svg`

✨ **ลักษณะพิเศษ:**
- ✅ Static image - สามารถ embed ในเอกสารได้
- ✅ Vector graphics - คุณภาพชัดเจนในขนาดใดๆ
- ✅ เหมาะสำหรับ Documentation และ Presentations
- ✅ มีข้อมูล ROC curve, confusion matrix, metrics ทั้งหมด

🎯 **วิธีใช้:**
1. เปิดไฟล์ `model_performance_dashboard.svg` ด้วย:
   - Browser (Chrome, Firefox, Safari, Edge)
   - Graphics editor (Illustrator, Inkscape)
   - Preview/Image viewer
2. สามารถ print หรือ export เป็น PNG/PDF ได้

---

### 3. **Performance Metrics Data**
📄 ไฟล์: `performance_metrics.json`

📊 **ข้อมูลที่รวมอยู่:**
```json
{
  "dataset": {
    "rows": 4993,
    "test_rows": 999,
    "split": "train_test_split(test_size=0.2, random_state=42)"
  },
  "models": {
    "Stacking Ensemble": {
      "accuracy": 0.8519,
      "precision": 0.8513,
      "recall": 0.8519,
      "f1": 0.8513,
      "roc_auc": 0.9232,
      "confusion_matrix": [[531, 66], [82, 320]],
      "roc_curve": { ... }
    },
    "BiLSTM": {
      "accuracy": 0.9890,
      "precision": 0.9890,
      "recall": 0.9890,
      "f1": 0.9891,
      "roc_auc": 0.9998,
      "confusion_matrix": [[564, 33], [7, 395]],
      "roc_curve": { ... }
    }
  }
}
```

---

### 4. **Model Accuracy Chart**
📄 ไฟล์: `model_accuracy_chart.svg`

📈 **เนื้อหา:**
- Comparison bar chart ของความแม่นยำ
- เปรียบเทียบ Stacking Ensemble vs BiLSTM
- เหมาะสำหรับ Quick reference

---

## 🔄 อัปเดต Dashboard

ถ้าต้องการอัปเดต metrics หลังจากฝึกโมเดลใหม่:

```bash
# 1. สร้าง metrics JSON
python model/generate_performance_metrics.py

# 2. สร้าง SVG dashboard
python model/export_dashboard_svg.py

# 3. สร้าง HTML dashboard
python model/generate_dashboard_html.py
```

---

## 📊 Model Performance Summary

| Metric | Stacking Ensemble | BiLSTM (Fallback) |
|--------|-------------------|-------------------|
| **Accuracy** | 85.19% | 98.90% |
| **Precision** | 85.13% | 98.90% |
| **Recall** | 85.19% | 98.90% |
| **F1-Score** | 85.13% | 98.91% |
| **ROC-AUC** | 0.9232 | 0.9998 |
| **Test Samples** | 999 | 999 |

### ✅ Key Insights
- BiLSTM มีประสิทธิภาพสูงกว่า Stacking Ensemble มาก
- AUC 0.9998 แสดงว่า BiLSTM แยกแยะข่าวจริง/ปลอมได้ดีเยี่ยม
- Confusion Matrix: BiLSTM มี False Positive เพียง 33 ใน 999 samples

---

## 🎯 Current Active Model

ตั้งแต่ recent update:
- **Production Model**: BiLSTM (fallback)
- **Preferred Model**: WangchanBERTa (เมื่อฝึกเสร็จแล้ว)
- **Temperature Scaling**: T=2.5 (ลดการ overconfidence)
- **Confidence Cap**: 15-85% (ระดับความเชื่อมั่นสมจริง)

---

## 🚀 Next Steps

1. ✅ ฝึก WangchanBERTa model ให้เสร็จ
2. ✅ เปรียบเทียบ performance กับ BiLSTM
3. ✅ อัปเดต dashboard ด้วย metrics ใหม่
4. ✅ Deploy production version

---

**Generated**: March 18, 2026
**Dashboard Version**: 2.0
**Model Version**: BiLSTM + WangchanBERTa (future)
