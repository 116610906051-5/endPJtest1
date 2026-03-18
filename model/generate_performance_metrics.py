import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from pythainlp.tokenize import word_tokenize


def evaluate():
    df = pd.read_csv("dataset.csv").dropna()
    texts = df["Title"]
    y = df["Verification_Status"].map({"ข่าวปลอม": 0, "ข่าวจริง": 1}).astype(int)

    _, X_test, _, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=42
    )

    stacking_model = joblib.load("svm_model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    X_test_vec = tfidf.transform(X_test)
    stack_pred = stacking_model.predict(X_test_vec)

    if hasattr(stacking_model, "predict_proba"):
        stack_score = stacking_model.predict_proba(X_test_vec)[:, 1]
    elif hasattr(stacking_model, "decision_function"):
        raw = stacking_model.decision_function(X_test_vec)
        stack_score = 1 / (1 + np.exp(-raw))
    else:
        stack_score = stack_pred.astype(float)

    stack_acc = float(accuracy_score(y_test, stack_pred))
    sp, sr, sf1, _ = precision_recall_fscore_support(
        y_test, stack_pred, average="weighted", zero_division=0
    )
    stack_auc = float(roc_auc_score(y_test, stack_score))
    stack_cm = confusion_matrix(y_test, stack_pred).tolist()
    stack_fpr, stack_tpr, _ = roc_curve(y_test, stack_score)

    class BiLSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.3):
            super(BiLSTMClassifier, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                bidirectional=True,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_dim * 2, 64)
            self.fc2 = nn.Linear(64, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            embedded = self.embedding(x)
            _, (hidden, _) = self.lstm(embedded)
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            hidden = self.dropout(hidden)
            out = self.relu(self.fc1(hidden))
            out = self.dropout(out)
            out = self.sigmoid(self.fc2(out))
            return out

    word2idx = joblib.load("word2idx.pkl")
    bilstm = BiLSTMClassifier(
        vocab_size=len(word2idx),
        embedding_dim=128,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
    )
    bilstm.load_state_dict(torch.load("lstm_model.pth", map_location="cpu"))
    bilstm.eval()

    MAX_LEN = 100

    def text_to_seq(text):
        tokens = word_tokenize(text, engine="newmm")
        seq = [word2idx.get(tok, 1) for tok in tokens]
        if len(seq) > MAX_LEN:
            seq = seq[:MAX_LEN]
        else:
            seq = seq + [0] * (MAX_LEN - len(seq))
        return seq

    test_seqs = np.array([text_to_seq(t) for t in X_test.tolist()])
    with torch.no_grad():
        bilstm_probs = bilstm(torch.LongTensor(test_seqs)).squeeze().numpy()

    bilstm_pred = (bilstm_probs > 0.5).astype(int)
    bilstm_acc = float(accuracy_score(y_test, bilstm_pred))
    bp, br, bf1, _ = precision_recall_fscore_support(
        y_test, bilstm_pred, average="weighted", zero_division=0
    )
    bilstm_auc = float(roc_auc_score(y_test, bilstm_probs))
    bilstm_cm = confusion_matrix(y_test, bilstm_pred).tolist()
    bilstm_fpr, bilstm_tpr, _ = roc_curve(y_test, bilstm_probs)

    metrics = {
        "dataset": {
            "rows": int(len(df)),
            "test_rows": int(len(X_test)),
            "split": "train_test_split(test_size=0.2, random_state=42)",
            "label_mapping": {"ข่าวปลอม": 0, "ข่าวจริง": 1},
        },
        "models": {
            "Stacking Ensemble": {
                "accuracy": stack_acc,
                "precision_weighted": float(sp),
                "recall_weighted": float(sr),
                "f1_weighted": float(sf1),
                "roc_auc": stack_auc,
                "confusion_matrix": stack_cm,
                "roc_curve": {"fpr": stack_fpr.tolist(), "tpr": stack_tpr.tolist()},
            },
            "BiLSTM": {
                "accuracy": bilstm_acc,
                "precision_weighted": float(bp),
                "recall_weighted": float(br),
                "f1_weighted": float(bf1),
                "roc_auc": bilstm_auc,
                "confusion_matrix": bilstm_cm,
                "roc_curve": {"fpr": bilstm_fpr.tolist(), "tpr": bilstm_tpr.tolist()},
            },
        },
    }

    with open("performance_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Saved performance_metrics.json")
    print(
        f"Stacking Ensemble: acc={stack_acc:.4f}, auc={stack_auc:.4f}, f1={float(sf1):.4f}"
    )
    print(f"BiLSTM: acc={bilstm_acc:.4f}, auc={bilstm_auc:.4f}, f1={float(bf1):.4f}")


if __name__ == "__main__":
    evaluate()
