import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


class ThaiNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def evaluate(model, dataloader, device):
    model.eval()
    preds, probs, trues = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            p = torch.softmax(logits, dim=-1)
            y_pred = torch.argmax(p, dim=-1)

            preds.extend(y_pred.cpu().numpy().tolist())
            probs.extend(p[:, 1].cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="weighted")
    auc = roc_auc_score(trues, probs)
    report = classification_report(trues, preds, target_names=["ข่าวปลอม", "ข่าวจริง"], digits=4)

    return {
        "accuracy": float(acc),
        "f1_weighted": float(f1),
        "roc_auc": float(auc),
        "report_text": report,
        "y_true": trues,
        "y_pred": preds,
        "y_prob": probs,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune WangchanBERTa for Thai fake news detection")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--base-model",
        type=str,
        default="airesearch/wangchanberta-base-att-spm-uncased",
        help="Hugging Face base model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="wangchanberta_model",
        help="Directory to save fine-tuned model artifacts",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🖥️ Loading dataset...")
    df = pd.read_csv(base_dir / "dataset.csv").dropna()

    texts = df["Title"]
    labels = df["Verification_Status"].map({"ข่าวปลอม": 0, "ข่าวจริง": 1}).astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )

    print(f"📊 Train size: {len(X_train)} | Validation size: {len(X_val)}")

    print(f"🔤 Loading tokenizer/model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)

    device = torch.device("cpu")
    model.to(device)

    train_ds = ThaiNewsDataset(X_train, y_train, tokenizer, max_length=args.max_length)
    val_ds = ThaiNewsDataset(X_val, y_val, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    best_acc = -1.0
    best_metrics = None

    print("🚀 Start fine-tuning WangchanBERTa...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_batch,
            )
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))
        metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={avg_loss:.4f} | "
            f"val_acc={metrics['accuracy']:.4f} | "
            f"val_f1={metrics['f1_weighted']:.4f} | "
            f"val_auc={metrics['roc_auc']:.4f}"
        )

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_metrics = metrics
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    print("\n✅ Fine-tuning complete")
    print(f"📁 Saved model to: {output_dir}")

    if best_metrics is None:
        best_metrics = evaluate(model, val_loader, device)

    metrics_payload = {
        "model": "WangchanBERTa (fine-tuned)",
        "base_model": args.base_model,
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "accuracy": float(best_metrics["accuracy"]),
        "f1_weighted": float(best_metrics["f1_weighted"]),
        "roc_auc": float(best_metrics["roc_auc"]),
        "label_mapping": {"ข่าวปลอม": 0, "ข่าวจริง": 1},
    }

    with (base_dir / "wangchanberta_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    with (base_dir / "wangchanberta_classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(best_metrics["report_text"])

    print("\n📈 Best validation metrics")
    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
