"""Train CNN model for Thai Fake News Classification with improved tuning pipeline."""

import argparse
import json
import pickle
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pythainlp.tokenize import word_tokenize
import scipy.stats as stats
from collections import Counter

# Fallback metrics functions if sklearn not available
def accuracy_score(y_true, y_pred):
    return (np.array(y_true) == np.array(y_pred)).mean()

def confusion_matrix(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return np.array([[tn, fp], [fn, tp]])

def f1_score(y_true, y_pred, pos_label=1):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = ((y_pred == pos_label) & (y_true == pos_label)).sum()
    fp = ((y_pred == pos_label) & (y_true != pos_label)).sum()
    fn = ((y_pred != pos_label) & (y_true == pos_label)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def precision_recall_fscore_support(y_true, y_pred, average=None):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    classes = np.unique(y_true)
    results = []
    for c in classes:
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        s = (y_true == c).sum()
        results.append((p, r, f1, s))
    return tuple(zip(*results)) if results else ((), (), (), ())

def roc_auc_score(y_true, y_score):
    y_true, y_score = np.array(y_true), np.array(y_score)
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_indices]
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    tp_cumsum = np.cumsum(y_true_sorted == 1)
    fp_cumsum = np.cumsum(y_true_sorted == 0)
    tpr = tp_cumsum / n_pos if n_pos > 0 else np.zeros_like(tp_cumsum)
    fpr = fp_cumsum / n_neg if n_neg > 0 else np.zeros_like(fp_cumsum)
    tpr = np.concatenate([[0], tpr, [1]])
    fpr = np.concatenate([[0], fpr, [1]])
    # Use numpy.trapezoid or trapz depending on numpy version
    try:
        auc = np.trapezoid(tpr, fpr)
    except AttributeError:
        auc = np.trapz(tpr, fpr)
    return auc

def roc_curve(y_true, y_score):
    y_true, y_score = np.array(y_true), np.array(y_score)
    thresholds = np.sort(np.unique(y_score))[::-1]
    tpr_list, fpr_list = [], []
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tpr_list.append(tp / n_pos if n_pos > 0 else 0)
        fpr_list.append(fp / n_neg if n_neg > 0 else 0)
    return np.array(fpr_list), np.array(tpr_list), thresholds

def train_test_split(X, y, test_size=0.2, random_state=None, stratify=None):
    """Simple stratified train_test_split without sklearn."""
    np.random.seed(random_state)
    indices = np.arange(len(X))
    
    if stratify is not None:
        stratify = np.array(stratify)
        # Group by class
        classes = np.unique(stratify)
        train_indices = []
        test_indices = []
        for cls in classes:
            cls_indices = np.where(stratify == cls)[0]
            n_test = max(1, int(len(cls_indices) * test_size))
            perm = np.random.permutation(len(cls_indices))
            test_indices.extend(cls_indices[perm[:n_test]])
            train_indices.extend(cls_indices[perm[n_test:]])
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
    else:
        perm = np.random.permutation(len(X))
        n_test = int(len(X) * test_size)
        test_indices = perm[:n_test]
        train_indices = perm[n_test:]
    
    X_train = X[train_indices] if isinstance(X, np.ndarray) else [X[i] for i in train_indices]
    X_test = X[test_indices] if isinstance(X, np.ndarray) else [X[i] for i in test_indices]
    y_train = y[train_indices] if isinstance(y, np.ndarray) else [y[i] for i in train_indices]
    y_test = y[test_indices] if isinstance(y, np.ndarray) else [y[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class CNNClassifier(nn.Module):
    """CNN-based text classifier for fake news detection."""
    def __init__(self, vocab_size, embedding_dim, num_filters=128, filter_sizes=[2, 3, 4, 5], dropout=0.4):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 1D Convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, 
                     out_channels=num_filters, 
                     kernel_size=fs) 
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = embedded.transpose(1, 2)  # (batch_size, embedding_dim, seq_len) for Conv1d
        
        # Apply multiple convolutional filters
        conv_outputs = []
        for conv in self.convs:
            # Apply convolution and ReLU activation
            conv_out = self.relu(conv(embedded))  # (batch_size, num_filters, seq_len - filter_size + 1)
            # Apply max pooling over time dimension
            pooled = torch.max(conv_out, dim=2)[0]  # (batch_size, num_filters)
            conv_outputs.append(pooled)
        
        # Concatenate all pooled outputs
        cat = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)
        
        # Fully connected layers (return logits for BCEWithLogitsLoss)
        cat = self.dropout(cat)
        out = self.relu(self.fc1(cat))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def build_vocab(texts, max_vocab=5000):
    """Build vocabulary from texts."""
    word_freq = {}
    for text in texts:
        tokens = word_tokenize(str(text), engine='newmm')
        for token in tokens:
            word_freq[token] = word_freq.get(token, 0) + 1
    
    # Sort by frequency and keep top max_vocab
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_vocab]
    word2idx = {word: idx + 2 for idx, (word, _) in enumerate(sorted_words)}
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1
    return word2idx


def text_to_sequence(text, word2idx, max_len=150):
    """Convert text to sequence of indices."""
    tokens = word_tokenize(str(text), engine='newmm')
    seq = [word2idx.get(token, 1) for token in tokens]  # 1 = <UNK>
    
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq = seq + [0] * (max_len - len(seq))  # 0 = <PAD>
    
    return seq


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_x, batch_y in tqdm(train_loader, desc="Training", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_x).squeeze(1)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, data_loader, device, threshold=0.5, temperature=1.0):
    """Evaluate model on data with configurable threshold and temperature."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_logits = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(data_loader, desc="Evaluating", leave=False):
            batch_x = batch_x.to(device)
            logits = model(batch_x).squeeze(1)
            probs = torch.sigmoid(logits / max(temperature, 1e-6))
            
            all_preds.append((probs >= threshold).cpu().numpy().astype(int).flatten())
            all_labels.append(batch_y.numpy())
            all_probs.append(probs.cpu().numpy().flatten())
            all_logits.append(logits.cpu().numpy().flatten())
    
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)
    
    accuracy = accuracy_score(labels, preds)
    prec_recall_f1 = precision_recall_fscore_support(labels, preds)
    if prec_recall_f1[0]:  # If we have results
        precision = np.mean(prec_recall_f1[0])  # Weighted average of precision
        recall = np.mean(prec_recall_f1[1])
        f1 = np.mean(prec_recall_f1[2])
    else:
        precision = recall = f1 = 0
    auc = roc_auc_score(labels, probs)
    cm = confusion_matrix(labels, preds)
    fpr, tpr, _ = roc_curve(labels, probs)
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "y_true": labels.tolist(),
        "y_prob": probs.tolist(),
        "y_logit": np.concatenate(all_logits).tolist(),
        "threshold": float(threshold),
        "temperature": float(temperature),
    }


def tune_threshold(y_true, y_prob):
    """Tune decision threshold on validation set by weighted F1."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.20, 0.80, 61):
        y_pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t, float(best_f1)


def calibrate_temperature(logits, labels):
    """Simple temperature scaling via grid search on validation logits."""
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss()

    best_temp = 1.0
    best_loss = float("inf")
    for temp in np.linspace(0.7, 3.0, 47):
        loss = criterion(logits_t / temp, labels_t).item()
        if loss < best_loss:
            best_loss = loss
            best_temp = float(temp)
    return best_temp, float(best_loss)


def main():
    parser = argparse.ArgumentParser(description="Train CNN for Thai Fake News Detection")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num-filters", type=int, default=128, help="Number of filters")
    parser.add_argument("--max-len", type=int, default=200, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate")
    parser.add_argument("--max-vocab", type=int, default=7000, help="Maximum vocabulary size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--early-stopping-patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum AUC improvement for early stopping")
    parser.add_argument("--dataset-path", type=str, default="", help="Optional custom dataset path")
    args = parser.parse_args()

    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data (robust absolute path)
    print("Loading dataset...")
    base_dir = Path(__file__).resolve().parent
    dataset_path = Path(args.dataset_path) if args.dataset_path else (base_dir / "dataset.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path).dropna()
    texts = df["Title"]
    labels = df["Verification_Status"].map({"ข่าวปลอม": 0, "ข่าวจริง": 1}).astype(int).values
    
    # Build vocabulary
    print("Building vocabulary...")
    word2idx = build_vocab(texts, max_vocab=args.max_vocab)
    print(f"Vocabulary size: {len(word2idx)}")
    
    # Convert texts to sequences
    print(" Converting texts to sequences...")
    sequences = np.array([text_to_sequence(text, word2idx, args.max_len) for text in texts])
    labels_tensor = torch.FloatTensor(labels)
    sequences_tensor = torch.LongTensor(sequences)
    
    # Stratified split (better than random_split)
    X_train, X_val, y_train, y_val = train_test_split(
        sequences,
        labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )

    train_dataset = TensorDataset(torch.LongTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.LongTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f" Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    print(" Creating CNN model...")
    model = CNNClassifier(
        vocab_size=len(word2idx),
        embedding_dim=args.embedding_dim,
        num_filters=args.num_filters,
        filter_sizes=[2, 3, 4, 5],
        dropout=args.dropout
    ).to(device)

    # BCEWithLogitsLoss + class weighting for imbalance
    n_pos = max(int(np.sum(y_train == 1)), 1)
    n_neg = max(int(np.sum(y_train == 0)), 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Training loop
    print(f" Starting training for {args.epochs} epochs...")
    best_auc = 0.0
    best_epoch = 0
    wait = 0
    best_state = None
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device, threshold=0.5, temperature=1.0)
        scheduler.step(val_metrics["auc"])
        
        print(f"\n Epoch {epoch + 1}/{args.epochs}")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"   Val F1: {val_metrics['f1']:.4f}")
        print(f"   Val AUC: {val_metrics['auc']:.4f}")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model + early stopping
        if val_metrics['auc'] > (best_auc + args.min_delta):
            best_auc = val_metrics['auc']
            best_epoch = epoch + 1
            wait = 0
            best_state = model.state_dict()
            torch.save(best_state, str(base_dir / "cnn_model.pth"))
            print(f"    Best model saved (AUC: {best_auc:.4f})")
        else:
            wait += 1
            if wait >= args.early_stopping_patience:
                print(f"    Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Calibration + threshold tuning on validation set
    base_eval = evaluate(model, val_loader, device, threshold=0.5, temperature=1.0)
    best_temp, best_temp_nll = calibrate_temperature(base_eval["y_logit"], base_eval["y_true"])
    prob_after_temp = 1 / (1 + np.exp(-np.array(base_eval["y_logit"]) / best_temp))
    best_threshold, tuned_f1 = tune_threshold(base_eval["y_true"], prob_after_temp)

    # Final evaluation
    print("\n Final Evaluation:")
    test_metrics = evaluate(model, val_loader, device, threshold=best_threshold, temperature=best_temp)
    for key, value in test_metrics.items():
        if key not in ["fpr", "tpr", "confusion_matrix", "y_true", "y_prob", "y_logit"]:
            print(f"   {key}: {value:.4f}")
    print(f"   best_epoch: {best_epoch}")
    print(f"   tuned_threshold: {best_threshold:.3f}")
    print(f"   calibrated_temperature: {best_temp:.3f}")
    print(f"   temperature_nll: {best_temp_nll:.5f}")
    print(f"   tuned_f1(weighted): {tuned_f1:.4f}")
    
    # Save artifacts
    print(" Saving artifacts...")
    with open(base_dir / "word2idx.pkl", 'wb') as f:
        pickle.dump(word2idx, f)
    
    metrics = {
        "model": "CNN",
        "embedding_dim": args.embedding_dim,
        "num_filters": args.num_filters,
        "max_len": args.max_len,
        "dropout": args.dropout,
        "max_vocab": args.max_vocab,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "temperature": best_temp,
        "temperature_nll": best_temp_nll,
        "accuracy": test_metrics['accuracy'],
        "precision": test_metrics['precision'],
        "recall": test_metrics['recall'],
        "f1": test_metrics['f1'],
        "auc": test_metrics['auc'],
        "confusion_matrix": test_metrics['confusion_matrix'],
        "roc_curve": {
            "fpr": test_metrics['fpr'],
            "tpr": test_metrics['tpr']
        }
    }

    with open(base_dir / "cnn_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    calibration = {
        "threshold": best_threshold,
        "temperature": best_temp,
        "note": "Use these values for inference-time calibration if needed",
    }
    with open(base_dir / "cnn_calibration.json", "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2, ensure_ascii=False)
    
    print(" Training complete!")
    print(f" Final AUC: {test_metrics['auc']:.4f}")
    print(f" Final Accuracy: {test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
