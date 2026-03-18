"""
Train CNN model for Thai Fake News Classification
CNN (Convolutional Neural Network) with multiple filter sizes for text classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import joblib
import argparse
from pathlib import Path
import json
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, roc_auc_score, roc_curve
)
from pythainlp.tokenize import word_tokenize
from tqdm import tqdm


class CNNClassifier(nn.Module):
    """CNN-based text classifier for fake news detection."""
    def __init__(self, vocab_size, embedding_dim, num_filters=100, filter_sizes=[3, 4, 5], dropout=0.3):
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
        self.sigmoid = nn.Sigmoid()
    
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
        
        # Fully connected layers
        cat = self.dropout(cat)
        out = self.relu(self.fc1(cat))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
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


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_x, batch_y in tqdm(train_loader, desc="Training", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, data_loader, device):
    """Evaluate model on data."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(data_loader, desc="Evaluating", leave=False):
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            
            all_preds.append((outputs > 0.5).cpu().numpy().flatten())
            all_labels.append(batch_y.numpy())
            all_probs.append(outputs.cpu().numpy().flatten())
    
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
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
        "tpr": tpr.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description="Train CNN for Thai Fake News Detection")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num-filters", type=int, default=100, help="Number of filters")
    parser.add_argument("--max-len", type=int, default=150, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Load data
    print("📂 Loading dataset...")
    df = pd.read_csv("dataset.csv").dropna()
    texts = df["Title"]
    labels = df["Verification_Status"].map({"ข่าวปลอม": 0, "ข่าวจริง": 1}).astype(int).values
    
    # Build vocabulary
    print("🔨 Building vocabulary...")
    word2idx = build_vocab(texts, max_vocab=5000)
    print(f"📊 Vocabulary size: {len(word2idx)}")
    
    # Convert texts to sequences
    print("🔄 Converting texts to sequences...")
    sequences = np.array([text_to_sequence(text, word2idx, args.max_len) for text in texts])
    labels_tensor = torch.FloatTensor(labels)
    sequences_tensor = torch.LongTensor(sequences)
    
    # Create dataset and split
    dataset = TensorDataset(sequences_tensor, labels_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    print(f"📊 Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    print("🔨 Creating CNN model...")
    model = CNNClassifier(
        vocab_size=len(word2idx),
        embedding_dim=args.embedding_dim,
        num_filters=args.num_filters,
        filter_sizes=[3, 4, 5],
        dropout=args.dropout
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    
    # Training loop
    print(f"🚀 Starting training for {args.epochs} epochs...")
    best_auc = 0.0
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"\n📊 Epoch {epoch + 1}/{args.epochs}")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"   Val F1: {val_metrics['f1']:.4f}")
        print(f"   Val AUC: {val_metrics['auc']:.4f}")
        
        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save(model.state_dict(), "cnn_model.pth")
            print(f"   ✅ Best model saved (AUC: {best_auc:.4f})")
    
    # Final evaluation
    print("\n🎯 Final Evaluation:")
    test_metrics = evaluate(model, val_loader, device)
    for key, value in test_metrics.items():
        if key not in ["fpr", "tpr", "confusion_matrix"]:
            print(f"   {key}: {value:.4f}")
    
    # Save artifacts
    print("💾 Saving artifacts...")
    joblib.dump(word2idx, "word2idx.pkl")
    
    metrics = {
        "model": "CNN",
        "embedding_dim": args.embedding_dim,
        "num_filters": args.num_filters,
        "max_len": args.max_len,
        "dropout": args.dropout,
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
    
    with open("cnn_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print("✅ Training complete!")
    print(f"📊 Final AUC: {test_metrics['auc']:.4f}")
    print(f"📊 Final Accuracy: {test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
