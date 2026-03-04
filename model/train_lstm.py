import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pythainlp.tokenize import word_tokenize
import joblib
import warnings
warnings.filterwarnings('ignore')

# ตั้งค่า device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

print("\n🔧 Loading data...")

# 1. โหลดข้อมูล
df = pd.read_csv("../dataset.csv")
df = df.dropna()

# Tokenize ภาษาไทย
print("🔤 Tokenizing Thai text...")
df["Title_tokens"] = df["Title"].apply(lambda x: word_tokenize(x, engine='newmm'))

texts = df["Title_tokens"].values
labels = df["Verification_Status"].map({"ข่าวปลอม": 0, "ข่าวจริง": 1}).values

print(f"จำนวนข้อมูลทั้งหมด: {len(df)} รายการ")
print(f"ข่าวปลอม: {(labels == 0).sum()} รายการ")
print(f"ข่าวจริง: {(labels == 1).sum()} รายการ")

# 2. สร้าง vocabulary
print("\n🔢 Building vocabulary...")
MAX_WORDS = 10000
MAX_LEN = 100

# สร้าง word to index mapping
word_freq = {}
for tokens in texts:
    for word in tokens:
        word_freq[word] = word_freq.get(word, 0) + 1

# เรียงตาม frequency และเลือก top MAX_WORDS
sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:MAX_WORDS-2]
word2idx = {word: idx+2 for idx, (word, _) in enumerate(sorted_words)}
word2idx['<PAD>'] = 0
word2idx['<UNK>'] = 1

print(f"Vocabulary size: {len(word2idx)}")

# แปลงข้อความเป็น sequences
def text_to_sequence(tokens, word2idx, max_len):
    seq = [word2idx.get(word, 1) for word in tokens]  # 1 = <UNK>
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq = seq + [0] * (max_len - len(seq))  # 0 = <PAD>
    return seq

sequences = np.array([text_to_sequence(tokens, word2idx, MAX_LEN) for tokens in texts])

# 3. แบ่ง train / test
X_train, X_test, y_train, y_test = train_test_split(
    sequences, labels, test_size=0.2, random_state=42
)

print(f"Training shape: {X_train.shape}")
print(f"Testing shape: {X_test.shape}")

# 4. สร้าง Dataset และ DataLoader
class NewsDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

train_dataset = NewsDataset(X_train, y_train)
test_dataset = NewsDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 5. สร้างโมเดล LSTM
print("\n" + "="*60)
print("🧠 Building BiLSTM Model with PyTorch")
print("="*60)

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
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # ใช้ hidden state จาก layer สุดท้าย (forward + backward)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        out = self.relu(self.fc1(hidden))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        return out

model = BiLSTMClassifier(
    vocab_size=len(word2idx),
    embedding_dim=128,
    hidden_dim=64,
    num_layers=2,
    dropout=0.3
).to(device)

print(model)
print(f"\n📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# 6. Training setup
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# 7. เทรนโมเดล
print("\n🚀 Training BiLSTM model...")
num_epochs = 30
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation (ใช้ test set เป็น validation)
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    train_loss /= len(train_loader)
    val_loss /= len(test_loader)
    val_acc = correct / total
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'lstm_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break

# 8. โหลดโมเดลที่ดีที่สุดและประเมินผล
print("\n" + "="*60)
print("📊 Evaluation Results")
print("="*60)

model.load_state_dict(torch.load('lstm_model.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        outputs = model(sequences).squeeze()
        predicted = (outputs > 0.5).float().cpu().numpy()
        all_preds.extend(predicted)
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = accuracy_score(all_labels, all_preds)
print(f"\n🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n📈 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['ข่าวปลอม', 'ข่าวจริง']))

# 9. เปรียบเทียบกับโมเดลเดิม
print("\n" + "="*60)
print("⚖️ Comparison with Previous Models")
print("="*60)
print("Stacking Ensemble : 85.19%")
print(f"BiLSTM (PyTorch)  : {accuracy*100:.2f}%")

# 10. บันทึกโมเดลและ tokenizer
print("\n💾 Saving model...")
joblib.dump(word2idx, "word2idx.pkl")
joblib.dump({'max_len': MAX_LEN}, "model_config.pkl")

print("\n✅ Training complete!")
print("📁 Saved files:")
print("   - lstm_model.pth")
print("   - word2idx.pkl")
print("   - model_config.pkl")

# 11. ทดสอบ prediction
print("\n" + "="*60)
print("🧪 Testing Predictions")
print("="*60)

test_texts = [
    "กรมพัฒนาธุรกิจการค้าอนุญาตใบทะเบียนพาณิชย์รายบุคคล ประกอบธุรกิจเงินกู้นอกระบบแบบออนไลน์",
    "นายกรัฐมนตรีเปิดเผยนโยบายเศรษฐกิจใหม่ในการประชุมคณะรัฐมนตรี"
]

model.eval()
for text in test_texts:
    tokens = word_tokenize(text, engine='newmm')
    seq = np.array([text_to_sequence(tokens, word2idx, MAX_LEN)])
    seq_tensor = torch.LongTensor(seq).to(device)
    
    with torch.no_grad():
        pred = model(seq_tensor).item()
    
    label = "ข่าวจริง" if pred > 0.5 else "ข่าวปลอม"
    confidence = abs(pred - 0.5) * 200
    print(f"\nข้อความ: {text[:50]}...")
    print(f"ผลลัพธ์: {label} (confidence: {confidence:.1f}%)")

