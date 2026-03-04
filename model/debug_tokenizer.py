from pythainlp.tokenize import word_tokenize
import joblib
import torch
import numpy as np

# โหลด word2idx และ model config
word2idx = joblib.load("word2idx.pkl")
model_config = joblib.load("model_config.pkl")
MAX_LEN = model_config['max_len']

def text_to_sequence(tokens, word2idx, max_len):
    seq = [word2idx.get(word, 1) for word in tokens]  # 1 = <UNK>
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq = seq + [0] * (max_len - len(seq))
    return seq

# ทดสอบกับข้อความจริง
test_fake = "กรมพัฒนาธุรกิจการค้าอนุญาตใบทะเบียนพาณิชย์รายบุคคล ประกอบธุรกิจเงินกู้นอกระบบแบบออนไลน์"
test_real = "นายกรัฐมนตรีเปิดเผยนโยบายเศรษฐกิจใหม่ในการประชุมคณะรัฐมนตรี"

tokens_fake = word_tokenize(test_fake, engine='newmm')
tokens_real = word_tokenize(test_real, engine='newmm')

print("Fake news tokens:", tokens_fake[:10])
print("Real news tokens:", tokens_real[:10])

seq_fake = text_to_sequence(tokens_fake, word2idx, MAX_LEN)
seq_real = text_to_sequence(tokens_real, word2idx, MAX_LEN)

print(f"\nFake sequence (first 10): {seq_fake[:10]}")
print(f"Real sequence (first 10): {seq_real[:10]}")

# ตรวจสอบว่ามีคำที่รู้จักหรือไม่
known_fake = sum(1 for token in tokens_fake if token in word2idx)
known_real = sum(1 for token in tokens_real if token in word2idx)

print(f"\nFake news: {known_fake}/{len(tokens_fake)} words in vocab ({known_fake/len(tokens_fake)*100:.1f}%)")
print(f"Real news: {known_real}/{len(tokens_real)} words in vocab ({known_real/len(tokens_real)*100:.1f}%)")

# ตรวจสอบตัวอย่างคำที่มี
print("\nSample words in vocabulary:")
for word in list(word2idx.keys())[:20]:
    print(f"  '{word}' -> {word2idx[word]}")
