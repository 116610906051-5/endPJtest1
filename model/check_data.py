import pandas as pd

df = pd.read_csv("dataset.csv")
df = df.dropna()

print("Total samples:", len(df))
print("\nValue counts:")
print(df["Verification_Status"].value_counts())

print("\n=== ตัวอย่างข่าวปลอม (5 samples) ===")
fake_samples = df[df["Verification_Status"] == "ข่าวปลอม"]["Title"].head(5)
for i, text in enumerate(fake_samples, 1):
    print(f"{i}. {text[:80]}...")

print("\n=== ตัวอย่างข่าวจริง (5 samples) ===")
real_samples = df[df["Verification_Status"] == "ข่าวจริง"]["Title"].head(5)
for i, text in enumerate(real_samples, 1):
    print(f"{i}. {text[:80]}...")
