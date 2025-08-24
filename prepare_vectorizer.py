from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Dataset contoh, ganti dengan data laporan asli
texts = ["laporan 1","laporan 2","jalan rusak","sampah berserakan"]

# Buat dan fit vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(texts)

# Simpan ke file .pkl
with open("vectorizer.pkl","wb") as f:
    pickle.dump(vectorizer,f)

print("vectorizer.pkl berhasil dibuat")
