import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# 1. Baca dataset
df = pd.read_csv("hasil_cluster_label.csv")

# 2. Vectorize kolom isi_clean
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['isi_clean'])

# 3. Latih K-Means
k = 4
model = KMeans(n_clusters=k, random_state=42, n_init=10)
model.fit(X)

# 4. Mapping cluster ke kategori manual
mapping = {}
sample_indices = {
    "kebersihan": df[df['label'] == "kebersihan"].index[0],
    "sosial": df[df['label'] == "sosial"].index[0],
    "keamanan": df[df['label'] == "keamanan"].index[0],
    "infrastruktur": df[df['label'] == "infrastruktur"].index[0]
}
for kategori, idx in sample_indices.items():
    mapping[model.labels_[idx]] = kategori

df['label_pred'] = [mapping[c] for c in model.labels_]

# 5. Hitung metrik (dalam persen)
accuracy = accuracy_score(df['label'], df['label_pred']) * 100
precision = precision_score(df['label'], df['label_pred'], average='macro') * 100
recall = recall_score(df['label'], df['label_pred'], average='macro') * 100
f1 = f1_score(df['label'], df['label_pred'], average='macro') * 100

print("=== Evaluasi Hasil Clustering (dari isi_clean) ===")
print(f"Akurasi   : {accuracy:.2f}%")
print(f"Presisi   : {precision:.2f}%")
print(f"Recall    : {recall:.2f}%")
print(f"F1-score  : {f1:.2f}%")

# 6. Confusion Matrix
cm = confusion_matrix(df['label'], df['label_pred'], labels=list(mapping.values()))
print("\n=== Confusion Matrix ===")
print(pd.DataFrame(cm, index=list(mapping.values()), columns=list(mapping.values())))
