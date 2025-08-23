import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# === Data laporan ===
data = {
    'laporan': [
        'sampah menumpuk belum diangkut, sediakan tempat pembuangan sampah perlu diadakan kebersihan',
        'butuh bantuan sosial untuk keluarga tidak mampu',
        'pencurian sepeda motor di parkiran masjid saat salat Jumat perlu tindak lanjut dari bagian keamanan',
        'jalan rusak parah di depan sekolah dasar perlu diadakan perbaikan fasilitas',
        'pencemaran air dari limbah peternakan warga perlu diadakan kebersihan'
    ]
}
df = pd.DataFrame(data)

# === Tahap 1: Preprocessing TF-IDF ===
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['laporan'])

# === Tahap 2: KMeans Clustering ===
k = 4
model = KMeans(n_clusters=k, random_state=42, n_init=10)
model.fit(X)

# === Tahap 3: Perhitungan jarak ke centroid ===
distances = pairwise_distances(X, model.cluster_centers_, metric='euclidean')

# === Tahap 4: Mapping label cluster ke kategori manual ===
# Indeks dokumen yang dipakai untuk menentukan kategori:
# D1 -> Kebersihan, D2 -> Sosial, D3 -> Keamanan, D4 -> Infrastruktur
mapping_manual = {}
cluster_examples = {
    "Kebersihan": 0,     # index D1
    "Sosial": 1,         # index D2
    "Keamanan": 2,       # index D3
    "Infrastruktur": 3   # index D4
}

for kategori, idx_doc in cluster_examples.items():
    label_asli = model.labels_[idx_doc]
    mapping_manual[label_asli] = kategori

# Tambahkan kategori hasil mapping ke dataframe
df['cluster_label'] = model.labels_
df['kategori'] = df['cluster_label'].map(mapping_manual)

# === Tahap 5: Cetak hasil ===
print("=== Jarak tiap laporan ke centroid (Euclidean) ===")
for i, row in enumerate(distances):
    jarak_str = " | ".join([f"C{j}: {dist:.4f}" for j, dist in enumerate(row)])
    print(f"D{i+1} -> {jarak_str}")

print("\n=== Hasil Clustering dengan Kategori ===")
for kategori in sorted(set(mapping_manual.values())):
    print(f"\n{kategori}:")
    for lap in df[df['kategori'] == kategori]['laporan']:
        print(f" - {lap}")
