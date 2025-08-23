import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import pickle
import joblib
import numpy as np
import re

# 1. Load data
df = pd.read_csv('hasil_cluster_label.csv')

# 2. Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['isi_clean'] = df['isi_clean'].astype(str).apply(clean_text)

# 3. TF-IDF
corpus = df['isi_clean'].tolist()
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# 4. Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
predicted_clusters = kmeans.fit_predict(X)

# 5. Mapping cluster hasil KMeans → label asli
df['predicted_cluster'] = predicted_clusters
true_labels = df['cluster'].tolist()
conf_matrix = confusion_matrix(true_labels, predicted_clusters)

# mapping[cluster_id] = label_name
mapping = {}
for true_label in sorted(set(true_labels)):
    mapped_cluster = np.argmax(conf_matrix[true_label])
    label_name = df[df['cluster'] == true_label]['label'].iloc[0]
    mapping[mapped_cluster] = label_name

print("Mapping KMeans Cluster → Label:", mapping)

# 6. Simpan model dan vectorizer
joblib.dump(kmeans, 'kmeans_model.pkl')
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('cluster_label_map.pkl', 'wb') as f:
    pickle.dump(mapping, f)

print("Model, vectorizer, dan mapping berhasil disimpan!")
