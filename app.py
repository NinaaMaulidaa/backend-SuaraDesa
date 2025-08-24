from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import joblib
import re
import os

app = Flask(__name__)
CORS(app)

# =======================
# Load Vectorizer, Model, dan Mapping
# =======================
try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except Exception as e:
    vectorizer = None
    print("ERROR: vectorizer.pkl tidak ditemukan atau corrupt:", e)

try:
    kmeans_model = joblib.load('kmeans_model.pkl')
except Exception as e:
    kmeans_model = None
    print("ERROR: kmeans_model.pkl tidak ditemukan atau corrupt:", e)

try:
    with open('cluster_label_map.pkl', 'rb') as f:
        label_mapping = pickle.load(f)
except Exception as e:
    label_mapping = None
    print("ERROR: cluster_label_map.pkl tidak ditemukan atau corrupt:", e)

# =======================
# Fungsi membersihkan teks
# =======================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =======================
# Root endpoint
# =======================
@app.route('/')
def index():
    return jsonify({"message": "Flask KMeans API is running"}), 200

# =======================
# Endpoint test
# =======================
@app.route('/test')
def test():
    if not vectorizer or not kmeans_model:
        return jsonify({"error": "Vectorizer atau model belum siap"}), 500
    try:
        sample = vectorizer.transform(["Contoh laporan pengaduan"])
        cluster = int(kmeans_model.predict(sample)[0])
        label = label_mapping.get(cluster, "Tidak diketahui") if label_mapping else "Tidak diketahui"
        return jsonify({"cluster": cluster, "label": label}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =======================
# Endpoint prediksi cluster
# =======================
@app.route('/predict', methods=['POST'])
def predict_cluster():
    if not vectorizer or not kmeans_model:
        return jsonify({"error": "Vectorizer atau model belum siap"}), 500
    try:
        data = request.get_json()
        text = data.get("text", "")
        cleaned = clean_text(text)
        vectorized = vectorizer.transform([cleaned])
        cluster = int(kmeans_model.predict(vectorized)[0])
        label = label_mapping.get(cluster, "Tidak diketahui") if label_mapping else "Tidak diketahui"

        return jsonify({
            "cluster": cluster,
            "label": label
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =======================
# Jalankan aplikasi
# =======================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
