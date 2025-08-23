from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import joblib
import re

app = Flask(__name__)
CORS(app)  # Izinkan semua domain (bisa dibatasi nanti)

# =======================
# Load Vectorizer, Model, dan Mapping
# =======================
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

kmeans_model = joblib.load('kmeans_model.pkl')

with open('cluster_label_map.pkl', 'rb') as f:
    label_mapping = pickle.load(f)

# =======================
# Fungsi untuk membersihkan teks
# =======================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =======================
# Endpoint root untuk test service
# =======================
@app.route('/')
def index():
    return jsonify({"message": "Flask KMeans API is running"}), 200

# =======================
# Endpoint untuk prediksi cluster
# =======================
@app.route('/predict', methods=['POST'])
def predict_cluster():
    try:
        data = request.get_json()
        text = data.get("text", "")
        cleaned = clean_text(text)
        vectorized = vectorizer.transform([cleaned])
        cluster = int(kmeans_model.predict(vectorized)[0])
        label = label_mapping.get(cluster, "Tidak diketahui")

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
    # Tidak perlu debug di server Render
    app.run(host='0.0.0.0', port=5000)
