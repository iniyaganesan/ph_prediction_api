from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model, scaler, and PCA
model = tf.keras.models.load_model('soil_ph_model.h5', compile=False)
scaler = joblib.load('rgb_scaler.joblib')
pca = joblib.load('pca_model.joblib')

# Optional: Check server status
@app.route('/', methods=['GET'])
def home():
    return "Flask Soil pH Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['rgb']  # JSON should have "rgb" key
        rgb_values = np.array([data])
        rgb_normalized = scaler.transform(rgb_values)
        pca_components = pca.transform(rgb_normalized)
        ph_pred = model.predict(pca_components, verbose=0)[0][0]
        return jsonify({'ph': float(ph_pred)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Flask Soil pH Prediction API is running!")  # <--- Add this line
    app.run(host='0.0.0.0', port=5000)
