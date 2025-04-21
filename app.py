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
















import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib
import logging
import time

tf.config.set_visible_devices([], 'GPU')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

try:
    model = tf.keras.models.load_model('soil_ph_model.h5', compile=False)
    scaler = joblib.load('rgb_scaler.joblib')
    pca = joblib.load('pca_model.joblib')
    logger.info("Model, scaler, and PCA loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model/scaler/pca: {str(e)}")
    raise

@app.route('/', methods=['GET'])
def home():
    return "Flask Soil pH Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'rgb' not in data:
            return jsonify({'error': 'Missing "rgb" key in JSON'}), 400
        rgb_values = np.array([data['rgb']], dtype=np.float32)
        logger.info(f"Received RGB values: {rgb_values}")

        start_time = time.time()  # Start timing
        rgb_normalized = scaler.transform(rgb_values)
        pca_components = pca.transform(rgb_normalized)
        ph_pred = model.predict(pca_components, verbose=0)[0][0]
        end_time = time.time()  # End timing

        logger.info(f"Predicted pH: {ph_pred}, Time taken: {end_time - start_time:.2f} seconds")
        return jsonify({'ph': float(ph_pred)})
    except Exception as e:
        logger.error(f"Error in /predict: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask Soil pH Prediction API")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)