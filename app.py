import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Must be before any TensorFlow import
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib
import logging

# Rest of your code...
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
        rgb_normalized = scaler.transform(rgb_values)
        pca_components = pca.transform(rgb_normalized)
        ph_pred = model.predict(pca_components, verbose=0)[0][0]
        logger.info(f"Predicted pH: {ph_pred}")
        return jsonify({'ph': float(ph_pred)})
    except Exception as e:
        logger.error(f"Error in /predict: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask Soil pH Prediction API")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)