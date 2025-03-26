import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU usage before importing TensorFlow
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the trained model, scaler, and PCA at startup
try:
    model = tf.keras.models.load_model('soil_ph_model.h5', compile=False)
    scaler = joblib.load('rgb_scaler.joblib')
    pca = joblib.load('pca_model.joblib')
    logger.info("Model, scaler, and PCA loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model/scaler/pca: {str(e)}")
    raise  # Stop the app if loading fails

# Optional: Check server status
@app.route('/', methods=['GET'])
def home():
    return "Flask Soil pH Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract RGB data from request
        data = request.get_json()
        if 'rgb' not in data:
            return jsonify({'error': 'Missing "rgb" key in JSON'}), 400
        
        rgb_values = np.array([data['rgb']], dtype=np.float32)  # Ensure float32 for TensorFlow
        logger.info(f"Received RGB values: {rgb_values}")

        # Preprocess the input
        rgb_normalized = scaler.transform(rgb_values)
        pca_components = pca.transform(rgb_normalized)
        
        # Predict pH
        ph_pred = model.predict(pca_components, verbose=0)[0][0]
        logger.info(f"Predicted pH: {ph_pred}")
        
        return jsonify({'ph': float(ph_pred)})
    except Exception as e:
        logger.error(f"Error in /predict: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask Soil pH Prediction API")
    port = int(os.environ.get('PORT', 5000))  # Use Render's PORT env var or default to 5000
    app.run(host='0.0.0.0', port=port)