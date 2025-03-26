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
        # Extract individual R, G, B values from JSON
        rgb = request.json['rgb']
        r = rgb['R']
        g = rgb['G']
        b = rgb['B']

        # Prepare input array in the correct format
        rgb_values = np.array([[r, g, b]])

        # Perform scaling and PCA
        rgb_normalized = scaler.transform(rgb_values)
        pca_components = pca.transform(rgb_normalized)

        # Predict pH
        ph_pred = model.predict(pca_components, verbose=0)[0][0]

        return jsonify({'ph': float(ph_pred)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Flask Soil pH Prediction API is running!")
    app.run(host='0.0.0.0', port=5000)

