from flask import Flask, request, jsonify
import pandas as pd
import joblib
from license_for_flask import extract_features
import os
import threading

app = Flask(__name__)

# Global variables for the model and scaler
model = None
scaler = None
model_loaded = False

def load_model_and_scaler():
    """Background function to load model and scaler."""
    global model, scaler, model_loaded
    print("Loading model and scaler...")
    model = joblib.load('trained_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_loaded = True
    print("Model and scaler loaded successfully.")

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': 'Model is still loading. Please try again shortly.'}), 503
    
    data = request.get_json()
    if 'plate_number' not in data:
        return jsonify({'error': 'Plate number is required.'}), 400

    plate_number = data['plate_number']

    # Extract features using the extract_features function
    features = pd.Series(extract_features(str(plate_number))).values.reshape(1, -1)

    # Scale the features using the loaded scaler
    features_scaled = scaler.transform(features)

    # Make prediction using the loaded model
    predicted_price = model.predict(features_scaled)[0]

    return jsonify({
        'plate_number': plate_number,
        'predicted_price': predicted_price
    })

if __name__ == '__main__':
    # Start a background thread to load the model and scaler after the server starts
    threading.Thread(target=load_model_and_scaler).start()

    # Start the Flask server
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
