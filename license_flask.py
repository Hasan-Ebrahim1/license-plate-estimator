from flask import Flask, request, jsonify
import pandas as pd
import joblib
from license_for_flask import extract_features
import os

app = Flask(__name__)

model = None
scaler = None

def load_model_and_scaler():
    """Load model and scaler only when needed."""
    global model, scaler
    if model is None or scaler is None:
        print("Loading model and scaler...")
        model = joblib.load('trained_model.pkl')
        scaler = joblib.load('scaler.pkl')
        print("Model and scaler loaded successfully.")

@app.route('/predict', methods=['POST'])
def predict():
    load_model_and_scaler()  # Load model and scaler if not already loaded

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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
