from flask import Flask, request, jsonify
from license_for_flask import predict_price

import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if 'plate_number' not in data:
        return jsonify({'error': 'Plate number is required.'}), 400
    
    plate_number = data['plate_number']
    predicted_price = predict_price(plate_number)
    
    return jsonify({
        'plate_number': plate_number,
        'predicted_price': predicted_price
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000)) 
    app.run(host='0.0.0.0', port=port)