import json
import joblib
import numpy as np
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load models and scalers
model1_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model2_path = os.path.join(os.path.dirname(__file__), 'heart_rate_emotion_model.pkl')
scaler1_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
scaler2_path = os.path.join(os.path.dirname(__file__), 'heart_rate_scaler.pkl')

model1 = joblib.load(model1_path)
model2 = joblib.load(model2_path)
scaler1 = joblib.load(scaler1_path)
scaler2 = joblib.load(scaler2_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_values = data.get('values', [])
        model_choice = data.get('model', 'model1')  # model1 or model2

        # Validate input length based on model
        if model_choice == 'model2':
            if len(input_values) != 17:
                
                return jsonify({'error': f'For model2, input must contain exactly 17 value. Received {len(input_values)}.'}), 400
        else:
            if len(input_values) != 187:
                return jsonify({'error': f'For model1, input must contain exactly 187 values. Received {len(input_values)}.'}), 400

        input_array = np.array(input_values).reshape(1, -1)

        # Select model and scaler
        if model_choice == 'model2':
            scaled_input = scaler2.transform(input_array)
            prediction = int(model2.predict(scaled_input)[0])
        else:
            scaled_input = scaler1.transform(input_array)
            prediction = int(model1.predict(scaled_input)[0])

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': f'Error making prediction: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
