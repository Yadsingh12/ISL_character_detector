from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
CORS(app)  # Allow all origins, you can configure this as needed

# Load the trained model
model_path = 'models/number_model_with_zero.keras'  # Update model path if necessary
if os.path.exists(model_path):
    number_model = tf.keras.models.load_model(model_path)
else:
    number_model = None
    print(f"Model not found at {model_path}")

# Define label encoder
number_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'noNumber']

# Route to serve the index page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not number_model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get the landmarks from the request
        data = request.get_json()
        if not data or 'landmarks' not in data:
            return jsonify({'error': 'Invalid input'}), 400

        landmarks = np.array(data['landmarks']).reshape(1, -1)  # Adjust shape as needed
        print(f"Received landmarks: {data['landmarks']}")
        
        # Make a prediction
        prediction = number_model.predict(landmarks)
        predicted_label = number_labels[np.argmax(prediction)]
        print(f"Prediction: {predicted_label}")
        
        # Check for the noNumber label
        if predicted_label == "noNumber":
            return jsonify({'prediction': "No number detected"})
        
        return jsonify({'prediction': int(predicted_label)})
    
    except Exception as e:
        print(f"Error: {str(e)}")  # Print the error to the server logs
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
