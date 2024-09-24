
import tensorflow as tf
import os

model_path = 'models/number_model_copy.keras'
if not os.path.exists(model_path):
    print(f"Model file not found at {model_path}")
else:
    print("Model file found. Proceeding to load.")

# Load number model
try:
    number_model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    number_model = None

number_model.save('models/number_model_copy.h5')
