from tensorflow.keras.models import load_model

import cv2
import numpy as np

# Load the trained model
model = load_model(r'C:\Users\yadau\Desktop\coding\python\openCV\sign_detect\version_2.keras')  # Update this path to your model's path

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to get the predicted class
def get_prediction(image, model, threshold=0.5):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    max_confidence = np.max(predictions)
    if max_confidence >= threshold:
        predicted_class = np.argmax(predictions)
        print(predictions, predicted_class)
        return predicted_class, max_confidence
    return None, None

# OpenCV setup for capturing video
cap = cv2.VideoCapture(0)

# Load the class names
class_names = [chr(i) for i in range(65, 91)]  # A-Z

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict the class of the sign
    predicted_class, confidence = get_prediction(frame, model)
    
    # Display the result
    if predicted_class is not None:
        text = f'{class_names[predicted_class]}: {confidence:.2f}'
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Sign Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
