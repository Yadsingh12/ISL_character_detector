import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(r'C:\Users\yadau\Desktop\coding\python\openCV\sign_detect\version_2.keras')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

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

    # Convert the BGR image to RGB before processing.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get the bounding box coordinates of the hand
            x_min = min([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])

            height, width, _ = frame.shape
            x, y = int(x_min * width), int(y_min * height)
            w, h = int((x_max - x_min) * width), int((y_max - y_min) * height)

            # Extract the hand region
            hand_region = frame[y:y+h, x:x+w]

            # Predict the class of the sign
            predicted_class, confidence = get_prediction(hand_region, model)
            
            # Display the result
            if predicted_class is not None:
                text = f'{class_names[predicted_class]}: {confidence:.2f}'
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display the original frame
    cv2.imshow('Sign Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
