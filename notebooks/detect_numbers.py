import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os

print("\n  hii\n")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

print("Loading the model...")

# Check if the model file exists
model_path = 'models/number_model.keras'
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

# Label encoders
number_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Start capturing video
cap = cv2.VideoCapture(0)

def process_landmarks(landmarks):
    flattened_landmarks = []
    for lm in landmarks:
        flattened_landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(flattened_landmarks)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        landmarks_list = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = process_landmarks(hand_landmarks.landmark)
            landmarks_list.append(landmarks)

        if len(landmarks_list) == 1:
            input_data = np.array(landmarks_list[0]).reshape(1, -1)
            print(f"Input data shape: {input_data.shape}")
            try:
                prediction = number_model.predict(input_data)
                predicted_label = number_labels[np.argmax(prediction)]
                print("Single Hand Detected:", predicted_label)
            except Exception as e:
                print(f"Error during prediction: {e}")
                predicted_label = ""
        else:
            predicted_label = ""

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, str(predicted_label), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        print("No Hands Detected")

    cv2.imshow('Single Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
