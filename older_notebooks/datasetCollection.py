import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to create directories if they do not exist
def setup_directories(class_name):
    os.makedirs(f'dataset/{class_name}', exist_ok=True)

# Function to save landmarks with custom filenames
def save_landmarks(landmarks, class_name):
    try:
        # Get the current count of files in the directory
        file_list = os.listdir(f'dataset_numbers/{class_name}')
        count = len(file_list)*8
        
        # Generate the new filename
        filename = f'dataset_numbers/{class_name}/{class_name}_{count + 1}.npy'
        
        # Save landmarks to the new file
        data = np.array(landmarks).flatten()
        np.save(filename, data)
    except Exception as e:
        print(f"Error saving landmarks: {e}")

# Take user input for class name
class_name = input("Enter class name: ").strip().upper()
setup_directories(class_name)

# Open a webcam feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not detected.")
    exit()

print("Press 'q' to stop the execution.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Error: Failed to grab frame.")
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract hand landmarks
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])

            # Save landmarks to the specified class folder
            save_landmarks(landmarks, class_name)

    cv2.imshow('MediaPipe Hands', image)

    # Stop the execution when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
