import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Maximum file limits
max_train_files = 800
max_val_files = 200
max_test_files = 200

# Function to create class subdirectories for train, validation, and test
def setup_directories(class_name):
    os.makedirs(f'dataset_numbers/train/{class_name}', exist_ok=True)
    os.makedirs(f'dataset_numbers/validation/{class_name}', exist_ok=True)
    os.makedirs(f'dataset_numbers/test/{class_name}', exist_ok=True)

# Function to save landmarks with custom filenames in class subfolders
def save_landmarks(landmarks, class_name, folder_type, count):
    try:
        # Generate the new filename in the class subfolder
        filename = f'dataset_numbers/{folder_type}/{class_name}/{class_name}_{count + 1}.npy'

        # Save landmarks to the new file
        data = np.array(landmarks).flatten()
        np.save(filename, data)
    except Exception as e:
        print(f"Error saving landmarks: {e}")

# Function to count the number of files in the class subfolders for each folder type
def count_files(class_name):
    train_count = len(os.listdir(f'dataset_numbers/train/{class_name}'))
    val_count = len(os.listdir(f'dataset_numbers/validation/{class_name}'))
    test_count = len(os.listdir(f'dataset_numbers/test/{class_name}'))
    return train_count, val_count, test_count

# Take user input for class name
class_name = input("Enter class name: ").strip().upper()
setup_directories(class_name)

# Get current file counts
train_count, val_count, test_count = count_files(class_name)

# Check if the maximum file limits are already reached
if train_count >= max_train_files and val_count >= max_val_files and test_count >= max_test_files:
    print(f"Already have enough files for class '{class_name}'.")
    print(f"Training: {train_count}, Validation: {val_count}, Testing: {test_count}")
    exit()

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

            # Save landmarks to the appropriate folder based on current counts
            if train_count < max_train_files:
                save_landmarks(landmarks, class_name, 'train', train_count)
                train_count += 1
                print(f"Saved to training. Total: {train_count}/{max_train_files}")
            elif val_count < max_val_files:
                save_landmarks(landmarks, class_name, 'validation', val_count)
                val_count += 1
                print(f"Saved to validation. Total: {val_count}/{max_val_files}")
            elif test_count < max_test_files:
                save_landmarks(landmarks, class_name, 'test', test_count)
                test_count += 1
                print(f"Saved to testing. Total: {test_count}/{max_test_files}")
            else:
                print(f"File limit reached for class '{class_name}'.")
                cap.release()
                cv2.destroyAllWindows()
                exit()

    cv2.imshow('MediaPipe Hands', image)

    # Stop the execution when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
