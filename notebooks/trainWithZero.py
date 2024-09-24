import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Directory structure
train_dir = 'dataset_numbers/train'
test_dir = 'dataset_numbers/test'

# Classes for digits 0-9 and 'noNumber'
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'noNumber']
num_classes = len(classes)

def load_data(directory):
    data = []
    labels = []
    for class_index, class_name in enumerate(classes):
        class_folder = os.path.join(directory, class_name)
        for npy_file in os.listdir(class_folder):
            file_path = os.path.join(class_folder, npy_file)
            hand_landmarks = np.load(file_path)

            # Debugging: Print the shape of the loaded landmarks
            print(f"Loaded {npy_file}: shape {hand_landmarks.shape}")

            if hand_landmarks.shape == (63,):
                data.append(hand_landmarks)  # Already in the correct shape
                labels.append(class_index)  # Only append label if data is valid
            else:
                print(f"Skipping {npy_file} due to unexpected shape: {hand_landmarks.shape}")

    return np.array(data), np.array(labels)


# Load training and test data
X_train, y_train = load_data(train_dir)
X_test, y_test = load_data(test_dir)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Define the model with input shape 63

model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),  # Input layer
    BatchNormalization(),  # Normalizing activations
    Dropout(0.3),  # Regularization
    Dense(64, activation='relu'),  # Hidden layer
    BatchNormalization(),  # Normalizing activations
    Dropout(0.3),  # Regularization
    Dense(32, activation='relu'),  # Additional hidden layer for more complexity
    Dropout(0.3),  # Regularization
    Dense(num_classes, activation='softmax')  # Output layer for 11 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Save the model
model.save('number_model_with_zero.keras')

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification report
print(classification_report(y_true, y_pred_classes))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
