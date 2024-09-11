import os
import numpy as np

def print_npy_shapes(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                try:
                    data = np.load(file_path)
                    print(f"File: {file_path}, Shape: {data.shape}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

# Path to the dataset directory
dataset_path = 'C:/Users/yadau/Desktop/coding/python/openCV/sign_detect/dataset_numbers/1'

# Print the shape of each .npy file in the dataset
print_npy_shapes(dataset_path)
