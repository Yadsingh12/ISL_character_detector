import numpy as np

# Function to load and print the contents of a .npy file
def view_npy_file(file_path):
    try:
        # Load the .npy file
        data = np.load(file_path)
        # Print the shape and contents of the array
        print(f"Shape of the array: {data.shape}")
        print("Contents of the array:")
        print(data)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# Path to the .npy file
file_path = 'C:/Users/yadau/Desktop/coding/python/openCV/sign_detect/dataset_numbers/1/1_93.npy'

# View the contents of the .npy file
view_npy_file(file_path)
