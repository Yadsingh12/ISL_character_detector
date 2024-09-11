import os
import shutil

print("hii")
def organize_dataset(source_dir, train_dir, test_dir, train_limit=650, test_limit=250):
    # List all character folders
    characters = [str(i) for i in range(1, 10)]
    
    for char in characters:
        char_source_dir = os.path.join(source_dir, char)
        char_train_dir = os.path.join(train_dir, char)
        char_test_dir = os.path.join(test_dir, char)

        # Create directories if they don't exist
        os.makedirs(char_train_dir, exist_ok=True)
        os.makedirs(char_test_dir, exist_ok=True)

        # Get all files for the character
        files = [f for f in os.listdir(char_source_dir) if f.endswith('.npy')]
        files.sort()  # Sort files to maintain consistency
        
        # Move files to train and test directories
        for i, file in enumerate(files):
            src_file = os.path.join(char_source_dir, file)
            if i < train_limit:
                dest_file = os.path.join(char_train_dir, file)
            elif i < train_limit + test_limit:
                dest_file = os.path.join(char_test_dir, file)
            else:
                continue
            
            shutil.move(src_file, dest_file)
            print(f"Moved {file} to {dest_file}")

# Paths to the dataset directories
source_directory = 'dataset_numbers'
train_directory = 'dataset_numbers/train'
test_directory = 'dataset_numbers/test'

# Organize the dataset
organize_dataset(source_directory, train_directory, test_directory)
