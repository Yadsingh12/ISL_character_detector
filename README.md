# Hand Gesture Recognition

This repository contains the code and resources for a hand gesture recognition project. The project uses a trained neural network model to recognize hand gestures and predict numbers from 0 to 9, including a "no number" detection.

## Repository Structure

- **history**
  - `history.txt`: Contains the history of the development of the repository.
- **older_notebooks**: Contains notebooks used for previous models.
- **older_models**: Contains older models.
- **models**: Contains the most recent model.
- **notebooks**
  - `datasetCollection.py`: Script for collecting the dataset.
  - `trainwithzero.py`: Script for training the model with zero.
- **website**
  - **templates**
    - `index.html`: HTML template for the web interface.
  - `app.py`: Flask application for serving the model and handling predictions.
  - **models**
    - `number_model_with_zero.py`: Model script for number recognition including zero.

## Getting Started

### Prerequisites

- Python 3.7+
- Flask
- TensorFlow
- NumPy
- Flask-CORS

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/hand-gesture-recognition.git
    cd hand-gesture-recognition
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1. Ensure the trained model is placed in the `models` directory with the name `number_model_with_zero.keras`.

2. Start the Flask application:
    ```bash
    python website/app.py
    ```

3. Open your browser and navigate to `http://127.0.0.1:5000` to access the hand gesture recognition interface.

### Usage

- The web interface allows you to use your webcam to capture hand gestures.
- The model predicts the number based on the hand gesture and displays the result on the screen.

### How website Works

User Interaction:
The user accesses the web interface at http://127.0.0.1:5000.
The index.html page is served, displaying the video feed and a canvas for drawing.
Prediction Process:
The user performs a hand gesture in front of the webcam.
The landmarks of the hand gesture are captured and sent to the /predict endpoint via a POST request.
The Flask application processes the request, uses the TensorFlow model to predict the number, and returns the result.
The prediction is displayed on the web interface.

### File Descriptions

- **app.py**: Main Flask application file. It loads the trained model, handles HTTP requests, and serves the web interface.
- **index.html**: HTML template for the web interface. It includes scripts for capturing video and displaying predictions.
- **datasetCollection.py**: Script for collecting hand gesture data.
- **trainwithzero.py**: Script for training the model with hand gesture data including zero.

### Contributing

Feel free to open issues or submit pull requests if you have any suggestions or improvements.
