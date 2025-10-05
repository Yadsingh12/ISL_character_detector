// === MODAL JAVASCRIPT LOGIC ===
const modal = document.getElementById("instructions-modal");
const btn = document.getElementById("help-button");
const closeButton = document.getElementsByClassName("close-button")[0];

// When the user clicks the Help button, open the modal 
btn.onclick = function() {
  modal.style.display = "block";
}

// When the user clicks on the close (x) button, close the modal
closeButton.onclick = function() {
  modal.style.display = "none";
}

// When the user clicks anywhere outside of the modal, close it
window.onclick = function(event) {
  if (event.target == modal) {
    modal.style.display = "none";
  }
}
// === END MODAL JAVASCRIPT LOGIC ===


// === MEDIAPIPE & PREDICTION LOGIC ===
const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
const predictionElement = document.getElementById('prediction');
const statusElement = document.getElementById('status');

let latestLandmarks = null;
let lastSentTime = 0;
// Using 1000ms (1 second) as the delay between prediction calls, as requested.
const sendInterval = 1000; 
let isSending = false;

// Function to handle MediaPipe results
function onResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Apply the flip for the canvas context for a mirrored user view
    canvasCtx.scale(-1, 1);
    canvasCtx.translate(-canvasElement.width, 0);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        statusElement.innerHTML = `Hand detected. Waiting for analysis interval...`;
        for (let i = 0; i < results.multiHandLandmarks.length; i++) {
            let landmarks = results.multiHandLandmarks[i];
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 5 });
            drawLandmarks(canvasCtx, landmarks, { color: '#FF0000', lineWidth: 2 });

            // Preprocessing: Extract absolute coords, then flip X (to match your existing model)
            // The X flip (1 - value) is crucial here because the model expects non-mirrored data.
            latestLandmarks = landmarks.map(lm => [lm.x, lm.y, lm.z]).flat();
            latestLandmarks = latestLandmarks.map((value, index) =>
                index % 3 === 0 ? 1 - value : value
            );
        }
    } else {
        statusElement.innerHTML = 'No hand detected. Please show your hand.';
        predictionElement.innerText = '?';
        latestLandmarks = null;
    }
    canvasCtx.restore();
}

// Function to send landmarks to the backend (now called recursively using setTimeout)
async function sendLandmarksToBackend() {
    
    if (latestLandmarks && !isSending) {
        
        isSending = true; 
        
        const spinner = `<div class="spinner"></div>`;
        // UX Improvement: Show sending status with spinner
        statusElement.innerHTML = `Analyzing... ${spinner}`;
        const spinnerElement = document.querySelector('.spinner');
        if (spinnerElement) spinnerElement.style.display = 'inline-block';
        
        try {
            // Using exponential backoff for API call robustness
            const maxRetries = 5;
            let currentDelay = 1000;
            let response = null;
            
            for (let i = 0; i < maxRetries; i++) {
                try {
                    response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ landmarks: latestLandmarks })
                    });
                    
                    if (response.status !== 429 && response.ok) {
                        break; // Success or non-throttling error, exit loop
                    }
                    if (response.status === 429 && i < maxRetries - 1) {
                        // Throttling error, wait and retry
                        await new Promise(resolve => setTimeout(resolve, currentDelay));
                        currentDelay *= 2; // Exponential backoff
                    } else {
                        throw new Error(`HTTP error! Status: ${response.status}`);
                    }
                } catch (fetchError) {
                    if (i === maxRetries - 1) throw fetchError;
                    // Wait and retry for network/transient errors
                    await new Promise(resolve => setTimeout(resolve, currentDelay));
                    currentDelay *= 2;
                }
            }
            
            if (!response || !response.ok) {
                 throw new Error(`Final request failed with status: ${response ? response.status : 'No response'}`);
            }
            
            const data = await response.json();
            
            predictionElement.innerText = data.prediction;
            statusElement.innerHTML = `Predicted: ${data.prediction}. Ready for next gesture.`;

        } catch (error) {
            console.error('Prediction Error:', error);
            statusElement.innerHTML = `Error: Cannot reach server. ${spinner}`;
            predictionElement.innerText = 'ERR';
        } finally {
            isSending = false;
            const spinnerElement = document.querySelector('.spinner');
            if (spinnerElement) spinnerElement.style.display = 'none';
        }
    }
    
    // Schedule the next check/send after the defined interval
    setTimeout(sendLandmarksToBackend, sendInterval);
}


// MediaPipe Hands setup
const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});
hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7
});
hands.onResults(onResults);

// Camera setup
const camera = new Camera(videoElement, {
    onFrame: async () => {
        // MediaPipe runs as fast as possible to get the latest landmarks
        await hands.send({ image: videoElement });
    },
    width: 640,
    height: 480
});

// Start the camera and initialize the prediction loop
camera.start().then(() => {
    statusElement.innerHTML = 'Camera started. Show your hand now.';
    // CRITICAL: Start the recursive prediction loop after the camera starts
    setTimeout(sendLandmarksToBackend, sendInterval); 
}).catch(e => {
    statusElement.innerHTML = 'ERROR: Camera permission denied or device not found.';
    console.error(e);
});
