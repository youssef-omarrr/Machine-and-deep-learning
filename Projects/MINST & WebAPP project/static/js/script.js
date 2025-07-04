/* ========== ELEMENT REFERENCES ========== */
// Grabbing references to canvas, buttons, and context
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clear');
const predictBtn = document.getElementById('predict');

// Grabbing references to result area and preview image
const resultEl = document.getElementById('result');
const previewImg = document.getElementById('preview-img');

// Global variable to store class probabilities
window.classProbabilities = [];  // Accessible from any other script

let drawing = false;


/* ========== CANVAS EVENT LISTENERS ========== */
// Handles when user is drawing with mouse
canvas.addEventListener('mousedown', () => drawing = true);
canvas.addEventListener('mouseup', () => drawing = false);
canvas.addEventListener('mouseleave', () => drawing = false);  // Stop drawing if mouse leaves canvas
canvas.addEventListener('mousemove', draw);


/* ========== DRAWING FUNCTION ========== */
// Draws a filled circle at the current mouse position while drawing
function draw(event) {
    if (!drawing) return;

    ctx.beginPath();
    ctx.arc(event.offsetX, event.offsetY, 8, 0, 2 * Math.PI);  // Circle with radius 8
    ctx.fillStyle = 'black';
    ctx.fill();
}


/* ========== CLEAR CANVAS FUNCTION ========== */
// Resets canvas to blank (white)
function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawing = false;

    // Reset result and preview image
    const oldPreview = document.getElementById('prediction-preview');
    if (oldPreview) {
        oldPreview.remove();
    }

    // Clear only the probability text (span), keep numbers in <p>
    for (let i = 0; i < 10; i++) {
        const probEl = document.getElementById(`prob${i}`);
        const nodeEl = document.getElementById(`node${i}`);
        const lineEl = document.getElementById(`line${i}`);
        if (probEl) {
            probEl.textContent = ""; // Remove just the probability label
            probEl.style.opacity = 1; // Hide the probability label
            nodeEl.style.opacity = 1; // Hide the node
            lineEl.style.opacity = 1; // Hide the line
        }
    }
}


/* ========== PREDICT FUNCTION ========== */
// This function captures the canvas drawing, sends it to the Flask backend server,
// receives the prediction result, and updates the UI with the prediction and probabilities.

function predict() {
    // Convert the canvas content to a Blob (binary large object) in PNG format
    canvas.toBlob(blob => {

        // Create a FormData object to hold the image file for sending via POST
        const formData = new FormData();
        formData.append('file', blob, 'image1.png'); // Add the blob as a file named 'image1.png'

        // Send the image to the Flask backend using a POST request
        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        })

        // Parse the JSON response from the server
        .then(response => response.json())

        // Handle the received data from the server
        .then(data => {
            if (data.Predicted !== undefined) {
                // Remove any existing preview
                const oldPreview = document.getElementById('prediction-preview');
                if (oldPreview) {
                    oldPreview.remove();
                }

                // Create wrapper for image + label
                const previewWrapper = document.createElement('div');
                previewWrapper.id = 'prediction-preview';
                previewWrapper.className = 'prediction-preview';

                // Add image element
                const img = document.createElement('img');
                img.src = '/static/received_input.png?t=' + new Date().getTime();
                img.className = 'preview-img';
                previewWrapper.appendChild(img);

                // Add label under image
                const label = document.createElement('div');
                label.textContent = 'Predicted: ' + data.Predicted;
                label.className = 'preview-label';
                previewWrapper.appendChild(label);

                // Attach preview to a fixed container
                document.body.appendChild(previewWrapper);

                // ====== Display Class Probabilities in the Nodes ======
                if (data.Probabilities) {
                    const probabilities = data.Probabilities;

                    // Store them globally for reuse
                    window.classProbabilities = probabilities;

                    // Update each node's <p> tag and apply opacity
                    // Get min and max from the list
                    const maxProb = Math.max(...probabilities);
                    const minProb = Math.min(...probabilities);

                    probabilities.forEach((prob, i) => {
                        const probEl = document.getElementById(`prob${i}`);
                        const nodeEl = document.getElementById(`node${i}`);
                        const lineEl = document.getElementById(`line${i}`);

                        // Normalize the probability between 0.2 and 1
                        let opacity = 1.0;
                        if (maxProb !== minProb) {
                            opacity = 0.2 + ((prob - minProb) / (maxProb - minProb)) * (1.0 - 0.2);
                        }

                        // Update probability text
                        if (probEl) {
                            probEl.textContent = `${(prob * 100).toFixed(2)}%`;
                            probEl.style.opacity = opacity; // Set opacity based on probability
                        }

                        if (nodeEl) {
                            nodeEl.style.opacity = opacity;
                        }

                        if (lineEl) {
                            lineEl.style.opacity = opacity;
                        }
                    });
                }
            } else {
                // If no prediction was returned, display an error
                alert("Error: Invalid response from server.");
            }
        })

        // Handle any error that occurred during fetch or processing
        .catch(error => {
            alert("Error: " + error);
        });

    }, 'image/png'); // Image format to encode canvas
}



/* ========== INITIALIZATION ========== */
// Clear the canvas on first load
window.onload = clearCanvas;
