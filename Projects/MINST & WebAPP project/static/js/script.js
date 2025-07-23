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
let hasDrawn = false;


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
    hasDrawn = true;       // <-- mark that we drew something
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
    hasDrawn = false;      // <-- reset whenever we clear

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

/* ========== FEEDBACK FUNCTION ========== */
function sendFeedback(correctLabel) {
    canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('file', blob, 'corrected.png');
        formData.append('correct_label', correctLabel);

        fetch('http://127.0.0.1:5000/feedback', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (response.ok) {
                alert('Thank you! Feedback received.');
            } else {
                alert('Error sending feedback.');
            }
        })
        .catch(err => {
            console.error(err);
            alert('Server error.');
        });
    }, 'image/png');
}

/* ========== PREDICT FUNCTION ========== */
// This function captures the canvas drawing, sends it to the Flask backend server,
// receives the prediction result, and updates the UI with the prediction and probabilities.

function predict() {
    if (!hasDrawn) {
        alert("Error: Canvas is empty. Please draw a digit first.");
        return;
    }

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

                // === Add correct/wrong buttons ===
                const correct = document.createElement('button');
                correct.id = 'correct_btn';
                correct.className = 'preview-btns';
                correct.setAttribute('aria-label', 'Correct prediction');
                correct.textContent = '✓';

                const wrong = document.createElement('button');
                wrong.id = 'wrong_btn';
                wrong.className = 'preview-btns';
                wrong.setAttribute('aria-label', 'Wrong prediction');
                wrong.textContent = '❌';

                const btnWrapper = document.createElement('div');
                btnWrapper.className = 'btn-wrapper';
                btnWrapper.appendChild(wrong);
                btnWrapper.appendChild(correct);

                previewWrapper.appendChild(btnWrapper);
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
                            opacity = 0.3 + ((prob - minProb) / (maxProb - minProb)) * (1.0 - 0.3);
                        }

                        // Update probability text
                        if (probEl) {
                            probEl.classList.add('probability');
                            probEl.textContent = `${(prob * 100).toFixed(2)}%`;
                            probEl.style.opacity = opacity; // Set opacity based on probability
                        }

                        if (nodeEl) {
                            nodeEl.classList.add('node');
                            nodeEl.style.opacity = opacity;
                        }

                        if (lineEl) {
                            lineEl.classList.add('line');
                            lineEl.style.opacity = opacity;
                        }
                    });
                }
            
            // === Add functionality ===
            // Right (✓) button – behaves like Clear
            correct.addEventListener('click', () => {
                clearCanvas();
            });

            // Left (❌) button – re-sends the canvas to backend
            wrong.addEventListener('click', () => {
                
                // Create the semi-transparent background overlay
                const overlay = document.createElement('div');
                overlay.className = 'keypad-overlay';

                // Create the container for keypad and label
                const keypadContainer = document.createElement('div');
                keypadContainer.className = 'keypad-container';

                // --- Close button (X) ---
                const closeBtn = document.createElement('button');
                closeBtn.className = 'keypad-close-btn';
                closeBtn.setAttribute('aria-label', 'Close keypad');
                closeBtn.textContent = '✕';
                closeBtn.addEventListener('click', () => {
                    clearCanvas();            // clear the canvas
                    document.body.removeChild(overlay);  // remove keypad
                });
                keypadContainer.appendChild(closeBtn);

                // Heading text
                const heading = document.createElement('div');
                heading.className = 'keypad-heading';
                heading.textContent = 'Send the correct answer:';
                keypadContainer.appendChild(heading);

                // Display where user's input will show
                const inputDisplay = document.createElement('div');
                inputDisplay.className = 'keypad-display';
                inputDisplay.textContent = '';
                keypadContainer.appendChild(inputDisplay);

                // Create the keypad grid
                const keypad = document.createElement('div');
                keypad.className = 'keypad-grid';

                // Define keypad buttons in display order
                const buttons = [
                    '1', '2', '3',
                    '4', '5', '6',
                    '7', '8', '9',
                    'clear', '0', 'send'
                ];

                // Generate and style each button
                buttons.forEach(label => {
                    const btn = document.createElement('button');
                    btn.textContent = label;
                    btn.setAttribute('aria-label', label);

                    // Apply special classes for styling
                    if (label === 'clear') {
                        btn.classList.add('keypad-clear');
                    } else if (label === 'send') {
                        btn.classList.add('keypad-send');
                    }

                    // Attach behavior for each type of button
                    if (label === 'clear') {
                        btn.addEventListener('click', () => {
                            inputDisplay.textContent = '';
                        });

                    } else if (label === 'send') {
                        btn.addEventListener('click', () => {
                            const input = inputDisplay.textContent.trim();
                            if (input.length === 1 && /^[0-9]$/.test(input)) {
                                sendFeedback(input); // send to Flask
                                document.body.removeChild(overlay); // remove keypad
                                clearCanvas(); // Clear canvas
                            } else {
                                alert('Please enter a single digit (0–9).');
                            }
                        });
                    } else {
                        // Digit button appends input if empty
                        btn.addEventListener('click', () => {
                            if (inputDisplay.textContent.length < 1) {
                                inputDisplay.textContent += label;
                            }
                        });
                    }

                    keypad.appendChild(btn);
                });

            // Assemble keypad and overlay
            keypadContainer.appendChild(keypad);
            overlay.appendChild(keypadContainer);
            document.body.appendChild(overlay);
        });



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
