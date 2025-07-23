from flask import Flask, request, jsonify, render_template
import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ImageOps
import os, time

from helpers import build_model, Preprocessing_input, REtrain_model

app = Flask(__name__)

# Create an instance of the model
model = build_model()

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "MNIST_Model.pth")
model.load_state_dict(torch.load(MODEL_PATH))

model.eval()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# Define pre-processing
transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Serve the drawing page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error' : 'No image file provided'}), 400
    
    file = request.files['file']
    
    img = Image.open(file.stream)
    img_tensor = Preprocessing_input(img, transform)
    
    if img_tensor is None:
        return jsonify({'prediction': 'blank input'})
    
    with torch.inference_mode():
        logits = model(img_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
    return jsonify({
        "Predicted" : int(predicted_class),
        "Probabilities": probabilities.squeeze().tolist()  # convert tensor to list
        })
    
# Route to receive feedback from the front-end
@app.route('/feedback', methods=['POST'])
def feedback():
    # Validate that both the image file and correct label are present in the POST request
    if 'file' not in request.files or 'correct_label' not in request.form:
        return jsonify({'error': 'Missing image or correct label'}), 400

    # Extract image and correct label from request
    file = request.files['file']
    correct_label = request.form['correct_label']

    # Load the image using PIL
    img = Image.open(file.stream)

    # Prepare directory and filename for saving feedback image
    timestamp = int(time.time())
    save_dir = 'feedback'
    os.makedirs(save_dir, exist_ok=True)  # Create folder if it doesn't exist
    save_path = os.path.join(save_dir, f'label_{correct_label}_{timestamp}.png')

    # Save the feedback image locally
    img.save(save_path)

    try:
        # Retrain the model using the new feedback
        REtrain_model(model, optimizer, transform, correct_label, img)

        # After retraining, delete the saved feedback image
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"Deleted feedback image: {save_path}")

        # Reload updated model weights into memory
        MODEL_PATH = "MNIST_Model.pth"
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH))
            model.eval()
            print("Reloaded updated model weights into memory.")

    except Exception as e:
        print(f"Error during feedback processing: {e}")
        return jsonify({'error': 'Internal training error'}), 500

    print(f"Processed and updated model with label: {correct_label}")
    return jsonify({'status': 'feedback received'}) 

    
if __name__ == '__main__':
    app.run(debug = True)