### 1. Imports and class names setup ### 
import gradio as gr
import os
import torch

import base64
from io import BytesIO
from PIL import Image

from vit_model import create_ViT
from timeit import default_timer as timer
from typing import Tuple, Dict
from info import *

### 2. Model and transforms preparation ###

# Create ViT model
vit, vit_transforms = create_ViT()

# Load saved weights
vit.load_state_dict(
    torch.load(
        f="My_ViT_16b.pth",
        map_location=torch.device("cpu"),  # load to CPU
    )
)

### 3. Predict function ###

def predict(img) -> Tuple[Dict, str, float]:
    """Transforms and performs a prediction on img and returns prediction dict, diagnosis, and time taken."""
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img = vit_transforms(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    vit.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(vit(img), dim=1)

    # Create a prediction label and probability dictionary
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Get the top predicted class
    top_class_index = pred_probs.argmax(dim=1).item()
    top_class = class_names[top_class_index]
    
    # Get potential diagnosis
    potential_diagnosis = diagnosis_map.get(top_class, "N/A")

    # Calculate prediction time
    pred_time = round(timer() - start_time, 5)

    # Return prediction dict, diagnosis string, and prediction time
    return pred_labels_and_probs, potential_diagnosis, pred_time

### 4. Gradio app ###

# ----------------------------- #
# Title, Description, Disclaimer
# ----------------------------- #
title = "# 🧫 CelluScan"
description = (
    "CelluScan is a deep learning model based on the **Vision Transformer (ViT B-16)** architecture. "
    "It was trained on a dataset of approximately **60,000 blood cell images** to classify various types of white blood cells "
    "and detect possible abnormalities.\n\n"
    "Designed for educational and research purposes, it can assist in exploring AI-based blood analysis."
)
disclaimer = (
    "⚠️ **DISCLAIMER:** This project was developed as part of a student-led academic initiative and is intended solely for **educational and research** use. "
    "It is *not* a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare provider for any medical concerns or decisions."
)

# Auto-generate labeled examples with base64 encoding
def image_to_base64(image_path):
    """Convert image to base64 for embedding in HTML"""
    try:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error converting {image_path} to base64: {e}")
        return None
    
example_dir = "examples"
example_list = []
labeled_examples_html = "<h2>🧪 Example Images (Ordered by Class)</h2><div style='display: flex; flex-wrap: wrap;'>\n"

if os.path.exists(example_dir):
    for fname in sorted(os.listdir(example_dir)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            prefix = fname.split('_')[0].upper()
            label = prefix_to_label.get(prefix, "Unknown")

            example_path = os.path.join(example_dir, fname)
            example_list.append([example_path])

            # Convert to base64 for HTML display
            base64_img = image_to_base64(example_path)
            if base64_img:
                labeled_examples_html += f"""
                <div style="margin: 10px; text-align: center;">
                    <div style="font-weight: bold;">{label}</div>
                    <img src="{base64_img}" alt="{label}"
                        style="width: 120px; border: 1px solid #ccc; border-radius: 6px; margin-top: 5px;" />
                </div>
                """

labeled_examples_html += "</div>"

# ----------------------------- #
#  Gradio Layout with Blocks
# ----------------------------- #
with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown(disclaimer)

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload a Blood Cell Image")
        
        with gr.Column():
            label_output = gr.Label(num_top_classes=3, label="Predictions")
            diagnosis_output = gr.Markdown(label="Potential Diagnosis")
            time_output = gr.Number(label="Prediction time (s)")

    gr.Markdown("### Try the Model with Clickable Examples")
    gr.Examples(
        examples=example_list,
        inputs=image_input,
        label="Click an Example to Predict",
        examples_per_page=14
    )
    
    gr.HTML(labeled_examples_html)

    def wrapped_predict(img):
        pred_probs, diagnosis_md, pred_time = predict(img)
        return pred_probs, diagnosis_md, pred_time

    image_input.change(fn=wrapped_predict, inputs=image_input, outputs=[label_output, diagnosis_output, time_output])


# Launch the demo!
demo.launch()

