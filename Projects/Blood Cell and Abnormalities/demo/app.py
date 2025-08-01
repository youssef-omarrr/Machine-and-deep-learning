### 1. Imports and class names setup ### 
import gradio as gr
import os
import torch

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

# Auto-generate labeled examples for Gradio
example_dir = "examples/"
example_list = []
labeled_examples_md = "### 🧪 Example Images\n"

for fname in sorted(os.listdir(example_dir)):
    filepath = os.path.join(example_dir, fname)
    prefix = fname.split('_')[0].upper()
    label = prefix_to_label.get(prefix, "Unknown")
    example_list.append([filepath])
    labeled_examples_md += f"- **{label}** — `{fname}`\n  \n  ![]({filepath})\n\n"

# ----------------------------- #
#  Gradio Layout with Blocks
# ----------------------------- #
with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown(disclaimer)
    gr.Markdown(labeled_examples_md)

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload a Blood Cell Image")

    with gr.Row():
        label_output = gr.Label(num_top_classes=3, label="Predictions")
        diagnosis_output = gr.Markdown(label="Potential Diagnosis")
        time_output = gr.Number(label="Prediction time (s)")

    gr.Examples(
        examples=example_list,
        inputs=image_input,
        label="Click an Example to Predict",
        examples_per_page=10
    )

    def wrapped_predict(img):
        pred_probs, diagnosis_md, pred_time = predict(img)
        return pred_probs, diagnosis_md, pred_time

    image_input.change(fn=wrapped_predict, inputs=image_input, outputs=[label_output, diagnosis_output, time_output])


# Launch the demo!
demo.launch()

