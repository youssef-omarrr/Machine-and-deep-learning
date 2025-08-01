### 1. Imports and class names setup ### 
import gradio as gr
import os
import torch

from vit_model import create_ViT
from timeit import default_timer as timer
from typing import Tuple, Dict

class_names = ['Band Neutrophil', 'Basophil', 'Eosinophil', 'Erythroblast', 
            'Immature Granulocyte', 'Lymphocyte', 'Metamyelocytes', 'Monocyte', 
            'Myeloblast', 'Myelocytes', 'Neutrophil', 'Platelets', 'Promyelocytes', 'Segmented Neutrophil']


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

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img = vit_transforms(img).unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    vit.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(vit(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time

### 4. Gradio app ###

# Create title, description and article strings
title = "CelluScan 🧫"
description = "CelluScan is a deep learning model for classifying blood cells (mainly white blood cells) and detecting abnormalities."

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"), # what are the outputs?
                            gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
                    # Create examples list from "examples/" directory
                    examples=example_list, 
                    title=title,
                    description=description)

# Launch the demo!
demo.launch()
