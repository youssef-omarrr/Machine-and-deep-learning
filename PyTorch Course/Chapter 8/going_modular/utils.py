"""
Contains various utility functions for PyTorch model training and saving.
"""

from pathlib import Path
import torch

def save_model(model: torch.nn.Module,
                target_dir: str,
                model_name: str):
    """
    Saves a PyTorch model's state_dict to a specified directory.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        target_dir (str): The directory where the model should be saved.
        model_name (str): The name for the saved model file. Should end with '.pt' or '.pth'.

    Example:
        save_model(model=model_0,
                target_dir="models",
                model_name="05_going_modular_tinyvgg_model.pth")
    """

    # Convert target_dir to a Path object and create the directory if it doesn't exist
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Check the file extension is valid
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), \
        "❌ Error: model_name should end with '.pt' or '.pth'"

    # Create the full path to where the model will be saved
    model_save_path = target_dir_path / model_name

    # Save only the state_dict (recommended for most use cases)
    torch.save(obj=model.state_dict(), f=model_save_path)

    print(f"[INFO] ✅ Model saved to: {model_save_path}")
    
    
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from typing import List
from torch.utils.data import DataLoader

def confusion_matrix_and_plot(model: torch.nn.Module,
                            dataloader: DataLoader,
                            class_names: List[str],
                            device: torch.device = torch.device("cpu"),
                            normalize: str = None,  # options: 'true', 'pred', 'all', or None
                            figsize: tuple = (10, 8),
                            cmap = 'Blues'):
    """
    Computes and plots the confusion matrix for a given model and dataset.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        dataloader (DataLoader): DataLoader containing test/validation data.
        class_names (List[str]): List of class names in correct order.
        device (torch.device): Target device for model and data.
        normalize (str, optional): How to normalize confusion matrix (per sklearn). Default is None.
        figsize (tuple): Size of the plot.
        cmap: Color map for the plot.
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds, normalize=normalize)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=figsize)
    disp.plot(ax=ax, cmap=cmap, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()


import os
import zipfile

from pathlib import Path

import requests

def download_data(source: str, 
                destination: str,
                remove_source: bool = True) -> Path:
    # sourcery skip: extract-method
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    
    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                    destination="pizza_steak_sushi")
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download pizza, steak, sushi data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...") 
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)
    
    return image_path


from torch.utils.tensorboard import SummaryWriter

def create_writer(experiment_name: str, 
                    model_name: str, 
                    extra: str=None) -> SummaryWriter:
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                                model_name="effnetb2",
                                extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    from datetime import datetime
    import os

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)


import torchvision
from torch import nn

def create_effnetb0(OUT_FEATURES: int,
                    device):
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights= weights)
    
    for param in model.features.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=1280, out_features=OUT_FEATURES)
    ).to(device)
    
    # Give the model a name
    model.name = "effnetb0"
    print(f"\033[32m[INFO] Created new {model.name} model.\033[0m")
    
    return model

def create_effnetb2(OUT_FEATURES: int,
                    device):
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights= weights)
    
    for param in model.features.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features=1408, out_features=OUT_FEATURES)
    ).to(device)
    
    # Give the model a name
    model.name = "effnetb2"
    print(f"\033[32m[INFO] Created new {model.name} model.\033[0m")
    
    return model

from typing import List, Tuple
import torch
from PIL import Image
from torchvision import transforms

# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str, 
                        class_names: List[str],
                        device,
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None):
    
    
    # 2. Open image
    img = Image.open(image_path)

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    ### Predict on image ### 

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # 7. Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability 
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max()*100:.3f}")
    plt.axis(False);