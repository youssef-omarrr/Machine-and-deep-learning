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

