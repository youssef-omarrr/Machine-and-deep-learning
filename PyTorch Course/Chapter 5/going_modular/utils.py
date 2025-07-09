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

