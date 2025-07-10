"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
from torchvision import transforms
from going_modular import data_setup, engine, model_builder, utils

def init(num_epochs: int = 3,
        batch_size: int = 32,
        hidden_units: int = 10,
        learning_rate: float = 0.001,
        train_dir: str = "data/pizza_steak_sushi/train",
        test_dir: str = "data/pizza_steak_sushi/test"):
    """
    Initializes training pipeline using modular components.

    Args:
        num_epochs (int): Number of epochs to train the model.
        batch_size (int): Number of samples per batch for training/testing.
        hidden_units (int): Number of hidden units in the TinyVGG model.
        learning_rate (float): Learning rate for optimizer.
        train_dir (str): Path to training data.
        test_dir (str): Path to testing data.
    """

    # Setup hyperparameters for easy tracking and reference
    HYPER_PARAMS = {
        'NUM_EPOCHS': num_epochs,
        'BATCH_SIZE': batch_size,
        'HIDDEN_UNITS': hidden_units,
        'LEARNING_RATE': learning_rate
    }

    # Detect the device (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define image transforms (resize + convert to tensor)
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize all images to 64x64
        transforms.ToTensor()         # Convert PIL image to PyTorch tensor
    ])

    # Create training and testing dataloaders
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloader(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=HYPER_PARAMS["BATCH_SIZE"]
    )
    print("✅ Dataloaders created successfully!")

    # Build the TinyVGG model based on the provided parameters
    model = model_builder.TinyVGG(
        input_shape=3,  # RGB images
        hidden_units=HYPER_PARAMS["HIDDEN_UNITS"],
        output_shape=len(class_names)  # Number of classes
    ).to(device)
    print("✅ Model built successfully!")

    # Set up the loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=HYPER_PARAMS["LEARNING_RATE"])

    # Train and evaluate the model using the engine module
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=HYPER_PARAMS["NUM_EPOCHS"],
        device=device
    )

    # Save the trained model using a utility function
    utils.save_model(
        model=model,
        target_dir="models",  # Save directory
        model_name="05_going_modular_script_mode_tinyvgg_model.pth"
    )
    print("✅ Model saved successfully!")

# When running the script directly, start training
if __name__ == "__main__":
    init()

    
    
