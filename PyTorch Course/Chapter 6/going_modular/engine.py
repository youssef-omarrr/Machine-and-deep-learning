"""
Contains functions for training and testing a PyTorch model.
"""

from typing import Dict, Tuple, List
import torch 
from torch import nn
from tqdm.auto import tqdm
import torchmetrics
from torchmetrics import Accuracy


# ------------------------ TRAINING STEP ------------------------
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               acc_fn: torchmetrics.Accuracy,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """
    Trains a PyTorch model for a single epoch.

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader providing batches of training data.
        loss_fn: The loss function used for optimization (e.g., CrossEntropyLoss).
        acc_fn: A torchmetrics Accuracy instance to track accuracy.
        optimizer: An optimizer instance (e.g., SGD, Adam).
        device: The device to run computations on (CPU or CUDA).

    Returns:
        Tuple of average training loss and accuracy over the epoch.
    """
    model.train()  # Set model to training mode
    loss_total, acc_total = 0, 0

    # Loop over training batches
    for batch, (x, y) in enumerate(dataloader):
        # Move input and target to the target device
        x, y = x.to(device), y.to(device)

        # Forward pass - get model predictions
        y_pred = model(x)

        # Compute loss
        loss = loss_fn(y_pred, y)
        loss_total += int(loss)

        # Update accuracy metric
        acc_fn.update(y_pred.argmax(dim=1), y)

        # Clear previous gradients
        optimizer.zero_grad()

        # Backpropagation - calculate gradients
        loss.backward()

        # Update model weights
        optimizer.step()

        # Optional logging every 400 batches
        if batch % 55 == 0:
            print(f"Looked at {batch * len(x)}/{len(dataloader.dataset)} samples")

    # Compute average loss and accuracy for the entire epoch
    loss_total /= len(dataloader)
    acc_total = acc_fn.compute().item() * 100  # Convert to %
    acc_fn.reset()  # Reset metric for next epoch

    return loss_total, acc_total


# ------------------------ TESTING STEP ------------------------
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              acc_fn: torchmetrics.Accuracy,
              device: torch.device) -> Tuple[float, float]:
    """
    Evaluates the model on the test dataset.

    Args:
        model: A PyTorch model to be evaluated.
        dataloader: A DataLoader for test data.
        loss_fn: Loss function to evaluate prediction error.
        acc_fn: Accuracy metric from torchmetrics.
        device: Device to compute on.

    Returns:
        Tuple of average test loss and accuracy.
    """
    model.to(device)
    model.eval()  # Turn off dropout, batchnorm, etc.
    loss_total, acc_total = 0, 0

    # Disable gradient calculation for inference
    with torch.inference_mode():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # Get predictions
            y_pred = model(x)

            # Compute loss and update accumulators
            loss = loss_fn(y_pred, y)
            loss_total += int(loss)
            acc_fn.update(y_pred.argmax(dim=1), y)

    # Compute epoch-level loss and accuracy
    loss_total /= len(dataloader)
    acc_total = acc_fn.compute().item() * 100  # Convert to %
    acc_fn.reset()  # Reset metric for reuse

    return loss_total, acc_total


# ------------------------ TRAINING LOOP ------------------------
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          device: torch.device = 'cpu') -> Dict[str, List[float]]:
    """
    Runs the full training loop: training + testing for multiple epochs.

    Args:
        model: PyTorch model to train and evaluate.
        train_dataloader: DataLoader for training data.
        test_dataloader: DataLoader for test data.
        optimizer: Optimization algorithm.
        loss_fn: Loss function (default = CrossEntropyLoss).
        epochs: Total number of epochs to train.
        device: Target computation device.

    Returns:
        Dictionary containing loss and accuracy history for training and testing.
    """

    # Initialize accuracy metric for classification task
    acc_fn = Accuracy(task="multiclass", num_classes=3).to(device)

    # Initialize history dictionary for storing results
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # Loop through epochs
    for epoch in tqdm(range(epochs)):
        # Training step
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            acc_fn=acc_fn,
            device=device
        )

        # Display training results
        print("\n")
        print("\033[91m======================================================\033[0m")
        print(f"\033[94mTrain loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%\033[0m")

        # Testing step
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            acc_fn=acc_fn,
            device=device
        )

        # Display testing results
        print(f"\033[92mTest loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\033[0m")
        print("\033[91m======================================================\033[0m")
        print("\n")

        # Save results to history (converted to CPU floats if needed)
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    # Return metrics for analysis/visualization
    return results
