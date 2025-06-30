import torch # type: ignore
from torch import nn # type: ignore
from torch.utils.data import DataLoader # type: ignore
from matplotlib import pyplot as plt # type: ignore
from timeit import default_timer as timer # type: ignore
from tqdm.auto import tqdm # type: ignore
from torchmetrics import ConfusionMatrix # type: ignore
from mlxtend.plotting import plot_confusion_matrix # type: ignore

def plot_example(train_data, class_names, rows=4, cols=4):
    """
    Plots a grid of random example images from the training dataset.

    Args:
        train_data (Dataset): The training dataset containing images and labels.
        class_names (list): List of class names for the dataset.
        rows (int): Number of rows in the plot grid.
        cols (int): Number of columns in the plot grid.

    Returns:
        None. Displays a matplotlib figure with example images.
    """
    fig = plt.figure(figsize=(9,9))

    for i in range(1, rows*cols +1):
        # Select a random index from the dataset
        random_index = torch.randint(0, len(train_data), size=[1]).item() # ".item()" to switch from tensor to int
        
        # Get the image and label at the random index
        img, label = train_data[random_index]
        
        # Add a subplot for this image
        fig.add_subplot(rows, cols, i)
        
        # Display the image in grayscale
        plt.imshow(img.squeeze(), cmap="gray")
        # Set the title as the class name
        plt.title(class_names[label])
        # Hide axis ticks
        plt.axis(False);



def train_fn (model:nn.Module,
                accuracy_function,
                loss_function:nn.Module,
                optimizer: torch.optim.Optimizer,
                dataloader:DataLoader,
                device:torch.device = "cpu"):
    """
    Trains a PyTorch model for one epoch.

    Args:
        model (nn.Module): The model to train.
        accuracy_function: Metric object for tracking accuracy.
        loss_function (nn.Module): Loss function to optimize.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        dataloader (DataLoader): DataLoader for training data.
        device (torch.device): Device to run training on.

    Returns:
        None. Prints training loss and accuracy.
    """
    losses, accuracy = 0,0
    
    model.to(device)
    model.train()  # Set model to training mode
    
    for batch, (x,y) in enumerate(dataloader):
        # Move data to the target device
        x, y = x.to(device), y.to(device)
        
        # 1) Forward pass: compute predictions
        y_pred = model(x)
        
        # 2) Compute loss
        loss = loss_function(y_pred, y)
        losses += loss
        # Update accuracy metric with predictions
        accuracy_function.update(y_pred.argmax(dim=1), y)
        
        # 3) Zero gradients from previous step
        optimizer.zero_grad()
        
        # 4) Backward pass: compute gradients
        loss.backward()
        
        # 5) Update model parameters
        optimizer.step()
        
        # Print progress every 400 batches
        if batch % 400 == 0:
            print(f"Looked at {batch * len(x)}/{len(dataloader.dataset)} samples")
        
    # Calculate average loss and accuracy for the epoch
    losses /= len(dataloader)
    accuracy = accuracy_function.compute().item()
    accuracy *= 100
    accuracy_function.reset()  
    
    print(f"Train loss: {losses:.5f} | Train accuracy: {accuracy:.2f}%")
    
    
    
    
def test_fn (model:nn.Module,
                accuracy_function,
                loss_function:nn.Module,
                dataloader:DataLoader,
                device:torch.device = "cpu"):
    """
    Evaluates a PyTorch model on a test/validation set.

    Args:
        model (nn.Module): The model to evaluate.
        accuracy_function: Metric object for tracking accuracy.
        loss_function (nn.Module): Loss function to compute.
        dataloader (DataLoader): DataLoader for test/validation data.
        device (torch.device): Device to run evaluation on.

    Returns:
        None. Prints test loss and accuracy.
    """
    losses, accuracy = 0,0
    
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    with torch.inference_mode():
        for x,y in dataloader:
            # Move data to the target device
            x, y = x.to(device), y.to(device)
            
            # 1) Forward pass: compute predictions
            y_pred = model(x)
            
            # 2) Compute loss
            losses += loss_function(y_pred, y)
            # Update accuracy metric with predictions
            accuracy_function.update(y_pred.argmax(dim=1), y)
            
        # Calculate average loss and accuracy for the epoch
        losses /= len(dataloader)
        accuracy = accuracy_function.compute().item()
        accuracy *= 100
        accuracy_function.reset()  
            
        print(f"Test loss: {losses:.5f} | Test accuracy: {accuracy:.2f}%\n")
        
        
        
        
def print_train_time(start: float,
                    end: float,
                    device: torch.device = "cpu"):
    """
    Prints and returns the time difference between start and end.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device (torch.device): Device that compute is running on.

    Returns:
        float: Time between start and end in seconds.
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time




def eval_model(model: torch.nn.Module,
                data_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                accuracy_fn,
                device: torch.device = "cpu"):
    """
    Evaluates a model on a dataset and returns loss and accuracy.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
        device (torch.device): Device to run evaluation on.

    Returns:
        dict: Dictionary with model name, average loss, and accuracy.
    """   
    loss, acc = 0,0
    model.eval()
    model.to(device)
    
    with torch.inference_mode():
        for X,y in data_loader:
            # Send data to the target device
            X, y = X.to(device), y.to(device)
            
            # Make predictions with the model
            y_pred = model(X)
            
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            accuracy_fn.update(y_pred.argmax(dim=1), y)  
            
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc = accuracy_fn.compute().item()
        acc *= 100 
        accuracy_fn.reset()     
        
        return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}




def make_predictions(model: nn.Module, data:list, device:torch.device = "cpu"):
    """
    Makes predictions on a list of data samples using a trained model.

    Args:
        model (nn.Module): Trained PyTorch model.
        data (list): List of input samples (not a DataLoader).
        device (torch.device): Device to run inference on.

    Returns:
        torch.Tensor: Stacked tensor of prediction probabilities for each sample.
    """
    pred_probs = []
    model.eval()
    
    with torch.inference_mode():
        for sample in data:
            # Prepare sample: add batch dimension and move to device
            sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device
            
            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)
            
            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 1, so can perform on dim=0)
            
            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())
            
    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)




def test_model_plot(test_samples, class_names, pred_classes, test_labels):
    """
    Plots a grid of test images with predicted and true labels.

    Args:
        test_samples (list): List of test image tensors.
        class_names (list): List of class names.
        pred_classes (Tensor): Tensor of predicted class indices.
        test_labels (list): List of true label indices.

    Returns:
        None. Displays a matplotlib figure with predictions and ground truth.
    """
    plt.figure(figsize=(12, 12))
    nrows = 4
    ncols = 4
    for i, sample in enumerate(test_samples):
        # Create a subplot
        plt.subplot(nrows, ncols, i+1)  
        # Plot the target image
        plt.imshow(sample.squeeze(), cmap="gray")   
        # Find the prediction label (in text form, e.g. "Sandal")
        pred_label = class_names[pred_classes[i]]   
        # Get the truth label (in text form, e.g. "T-shirt")
        truth_label = class_names[test_labels[i]]   
        # Create the title text of the plot
        title_text = f"Pred: {pred_label} | Truth: {truth_label}"

        # Check for equality and change title colour accordingly
        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g") # green text if correct
        else:
            plt.title(title_text, fontsize=10, fontweight = "bold", c="r") # red text if wrong
        plt.axis(False);



def plot_conffusion_matrix(model, dataloader, class_names, test_data):
    """
    Plots a confusion matrix for the predictions of a given model on a test dataset.

    Args:
        model (nn.Module): Trained PyTorch model to evaluate.
        dataloader (DataLoader): DataLoader for the test dataset.
        class_names (list): List of class names for the dataset.
        test_data (Dataset): The original test dataset (used for true labels).

    Steps:
    1. Make predictions on the test set using the model.
    2. Collect all predicted labels into a single tensor.
    3. Create a confusion matrix comparing predictions to true labels.
    4. Plot the confusion matrix using matplotlib.

    Returns:
        None. Displays the confusion matrix plot.
    """
    # 1. Make predictions with trained model
    y_preds = []  # List to store batch predictions
    number_model = model  # Alias for clarity
    number_model.eval()   # Set model to evaluation mode (disables dropout, etc.)

    # Disable gradient calculation for inference
    with torch.inference_mode():
        # Iterate over batches in the dataloader
        for X, y in tqdm(dataloader, desc="Making predictions"):
            # Do the forward pass to get logits
            y_logits = number_model(X)
            # Convert logits to probabilities, then get predicted class indices
            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
            # Append predictions for this batch
            y_preds.append(y_pred)

        # Concatenate list of predictions into a single tensor
        y_pred_tensor = torch.cat(y_preds)
        
    # 2. Setup confusion matrix instance and compare predictions to targets
    # ConfusionMatrix from torchmetrics computes the confusion matrix
    con_mat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
    # test_data.targets contains the true labels for the test set
    con_mat_tensor = con_mat(preds=y_pred_tensor, target=test_data.targets)

    # 3. Plot the confusion matrix
    # plot_confusion_matrix from mlxtend creates a nice matplotlib plot
    fig, ax = plot_confusion_matrix(
        conf_mat=con_mat_tensor.numpy(),  # Convert tensor to numpy array for plotting
        class_names=class_names,          # Use class names for axis labels
        figsize=(10, 7)                   # Set figure size
    )
    # The plot will be displayed automatically in Jupyter notebooks
