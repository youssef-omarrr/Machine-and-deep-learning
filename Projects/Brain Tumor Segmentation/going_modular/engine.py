from tqdm import tqdm
import torch.nn.functional as F
import torch
import os

# ======================================================================================
# 1. DEFINE THE TRAINING FUNCTION FOR ONE EPOCH
# ======================================================================================
def train_one_epoch(model, 
                    loader, 
                    optimizer, 
                    loss_fn, 
                    scaler, 
                    device, 
                    epoch_num):
    """
    Performs one full training epoch.
    
    Args:
        model (torch.nn.Module): The model to be trained.
        loader (DataLoader): The DataLoader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer for updating weights.
        loss_fn: The loss function.
        scaler (torch.cuda.amp.GradScaler): The gradient scaler for mixed precision.
        device (torch.device): The device to train on (e.g., "cuda").
        epoch_num (int): The current epoch number, for logging.
        
    Returns:
        float: The average training loss for the epoch.
    """
    # 1. Set the model to training mode
    # This enables features like dropout and batch normalization to work correctly during training.
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(loader, desc=f"Train E-{epoch_num}")

    use_amp = (device is not None and getattr(device, "type", None) == "cuda")
    for batch in progress_bar:
        # Accept either dict-style batch or tuple/list (image, mask)
        if isinstance(batch, dict):
            imgs = batch.get("image")
            gts = batch.get("mask")
        else:
            # assume (images, masks) or similar
            imgs, gts = batch

        imgs = imgs.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Use AMP only when on CUDA and scaler provided
        if use_amp and scaler is not None:
            with torch.amp.autocast(enabled=True, device_type="cuda" if torch.cuda.is_available() else "cpu"):
                logits = model(imgs)['out']
                loss = loss_fn(logits, gts)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard FP32 path (CPU or no scaler)
            logits = model(imgs)['out']
            loss = loss_fn(logits, gts)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return epoch_loss / len(loader)


# ======================================================================================
# 2. DEFINE THE EVALUATION FUNCTION
# ======================================================================================
def evaluate(model, 
             loader,
             metric,
             device,
             epoch_num):
    """
    Evaluates the model on the validation set.
    
    Args:
        model (torch.nn.Module): The model to be evaluated.
        loader (DataLoader): The DataLoader for the validation data.
        metric (monai.metrics.DiceMetric): The MONAI metric to compute the Dice score.
        device (torch.device): The device to evaluate on.
        epoch_num (int): The current epoch number, for logging.
        
    Returns:
        tuple[float, float]: A tuple containing the average validation Dice score
                            and the average validation accuracy.
    """
    # 1. Set the model to evaluation mode
    # This disables layers like Dropout and ensures BatchNorm uses running stats.
    model.eval()
    metric.reset()  # Reset the metric at the start of each evaluation
    val_acc = []
    progress_bar = tqdm(loader, desc=f"Validation E-{epoch_num}")
    

    use_amp = (device is not None and getattr(device, "type", None) == "cuda")
    
    with torch.inference_mode():
        for batch_data in progress_bar:
            if isinstance(batch_data, dict):
                test_img = batch_data.get("image")
                test_gts = batch_data.get("mask")
            else:
                test_img, test_gts = batch_data

            test_img = test_img.to(device, non_blocking=True)
            test_gts = test_gts.to(device, non_blocking=True) # shape (B,1,H,W)

            # forward
            if use_amp:
                with torch.amp.autocast(enabled=True, device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    logits = model(test_img)['out'] # (B,2,H,W)
            else:
                logits = model(test_img)['out']


            # -----------------------
            # explicit, robust post-processing
            # -----------------------
            # predictions: (B,H,W) indices
            preds_idx = torch.argmax(logits, dim=1)                # (B,H,W)

            # labels: squeeze channel if present -> (B,H,W)
            labels_idx = test_gts.squeeze(1).long().to(device)     # (B,H,W)


            # if labels accidentally use 255 or other values: normalize
            if torch.max(labels_idx) > 1:
                labels_idx = (labels_idx > 0).long()

            num_classes = logits.shape[1]                         # 2

            # one-hot: (B, H, W, C) -> permute -> (B, C, H, W)
            pred_onehot = F.one_hot(preds_idx.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
            label_onehot = F.one_hot(labels_idx.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

            # update metric (MONAI expects float one-hot tensors shaped (B,C,H,W))
            metric(y_pred=pred_onehot, y=label_onehot)

            # pixel accuracy
            correct_pixels = (preds_idx == labels_idx).sum()
            total_pixels = labels_idx.numel()
            batch_accuracy = (correct_pixels / total_pixels).item()
            val_acc.append(batch_accuracy)
            progress_bar.set_postfix(acc=batch_accuracy)

        # end loop
        
    # print("logits", logits.shape, "preds_idx unique:", torch.unique(preds_idx))
    # print("labels", labels_idx.shape, "labels unique:", torch.unique(labels_idx))
    # print("pred_onehot sum per class:", pred_onehot.sum(dim=[0,2,3]))
    # print("label_onehot sum per class:", label_onehot.sum(dim=[0,2,3]))

    mean_dice = metric.aggregate().item()  # scalar
    metric.reset()
    mean_accuracy = sum(val_acc) / len(val_acc) if len(val_acc) > 0 else 0.0

    return mean_dice, mean_accuracy


# ======================================================================================
# 3. DEFINE THE MAIN TRAINING LOOP
# ======================================================================================
def train(model,
        train_loader,
        test_loader,
        loss_fn,
        metric,
        optimizer,
        scheduler,
        scaler,
        device = "cuda" if torch.cuda.is_available() else "cpu",
        epochs:int = 20,
        checkpoint_dir:str = "checkpoints/"):
    """
    Coordinates the full training process over multiple epochs.

    Args:
        model (torch.nn.Module): The model to be trained.
        
        train_loader (DataLoader): DataLoader for the training data.
        test_loader (DataLoader): DataLoader for the validation data.
        
        loss_fn: The loss function.
        metric (monai.metrics.DiceMetric): The metric for evaluation.
        optimizer (torch.optim.Optimizer): The optimizer for updating weights.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision.
        
        device (torch.device): The device to train on.
        epochs (int): The total number of epochs to train for.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains the
                    training and validation metrics for an epoch.
    """
    
    # Initialize a list to store results from each epoch for later analysis or plotting
    results = []
    best_val_dice = 0.0

    # Ensure checkpoint dir exists (clean trailing spaces)
    checkpoint_dir = checkpoint_dir.strip()
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Loop through the specified number of epochs
    for epoch in range(1, epochs + 1):
        print(f"\n===== Starting Epoch {epoch}/{epochs} =====")
        
        # 2. Run the training function for one epoch
        # This will update the model's weights based on the training data.
        train_loss = train_one_epoch(model,
                                    train_loader,
                                    optimizer,
                                    loss_fn,
                                    scaler,
                                    device,
                                    epoch)
        
        # 3. Run the evaluation function on the validation set
        # This assesses the model's performance on unseen data.
        val_dice, val_acc = evaluate(model,
                                     test_loader,
                                     metric,
                                     device,
                                     epoch)
        
        # 4. Update the learning rate scheduler
        # Schedulers like ReduceLROnPlateau adjust the LR based on a monitored metric.
        if scheduler:
            # We use `1.0 - val_dice` because schedulers typically aim to *minimize* a metric.
            # Since a higher Dice score is better, minimizing (1 - Dice) is equivalent to maximizing Dice.
            scheduler.step(1.0 - val_dice)
        
        # 5. Save model checkpoints
        # Save a checkpoint for the current epoch for traceability.
        if checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)

        # Save the model if it has the best validation Dice score so far.
        if val_dice > best_val_dice and checkpoint_dir:
            best_val_dice = val_dice
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"\n  New best validation Dice: {val_dice:.4f}. Model saved to {best_model_path}")
        
        # 6. Store the results for the current epoch
        epoch_results = {
            "Train Loss": train_loss,
            "Validation Dice": val_dice,
            "Validation Acc": val_acc
        }
        results.append(epoch_results)
        
        # 7. Print a summary of the epoch's performance
        print(f"\n--- Epoch {epoch} Summary ---")
        print(f"  Train Loss:      {train_loss:.4f}")
        print(f"  Validation Dice:   {val_dice:.4f}")
        print(f"  Validation Acc:    {val_acc*100:.2f}%")
        if checkpoint_path:
            print(f"  Checkpoint saved:  {checkpoint_path}")
        print("=" * (55 + len(str(epoch)) + len(str(epochs))))

    # 8. Final message when training is complete
    print("\nTraining finished!")
    
    # 9. Return the collected results for plotting or analysis
    return results


# ======================================================================================
# 4. PLOT THE RESULTS
# ======================================================================================

import matplotlib.pyplot as plt
def plot_results(results: list[dict]):
    """
    Plots the training and validation metrics from the results dictionary.

    Args:
        results (list[dict]): A list of dictionaries, where each dictionary
                              contains the metrics for one epoch. Expected keys
                              are "Train Loss", "Validation Dice", "Validation Acc".
    """
    # 1. Extract the metrics from the results list
    # We use list comprehensions for a concise way to create lists for each metric.
    train_loss = [r["Train Loss"] for r in results]
    val_dice = [r["Validation Dice"] for r in results]
    # We multiply accuracy by 100 to display it as a percentage.
    val_acc = [r["Validation Acc"] * 100 for r in results]
    epochs = range(1, len(results) + 1)

    # 2. Create the plot with two y-axes
    # `subplots` creates a figure and a set of subplots.
    # `ax1` will handle the loss, and `ax2` will handle accuracy and Dice score.
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 3. Plot Training Loss on the primary y-axis (left)
    color_loss = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color_loss)
    ax1.plot(epochs, train_loss, 'o-', color=color_loss, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')

    # 4. Create a secondary y-axis (right) for validation metrics
    # `twinx()` creates a new y-axis that shares the same x-axis.
    ax2 = ax1.twinx()
    color_acc = 'tab:blue'
    color_dice = 'tab:green'
    ax2.set_ylabel('Validation Metrics', color=color_acc)
    # Plot Validation Accuracy
    ax2.plot(epochs, val_acc, 's--', color=color_acc, label='Validation Accuracy (%)')
    # Plot Validation Dice Score
    ax2.plot(epochs, val_dice, '^-', color=color_dice, label='Validation Dice Score')
    ax2.tick_params(axis='y', labelcolor=color_acc)

    # 5. Add title and legend
    plt.title('Training Loss and Validation Metrics per Epoch', fontsize=16)
    # Ask Matplotlib to combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    fig.tight_layout()
    plt.show()
