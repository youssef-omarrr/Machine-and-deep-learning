############################################################
# IMPORTS
############################################################
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import defaultdict

from transformers import get_linear_schedule_with_warmup

from datasets import load_dataset


############################################################
# CONSTANTS / CONFIG
############################################################
MODEL_NAME = "gpt2-medium"
DOWNLOAD_PATH = "../MODELS/"
TRAIN_FILE = "../dataset/train_file.txt"
TEST_FILE = "../dataset/test_file.txt"
OUTPUT_DIR = "../MODELS/Project_Mozart_gpt2-medium"

# Enhanced training hyperparams
EPOCHS = 5
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 1.5e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500

MAX_GRAD_NORM = 1.0
USE_FP16 = True
SAVE_EVERY_N_STEPS = 1000
EVAL_EVERY_N_STEPS = 500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_TARGET_MODULES = ["c_attn", "c_proj", "c_fc"]

# Custom loss weights - only penalize consecutive rests
CONSECUTIVE_REST_PENALTY = 5.0  # Heavy penalty for consecutive rests
NOTE_DENSITY_REWARD = 0.3       # Mild reward for more notes
MIN_NOTES_THRESHOLD = 10         # Minimum notes per sequence

############################################################
# Custom Loss Functions - Penalize Consecutive Rests Only
############################################################
class ConsecutiveRestLoss(nn.Module):
    def __init__(self, tokenizer, consecutive_penalty=CONSECUTIVE_REST_PENALTY,
                                    density_reward=NOTE_DENSITY_REWARD):
        super().__init__()
        self.tokenizer = tokenizer
        self.consecutive_penalty = consecutive_penalty
        self.density_reward = density_reward
        
        
        # Identify rest note tokens
        self.rest_token_ids = []
        for token, token_id in tokenizer.get_vocab().items():
            if any(rest_word in token for rest_word in ['Rest_', 'rest_', 'REST_', 'pause_']):
                self.rest_token_ids.append(token_id)
        print(f"Found {len(self.rest_token_ids)} rest tokens")


    def detect_consecutive_rests(self, sequence):
        """Detect positions with consecutive rest notes"""
        if len(self.rest_token_ids) == 0:
            return torch.zeros_like(sequence, dtype=torch.bool)
        
        # Create mask for rest tokens
        is_rest = torch.isin(sequence, torch.tensor(self.rest_token_ids).to(sequence.device))
        
        # Find consecutive rests (current token is rest AND previous token is rest)
        consecutive_rests = torch.zeros_like(is_rest, dtype=torch.bool)
        consecutive_rests[:, 1:] = is_rest[:, 1:] & is_rest[:, :-1]
        
        return consecutive_rests


    def forward(self, logits, labels, attention_mask=None):
        # Standard cross entropy loss (use the model's built-in loss calculation)
        # We'll compute this properly by shifting
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1), 
            reduction='mean'
        )
        
        # Penalize only consecutive rests in predictions
        # Initialize loss components as tensors on the same device
        consecutive_penalty_loss = torch.tensor(0.0, device=logits.device)
        density_reward = torch.tensor(0.0, device=logits.device)
        consecutive_rests_detected = 0
        
        if len(self.rest_token_ids) > 0:
            with torch.no_grad():
                # Get predicted tokens (greedy)
                pred_tokens = torch.argmax(logits, dim=-1)
                
                # Detect consecutive rests in predictions
                consecutive_rests_mask = self.detect_consecutive_rests(pred_tokens)
                consecutive_rests_detected = consecutive_rests_mask.sum().item()
                
                # Apply penalty only to positions with consecutive rests
                if consecutive_rests_mask.any():
                    # Simple penalty based on count of consecutive rests
                    # This avoids the complex indexing that was causing errors
                    consecutive_penalty_loss = consecutive_rests_mask.sum().float() * self.consecutive_penalty * 0.001
                
                # Reward note density (encourage more non-rest notes)
                if attention_mask is not None:
                    # Count non-rest tokens in labels (ignore padding and -100)
                    non_rest_mask = ~torch.isin(labels, torch.tensor(self.rest_token_ids).to(labels.device))
                    valid_mask = (labels != self.tokenizer.pad_token_id) & (labels != -100)
                    note_count = (non_rest_mask & valid_mask).sum(dim=-1).float()
                    
                    # Reward sequences with more notes
                    density_reward = -torch.relu(MIN_NOTES_THRESHOLD - note_count).mean() * self.density_reward
        
        total_loss = ce_loss + consecutive_penalty_loss + density_reward
        
        return total_loss, {
            'ce_loss': ce_loss.item(),
            'consecutive_rest_penalty': consecutive_penalty_loss.item(),
            'density_reward': density_reward.item(),
            'consecutive_rests_detected': consecutive_rests_detected
        }

############################################################
# Enhanced Dataset Preparation
############################################################
def prepare_dataset(tokenizer):
    """Enhanced dataset preparation"""
    
    dataset = load_dataset("text", data_files={
        "train": TRAIN_FILE,
        "validation": TEST_FILE
    })

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset"
    )

    return tokenized

############################################################
# Enhanced Collate Function
############################################################
def collate_fn(samples, tokenizer):
    """Enhanced collate function"""
    
    samples.sort(key=lambda x: len(x["input_ids"]), reverse=True)
    
    input_ids = [torch.tensor(s["input_ids"], dtype=torch.long) for s in samples]
    attention_mask = [torch.tensor(s["attention_mask"], dtype=torch.long) for s in samples]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

############################################################
# Enhanced Training Loop
############################################################
def train_loop(model, tokenizer, tokenized):
    """Enhanced training loop with consecutive rest penalty"""
    
    custom_loss_fn = ConsecutiveRestLoss(tokenizer, CONSECUTIVE_REST_PENALTY, NOTE_DENSITY_REWARD)

    train_dataloader = DataLoader(
        tokenized["train"],
        batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda s: collate_fn(s, tokenizer),
        pin_memory=True,
    )
    
    val_dataloader = DataLoader(
        tokenized["validation"],
        batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda s: collate_fn(s, tokenizer),
        pin_memory=True,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Learning rate scheduler
    total_steps = len(train_dataloader) * EPOCHS // GRAD_ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler(device=DEVICE) if USE_FP16 and DEVICE == "cuda" else None

    # Training metrics
    metrics_history = defaultdict(list)
    global_step = 0
    model.train()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        steps_in_epoch = 0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for step, batch in enumerate(pbar):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            with torch.amp.autocast(enabled=USE_FP16, device_type=DEVICE):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                # Apply custom loss
                total_loss, loss_components = custom_loss_fn(
                    outputs.logits,
                    batch["labels"],
                    batch["attention_mask"]
                )
                
                loss_reduced = total_loss / GRAD_ACCUM_STEPS

            # Backward pass
            if scaler is not None:
                scaler.scale(loss_reduced).backward()
            else:
                loss_reduced.backward()

            epoch_loss += total_loss.item()
            steps_in_epoch += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'rest_pen': f"{loss_components['consecutive_rest_penalty']:.4f}",
                'rests': loss_components['consecutive_rests_detected']
            })

            # Gradient accumulation step
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Save checkpoint
                if global_step % SAVE_EVERY_N_STEPS == 0:
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    model.save_pretrained(OUTPUT_DIR)
                    tokenizer.save_pretrained(OUTPUT_DIR)
                    print(f"\nSaved checkpoint at step {global_step}")

                # Validation
                if global_step % EVAL_EVERY_N_STEPS == 0:
                    val_metrics = evaluate_model(model, custom_loss_fn, val_dataloader)
                    print(f"\nValidation at step {global_step}: {val_metrics}")
                    metrics_history['val_loss'].append(val_metrics['val_loss'])

        # End of epoch
        avg_train_loss = epoch_loss / max(1, steps_in_epoch)
        metrics_history['train_loss'].append(avg_train_loss)
        
        # Full validation at end of epoch
        val_metrics = evaluate_model(model, custom_loss_fn, val_dataloader)
        print(f"\nEpoch {epoch+1} finished. Train Loss: {avg_train_loss:.4f}, Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"Consecutive rests detected: {val_metrics['consecutive_rests_detected']}")

        # Save epoch checkpoint
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

    return metrics_history

def evaluate_model(model, loss_fn, dataloader):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    total_steps = 0
    total_consecutive_rests = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            with torch.amp.autocast(enabled=USE_FP16, device_type=DEVICE):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                loss, loss_components = loss_fn(
                    outputs.logits,
                    batch["labels"],
                    batch["attention_mask"]
                )
                
                total_loss += loss.item()
                total_consecutive_rests += loss_components.get('consecutive_rests_detected', 0)
                total_steps += 1
    
    model.train()
    return {
        'val_loss': total_loss / max(1, total_steps),
        'consecutive_rests_detected': total_consecutive_rests
    }

############################################################
# Entry Point
############################################################
def train_lora(model, tokenizer):
    """Main training function"""
    
    # Set model to training mode
    model.train()
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")
    
    tokenized = prepare_dataset(tokenizer)
    metrics_history = train_loop(model, tokenizer, tokenized)
    
    return metrics_history

############################################################
# Plotting
############################################################
import matplotlib.pyplot as plt
def plot_metrics(metrics_history):
    """Plot training metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(metrics_history['train_loss'], label='Train Loss', marker='o')
    if 'val_loss' in metrics_history:
        axes[0].plot(metrics_history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Additional metrics can be added here
    axes[1].set_title('Training Metrics')
    axes[1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_metrics.png'))
    plt.show()