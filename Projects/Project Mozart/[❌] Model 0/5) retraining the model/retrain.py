############################################################
# IMPORTS
############################################################
import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import defaultdict

from transformers import get_linear_schedule_with_warmup

from filter import *
from loss_fn import MusicTokenEnforcementLoss
from prepare_dataset import prepare_dataset, MUSIC_NOTES

############################################################
# CONSTANTS / CONFIG
############################################################
MODEL_NAME = "gpt2-medium"
DOWNLOAD_PATH = "../MODELS/"
TRAIN_FILE = "../dataset/train_file_2.txt"
TEST_FILE = "../dataset/test_file_2.txt"
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

MASK_TOKEN = "<MASK>"
MUSIC_TOKEN_IDS = None  # Will be populated during initialization


def get_music_token_ids(tokenizer):
    """Get token IDs for MUSIC_NOTES, but filter out invalid / missing ones."""
    ids = []
    missing_tokens = []
    for note in MUSIC_NOTES:
        tid = tokenizer.convert_tokens_to_ids(note)
        # Some tokenizers return tokenizer.unk_token_id for unknown tokens, or -1.
        if tid is None or tid < 0 or tid >= tokenizer.vocab_size:
            missing_tokens.append((note, tid))
        else:
            ids.append(int(tid))
    if missing_tokens:
        print(f"[WARN] Some MUSIC_NOTES are missing or invalid in tokenizer: {missing_tokens[:10]} "
                f"(showing up to 10). Filtered them out.")
    return ids

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
    
# Modify collate_fn to include musical_attention_mask
def musical_collate_fn(samples, tokenizer=None):
    result = collate_fn(samples, tokenizer)
    
    # Add musical attention mask
    musical_masks = [torch.tensor(s["musical_attention_mask"], dtype=torch.long) for s in samples]
    musical_attention_mask = torch.nn.utils.rnn.pad_sequence(
        musical_masks, batch_first=True, padding_value=0
    )
    
    result["musical_attention_mask"] = musical_attention_mask
    return result

############################################################
# Enhanced Training Loop
############################################################
def train_loop(model, tokenizer, tokenized):
    """Enhanced training loop with masked token prediction"""
    global MUSIC_TOKEN_IDS
    MUSIC_TOKEN_IDS = get_music_token_ids(tokenizer)
    
    custom_loss_fn = MusicTokenEnforcementLoss(tokenizer, MUSIC_TOKEN_IDS, non_music_penalty=100.0)

    train_dataloader = DataLoader(
        tokenized["train"],
        batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda s: musical_collate_fn(s, tokenizer),
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
                
                # Add strict masking to the training loop after model forward pass:
                # Replace the current masking code with:
                # STRICT masking: only allow music tokens in predictions
                logits = outputs.logits

                # Create strict music-only mask
                music_token_mask = torch.zeros(logits.size(-1), device=logits.device)
                music_token_mask[MUSIC_TOKEN_IDS] = 1
                music_token_mask[tokenizer.pad_token_id] = 1
                music_token_mask[tokenizer.eos_token_id] = 1

                # Apply mask to logits - set non-music tokens to very negative value
                logits = logits + (1 - music_token_mask) * -1e9
                outputs.logits = logits
                
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
        model.save_pretrained(OUTPUT_DIR, save_embedding_layers=True)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        print(f"Model and tokenizer for epoch {epoch+1} saved at {OUTPUT_DIR}")

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


