############################################################
# IMPORTS
############################################################
import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import defaultdict

from transformers import get_linear_schedule_with_warmup

from filter import get_unique_tokens
from loss_fn import MusicTokenEnforcementLoss
from prepare_dataset import prepare_dataset

############################################################
# CONSTANTS / CONFIG
############################################################
OUTPUT_DIR = "../../MODELS/Project_Mozart_bart-small"

# Enhanced training hyperparams
EPOCHS = 5
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 1e-5  # Even lower learning rate
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500

MAX_GRAD_NORM = 1.0
USE_FP16 = True
SAVE_EVERY_N_STEPS = 1000
EVAL_EVERY_N_STEPS = 500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]

# Custom loss weights
NOTE_DENSITY_REWARD = 0.3       # Mild reward for more notes
MIN_NOTES_THRESHOLD = 50         # Minimum notes per sequence

MASK_TOKEN = "<MASK>"
MUSIC_NOTES = get_unique_tokens()
MUSIC_TOKEN_IDS = None  # Will be populated during initialization


############################################################
# UTILITIES
############################################################
def get_music_token_ids(tokenizer):
    """Convert music tokens to token IDs"""
    ids = []
    missing_tokens = []
    
    # Get the actual vocabulary size that includes added tokens
    vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab())
    print(f"\nTokenizer vocab_size: {tokenizer.vocab_size}, added tokens: {len(tokenizer.get_added_vocab())}, total: {vocab_size}")
    
    for note in MUSIC_NOTES:
        tid = tokenizer.convert_tokens_to_ids(note)
        # Check if token exists in the expanded vocabulary
        if tid is not None and tid != tokenizer.unk_token_id and tid < vocab_size:
            ids.append(int(tid))
        else:
            missing_tokens.append((note, tid))
    
    if missing_tokens:
        print(f"[WARN] Some MUSIC_NOTES missing in tokenizer: {missing_tokens[:10]} (showing up to 10), total of {len(missing_tokens)}")
    
    print(f"Found {len(ids)} music token IDs")
    return ids


def collate_fn(samples, tokenizer):
    """
    Collate function for input-target pairs.
    """
    input_ids = [torch.tensor(s["input_ids"], dtype=torch.long) for s in samples]
    attention_mask = [torch.tensor(s["attention_mask"], dtype=torch.long) for s in samples]
    
    labels = [torch.tensor(s["labels"], dtype=torch.long) for s in samples]
    mask_positions = [torch.tensor(s["mask_positions"], dtype=torch.long) for s in samples]
    
    # Pad to the same length
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    mask_positions = torch.nn.utils.rnn.pad_sequence(mask_positions, batch_first=True, padding_value=0)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "mask_positions": mask_positions
    }


############################################################
# TRAIN / EVAL LOOP
############################################################

def train_loop(model, tokenizer, tokenized):
    
    global MUSIC_TOKEN_IDS
    MUSIC_TOKEN_IDS = get_music_token_ids(tokenizer)
    print(f"Found {len(MUSIC_TOKEN_IDS)} music token IDs\n")

    # CRITICAL FIX: Much higher penalty to force music token selection
    custom_loss_fn = MusicTokenEnforcementLoss(
        tokenizer, MUSIC_TOKEN_IDS, non_music_penalty=50000.0  # Increased penalty
    )

    # Unified dataloaders
    train_dataloader = DataLoader(
        tokenized["train"],
        batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda s: collate_fn(s, tokenizer),
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        tokenized["test"],
        batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda s: collate_fn(s, tokenizer),
        pin_memory=True,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Scheduler and Scaler
    total_steps = len(train_dataloader) * EPOCHS // GRAD_ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler(device=DEVICE) if USE_FP16 and DEVICE == "cuda" else None

    metrics_history = defaultdict(list)
    global_step = 0
    model.train()
    
    # TEST THE LOSS FUNCTION FIRST
    print("\n🧪 TESTING LOSS FUNCTION ON FIRST BATCH:")
    test_batch = next(iter(train_dataloader))
    test_batch = {k: v.to(DEVICE) for k, v in test_batch.items()}

    with torch.no_grad():
        test_outputs = model(**{k: v for k, v in test_batch.items() if k in ['input_ids', 'attention_mask', 'labels']})
        
        # Test loss function
        test_loss, test_components = custom_loss_fn(
            test_outputs.logits,
            test_batch["labels"],
            test_batch["attention_mask"],
            mask_positions=test_batch["mask_positions"]
        )
        
        print(f"Test loss: {test_loss.item():.4f}")
        print(f"Test components: {test_components}")
        
        # Check predictions
        pred_ids = test_outputs.logits.argmax(dim=-1)
        mask_pos = test_batch["mask_positions"].bool()
        
        if mask_pos.sum().item() > 0:
            masked_preds = pred_ids[mask_pos]
            music_token_set = set(MUSIC_TOKEN_IDS)
            
            # Count different types of predictions
            eos_count = sum(1 for pid in masked_preds if pid.item() == tokenizer.eos_token_id)
            music_count = sum(1 for pid in masked_preds if pid.item() in music_token_set)
            other_count = masked_preds.numel() - eos_count - music_count
            
            print(f"Predictions breakdown:")
            print(f"  Music tokens: {music_count}/{masked_preds.numel()} ({music_count/masked_preds.numel()*100:.1f}%)")
            print(f"  </s> tokens: {eos_count}/{masked_preds.numel()} ({eos_count/masked_preds.numel()*100:.1f}%)")
            print(f"  Other tokens: {other_count}/{masked_preds.numel()} ({other_count/masked_preds.numel()*100:.1f}%)")
            
            # Show some example predictions
            print("First 10 predictions:")
            for i, pid in enumerate(masked_preds[:10]):
                token = tokenizer.decode([pid.item()])
                token_type = "MUSIC" if pid.item() in music_token_set else ("EOS" if pid.item() == tokenizer.eos_token_id else "OTHER")
                print(f"  {i}: '{token}' ({token_type})")
        
    print("=" * 80)

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
                    labels=batch["labels"],
                )
                
                # Debug predictions every 100 steps
                if step % 100 == 0:
                    with torch.no_grad():
                        pred_ids = outputs.logits.argmax(dim=-1)
                        mask_pos = batch["mask_positions"].bool()
                        
                        if mask_pos.sum().item() > 0:
                            masked_preds = pred_ids[mask_pos]
                            music_token_set = set(MUSIC_TOKEN_IDS)
                            
                            eos_count = sum(1 for pid in masked_preds if pid.item() == tokenizer.eos_token_id)
                            music_count = sum(1 for pid in masked_preds if pid.item() in music_token_set)
                            
                            print(f"\nStep {step} predictions: Music={music_count}/{masked_preds.numel()}, EOS={eos_count}/{masked_preds.numel()}")
                
                total_loss, loss_components = custom_loss_fn(
                    outputs.logits,
                    batch["labels"],
                    batch["attention_mask"],
                    mask_positions=batch["mask_positions"],
                )

            # Backward
            loss_reduced = total_loss / GRAD_ACCUM_STEPS
            if scaler is not None:
                scaler.scale(loss_reduced).backward()
            else:
                loss_reduced.backward()

            epoch_loss += total_loss.item()
            steps_in_epoch += 1

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "ce": f"{loss_components['ce_loss']:.4f}",
                    "penalty": f"{loss_components['non_music_penalty']:.4f}",
                }
            )

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                if scaler:
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

                if global_step % SAVE_EVERY_N_STEPS == 0:
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    model.save_pretrained(OUTPUT_DIR, save_embedding_layers=True)
                    tokenizer.save_pretrained(OUTPUT_DIR)
                    print(f"\nSaved Model and tokenizer checkpoint at step {global_step}")

                if global_step % EVAL_EVERY_N_STEPS == 0:
                    val_metrics = evaluate_model(model, custom_loss_fn, val_dataloader, tokenizer)

                    print(f"\nValidation at step {global_step}:")
                    print(f"  Total Loss: {val_metrics['val_loss']:.4f}")
                    print(f"  CE Loss: {val_metrics['val_ce_loss']:.4f}")
                    print(f"  Penalty Loss: {val_metrics['val_penalty_loss']:.4f}")
                    print(f"  Non-music rate: {val_metrics['non_music_rate']:.3%}")
                    print(f"  Music rate: {val_metrics['music_rate']:.3%}")
                    print(f"  EOS rate: {val_metrics['eos_rate']:.3%}")
                    print(f"  Exact-match rate: {val_metrics['exact_match_rate']:.3%}")

                    metrics_history["val_loss"].append(val_metrics["val_loss"])

        avg_train_loss = epoch_loss / max(1, steps_in_epoch)
        metrics_history["train_loss"].append(avg_train_loss)

        val_metrics = evaluate_model(model, custom_loss_fn, val_dataloader, tokenizer)

        print(f"\nEpoch {epoch+1} finished.")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Total Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Val CE Loss: {val_metrics['val_ce_loss']:.4f}")
        print(f"  Val Penalty Loss: {val_metrics['val_penalty_loss']:.4f}")
        print(f"  Non-music rate: {val_metrics['non_music_rate']:.3%}")
        print(f"  Music rate: {val_metrics['music_rate']:.3%}")
        print(f"  EOS rate: {val_metrics['eos_rate']:.3%}")
        print(f"  Exact-match rate: {val_metrics['exact_match_rate']:.3%}")

        # Save model after each epoch
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR, save_embedding_layers=True)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Model and tokenizer saved at {OUTPUT_DIR}")

    return metrics_history


def evaluate_model(model, loss_fn, dataloader, tokenizer=None):
    """
    Enhanced evaluation with detailed prediction breakdown
    """
    model.eval()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_penalty_loss = 0.0
    total_steps = 0

    total_mask_positions = 0
    total_music_preds = 0
    total_eos_preds = 0
    total_other_preds = 0
    total_exact_matches = 0

    music_token_set = set(MUSIC_TOKEN_IDS)
    eos_token_id = tokenizer.eos_token_id if tokenizer else None

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            batch_mask_count = int(batch["mask_positions"].sum().item())
            total_mask_positions += batch_mask_count

            with torch.amp.autocast(enabled=USE_FP16, device_type=DEVICE):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                loss, loss_components = loss_fn(
                    outputs.logits,
                    batch["labels"],
                    batch["attention_mask"],
                    mask_positions=batch["mask_positions"],
                )

            total_loss += loss.item()
            total_ce_loss += loss_components.get("ce_loss", 0.0)
            total_penalty_loss += loss_components.get("non_music_penalty", 0.0)
            total_steps += 1

            # Detailed prediction analysis
            pred_ids = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            mask_pos = batch["mask_positions"].bool()

            pred_flat = pred_ids[mask_pos]
            label_flat = labels[mask_pos]

            if pred_flat.numel() > 0:
                pred_np = pred_flat.detach().cpu().numpy()
                label_np = label_flat.detach().cpu().numpy()

                for p, l in zip(pred_np, label_np):
                    p_int, l_int = int(p), int(l)
                    
                    # Count prediction types
                    if p_int in music_token_set:
                        total_music_preds += 1
                    elif p_int == eos_token_id:
                        total_eos_preds += 1
                    else:
                        total_other_preds += 1
                    
                    # Count exact matches
                    if l_int != -100 and p_int == l_int:
                        total_exact_matches += 1

    model.train()

    # Calculate rates
    mask_pos_total = total_mask_positions if total_mask_positions > 0 else 1
    music_rate = total_music_preds / mask_pos_total
    eos_rate = total_eos_preds / mask_pos_total
    other_rate = total_other_preds / mask_pos_total
    exact_match_rate = total_exact_matches / mask_pos_total

    return {
        "val_loss": total_loss / max(1, total_steps),
        "val_ce_loss": total_ce_loss / max(1, total_steps),
        "val_penalty_loss": total_penalty_loss / max(1, total_steps),
        "non_music_predictions": total_eos_preds + total_other_preds,
        "total_mask_positions": total_mask_positions,
        "music_rate": music_rate,
        "eos_rate": eos_rate,
        "non_music_rate": eos_rate + other_rate,  # EOS + other non-music tokens
        "exact_match_rate": exact_match_rate,
        "error_rate": eos_rate + other_rate
    }


############################################################
# ENTRY POINT
############################################################
def train_lora(model, tokenizer):
    """Main training function"""
    model.train()

    # Prepare dataset - now returns separate input and target datasets
    dataset = prepare_dataset(tokenizer)
    
    # Verify we have all expected datasets
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    metrics_history = train_loop(model, tokenizer, dataset)
    
    return metrics_history