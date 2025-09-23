# -------------------------------------------------------
# Imports
# -------------------------------------------------------
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from notes_utils import build_vocab_maps
import math


# -------------------------------------------------------
# Build type maps & allowed transitions
# -------------------------------------------------------
def build_structural_constraints(tokenizer, device):
    vocab, id_to_token = build_vocab_maps(tokenizer)
    vocab_size = len(vocab)

    expected_transitions = {
        "Bar": ["Position"],
        "Position": ["Pitch"],
        "Pitch": ["Velocity"],
        "Velocity": ["Duration"],
        "Duration": ["Position", "Bar"]
    }

    # Create type index mapping
    types = list(expected_transitions.keys())
    type2idx = {t: i for i, t in enumerate(types)}

    # Map token_id -> type_idx (or -1 if no type)
    token_type_arr = torch.full((vocab_size,), -1, dtype=torch.long)
    for tid, tname in id_to_token.items():
        for t in types:
            if tname.startswith(f"{t}_"):
                token_type_arr[tid] = type2idx[t]
                break

    # Allowed next-token masks
    allowed_next_mask = torch.zeros((len(types), vocab_size), dtype=torch.bool)
    for t, next_types in expected_transitions.items():
        t_idx = type2idx[t]
        for tid, tname in id_to_token.items():
            if any(tname.startswith(f"{nt}_") for nt in next_types):
                allowed_next_mask[t_idx, tid] = True

    return vocab_size, token_type_arr.to(device), allowed_next_mask.to(device)


# -------------------------------------------------------
# Structural Loss
# -------------------------------------------------------
def compute_structural_loss(
    logits_flat, targets_flat, x, token_type_arr, allowed_next_mask,
    pad_id, structural_weight, device
):
    struct_loss = torch.tensor(0.0, device=device)

    if structural_weight > 0:
        curr_type_idx = token_type_arr[x[:, :-1]]          # (B, L-1)
        curr_type_flat = curr_type_idx.contiguous().view(-1)

        valid_pos_mask = (curr_type_flat >= 0) & (targets_flat != pad_id)

        if valid_pos_mask.any():
            logits_valid = logits_flat[valid_pos_mask]       # (N_valid, V)
            types_valid = curr_type_flat[valid_pos_mask]    # (N_valid,)
            probs_valid = torch.softmax(logits_valid, dim=1)

            allowed_masks = allowed_next_mask[types_valid]   # (N_valid, V)
            allowed_prob = (probs_valid * allowed_masks.float()).sum(dim=1)

            eps = 1e-9
            struct_loss = -torch.log(allowed_prob + eps).mean()

    return struct_loss


# -------------------------------------------------------
# Training Loop
# -------------------------------------------------------
def train_one_epoch(model, train_loader, optimizer, scheduler,
                    token_type_arr, allowed_next_mask,
                    main_loss_fn, structural_weight, pad_id,
                    device, epoch, epochs, global_step):
    model.train()
    total_main, total_struct = 0.0, 0.0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f"Epoch {epoch+1}/{epochs}")
    for step, (x, y) in pbar:
        x, y = x.to(device), y.to(device)

        logits = model(x[:, :-1])  # (B, L-1, V)
        B, Lm1, V = logits.shape
        logits_flat = logits.view(-1, V)
        targets_flat = y[:, 1:].contiguous().view(-1)

        # Cross-entropy
        main_loss = main_loss_fn(logits_flat, targets_flat)

        # Structural loss
        struct_loss = compute_structural_loss(
            logits_flat, targets_flat, x,
            token_type_arr, allowed_next_mask,
            pad_id, structural_weight, device
        )

        # Combine losses
        loss = main_loss + structural_weight * struct_loss

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_main += main_loss.item()
        total_struct += struct_loss.item()
        global_step += 1

        pbar.set_postfix({
            "main_loss": f"{total_main / (step+1):.4f}",
            "struct_loss": f"{total_struct / (step+1):.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })

    return total_main / len(train_loader), total_struct / len(train_loader), global_step


# -------------------------------------------------------
# Validation Loop
# -------------------------------------------------------
def validate(model, val_loader, main_loss_fn, device):
    model.eval()
    val_loss = 0.0
    with torch.inference_mode():
        for val_x, val_y in val_loader:
            val_x, val_y = val_x.to(device), val_y.to(device)
            v_logits = model(val_x[:, :-1])
            Bv, Lm1v, Vv = v_logits.shape
            v_logits_flat = v_logits.view(-1, Vv)
            v_targets_flat = val_y[:, 1:].contiguous().view(-1)
            val_loss += main_loss_fn(v_logits_flat, v_targets_flat).item()
    return val_loss / len(val_loader)


# -------------------------------------------------------
# Full Training Function
# -------------------------------------------------------
def train(model, train_loader, val_loader, tokenizer,
          epochs=20,  # Increase epochs - music models need more training
          save_dir="checkpoints/",
          learning_rate=5e-5,  # Reduce learning rate for better convergence
          warmup_steps=2000,   # Increase warmup steps
          structural_weight=0.1,  # Reduce structural weight - it might be too strong
          resume_training=True):
    """
    Train (or resume training) for a MusicTransformer.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Structural constraints
    _, token_type_arr, allowed_next_mask = build_structural_constraints(tokenizer, device)

    # Optimizer & scheduler with better settings
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=learning_rate, 
                                  weight_decay=0.01,
                                  betas=(0.9, 0.95))  # Better betas for music
    
    # Cosine decay after warmup instead of constant
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (len(train_loader) * epochs - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss with less label smoothing for music
    pad_id = getattr(tokenizer, "pad_token_id", 0)
    main_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.05)  # Reduce from 0.1

    global_step, best_loss = 0, float("inf")
    
    # Load checkpoint if exists
    if resume_training:
        ckpt_path = "checkpoints/best_model.pt"
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            print(f"✅ Loaded best model (val_loss={checkpoint['val_loss']:.4f})")

    for epoch in range(epochs):
        train_main, train_struct, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            token_type_arr, allowed_next_mask,
            main_loss_fn, structural_weight, pad_id,
            device, epoch, epochs, global_step
        )

        val_loss = validate(model, val_loader, main_loss_fn, device)
        print(f"Epoch {epoch+1} finished | Train main: {train_main:.4f} | Train struct: {train_struct:.4f} | Val: {val_loss:.4f}")

        # Save checkpoints
        ckpt_path = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt")
        torch.save({
            "epoch": epoch+1,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_loss": val_loss,
        }, ckpt_path)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": best_loss,
            }, os.path.join(save_dir, "best_model.pt"))
            print(f"New best model saved with validation loss: {best_loss:.4f}")

    print(f"Training completed! Best validation loss: {best_loss:.4f}")
