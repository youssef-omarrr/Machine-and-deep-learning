############################################################
# IMPORTS
############################################################
import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import AutoModelForCausalLM

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from datasets import load_dataset


############################################################
# CONSTANTS / CONFIG
############################################################
MODEL_NAME = "gpt2-medium"     # test_1 uses: 'distilgpt2', test_2 uses: 'gpt2-medium'
DOWNLOAD_PATH = "../MODELS/"
TRAIN_FILE = "../dataset/train.txt"
OUTPUT_DIR = "../MODELS/Project_Mozart_gpt2-medium"

# training hyperparams
EPOCHS = 20

PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4

MAX_GRAD_NORM = 1.0
USE_FP16 = True
SAVE_EVERY_N_STEPS = 2000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_TARGET_MODULES = ["c_attn", "c_proj"]

############################################################
# STEP 1: Prepare model with LoRA
############################################################
def prepare_model(tokenizer):
    """
    Loads distilgpt2 in half precision, applies LoRA adapters,
    resizes embeddings to match tokenizer.
    """
    # 1) Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=DOWNLOAD_PATH,
        dtype=torch.float16 if USE_FP16 else torch.float32,
    ).to(DEVICE)

    # 2) Prepare model for k-bit training (LoRA setup)
    model = prepare_model_for_kbit_training(model)

    # 3) Resize embeddings for tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # 4) Define LoRA config
    lora_config = LoraConfig(
        r=8,                                    # Rank (low-rank adaptation)
        lora_alpha=16,                          # Scaling factor
        target_modules=DEFAULT_TARGET_MODULES,  # GPT-2 layers where LoRA is applied
        lora_dropout=0.05,                      # Dropout for regularization
        bias="none",                            # Bias not trained
        task_type="CAUSAL_LM",                  # Task: causal language modeling
    )

    # 5) Wrap model with LoRA adapters
    model = get_peft_model(model, lora_config)

    # Count params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params

    print(f"Trainable params: {trainable_params} / {total_params} ({trainable_percent:.4f}%)")

    return model


############################################################
# STEP 2: Load and tokenize dataset
############################################################
def prepare_dataset(tokenizer):
    """
    Loads the dataset from text file and tokenizes it.
    """
    # 1) Load dataset (expects a plain text file)
    dataset = load_dataset("text", data_files={"train": TRAIN_FILE})
    
    # split into train/validation
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    # 2) Tokenize function
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,  # keep modest length for 6GB GPU
        )

    # 3) Apply tokenizer
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )

    return tokenized


############################################################
# STEP 3: Collate function for DataLoader
############################################################
def collate_fn(samples, tokenizer=None):
    """
    Pads sequences in a batch and prepares input_ids, attention_mask, labels.
    """
    input_ids = [torch.tensor(s["input_ids"], dtype=torch.long) for s in samples]
    attention_mask = [torch.tensor(s["attention_mask"], dtype=torch.long) for s in samples]

    # Pad to max length in batch
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    labels = input_ids.clone()

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


############################################################
# STEP 4: Training loop (with validation)
############################################################
def train_loop(model, tokenizer, tokenized):
    """
    Manual PyTorch training loop with tqdm progress bar and validation.
    """

    # 1) DataLoader
    train_dataset = tokenized["train"]
    val_dataset = tokenized["test"]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda s: collate_fn(s, tokenizer),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda s: collate_fn(s, tokenizer),
    )

    # 2) Optimizer (only trainable LoRA params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)

    # 3) Mixed precision
    scaler = torch.amp.GradScaler(DEVICE) if USE_FP16 else None

    # 4) Training loop
    global_step = 0
    model.train()
    
    # === NEW: track losses ===
    all_train_losses, all_val_losses, epoch_train_losses, epoch_val_losses = [], [], [], []

    for epoch in range(EPOCHS):
        ################
        # Training
        ################
        epoch_loss, steps_in_epoch = 0.0, 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=True)

        for step, batch in enumerate(pbar):
            
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            with torch.amp.autocast(enabled=USE_FP16,
                                    device_type= DEVICE):
                outputs = model(input_ids=input_ids, 
                                attention_mask=attention_mask, 
                                labels=labels)
                loss = outputs.loss
                loss_reduced = loss / GRAD_ACCUM_STEPS

            if scaler is not None:
                scaler.scale(loss_reduced).backward()
            else:
                loss_reduced.backward()

            epoch_loss += loss.item()
            steps_in_epoch += 1
            all_train_losses.append(loss.item())
            pbar.set_postfix({'train_loss': f"{loss.item():.4f}"})

            # Step optimizer every GRAD_ACCUM_STEPS
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                    optimizer.step()

                optimizer.zero_grad()
                global_step += 1

                # Optional checkpoint save
                if global_step % SAVE_EVERY_N_STEPS == 0:
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    model.save_pretrained(OUTPUT_DIR)
                    tokenizer.save_pretrained(OUTPUT_DIR)
                    print(f"\nSaved checkpoint at step {global_step}")
                    
        # FInal average losses
        avg_train_loss = epoch_loss / max(1, steps_in_epoch)
        epoch_train_losses.append(avg_train_loss)

        ##############
        # Validation
        ###############
        model.eval()
        val_loss, val_steps = 0.0, 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                with torch.amp.autocast(enabled=USE_FP16, device_type=DEVICE):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                val_loss += loss.item()
                val_steps += 1
                all_val_losses.append(loss.item())

        avg_val_loss = val_loss / max(1, val_steps)
        epoch_val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1} finished. train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        model.train()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        print(f"Saved epoch {epoch+1} checkpoint to {OUTPUT_DIR}")

    print("Training finished. Final model saved to", OUTPUT_DIR)
    return all_train_losses, all_val_losses, epoch_train_losses, epoch_val_losses

############################################################
# STEP 5: Entry point
############################################################
def train_lora(tokenizer):
    """
    Full pipeline: prepare model, dataset, and train.
    """
    # 1) Prepare model
    model = prepare_model(tokenizer)

    # 2) Prepare dataset
    tokenized = prepare_dataset(tokenizer)

    # 3) Train loop
    all_train_losses, all_val_losses, epoch_train_losses, epoch_val_losses = train_loop(model, tokenizer, tokenized)
    
    # === NEW: return losses ===
    return all_train_losses, all_val_losses, epoch_train_losses, epoch_val_losses



############################################################
# STEP 5: Plot output
############################################################
import matplotlib.pyplot as plt
def plot_losses(all_train_losses, all_val_losses, epoch_train_losses, epoch_val_losses):
    """
    Plots training and validation loss curves (per step and per epoch).
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Step-level losses
    axs[0].plot(all_train_losses, label="Train Loss (per step)", alpha=0.7)
    axs[0].plot(all_val_losses, label="Val Loss (per step)", alpha=0.7)
    axs[0].set_title("Step-level Losses")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True, linestyle="--", alpha=0.6)

    # Epoch-level average losses
    axs[1].plot(epoch_train_losses, label="Train Loss (per epoch)", marker="o")
    axs[1].plot(epoch_val_losses, label="Val Loss (per epoch)", marker="o")
    axs[1].set_title("Epoch-level Average Losses")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    axs[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()
