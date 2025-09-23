#!/usr/bin/env python3
"""
Create tokenizer + model with:
    - special tokens added
    - all tokens from unique_notes.txt added to tokenizer vocab
    - model & tokenizer downloaded to DOWNLOAD_PATH (not default HF cache)
    - tokenizer & model saved to OUTPUT_DIR for reuse
"""

import os
from pathlib import Path
import torch

# --- set HF cache env vars to DOWNLOAD_PATH BEFORE transformers does anything ---
# this forces downloads to the folder you control instead of the default cache.
DOWNLOAD_PATH = Path("../../MODELS").resolve()
os.environ["TRANSFORMERS_CACHE"] = str(DOWNLOAD_PATH)
os.environ["HF_HOME"] = str(DOWNLOAD_PATH)            # general HF home
os.environ["HF_DATASETS_CACHE"] = str(DOWNLOAD_PATH)  # datasets cache if used
os.environ["XDG_CACHE_HOME"] = str(DOWNLOAD_PATH)     # some libs respect this

# --- now import transformer classes (env set above) ---
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ----------------- CONFIG -----------------
MODEL_NAME = "facebook/bart-base"  
UNIQUE_NOTES_FILE = Path("../../dataset/unique_notes.txt")
OUTPUT_DIR = Path("../../MODELS/Project_Mozart_bart").resolve()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LoRA target modules etc. (tweak for your model if needed)
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]

# Create dirs
DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_unique_tokens(path: Path):
    """Load unique tokens from file, one-per-line, ignoring blanks."""
    if not path.exists():
        print(f"Warning: unique tokens file not found at {path}. No extra tokens will be added.")
        return []
    toks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                toks.append(t)
    # deduplicate while preserving order (stable)
    seen = set()
    uniq = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    print(f"Loaded {len(uniq)} unique tokens from {path}")
    return uniq


def create_model_and_tokenizer(
    model_name: str = MODEL_NAME,
    unique_tokens_file: Path = UNIQUE_NOTES_FILE,
    download_dir: Path = DOWNLOAD_PATH,
    output_dir: Path = OUTPUT_DIR,
):
    # Load tokenizer (force use of cache_dir so files are placed in DOWNLOAD_PATH)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(download_dir),
        use_fast=True,
        local_files_only=False,  # set to True if you only want local files
    )

    # --- Add special tokens ---
    special_tokens = {
        "bos_token": "<|startofpiece|>",
        "eos_token": "<|endofpiece|>",
        "pad_token": "<pad>",
        "additional_special_tokens": [
            "<TRACKS>",
            "<TRACKSEP>",
            "<MASK>",
            "<NAME=", "<BPM=", "<DURATION_BEATS=", "<DURATION_MINUTES=",
        ],
    }
    tokenizer.add_special_tokens(special_tokens)

    # --- Load unique tokens and add them to tokenizer ---
    unique_tokens = load_unique_tokens(unique_tokens_file)
    
    # Only add tokens that are not already in the tokenizer
    tokens_to_add = [t for t in unique_tokens if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
    if tokens_to_add:
        added = tokenizer.add_tokens(tokens_to_add)
        print(f"Added {added} new tokens to the tokenizer vocab (requested {len(tokens_to_add)})")
    else:
        print("No new unique tokens to add (all present in tokenizer).")


    # ensure pad token set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"
        

    # Save tokenizer right away to OUTPUT_DIR (so it's available even if model load later fails)
    tokenizer.save_pretrained(str(output_dir))
    print(f"Tokenizer saved to {output_dir}")
    

    # --- Load model ---
    # choose dtype: use float16 if GPU available, otherwise float32 to avoid CPU fp16 issues
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        cache_dir=str(download_dir),
        dtype=torch_dtype,
        local_files_only=False,
    )
    print("Loaded model with AutoModelForSeq2SeqLM (for BART).")  # CHANGED

    # move to device
    model.to(DEVICE)

    # If tokenizer added tokens, resize embeddings before LoRA wrapping
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
        print("Resized model token embeddings to", len(tokenizer))

    # Prepare model for k-bit training (LoRA helper)
    try:
        model = prepare_model_for_kbit_training(model)
        print("Prepared model for k-bit training.")
    except Exception as e:
        # Not fatal — print warning and continue
        print("Warning: prepare_model_for_kbit_training failed or is not applicable. Continuing. Error:", e)

    # LoRA config (you can tune these)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=DEFAULT_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM", 
    )

    # Wrap model with LoRA adapters (this is a lightweight operation)
    model = get_peft_model(model, lora_config)

    # Ensure model.config.pad_token_id set
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Final save: save both model weights (including PEFT adapters) and tokenizer to OUTPUT_DIR
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))  # save again to ensure matching tokenizer files

    # Report parameter counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params if total_params else 0.0
    print(f"Trainable params: {trainable_params} / {total_params} ({trainable_percent:.4f}%)")
    print(f"Model and tokenizer saved to {output_dir}")

    return model, tokenizer

    
