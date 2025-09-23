############################################################
# Imports
# - Hugging Face Transformers: model, trainer, args, collator
# - PEFT: for LoRA fine-tuning with parameter-efficient methods
# - Datasets: dataset loading utilities
# - Torch: for tensor operations & dtype settings
############################################################
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

from datasets import load_dataset
import torch


############################################################
# Constants / Configurations
############################################################
MODEL_NAME = "distilgpt2"      # Base model (small GPT-2 distilled)
DOWNLOAD_PATH = "../MODELS/"              # Cache directory for model
TRAIN_FILE = "../dataset/train.txt"       # Training text dataset
OUTPUT_DIR = "../MODELS/Project_Mozart_distilgpt2"  # Output path for fine-tuned model


# Notebook mode: set to True only if you have bitsandbytes installed AND plan to run with accelerate
USE_8BIT = False

# max length - reduce while debugging
MAX_LENGTH = 512

# -------- Helpers --------
def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, p in model.named_parameters():
        all_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    print(f"Trainable params: {trainable_params} / {all_params} "
          f"({100 * trainable_params / all_params:.6f}%)")

# auto-detect reasonable LoRA target modules for GPT-2 style models
DEFAULT_TARGET_MODULES = ["c_attn", "c_proj"]

############################################################
# Step 1: Prepare Model, Dataset, and Training Setup
############################################################
def prepare_units(tokenizer):
    """
    Prepares the model, dataset, training arguments, and data collator.
    Args:
        tokenizer: The tokenizer used for text preprocessing.
    Returns:
        model: The LoRA-wrapped model ready for training.
        training_args: TrainingArguments for Hugging Face Trainer.
        tokenized: Tokenized dataset for training.
        data_collator: Collator for language modeling.
    """
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Ensure tokenizer has a pad token (GPT2 doesn't have one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
        
    # 1) Load base model in 8-bit precision (memory efficient)
    if USE_8BIT:
        # NOTE: 8-bit loading typically requires bitsandbytes + device_map="auto"
        print("Loading model in 8-bit (requires bitsandbytes + accelerate recommended).")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=DOWNLOAD_PATH,
            load_in_8bit=True,
            device_map="auto",
        )
        # prepare for k-bit training (required for PEFT with 8-bit)
        model = prepare_model_for_kbit_training(model)
        
    else:
        # Standard fp16 load for single-GPU notebook
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=DOWNLOAD_PATH,
            dtype=torch.float16,   # use float16 on GPU to save memory
            low_cpu_mem_usage=True
        )
        # move to device AFTER model is loaded and before training
        model.to(device)

    
    # 2) Resize embeddings to match tokenizer vocabulary
    model.resize_token_embeddings(len(tokenizer))
    
    # 3) LoRA configuration (lightweight fine-tuning)
    lora_config = LoraConfig(
        r=8,                                    # Rank (low-rank adaptation)
        lora_alpha=16,                          # Scaling factor
        target_modules=DEFAULT_TARGET_MODULES,  # GPT-2 layers where LoRA is applied
        lora_dropout=0.05,                      # Dropout for regularization
        bias="none",                            # Bias not trained
        task_type="CAUSAL_LM",                  # Task: causal language modeling
    )
    
    # Wrap base model with LoRA
    model = get_peft_model(model, lora_config)
    
    # Debug: confirm some params are trainable
    print_trainable_parameters(model)
    
    # 4) Load dataset (expects a plain text file)
    dataset = load_dataset("text", data_files={"train": TRAIN_FILE})
    
    # 5) Tokenization function
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,          # Truncate to max length
            padding=False,
            max_length=MAX_LENGTH     # Maximum sequence length
        )
    
    # Apply tokenization across dataset
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,                # Process multiple examples at once
        remove_columns=["text"]      # Remove raw text to save memory
    )
    
    # Data collator: groups tokenized samples into batches
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,                     # No masked LM; causal LM task
        pad_to_multiple_of=None,
    )
    
    # 6) Training arguments - tuned for single-GPU notebook
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,        # Where to save checkpoints/models
        
        per_device_train_batch_size=1,   # Small batch size due to memory limits
        gradient_accumulation_steps=8,   # Simulates larger batch size (1*8=8)
        
        num_train_epochs=10,          # Number of training epochs
        save_strategy="epoch",        # Save model after each epoch
        logging_steps=50,             # Log training progress every 50 steps
        disable_tqdm=False,           # <- ensure tqdm progress bar shows
        fp16=True,                    # Mixed precision (faster + memory efficient)
        
        optim="adamw_torch",          # use CPU/GPU-friendly optimizer in notebook
        learning_rate=2e-4,           # Higher LR for LoRA fine-tuning
        weight_decay=0.01,            # Regularization
        
        lr_scheduler_type="cosine",   # Cosine decay schedule
        warmup_steps=50,              # Warmup to stabilize training
        
        gradient_checkpointing=True,  # Saves memory by recomputing gradients
        push_to_hub=False,            # Disable auto-upload
        report_to="none",             # No logging integration (e.g., wandb)
        remove_unused_columns=False,
    )
    
    return model, training_args, tokenized, data_collator
        

############################################################
# Step 2: Train LoRA Model
############################################################
def train_lora(tokenizer):
    """
    Trains the model with LoRA fine-tuning.
    Args:
        tokenizer: The tokenizer used for preprocessing the dataset.
    """
    
    # Prepare model, data, and training arguments
    model, args, tokenized, data_collator = prepare_units(tokenizer)
    
    
    # Quick smoke test (single forward/backward) to ensure gradients exist
    print("Performing a single-step smoke test to verify backward...")
    sample = tokenized["train"][0]
    # ensure input_ids in sample (tokenized mapping yields input_ids)
    input_ids = torch.tensor(sample["input_ids"], dtype=torch.long).unsqueeze(0).to(model.device)
    attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long).unsqueeze(0).to(model.device)
    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        print("Smoke test loss:", outputs.loss.item())
        outputs.loss.backward()
        print("Smoke test backward OK ✅")
    except Exception as e:
        print("Smoke test failed:", e)
        raise
    
    batch = data_collator([tokenized["train"][0], tokenized["train"][1]])
    print(batch.keys())
    print(batch["input_ids"].shape)
    print(batch["labels"].shape)
    print(batch["labels"][0][:50])  # inspect first 50 labels
    
    # 7) Initialize Trainer (Hugging Face training loop)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    trainer.train()
    
    # Save trained LoRA model to output directory
    trainer.save_model(OUTPUT_DIR)
