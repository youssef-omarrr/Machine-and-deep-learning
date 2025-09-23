import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from filter import MusicalTokenizerWrapper, is_musical_token, add_musical_tokens_to_tokenizer


DEFAULT_BASE_MODEL = "../MODELS/gpt2-medium-local/"            # local base model folder (saved_pretrained)
MY_MODEL_DIR = "../MODELS/Project_Mozart_gpt2-medium"        # where LoRA adapters live
CACHE_DIR = "../MODELS/"   
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(
    my_model_dir: str = MY_MODEL_DIR,
    base_model_dir: str = DEFAULT_BASE_MODEL,
    cashed_dir: str = CACHE_DIR,
    device: str =   DEVICE,
    for_training:bool = False):
    
    # 1. Load tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(
                                            pretrained_model_name_or_path = my_model_dir,
                                            local_files_only = True,
                                            cache_dir = cashed_dir,
                                            )
    

    tokenizer = MusicalTokenizerWrapper(base_tokenizer)

    # 2. Load model
    base_model  = AutoModelForCausalLM.from_pretrained(
                                            pretrained_model_name_or_path = base_model_dir,
                                            cache_dir=cashed_dir,
                                            dtype=torch.float16,
                                        ).to(device)

    # Resize embeddings if tokenizer has added tokens
    base_model.resize_token_embeddings(len(tokenizer),
                                        mean_resizing=False)
    
    # 3. Try to load LoRA with ignore_mismatched_sizes
    try:
        model = PeftModel.from_pretrained(
            base_model,
            my_model_dir,
            local_files_only=True,
            ignore_mismatched_sizes=True  # This allows size mismatches
        ).to(device)
        
    except Exception as e:
        print(f"Could not load adapters: {e}. Creating new adapters...")
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
        
    # Resize embeddings if tokenizer has added tokens
    model.resize_token_embeddings(len(tokenizer),
                                        mean_resizing=False)

    
    # Only set to eval mode if NOT for training
    if not for_training:
        model.eval()
    else:
        model.train()  # Set to training mode
        # Ensure LoRA parameters are trainable
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
    
    return model, tokenizer


def update_model_with_mask_token():
    """Update existing model and tokenizer with MASK token"""
    
    # 1. Load and update tokenizer
    print("Loading tokenizer...")
    base_tokenizer = AutoTokenizer.from_pretrained(
        MY_MODEL_DIR,
        local_files_only=True,
        cache_dir=CACHE_DIR,
    )
    
    # Wrap tokenizer
    tokenizer = MusicalTokenizerWrapper(base_tokenizer)
    add_musical_tokens_to_tokenizer(tokenizer)
    
    
    # Add MASK token if not already present
    if "<MASK>" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["<MASK>"])
        print(f"Added <MASK> token. New vocab size: {len(tokenizer)}")
    else:
        print("<MASK> token already exists in vocabulary")
        
    
    print("Loading base model with correct vocab size...")
    # Load base model first and resize embeddings
    base_model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_BASE_MODEL,
        cache_dir=CACHE_DIR,
        dtype=torch.float16,
    )
    
    # Resize base model embeddings FIRST
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Now load the LoRA adapters
    print("Loading LoRA adapters...")
    try:
        # Try to load with ignore_mismatched_sizes
        model = PeftModel.from_pretrained(
            base_model,
            MY_MODEL_DIR,
            local_files_only=True,
            ignore_mismatched_sizes=True  # This is key!
        )
    except:
        print("Could not load adapters with mismatched sizes. Creating new adapters...")
        # If that fails, create new adapters from scratch
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            loss_type="CE",                         # Explicitly set loss type to Cross Entropy
        )
        model = get_peft_model(base_model, lora_config)
    
    model = model.to(DEVICE)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    print("Saving updated model...")
    model.save_pretrained(MY_MODEL_DIR, save_embedding_layers=True)
    tokenizer.save_pretrained(MY_MODEL_DIR)
    
    print("Update complete!")
    
    
# Add a function to analyze the tokenizer
def analyze_tokenizer(tokenizer):
    """Analyze the tokenizer's musical vocabulary"""
    if hasattr(tokenizer, 'base_tokenizer'):
        base_tokenizer = tokenizer.base_tokenizer
    else:
        base_tokenizer = tokenizer
    
    musical_tokens = []
    for token, token_id in base_tokenizer.get_vocab().items():
        if is_musical_token(token):
            musical_tokens.append((token, token_id))
    
    print(f"Musical tokens found: {len(musical_tokens)}")
    print("Sample musical tokens:")
    for token, token_id in musical_tokens[:20]:
        print(f"  {token_id}: '{token}'")
    
    return musical_tokens


############################################################
# Plotting
############################################################
import matplotlib.pyplot as plt
import os
OUTPUT_DIR = "../MODELS/Project_Mozart_gpt2-medium"

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