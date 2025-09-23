"""
Debug script to isolate the model loading issue.
Run this step by step to find where the problem occurs.
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Your paths
DEFAULT_BASE_MODEL = "../MODELS/gpt2-medium-local/"
SAVED_MODEL_DIR = "../MODELS/Project_Mozart_gpt2-medium"
CACHE_DIR = "../MODELS/"

def test_tokenizer_only():
    """Step 1: Test tokenizer loading"""
    print("=== STEP 1: Testing tokenizer loading ===")
    try:
        # Try loading tokenizer from saved dir first
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                SAVED_MODEL_DIR,
                use_fast=True,
                local_files_only=True,
                cache_dir=CACHE_DIR,
            )
            print("✓ Loaded tokenizer from saved model dir")
        except Exception as e:
            print(f"× Failed to load from saved dir: {e}")
            print("Trying base model...")
            tokenizer = AutoTokenizer.from_pretrained(
                DEFAULT_BASE_MODEL,
                use_fast=True,
                local_files_only=True,
                cache_dir=CACHE_DIR,
            )
            print("✓ Loaded tokenizer from base model")
        
        # Add special tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        special_tokens = {
            "additional_special_tokens": [
                "<|startofpiece|>", "<|endofpiece|>",
                "<TRACKS>", "<TRACKSEP>", 
                "<NAME=", "<BPM=", "<DURATION_BEATS=", "<DURATION_MINUTES="
            ]
        }
        tokenizer.add_special_tokens(special_tokens)
        
        print(f"✓ Tokenizer vocab size: {len(tokenizer)}")
        print(f"✓ Pad token: {tokenizer.pad_token}")
        print(f"✓ EOS token: {tokenizer.eos_token}")
        
        return tokenizer
        
    except Exception as e:
        print(f"× Tokenizer loading failed: {e}")
        return None

def test_base_model_only(tokenizer):
    """Step 2: Test base model loading (no PEFT)"""
    print("\n=== STEP 2: Testing base model loading (CPU first) ===")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_BASE_MODEL,
            local_files_only=True,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float32,  # Use float32 for CPU
        )
        print(f"✓ Base model loaded on CPU")
        print(f"✓ Model embedding size: {model.get_input_embeddings().num_embeddings}")
        print(f"✓ Tokenizer size: {len(tokenizer)}")
        
        # Check size mismatch
        if model.get_input_embeddings().num_embeddings != len(tokenizer):
            print(f"⚠ SIZE MISMATCH DETECTED!")
            print(f"  Model embeddings: {model.get_input_embeddings().num_embeddings}")
            print(f"  Tokenizer vocab: {len(tokenizer)}")
            print("  Resizing embeddings...")
            model.resize_token_embeddings(len(tokenizer))
            print("✓ Embeddings resized")
        
        return model
        
    except Exception as e:
        print(f"× Base model loading failed: {e}")
        return None

def test_cuda_transfer(model):
    """Step 3: Test moving to CUDA"""
    print("\n=== STEP 3: Testing CUDA transfer ===")
    if not torch.cuda.is_available():
        print("× CUDA not available, skipping")
        return model
    
    try:
        print("Moving model to CUDA...")
        model = model.to('cuda')
        print("✓ Model moved to CUDA successfully")
        return model
    except Exception as e:
        print(f"× CUDA transfer failed: {e}")
        print("This is where your error occurs!")
        return None

def test_peft_loading(base_model):
    """Step 4: Test PEFT adapter loading"""
    print("\n=== STEP 4: Testing PEFT loading ===")
    try:
        from peft import PeftModel
        if os.path.isdir(SAVED_MODEL_DIR) and os.listdir(SAVED_MODEL_DIR):
            model = PeftModel.from_pretrained(
                base_model, 
                SAVED_MODEL_DIR, 
                local_files_only=True
            )
            print("✓ PEFT adapters loaded")
            return model
        else:
            print("× No PEFT adapters found")
            return base_model
    except Exception as e:
        print(f"× PEFT loading failed: {e}")
        return base_model

def test_simple_generation(model, tokenizer):
    """Step 5: Test simple generation"""
    print("\n=== STEP 5: Testing simple generation ===")
    try:
        device = next(model.parameters()).device
        test_text = "Hello"
        inputs = tokenizer.encode(test_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_new_tokens=5,
                do_sample=False
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Generation test passed: '{result}'")
        return True
    except Exception as e:
        print(f"× Generation test failed: {e}")
        return False

def run_full_diagnosis():
    """Run complete diagnosis"""
    print("Starting model loading diagnosis...\n")
    
    # Step 1: Tokenizer
    tokenizer = test_tokenizer_only()
    if tokenizer is None:
        print("DIAGNOSIS: Tokenizer loading failed - check file paths")
        return
    
    # Step 2: Base model
    model = test_base_model_only(tokenizer)
    if model is None:
        print("DIAGNOSIS: Base model loading failed - check model files")
        return
    
    # Step 3: CUDA transfer
    cuda_model = test_cuda_transfer(model)
    if cuda_model is None:
        print("DIAGNOSIS: CUDA transfer failed - this is your main issue!")
        print("\nSOLUTIONS TO TRY:")
        print("1. Use CPU instead: device='cpu'")
        print("2. Clear CUDA cache: torch.cuda.empty_cache()")
        print("3. Restart Python kernel")
        print("4. Check GPU memory usage")
        return
    
    # Step 4: PEFT
    final_model = test_peft_loading(cuda_model)
    
    # Step 5: Generation
    test_simple_generation(final_model, tokenizer)
    
    print("\n=== DIAGNOSIS COMPLETE ===")

if __name__ == "__main__":
    run_full_diagnosis()