############################################################
# TEST LOSS FUNCTION
############################################################
import torch
from transformers import AutoTokenizer
from loss_fn import MusicTokenEnforcementLoss
from filter import get_unique_tokens
from load_model_tokenizer import load_model_and_tokenizer

def test_loss():
    # Load your tokenizer
    _, tokenizer = load_model_and_tokenizer()
    # Add special tokens if they don't exist
    special_tokens = ["<MASK>", "<|startofpiece|>", "<|endofpiece|>", "<TRACKS>", "<TRACKSEP>", "<NAME=", "<BPM=", "<DURATION_"]
    tokenizer.add_tokens(special_tokens)
    
    MUSIC_NOTES = get_unique_tokens()

    # Get music token IDs
    def get_music_token_ids(tokenizer):
        ids = []
        vocab_size = tokenizer.vocab_size
        for note in MUSIC_NOTES:
            tid = tokenizer.convert_tokens_to_ids(note)
            if tid is not None and tid >= 0 and tid < vocab_size:
                ids.append(int(tid))
        return ids

    MUSIC_TOKEN_IDS = get_music_token_ids(tokenizer)
    print(f"Found {len(MUSIC_TOKEN_IDS)} music token IDs")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Initialize the loss function
    loss_fn = MusicTokenEnforcementLoss(
        tokenizer, MUSIC_TOKEN_IDS, non_music_penalty=100000.0
    )

    # Use the actual mask token from your tokenizer
    mask_token = tokenizer.mask_token if tokenizer.mask_token else "<MASK>"
    test_input = f"<|startofpiece|><NAME=Symphony No. 41 in C, K.551, Jupiter><BPM=150.0><DURATION_BEATS=49962.0><DURATION_MINUTES=46.26><TRACKS>{mask_token} <TRACKSEP> StringInstrument_3: {mask_token}<|endofpiece|>"

    # Tokenize the input
    inputs = tokenizer(
        test_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # Create mock logits and labels - ensure they match vocabulary size
    batch_size, seq_len = inputs["input_ids"].shape
    vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab())

    print(f"Batch size: {batch_size}, Sequence length: {seq_len}, Vocab size: {vocab_size}")

    # Create random logits with correct vocabulary size
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # Create labels (replace mask tokens with actual music tokens)
    labels = inputs["input_ids"].clone()
    
    # Find mask positions and replace with valid music token IDs
    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id)
    
    # Replace mask tokens with valid labels (use first music token ID)
    if mask_positions.any() and MUSIC_TOKEN_IDS:
        # Use actual music tokens for labels
        valid_music_id = MUSIC_TOKEN_IDS[0] if MUSIC_TOKEN_IDS else tokenizer.unk_token_id
        labels[mask_positions] = valid_music_id
        
    # Set non-mask positions to -100 (ignore index)
    labels[~mask_positions] = -100

    print("=" * 80)
    print("TESTING LOSS FUNCTION")
    print("=" * 80)
    print(f"Input: {test_input}")
    print(f"Input IDs: {inputs['input_ids'].tolist()}")
    print(f"Mask positions: {mask_positions.tolist()}")
    print(f"Labels: {labels.tolist()}")

    # Verify all labels are within bounds
    valid_labels = labels[labels != -100]
    if valid_labels.numel() > 0:
        max_label = valid_labels.max().item()
        min_label = valid_labels.min().item()
        print(f"Label range: {min_label} to {max_label}")
        
        if max_label >= vocab_size:
            print(f"ERROR: Label {max_label} exceeds vocab size {vocab_size}")
            # Fix out-of-bounds labels
            labels[labels >= vocab_size] = tokenizer.unk_token_id

    # Test the loss function
    try:
        with torch.no_grad():
            total_loss, loss_components = loss_fn(
                logits,
                labels,
                inputs["attention_mask"],
                mask_positions=mask_positions
            )

        print(f"Total loss: {total_loss.item():.4f}")
        print(f"Loss components: {loss_components}")
        
    except Exception as e:
        print(f"Error in loss function: {e}")
        print("Trying with simplified test...")
        return test_simple_loss(loss_fn, tokenizer, vocab_size)

    # Test context-aware masking
    print("\n" + "=" * 80)
    print("TESTING CONTEXT-AWARE MASKING")
    print("=" * 80)

    try:
        # Get the context-aware mask
        context_mask = loss_fn._create_context_aware_mask(inputs["input_ids"], "cpu")

        # Check allowed tokens at different positions
        positions_to_check = [0, 5, 10, -1]  # Start, middle, and end positions

        for pos in positions_to_check:
            if pos < 0:
                pos = seq_len + pos  # Convert negative index
            
            if pos >= seq_len:
                continue
                
            print(f"\nPosition {pos}:")
            token_id = inputs["input_ids"][0, pos].item()
            token = tokenizer.decode([token_id])
            context_type = loss_fn._get_context_type(inputs["input_ids"], pos)
            print(f"  Token: '{token}' (ID: {token_id})")
            print(f"  Context: {context_type}")
            
            # Check if space is allowed
            space_allowed = loss_fn._should_allow_space(inputs["input_ids"], pos)
            print(f"  Space allowed: {space_allowed}")
            
            # Count allowed tokens
            allowed_tokens = context_mask[0, pos].sum().item()
            print(f"  Total allowed tokens: {allowed_tokens}")
            
            # Show top 10 allowed tokens
            allowed_indices = context_mask[0, pos].nonzero(as_tuple=True)[0]
            if len(allowed_indices) > 0:
                print("  Sample allowed tokens:")
                for i, token_id in enumerate(allowed_indices[:5]):  # Show first 5
                    token = tokenizer.decode([token_id])
                    print(f"    {token} (ID: {token_id})")
                    
    except Exception as e:
        print(f"Error in context-aware masking: {e}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

def test_simple_loss(loss_fn, tokenizer, vocab_size):
    """Simplified test for the loss function"""
    print("Running simplified test...")
    
    # Create simple test data
    batch_size, seq_len, vocab_size = 1, 10, vocab_size
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    
    # Set a few positions to valid labels
    labels[0, 3:6] = tokenizer.convert_tokens_to_ids(["C4", "D4", "E4"])[0]  # Use first valid music token
    
    attention_mask = torch.ones(batch_size, seq_len)
    mask_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask_positions[0, 3:6] = True
    
    try:
        with torch.no_grad():
            total_loss, loss_components = loss_fn(
                logits,
                labels,
                attention_mask,
                mask_positions=mask_positions
            )
        
        print(f"Simple test - Total loss: {total_loss.item():.4f}")
        print(f"Simple test - Loss components: {loss_components}")
        return True
        
    except Exception as e:
        print(f"Simple test also failed: {e}")
        return False
