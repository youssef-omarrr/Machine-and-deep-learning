"""
Debug script to analyze tokenizer and identify token issues
"""
import torch
from filter import get_unique_tokens, is_music_token

def debug_tokenizer(tokenizer):
    """Comprehensive tokenizer analysis"""
    
    print("=" * 80)
    print("TOKENIZER DEBUG ANALYSIS")
    print("=" * 80)
    
    # Basic tokenizer info
    print(f"Tokenizer type: {type(tokenizer).__name__}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Added vocab size: {len(tokenizer.get_added_vocab())}")
    print(f"Total vocab size: {tokenizer.vocab_size + len(tokenizer.get_added_vocab())}")
    
    # Special tokens
    special_tokens = {
        'pad_token': tokenizer.pad_token,
        'unk_token': tokenizer.unk_token,
        'eos_token': tokenizer.eos_token,
        'bos_token': tokenizer.bos_token,
        'mask_token': getattr(tokenizer, 'mask_token', None),
    }
    
    print("\nSPECIAL TOKENS:")
    for name, token in special_tokens.items():
        if token is not None:
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"  {name}: '{token}' (ID: {token_id})")
    
    # All special token IDs
    print(f"\nAll special token IDs: {tokenizer.all_special_ids}")
    
    # Check </s> token specifically
    eos_variants = ['</s>', '<s>', '<|endoftext|>', 'EOS', '<eos>', '<|endofpiece|>']
    print(f"\nEND-OF-SEQUENCE TOKEN VARIANTS:")
    for variant in eos_variants:
        token_id = tokenizer.convert_tokens_to_ids(variant)
        if token_id != tokenizer.unk_token_id:
            print(f"  '{variant}' -> ID: {token_id}")
    
    # Check music tokens
    music_notes = get_unique_tokens()
    print(f"\nMUSIC TOKENS ANALYSIS:")
    print(f"Total music tokens from filter: {len(music_notes)}")
    
    valid_music_ids = []
    invalid_music_tokens = []
    
    for note in music_notes[:20]:  # Check first 20
        token_id = tokenizer.convert_tokens_to_ids(note)
        if token_id is None or token_id == tokenizer.unk_token_id:
            invalid_music_tokens.append(note)
        else:
            valid_music_ids.append(token_id)
            if len(valid_music_ids) <= 10:  # Show first 10
                print(f"  '{note}' -> ID: {token_id}")
    
    print(f"Valid music token IDs (first 10): {valid_music_ids[:10]}")
    print(f"Invalid music tokens (first 5): {invalid_music_tokens[:5]}")
    
    # Check custom tokens
    custom_tokens = ['<MASK>', '<TRACKS>', '<|startofpiece|>', '<|endofpiece|>']
    print(f"\nCUSTOM TOKENS:")
    for token in custom_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            print(f"  '{token}' -> ID: {token_id}")
        else:
            print(f"  '{token}' -> NOT FOUND")
    
    # Test a sample music sequence
    test_sequence = "A1.C2.E-2_9.0 B5.D6_0.75 <MASK> C4.E4.G4_q"
    print(f"\nTEST TOKENIZATION:")
    print(f"Input: {test_sequence}")
    
    tokens = tokenizer.tokenize(test_sequence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    print("Tokenization:")
    for i, (token, tid) in enumerate(zip(tokens, token_ids)):
        is_music = is_music_token(token)
        is_special = tid in tokenizer.all_special_ids
        print(f"  {i}: '{token}' -> ID: {tid} [music: {is_music}, special: {is_special}]")
    
    # Check what happens during encoding/decoding
    encoded = tokenizer.encode(test_sequence, add_special_tokens=True)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nENCODING TEST:")
    print(f"Original: {test_sequence}")
    print(f"Encoded IDs: {encoded}")
    print(f"Decoded: {decoded}")
    
    return {
        'total_vocab_size': tokenizer.vocab_size + len(tokenizer.get_added_vocab()),
        'valid_music_ids': valid_music_ids,
        'eos_token_id': tokenizer.eos_token_id,
        'special_token_ids': tokenizer.all_special_ids,
    }


def test_loss_function_setup(tokenizer):
    """Test the loss function token categorization"""
    
    print("\n" + "=" * 80)
    print("LOSS FUNCTION SETUP TEST")
    print("=" * 80)
    
    from filter import get_unique_tokens
    
    # Get music token IDs
    music_notes = get_unique_tokens()
    music_token_ids = []
    
    vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab())
    print(f"Working with vocab size: {vocab_size}")
    
    for note in music_notes:
        tid = tokenizer.convert_tokens_to_ids(note)
        if tid is not None and tid >= 0 and tid < vocab_size:
            music_token_ids.append(tid)
    
    print(f"Valid music token IDs: {len(music_token_ids)}")
    
    # Test allowed special tokens
    allowed_special = ['<MASK>', '<pad>', '<unk>', '<TRACKS>', '<|startofpiece|>', '<|endofpiece|>']
    allowed_special_ids = []
    
    for token in allowed_special:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and tid != tokenizer.unk_token_id:
            allowed_special_ids.append(tid)
            print(f"Allowed special: '{token}' -> ID: {tid}")
    
    # Test excluded tokens
    excluded_tokens = ['</s>', '<s>', '<|endoftext|>', 'EOS', '<eos>']
    excluded_ids = []
    
    for token in excluded_tokens:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and tid != tokenizer.unk_token_id:
            excluded_ids.append(tid)
            print(f"Excluded: '{token}' -> ID: {tid}")
    
    # Create the allowed mask like in the loss function
    allowed_mask = torch.zeros(vocab_size, dtype=torch.bool)
    
    # Allow music tokens
    if music_token_ids:
        allowed_mask[music_token_ids] = True
        print(f"Allowed {len(music_token_ids)} music tokens")
    
    # Allow special tokens
    if allowed_special_ids:
        allowed_mask[allowed_special_ids] = True
        print(f"Allowed {len(allowed_special_ids)} special tokens")
    
    # Exclude end tokens
    if excluded_ids:
        allowed_mask[excluded_ids] = False
        print(f"Excluded {len(excluded_ids)} end tokens")
    
    # Check </s> specifically
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        eos_allowed = allowed_mask[eos_id].item()
        print(f"\n</s> token (ID {eos_id}) is allowed: {eos_allowed}")
        if eos_allowed:
            print("❌ PROBLEM: </s> token should NOT be allowed!")
        else:
            print("✅ GOOD: </s> token is properly excluded")
    
    total_allowed = allowed_mask.sum().item()
    print(f"\nTotal allowed tokens: {total_allowed}/{vocab_size} ({total_allowed/vocab_size*100:.1f}%)")
    
    # Show some allowed tokens
    allowed_indices = torch.where(allowed_mask)[0][:20]  # First 20 allowed tokens
    print(f"\nFirst 20 allowed tokens:")
    for i, idx in enumerate(allowed_indices):
        token = tokenizer.convert_ids_to_tokens([idx.item()])[0]
        print(f"  {i}: ID {idx.item()} -> '{token}'")


if __name__ == "__main__":
    # Example usage - you'll need to load your tokenizer
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("your_model_path")
    # 
    # debug_info = debug_tokenizer(tokenizer)
    # test_loss_function_setup(tokenizer)
    
    print("Run this script with your tokenizer loaded to debug token issues")