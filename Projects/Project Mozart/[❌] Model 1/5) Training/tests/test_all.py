"""
Validation script to test the fixes before full training
"""
import torch

def test_preprocessing_fix(tokenizer):
    """Test that the new preprocessing preserves all necessary tokens"""
    
    print("=" * 80)
    print("TESTING PREPROCESSING FIXES")
    print("=" * 80)
    
    from prepare_dataset import pre_tokenize_musical_text
    
    # Test with your actual problematic example
    sample_input = "<|startofpiece|><NAME=Symphony No. 41 in C, K.551, Jupiter><BPM=150.0><DURATION_BEATS=49962.0><DURATION_MINUTES=46.26><TRACKS><MASK> <TRACKSEP> StringInstrument_3: <MASK><|endofpiece|>"
    
    sample_target = "<|startofpiece|><NAME=Symphony No. 41 in C, K.551, Jupiter><BPM=150.0><DURATION_BEATS=49962.0><DURATION_MINUTES=46.26><TRACKS>G4_1.5 B-4_e E-4_1.5 C5_e A4_e E-5_e C5_s Rest_s A4_s Rest_s F#4_s Rest_s G4_s Rest_s A4_s Rest_s B-4_s Rest_s G4_s Rest_0.75 G4_s Rest_1.75 E4_s Rest_0.75 Rest_q C#4_s Rest_1.75 A3_s Rest_0.75 D5_e F5_e <TRACKSEP> StringInstrument_3: B4_s Rest_s D5_s Rest_s G#4_s Rest_s B4_s Rest_s E4_s Rest_s E5_s Rest_s C5_s Rest_0.75 A4_s Rest_0.75 G4_s<|endofpiece|>"
    
    print("ORIGINAL INPUT:")
    print(sample_input)
    print()
    
    print("ORIGINAL TARGET:")
    print(sample_target)
    print()
    
    # Test new preprocessing
    processed_input = pre_tokenize_musical_text(sample_input)
    processed_target = pre_tokenize_musical_text(sample_target)
    
    print("PROCESSED INPUT:")
    print(processed_input)
    print()
    
    print("PROCESSED TARGET:")
    print(processed_target)
    print()
    
    # Check if mask tokens are preserved
    input_mask_count = processed_input.count('<MASK>')
    target_mask_count = processed_target.count('<MASK>')
    
    print(f"INPUT <MASK> count: {input_mask_count}")
    print(f"TARGET <MASK> count: {target_mask_count}")
    
    if input_mask_count == 0:
        print("❌ ERROR: All <MASK> tokens were removed from input!")
        return False
    
    # Tokenize and check alignment
    input_tokens = tokenizer.tokenize(processed_input)
    target_tokens = tokenizer.tokenize(processed_target)
    
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    
    # Find mask positions
    mask_token_id = tokenizer.convert_tokens_to_ids('<MASK>')
    mask_positions = [i for i, tid in enumerate(input_ids) if tid == mask_token_id]
    
    print(f"\nTOKENIZATION RESULTS:")
    print(f"  Input tokens: {len(input_tokens)}")
    print(f"  Target tokens: {len(target_tokens)}")
    print(f"  Mask positions: {mask_positions}")
    
    # Check what should be predicted at mask positions
    print(f"\nEXPECTED PREDICTIONS:")
    for pos in mask_positions:
        if pos < len(target_ids):
            expected_id = target_ids[pos]
            expected_token = tokenizer.convert_ids_to_tokens([expected_id])[0]
            print(f"  Position {pos}: {expected_token} (ID: {expected_id})")
        else:
            print(f"  Position {pos}: OUT OF BOUNDS!")
            return False
    
    return len(mask_positions) > 0


def test_loss_function_fix(tokenizer):
    """Test that the loss function properly excludes problematic tokens"""
    
    print("\n" + "=" * 80)
    print("TESTING LOSS FUNCTION FIXES")
    print("=" * 80)
    
    from loss_fn import MusicTokenEnforcementLoss
    from filter import get_unique_tokens
    
    # Get music token IDs
    music_notes = get_unique_tokens()
    music_token_ids = []
    
    vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab())
    
    for note in music_notes:
        tid = tokenizer.convert_tokens_to_ids(note)
        if tid is not None and tid >= 0 and tid < vocab_size:
            music_token_ids.append(tid)
    
    # Create loss function
    loss_fn = MusicTokenEnforcementLoss(tokenizer, music_token_ids, non_music_penalty=10000.0)
    
    # Test with dummy data to initialize the mask
    dummy_logits = torch.randn(1, 10, vocab_size)
    dummy_labels = torch.full((1, 10), -100, dtype=torch.long)
    dummy_attention = torch.ones(1, 10, dtype=torch.long)
    dummy_mask_pos = torch.zeros(1, 10, dtype=torch.long)
    dummy_mask_pos[0, 5] = 1  # One mask position
    
    # This will initialize the allowed_mask
    _, _ = loss_fn(dummy_logits, dummy_labels, dummy_attention, dummy_mask_pos)
    
    # Now check the allowed_mask
    allowed_mask = loss_fn.allowed_mask
    
    # Check specific problematic tokens
    problematic_tokens = [
        ('<|endofpiece|>', tokenizer.eos_token_id),
        ('Ġ', tokenizer.convert_tokens_to_ids('Ġ')),
        ('</s>', tokenizer.convert_tokens_to_ids('</s>')),
        ('<MASK>', tokenizer.convert_tokens_to_ids('<MASK>')),
        ('<TRACKS>', tokenizer.convert_tokens_to_ids('<TRACKS>'))
    ]
    
    print("CHECKING PROBLEMATIC TOKENS:")
    all_good = True
    
    for token_name, token_id in problematic_tokens:
        if token_id is not None and 0 <= token_id < vocab_size:
            is_allowed = allowed_mask[token_id].item()
            expected_allowed = token_name in ['<pad>', '<unk>']  # Only these should be allowed
            
            status = "✅" if is_allowed == expected_allowed else "❌"
            print(f"  {status} {token_name} (ID: {token_id}): allowed={is_allowed} (expected: {expected_allowed})")
            
            if is_allowed != expected_allowed:
                all_good = False
    
    # Check that music tokens are allowed
    music_allowed_count = sum(1 for mid in music_token_ids if allowed_mask[mid].item())
    print(f"\nMUSIC TOKENS: {music_allowed_count}/{len(music_token_ids)} allowed")
    
    if music_allowed_count < len(music_token_ids) * 0.9:  # At least 90% should be allowed
        print("❌ ERROR: Too many music tokens are being excluded!")
        all_good = False
    
    total_allowed = allowed_mask.sum().item()
    print(f"TOTAL ALLOWED: {total_allowed}/{vocab_size} ({total_allowed/vocab_size*100:.1f}%)")
    
    return all_good


def run_validation_tests(tokenizer):
    """Run all validation tests"""
    
    print("RUNNING VALIDATION TESTS FOR MUSIC MODEL FIXES\n")
    
    test1_passed = test_preprocessing_fix(tokenizer)
    test2_passed = test_loss_function_fix(tokenizer)
    
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    
    print(f"Preprocessing fix: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Loss function fix: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! The fixes should work.")
        print("\nNext steps:")
        print("1. Replace your prepare_dataset.py with the fixed version")
        print("2. Replace your loss_fn.py with the fixed version")
        print("3. Run training with the fixed scripts")
    else:
        print("\n❌ SOME TESTS FAILED! Please check the issues above.")
    
    return test1_passed and test2_passed


if __name__ == "__main__":
    print("Load your tokenizer and run: run_validation_tests(tokenizer)")