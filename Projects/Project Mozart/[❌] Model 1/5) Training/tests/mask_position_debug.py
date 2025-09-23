"""
Script to debug mask positions and understand what the model should be predicting
"""
import torch

def debug_mask_positions(input_text, target_text, tokenizer):
    """Debug mask positions to understand the training setup"""
    
    print("=" * 80)
    print("MASK POSITION DEBUGGING")
    print("=" * 80)
    
    print(f"Input text: {input_text}")
    print(f"Target text: {target_text}")
    print()
    
    # Tokenize both
    input_tokens = tokenizer.tokenize(input_text)
    target_tokens = tokenizer.tokenize(target_text)
    
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    
    print("INPUT TOKENIZATION:")
    for i, (token, tid) in enumerate(zip(input_tokens, input_ids)):
        print(f"  {i:2d}: '{token}' -> {tid}")
    
    print("\nTARGET TOKENIZATION:")
    for i, (token, tid) in enumerate(zip(target_tokens, target_ids)):
        print(f"  {i:2d}: '{token}' -> {tid}")
    
    # Find mask positions
    mask_token_id = tokenizer.convert_tokens_to_ids('<MASK>')
    mask_positions = [i for i, tid in enumerate(input_ids) if tid == mask_token_id]
    
    print(f"\nMASK TOKEN ID: {mask_token_id}")
    print(f"MASK POSITIONS: {mask_positions}")
    
    # Show what should be predicted at each mask position
    print("\nEXPECTED PREDICTIONS AT MASK POSITIONS:")
    min_len = min(len(input_ids), len(target_ids))
    
    for pos in mask_positions:
        if pos < min_len:
            target_token_id = target_ids[pos]
            target_token = tokenizer.convert_ids_to_tokens([target_token_id])[0]
            print(f"  Position {pos}: Should predict '{target_token}' (ID: {target_token_id})")
        else:
            print(f"  Position {pos}: OUT OF BOUNDS in target!")
    
    # Analyze the target sequence to see what types of tokens should be predicted
    from filter import is_music_token
    
    music_tokens_in_target = []
    special_tokens_in_target = []
    other_tokens_in_target = []
    
    for token, tid in zip(target_tokens, target_ids):
        if is_music_token(token):
            music_tokens_in_target.append((token, tid))
        elif tid in tokenizer.all_special_ids:
            special_tokens_in_target.append((token, tid))
        else:
            other_tokens_in_target.append((token, tid))
    
    print(f"\nTARGET SEQUENCE ANALYSIS:")
    print(f"  Music tokens: {len(music_tokens_in_target)}")
    print(f"  Special tokens: {len(special_tokens_in_target)}")
    print(f"  Other tokens: {len(other_tokens_in_target)}")
    
    # Show examples of each type
    print(f"\nMUSIC TOKENS IN TARGET (first 10):")
    for i, (token, tid) in enumerate(music_tokens_in_target[:10]):
        print(f"  {token} -> {tid}")
    
    print(f"\nSPECIAL TOKENS IN TARGET:")
    for token, tid in special_tokens_in_target:
        print(f"  {token} -> {tid}")
    
    print(f"\nOTHER TOKENS IN TARGET (first 10):")
    for i, (token, tid) in enumerate(other_tokens_in_target[:10]):
        print(f"  '{token}' -> {tid}")
    
    return {
        'mask_positions': mask_positions,
        'input_ids': input_ids,
        'target_ids': target_ids,
        'music_token_count': len(music_tokens_in_target),
        'special_token_count': len(special_tokens_in_target),
        'other_token_count': len(other_tokens_in_target)
    }


def simulate_training_batch(input_text, target_text, tokenizer, max_length=1024):
    """Simulate how the training batch would be created"""
    
    print("\n" + "=" * 80)
    print("TRAINING BATCH SIMULATION")
    print("=" * 80)
    
    # Pre-tokenize (like in prepare_dataset.py)
    def pre_tokenize_musical_text(text):
        from filter import is_music_token
        tokens = text.split()
        filtered = [tok for tok in tokens if is_music_token(tok) or tok.startswith("<")]
        return " ".join(filtered)
    
    input_filtered = pre_tokenize_musical_text(input_text)
    target_filtered = pre_tokenize_musical_text(target_text)
    
    print(f"INPUT AFTER FILTERING: {input_filtered}")
    print(f"TARGET AFTER FILTERING: {target_filtered}")
    
    # Tokenize
    enc_input = tokenizer(
        input_filtered,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    enc_target = tokenizer(
        target_filtered,
        truncation=True,
        max_length=max_length,
        padding="max_length", 
        return_tensors="pt"
    )
    
    input_ids = enc_input["input_ids"][0]  # Remove batch dimension
    target_ids = enc_target["input_ids"][0]
    attention_mask = enc_input["attention_mask"][0]
    
    # Create labels and mask_positions
    mask_token_id = tokenizer.convert_tokens_to_ids('<MASK>')
    labels = torch.full_like(input_ids, -100)
    mask_positions = torch.zeros_like(input_ids)
    
    for i, tok in enumerate(input_ids):
        if tok == mask_token_id:
            labels[i] = target_ids[i]
            mask_positions[i] = 1
    
    print(f"\nBATCH STATISTICS:")
    print(f"  Input length: {(input_ids != tokenizer.pad_token_id).sum().item()}")
    print(f"  Target length: {(target_ids != tokenizer.pad_token_id).sum().item()}")
    print(f"  Attention mask sum: {attention_mask.sum().item()}")
    print(f"  Mask positions: {mask_positions.sum().item()}")
    print(f"  Label positions (not -100): {(labels != -100).sum().item()}")
    
    # Show mask positions and their expected labels
    mask_indices = torch.where(mask_positions == 1)[0]
    print(f"\nMASK POSITIONS AND EXPECTED LABELS:")
    for idx in mask_indices[:10]:  # Show first 10
        input_token = tokenizer.convert_ids_to_tokens([input_ids[idx].item()])[0]
        expected_token = tokenizer.convert_ids_to_tokens([labels[idx].item()])[0] if labels[idx] != -100 else "N/A"
        print(f"  Position {idx}: '{input_token}' -> should predict '{expected_token}'")
    
    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'labels': labels,
        'mask_positions': mask_positions,
        'attention_mask': attention_mask
    }


# Example usage
if __name__ == "__main__":
    sample_input = "<|startofpiece|><NAME=Symphony No. 41 in C, K.551, Jupiter><BPM=150.0><DURATION_BEATS=49962.0><DURATION_MINUTES=46.26><TRACKS><MASK> <TRACKSEP> StringInstrument_3: <MASK><|endofpiece|>"
    
    sample_target = "<|startofpiece|><NAME=Symphony No. 41 in C, K.551, Jupiter><BPM=150.0><DURATION_BEATS=49962.0><DURATION_MINUTES=46.26><TRACKS>G4_1.5 B-4_e E-4_1.5 C5_e A4_e E-5_e C5_s Rest_s A4_s Rest_s F#4_s Rest_s G4_s Rest_s A4_s Rest_s B-4_s Rest_s G4_s Rest_0.75 G4_s Rest_1.75 E4_s Rest_0.75 Rest_q C#4_s Rest_1.75 A3_s Rest_0.75 D5_e F5_e <TRACKSEP> StringInstrument_3: B4_s Rest_s D5_s Rest_s G#4_s Rest_s B4_s Rest_s E4_s Rest_s E5_s Rest_s C5_s Rest_0.75 A4_s Rest_0.75 G4_s<|endofpiece|>"
    
    print("Load your tokenizer and run:")
    print("debug_mask_positions(sample_input, sample_target, tokenizer)")
    print("simulate_training_batch(sample_input, sample_target, tokenizer)")