############################################################
# TEST SCRIPT FOR ENHANCED LOSS FUNCTION
############################################################
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from loss_fn import MusicTokenEnforcementLoss
from filter import get_unique_tokens
from load_model_tokenizer import load_model_and_tokenizer

def test_enhanced_loss_function():
    """Comprehensive test of the enhanced loss function"""
    
    print("🧪 TESTING ENHANCED LOSS FUNCTION")
    print("=" * 80)
    
    _, tokenizer = load_model_and_tokenizer()
    
    # Get music token IDs
    music_notes = get_unique_tokens()
    music_token_ids = []
    for note in music_notes:
        tid = tokenizer.convert_tokens_to_ids(note)
        if tid is not None and tid >= 0:
            music_token_ids.append(int(tid))
    
    print(f"✅ Found {len(music_token_ids)} music token IDs")
    
    # Initialize loss function
    loss_fn = MusicTokenEnforcementLoss(tokenizer, music_token_ids, non_music_penalty=10000.0)
    print("✅ Initialized loss function")
    
    # Test cases
    test_cases = [
        {
            "name": "Normal music sequence with spaces",
            "input": "<|startofpiece|><NAME=Test><TRACKS><MASK> <|endofpiece|>",
            "target": "<|startofpiece|><NAME=Test><TRACKS>G4_1.5 B-4_e E-4_1.5<|endofpiece|>",
            "expected": "Should allow music notes and spaces between them"
        },
        {
            "name": "Track separator case", 
            "input": "<|startofpiece|><TRACKS><MASK> <TRACKSEP> StringInstrument_3: <MASK><|endofpiece|>",
            "target": "<|startofpiece|><TRACKS>G4_1.5 <TRACKSEP> StringInstrument_3: B4_s<|endofpiece|>",
            "expected": "Should handle track separators correctly"
        },
        {
            "name": "Metadata section",
            "input": "<|startofpiece|><NAME=<MASK>><BPM=<MASK>><TRACKS>test<|endofpiece|>",
            "target": "<|startofpiece|><NAME=Test Song><BPM=150.0><TRACKS>test<|endofpiece|>", 
            "expected": "Should handle metadata correctly"
        }
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📋 TEST CASE {i+1}: {test_case['name']}")
        print("-" * 60)
        print(f"Input:  {test_case['input']}")
        print(f"Target: {test_case['target']}")
        print(f"Expected: {test_case['expected']}")
        
        try:
            # Tokenize
            input_encoding = tokenizer(test_case['input'], return_tensors='pt', padding=True)
            target_encoding = tokenizer(test_case['target'], return_tensors='pt', padding=True)
            
            input_ids = input_encoding['input_ids'].to(device)
            attention_mask = input_encoding['attention_mask'].to(device)
            target_ids = target_encoding['input_ids'].to(device)
            
            print(f"Input tokens: {input_ids.shape}")
            print(f"Target tokens: {target_ids.shape}")
            
            # Create labels (only predict where there are <MASK> tokens)
            mask_token_id = tokenizer.convert_tokens_to_ids('<MASK>')
            labels = torch.full_like(input_ids, -100)
            mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)
            
            for b in range(input_ids.shape[0]):
                for pos in range(input_ids.shape[1]):
                    if input_ids[b, pos] == mask_token_id:
                        if pos < target_ids.shape[1]:
                            labels[b, pos] = target_ids[b, pos]
                            mask_positions[b, pos] = True
            
            num_masks = mask_positions.sum().item()
            print(f"Found {num_masks} mask positions")
            
            if num_masks == 0:
                print("⚠️  WARNING: No mask positions found!")
                continue
            
            # Create dummy logits (simulate model output)
            vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab())
            batch_size, seq_len = input_ids.shape
            
            # Create logits that initially prefer </s> (the problematic behavior)
            logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
            
            # Make </s> token have high probability initially (simulate the problem)
            eos_id = tokenizer.eos_token_id
            if eos_id is not None:
                logits[:, :, eos_id] += 5.0  # Bias toward </s>
            
            # Test the loss function
            print("\n🔍 TESTING LOSS FUNCTION:")
            
            loss, components = loss_fn(
                logits=logits,
                labels=labels,
                attention_mask=attention_mask,
                mask_positions=mask_positions.long()
            )
            
            print(f"✅ Loss computed successfully: {loss.item():.4f}")
            print(f"   - CE Loss: {components['ce_loss']:.4f}")
            print(f"   - Penalty Loss: {components['non_music_penalty']:.4f}")
            print(f"   - Non-music predictions: {components['non_music_predictions']}")
            
            # Test predictions after masking
            print("\n🎯 ANALYZING PREDICTIONS:")
            with torch.no_grad():
                # Get top predictions at mask positions
                mask_logits = logits[mask_positions]
                if mask_logits.numel() > 0:
                    top_vals, top_ids = torch.topk(mask_logits, k=5, dim=-1)
                    
                    print("Top 5 predictions at mask positions:")
                    for mask_idx in range(min(3, mask_logits.shape[0])):  # Show first 3 masks
                        print(f"  Mask position {mask_idx}:")
                        for k in range(5):
                            token_id = top_ids[mask_idx, k].item()
                            token = tokenizer.convert_ids_to_tokens([token_id])[0]
                            prob = F.softmax(top_vals[mask_idx], dim=0)[k].item()
                            
                            # Check token type
                            is_music = token_id in music_token_ids
                            is_space = token_id == loss_fn.space_token_id
                            is_eos = token_id == tokenizer.eos_token_id
                            is_special = token_id in [info['id'] for info in loss_fn.special_tokens.values() if info['id'] is not None]
                            
                            token_type = "MUSIC" if is_music else ("SPACE" if is_space else ("EOS" if is_eos else ("SPECIAL" if is_special else "OTHER")))
                            
                            print(f"    {k+1}. '{token}' ({token_type}) - prob: {prob:.3f}")
            
            # Test space token handling
            if loss_fn.space_token_id is not None:
                print(f"\n🔤 SPACE TOKEN ANALYSIS:")
                print(f"Space token ID: {loss_fn.space_token_id}")
                space_token = tokenizer.convert_ids_to_tokens([loss_fn.space_token_id])[0]
                print(f"Space token representation: '{space_token}'")
                
                # Test space allowance in different contexts
                for pos in range(min(5, seq_len)):
                    should_allow = loss_fn._should_allow_space(input_ids, pos)
                    context = loss_fn._get_context_type(input_ids, pos)
                    pos_token = tokenizer.convert_ids_to_tokens([input_ids[0, pos].item()])[0]
                    print(f"  Position {pos} ('{pos_token}'): context='{context}', allow_space={should_allow}")
            
            print("✅ Test case completed successfully")
            
        except Exception as e:
            print(f"❌ ERROR in test case: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("🎉 LOSS FUNCTION TESTING COMPLETED")
    
    # Final recommendations
    print("\n📋 RECOMMENDATIONS BEFORE TRAINING:")
    print("1. ✅ If all test cases passed, the loss function is ready")
    print("2. ⚠️  If space tokens aren't detected correctly, check tokenizer")
    print("3. ⚠️  If context detection seems wrong, review the context logic")
    print("4. 🔧 If penalties aren't strong enough, increase non_music_penalty")
    print("5. 📊 Monitor the prediction breakdowns during initial training")
    
    return True

def test_tokenizer_space_detection():
    """Separate test for space token detection"""
    print("\n🔍 TESTING SPACE TOKEN DETECTION")
    print("-" * 40)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("../../MODELS/Project_Mozart_bart-small")
        
        # Test various space representations
        test_strings = [
            "G4_1.5 B-4_e",
            "G4_1.5  B-4_e",  # Double space
            " G4_1.5",  # Leading space
            "G4_1.5 ",  # Trailing space
        ]
        
        for test_str in test_strings:
            tokens = tokenizer.tokenize(test_str)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            print(f"'{test_str}' -> {tokens} -> {token_ids}")
        
        # Direct space token tests
        space_variants = [' ', 'Ġ', 'Ä', 'Ä ', '▁']
        for variant in space_variants:
            token_id = tokenizer.convert_tokens_to_ids(variant)
            print(f"Space variant '{variant}': ID = {token_id}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_tokenizer_space_detection()
    test_enhanced_loss_function()