from prompt_utils import *
from output_utils import save_generated_piece
from filter import is_music_token  
from load_model_tokenizer import load_model_and_tokenizer
import torch

def clean_generated_text(text: str) -> str:
    """
    Post-process generated text to ensure only musical tokens remain in tracks
    """
    # Find the tracks section
    tracks_start = text.find("<TRACKS>")
    if tracks_start == -1:
        return text
        
    before_tracks = text[:tracks_start + len("<TRACKS>")]
    after_tracks = text[tracks_start + len("<TRACKS>"):]
    
    # Process the tracks section
    lines = []
    for line in after_tracks.split('<TRACKSEP>'):
        if ':' in line:
            inst, notes = line.split(':', 1)
            # Filter tokens
            clean_tokens = []
            for token in notes.split():
                if token != '<MASK>' and is_music_token(token):
                    clean_tokens.append(token)
            
            if clean_tokens:  # Only add if we have musical tokens
                lines.append(f"{inst.strip()}: {' '.join(clean_tokens)}")
        elif line.strip():  # Handle lines without colons
            clean_tokens = []
            for token in line.split():
                if token != '<MASK>' and is_music_token(token):
                    clean_tokens.append(token)
            if clean_tokens:
                lines.append(' '.join(clean_tokens))
    
    clean_after_tracks = ' <TRACKSEP> '.join(lines)
    return before_tracks + clean_after_tracks

def generate_text(
    prompt: str = "Say Hello World!",
    saved_txt_dir: str = "../../example_outputs/model_output/",
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    
    # 1. Get model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # 2. Prepare prompt
    out_dict = parse_user_instruction(prompt)
    prompt, metadata = build_structured_prompt(out_dict)
    
    print("Generated prompt:", prompt)
    
    # 3. Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"].clone()
    
    # 4. FIND ALL MASK POSITIONS
    mask_token_id = tokenizer.convert_tokens_to_ids("<MASK>")
    mask_positions = (input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]
    
    print(f"Found {len(mask_positions)} mask positions: {mask_positions.tolist()}")
    
    if len(mask_positions) == 0:
        print("No <MASK> tokens found in input!")
        return None
    
    # 5. PREDICT ALL MASKS SIMULTANEOUSLY (like during training)
    model.eval()
    with torch.no_grad():
        # Get predictions for the original masked input
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Replace ALL mask tokens at once using their predictions
        for mask_pos in mask_positions:
            mask_logits = logits[0, mask_pos]
            
            if do_sample:
                # Apply temperature
                scaled_logits = mask_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                
                # Apply top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs[indices_to_remove] = 0
                    if probs.sum() > 0:
                        probs = probs / probs.sum()
                    else:
                        probs = torch.softmax(mask_logits, dim=-1)  # Fallback
                
                predicted_id = torch.multinomial(probs, 1).item()
            else:
                predicted_id = torch.argmax(mask_logits).item()
            
            # Replace this mask token
            input_ids[0, mask_pos] = predicted_id
            predicted_token = tokenizer.decode(predicted_id)
            print(f"Position {mask_pos}: predicted '{predicted_token}' (ID: {predicted_id})")
    
    # 6. Decode the final result
    text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    print("\nGenerated text before cleaning:", repr(text))  # Use repr to see special chars
    
    # 7. Clean the generated text
    text = clean_generated_text(text)
    print("\nGenerated text after cleaning:", text)
    
    # 8. Save output
    file_path = save_generated_piece(
        metadata=metadata,
        generated_text=text,
        save_dir=saved_txt_dir
    )
    
    return file_path

