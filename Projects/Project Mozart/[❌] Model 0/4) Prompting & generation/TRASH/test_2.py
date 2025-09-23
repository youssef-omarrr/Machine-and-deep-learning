"""
Working music generator that fixes the CUDA issues.
This version uses the correct parameters and handles attention masks properly.
"""

import re
import os
import time
import random
import json
import torch
from typing import Optional, Tuple, Dict, Any, List

# Import your existing functions
from prompt_utils import (
    load_model_and_tokenizer, 
    OUTPUT_GENERATIONS_DIR,
    DEFAULT_BPM
)

def parse_user_instruction(text: str) -> Dict[str, Any]:
    """Parse user instructions into structured parameters."""
    t = text.lower()
    
    # BPM parsing
    bpm_match = re.search(r"(\d{2,3})(?:\s*)(?:bpm\b)?", t)
    bpm = None
    if bpm_match:
        bpm = int(bpm_match.group(1))

    # Duration parsing
    duration_minutes = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:minutes|minute|mins|min)\b", t)
    if m:
        duration_minutes = float(m.group(1))
    else:
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds|second|secs|sec)\b", t)
        if m2:
            duration_minutes = float(m2.group(1)) / 60.0

    # Instrument detection
    instruments = []
    basic_instruments = ["piano", "violin", "cello", "flute", "drums", "trumpet", "guitar"]
    
    for inst in basic_instruments:
        if inst in t:
            instruments.append(inst)
    
    # Special cases
    if "many instruments" in t or "orchestra" in t:
        instruments = ["piano", "violin", "flute"]
    elif "only" in t:
        for inst in basic_instruments:
            if inst in t:
                instruments = [inst]
                break

    return {
        "bpm": bpm,
        "duration_minutes": duration_minutes,
        "instruments": instruments if instruments else None,
        "raw_text": text,
    }

def build_music_prompt(
    tokenizer,
    *,
    bpm: Optional[float] = None,
    duration_minutes: Optional[float] = None,
    instruments: Optional[List[str]] = None,
    name: Optional[str] = None,
) -> str:
    """Build a music generation prompt in the correct format."""
    
    if name is None:
        name = f"Generated_{random.randint(1000,9999)}"
    
    # Set defaults
    if bpm is None:
        bpm = DEFAULT_BPM
    if duration_minutes is None:
        duration_minutes = 1.0
    
    # Calculate duration in beats
    duration_beats = int(round(duration_minutes * bpm))
    
    # Map instruments to your dataset format
    instrument_mapping = {
        "piano": "Piano",
        "violin": "Violin", 
        "cello": "Cello",
        "flute": "Flutes",
        "drums": "Drums", 
        "trumpet": "Trumpet",
        "guitar": "Guitar"
    }
    
    # Build tracks section
    if instruments:
        track_lines = []
        for inst in instruments[:3]:  # Limit to 3 instruments
            mapped_name = instrument_mapping.get(inst.lower(), "Piano")
            track_lines.append(f"{mapped_name}: Rest_q Rest_q")
        tracks_section = " <TRACKSEP> ".join(track_lines)
    else:
        tracks_section = "Piano: Rest_q Rest_q"
    
    # Build complete prompt
    prompt = (
        f"<|startofpiece|>"
        f"<NAME={name}>"
        f"<BPM={int(bpm)}>"
        f"<DURATION_BEATS={duration_beats}>"
        f"<DURATION_MINUTES={duration_minutes:.1f}>"
        f"<TRACKS>{tracks_section}"
    )
    
    return prompt

def generate_with_proper_attention(
    model,
    tokenizer, 
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
) -> str:
    """Generate text with proper attention mask handling."""
    
    # Tokenize with proper attention mask
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=False
    )
    
    # Move to device
    device = next(model.parameters()).device
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    # Generate with proper parameters
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,  # This fixes the attention mask warning
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
        )
    
    # Decode result
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Add end token if missing
    if "<|endofpiece|>" not in generated_text:
        generated_text = generated_text + " <|endofpiece|>"
    
    return generated_text

def generate_music_piece(
    user_instruction: str,
    *,
    saved_model_dir: str = "../MODELS/Project_Mozart_gpt2-medium",
    base_model_name: str = "../MODELS/gpt2-medium-local/", 
    max_new_tokens: int = 300,
    temperature: float = 0.8,
    filename_prefix: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], str]:
    """
    Main function to generate music from natural language instructions.
    
    Examples:
        generate_music_piece("create a 1 minute piano piece at 120 BPM")
        generate_music_piece("make a short violin and cello duet")
        generate_music_piece("generate a piece with many instruments around 90 BPM")
    """
    
    print(f"Processing: {user_instruction}")
    
    # Parse the instruction
    parsed = parse_user_instruction(user_instruction)
    print(f"Parsed: {parsed}")
    
    # Load model and tokenizer
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        saved_model_dir=saved_model_dir,
        base_model_name=base_model_name,
    )
    
    # Build prompt
    prompt = build_music_prompt(
        tokenizer,
        bpm=parsed.get("bpm"),
        duration_minutes=parsed.get("duration_minutes"),
        instruments=parsed.get("instruments"),
    )
    
    print(f"Prompt: {prompt}")
    
    # Generate the piece
    print("Generating music...")
    generated_text = generate_with_proper_attention(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    
    # Save the result
    timestamp = int(time.time())
    prefix = filename_prefix or "music_gen"
    filename = f"{prefix}_{timestamp}.txt"
    filepath = os.path.join(OUTPUT_GENERATIONS_DIR, filename)
    
    os.makedirs(OUTPUT_GENERATIONS_DIR, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(generated_text)
    
    # Extract metadata for return
    metadata = {
        "instruction": user_instruction,
        "parsed_params": parsed,
        "prompt_used": prompt,
        "generated_length": len(generated_text),
        "file_path": filepath
    }
    
    print(f"✓ Generated {len(generated_text)} characters")
    print(f"✓ Saved to: {filepath}")
    
    return generated_text, metadata, filepath

def test_working_generator():
    """Test the working generator with simple prompts."""
    
    test_prompts = [
        "generate a short piano piece",
        "create a 30 second piece at 120 BPM", 
        "make a violin piece",
        "generate a piece with piano and flute"
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Testing: {prompt}")
        print('='*60)
        
        try:
            text, metadata, filepath = generate_music_piece(
                prompt,
                max_new_tokens=150  # Start small for testing
            )
            
            print(f"✓ SUCCESS!")
            print(f"Generated preview:")
            print(text[:200] + "..." if len(text) > 200 else text)
            
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_working_generator()