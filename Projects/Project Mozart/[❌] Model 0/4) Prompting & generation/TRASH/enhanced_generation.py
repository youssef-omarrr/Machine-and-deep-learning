"""
Enhanced generation wrapper that uses the existing prompt_utils.py
This file adds better prompt parsing while keeping your core engine intact.
"""

import re
import os
import time
import json
import torch
from typing import Optional, Tuple, Dict, Any, List

# Import your existing functions - THIS IS THE KEY LINE
from prompt_utils import (
    load_model_and_tokenizer, 
    generate_piece_strict, 
    OUTPUT_GENERATIONS_DIR,
    DEFAULT_BPM,
    build_structured_prompt  # We'll enhance this but keep original as backup
)

def _parse_user_instruction_enhanced(text: str) -> Dict[str, Any]:
    """
    Enhanced parsing of user instructions with better handling of various prompt formats.
    This REPLACES the basic parsing in prompt_utils with more sophisticated logic.
    """
    t = text.lower()
    
    # BPM parsing with better context awareness
    bpm_match = re.search(r"(\d{2,3})(?:\s*)(?:bpm\b)?", t)
    bpm = None
    strict_bpm = False
    if bpm_match:
        span = bpm_match.span()
        window = t[max(0, span[0]-15): span[1]+15]
        bpm = int(bpm_match.group(1))
        # Check for approximation keywords
        strict_bpm = not bool(re.search(r"\baround\b|\bapprox\b|\bapproximately\b|\b~\b|\babout\b", window))

    # Duration parsing (minutes and seconds)
    duration_minutes = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:minutes|minute|mins|min)\b", t)
    if m:
        duration_minutes = float(m.group(1))
    else:
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds|second|secs|sec)\b", t)
        if m2:
            duration_minutes = float(m2.group(1)) / 60.0

    # Enhanced instrument detection
    instruments = []
    instrument_patterns = {
        "piano": r"\bpiano\b",
        "violin": r"\bviolin\b",
        "cello": r"\bcello\b",
        "flute": r"\bflute\b|\bflutes\b",
        "clarinet": r"\bclarinet\b",
        "guitar": r"\bguitar\b",
        "drums": r"\bdrum\b|\bdrums\b|\bpercussion\b",
        "saxophone": r"\bsaxophone\b|\bsax\b",
        "trumpet": r"\btrumpet\b",
        "organ": r"\borgan\b",
        "bass": r"\bbass\b",
        "harp": r"\bharp\b",
        "synth": r"\bsynth\b|\bsynthesizer\b",
        "strings": r"\bstrings\b|\bstring section\b",
        "brass": r"\bbrass\b|\bbrass section\b",
        "woodwinds": r"\bwoodwinds\b|\bwoodwind\b",
        "orchestra": r"\borchestra\b|\bfull orchestra\b",
    }
    
    for instrument, pattern in instrument_patterns.items():
        if re.search(pattern, t):
            instruments.append(instrument)
    
    # Special cases
    if re.search(r"\bmany instruments\b|\bmultiple instruments\b|\borchestral\b", t):
        instruments = ["piano", "violin", "cello", "flute", "trumpet", "drums"]
    elif re.search(r"\bsolo\b|\bonly\b.*\b(piano|violin|guitar|flute)\b", t):
        solo_match = re.search(r"\b(piano|violin|guitar|flute|cello|trumpet|drums)\b.*\bonly\b|\bonly\b.*\b(piano|violin|guitar|flute|cello|trumpet|drums)\b", t)
        if solo_match:
            solo_instrument = solo_match.group(1) or solo_match.group(2)
            instruments = [solo_instrument]

    return {
        "bpm": bpm,
        "strict_bpm": strict_bpm,
        "duration_minutes": duration_minutes,
        "instruments": instruments if instruments else None,
        "raw_text": text,
    }

def build_enhanced_prompt(tokenizer, **kwargs):
    """
    Enhanced version of your build_structured_prompt with better instrument handling.
    Uses your existing function but adds preprocessing.
    """
    instruments = kwargs.get('instruments')
    
    # Map common names to your dataset format
    if instruments:
        instrument_mapping = {
            "piano": "Piano",
            "violin": "Violin", 
            "cello": "Cello",
            "flute": "Flutes",
            "drums": "Drums",
            "trumpet": "Trumpet",
            "organ": "Pipe Organ",
            "guitar": "Guitar",
            "saxophone": "Saxophone",
            "bass": "Bass",
        }
        
        mapped_instruments = []
        for inst in instruments:
            mapped = instrument_mapping.get(inst.lower(), inst.capitalize())
            mapped_instruments.append(mapped)
        kwargs['instruments'] = mapped_instruments
    
    # Use your existing function with enhancements
    return build_structured_prompt(tokenizer, **kwargs)

def generate_music_from_prompt(
    user_instruction: str,
    *,
    saved_model_dir: str = "../MODELS/Project_Mozart_gpt2-medium",
    base_model_name: str = "../MODELS/gpt2-medium-local/",
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.05,
    chunk_max_new_tokens: int = 64,
    enforce_user_bpm: bool = True,
    filename_prefix: Optional[str] = None,
    output_format: str = "both",
) -> Tuple[str, Dict[str, Any], str]:
    """
    Enhanced music generation that uses your existing engine with better prompt handling.
    
    This is your NEW main function to use instead of generate_from_text.
    """
    print(f"Processing instruction: {user_instruction}")
    
    # Use enhanced parsing instead of basic parsing
    parsed = _parse_user_instruction_enhanced(user_instruction)
    
    bpm = parsed.get("bpm") or DEFAULT_BPM
    duration_minutes = parsed.get("duration_minutes")
    instruments = parsed.get("instruments")
    
    # Set reasonable defaults
    if duration_minutes is None:
        if "long" in user_instruction.lower():
            duration_minutes = 3.0
        elif "short" in user_instruction.lower():
            duration_minutes = 0.5
        else:
            duration_minutes = 1.0
    
    duration_beats = None
    if duration_minutes is not None and bpm is not None:
        duration_beats = int(round(duration_minutes * bpm))

    # Load model using YOUR existing function
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    
    model, tokenizer = load_model_and_tokenizer(
        saved_model_dir=saved_model_dir,
        base_model_name=base_model_name,
        device=device,
        dtype=dtype,
    )

    # Build prompt using enhanced version of YOUR function
    prompt, meta = build_enhanced_prompt(
        tokenizer,
        bpm=bpm,
        duration_beats=duration_beats,
        duration_minutes=duration_minutes,
        instruments=instruments,
    )

    print(f"Built prompt: {prompt[:100]}...")
    
    target_beats = duration_beats or int(round(duration_minutes * bpm))

    # Generate using YOUR existing function
    piece_text = generate_piece_strict(
        model,
        tokenizer,
        prompt=prompt,
        target_beats=float(target_beats),
        chunk_max_new_tokens=chunk_max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        device=device,
    )

    # Post-process (using your existing logic)
    if parsed.get("bpm") is not None and enforce_user_bpm:
        user_bpm = int(parsed.get("bpm"))
        piece_text = re.sub(r"<BPM=[0-9.]*>", f"<BPM={user_bpm}>", piece_text)

    if duration_minutes is not None and bpm is not None:
        computed_beats = int(round(duration_minutes * bpm))
        piece_text = re.sub(r"<DURATION_BEATS=[0-9.]*>", f"<DURATION_BEATS={computed_beats}>", piece_text)
        piece_text = re.sub(r"<DURATION_MINUTES=[0-9.]*>", f"<DURATION_MINUTES={duration_minutes:.2f}>", piece_text)

    # Parse back to structured format
    metadata = parse_generated_piece(piece_text)
    
    # Save files
    ts = int(time.time())
    prefix = filename_prefix or "enhanced"
    
    # Save text format using your existing directory
    filename_txt = f"{prefix}_{ts}.txt"
    filepath_txt = os.path.join(OUTPUT_GENERATIONS_DIR, filename_txt)
    with open(filepath_txt, "w", encoding="utf-8") as f:
        f.write(piece_text)
    
    # Optionally save JSON format too
    if output_format in ["json", "both"]:
        filename_json = f"{prefix}_{ts}.json"
        filepath_json = os.path.join(OUTPUT_GENERATIONS_DIR, filename_json)
        with open(filepath_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    print(f"✓ Generated and saved to: {filepath_txt}")
    print(f"✓ BPM: {metadata.get('bpm')}, Duration: {metadata.get('duration_minutes')}min")
    print(f"✓ Instruments: {list(metadata.get('tracks', {}).keys())}")
    
    return piece_text, metadata, filepath_txt

def parse_generated_piece(piece_text: str) -> Dict[str, Any]:
    """Convert generated text back to JSON structure like your training data."""
    metadata = {}
    
    # Extract metadata using regex
    name_match = re.search(r"<NAME=([^>]+)>", piece_text)
    if name_match:
        metadata["name"] = name_match.group(1)
    
    bpm_match = re.search(r"<BPM=([0-9.]+)>", piece_text)
    if bpm_match:
        metadata["bpm"] = float(bpm_match.group(1))
    
    duration_beats_match = re.search(r"<DURATION_BEATS=([0-9.]+)>", piece_text)
    if duration_beats_match:
        metadata["duration_beats"] = float(duration_beats_match.group(1))
    
    duration_minutes_match = re.search(r"<DURATION_MINUTES=([0-9.]+)>", piece_text)
    if duration_minutes_match:
        metadata["duration_minutes"] = float(duration_minutes_match.group(1))
    
    # Extract tracks
    tracks_match = re.search(r"<TRACKS>(.*?)(?:<\|endofpiece\|>|$)", piece_text, re.DOTALL)
    if tracks_match:
        tracks_text = tracks_match.group(1).strip()
        tracks = {}
        
        # Split by track separator
        track_sections = tracks_text.split("<TRACKSEP>")
        
        for section in track_sections:
            section = section.strip()
            if ":" in section:
                parts = section.split(":", 1)
                instrument = parts[0].strip()
                tokens = parts[1].strip().split()
                tracks[instrument] = [token for token in tokens if token.strip()]
        
        metadata["tracks"] = tracks
    
    return metadata

# Simple test function
def test_enhanced_generation():
    """Test the enhanced generation with various prompts."""
    test_prompts = [
        "generate a piece that is around 1 minute long",
        "generate a piece that has 120 bpm", 
        "generate a piece around 60 bpm that is 2 minutes long and uses piano only",
        "create a short piece with violin and cello",
    ]
    
    print("Testing enhanced generation...")
    for prompt in test_prompts:
        print(f"\n=== Testing: {prompt} ===")
        try:
            text, metadata, filepath = generate_music_from_prompt(prompt)
            print(f"✓ Success! Saved to: {filepath}")
        except Exception as e:
            print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_enhanced_generation()