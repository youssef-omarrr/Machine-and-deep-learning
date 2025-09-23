import os
import re
import json
from filter import is_musical_token  # Import the musical token checker


def save_generated_piece(metadata: dict, generated_text: str,
                        save_dir: str = "../example_outputs/decoded_pieces/") -> str:
    """
    Parse generated_text and write JSON file:
    {
        "name": ...,
        "bpm": ...,
        "duration_beats": ...,
        "duration_minutes": ...,
        "tracks": { "piano": ["Rest_q", ...], "cello": [...], ... }
    }
    
    """
    os.makedirs(save_dir, exist_ok=True)

    # Clean string
    s = (generated_text or "").strip()
    s = re.sub(r'\s+', ' ', s)  # normalize spaces

    # Extract the generated part (after the prompt)
    prompt_end = s.find("<TRACKS>") + len("<TRACKS>")
    if prompt_end > 0:
        s = s[prompt_end:].strip()

    tracks = {}

    # Improved regex pattern to handle the actual generated content
    # Pattern to match "Instrument: notes..." where notes can contain various characters
    pattern = re.compile(r'([A-Za-z][A-Za-z0-9_#\- ]{0,40}?):\s*([^<]+?)(?=\s+[A-Za-z][A-Za-z0-9_#\- ]{0,40}?:|$| <TRACKSEP>)')
    matches = pattern.findall(s)

    print(f"Found {len(matches)} instrument matches")  # Debug info

    for inst, notes in matches:
        inst_key = inst.strip().lower()
        # Clean up the notes - remove any trailing <TRACKSEP> or other artifacts
        notes_clean = notes.strip()
        
        # Remove any <TRACKSEP> that might be at the end
        if notes_clean.endswith('<TRACKSEP>'):
            notes_clean = notes_clean[:-len('<TRACKSEP>')].strip()
        
        # Split into tokens, filtering out empty strings
        raw_tokens = [tok for tok in notes_clean.split() if tok and tok != '<TRACKSEP>']
        
        # FILTER TOKENS: Only keep musical tokens and remove <MASK>
        filtered_tokens = []
        for token in raw_tokens:
            # Skip <MASK> tokens completely
            if token == '<MASK>':
                continue
            # Only keep musical tokens
            if is_musical_token(token):
                filtered_tokens.append(token)
            else:
                print(f"Filtered out non-musical token: '{token}'")  # Debug info
        
        # Only add if we have actual musical notes
        if filtered_tokens:
            tracks[inst_key] = filtered_tokens
            print(f"Added {inst_key}: {filtered_tokens}")  # Debug info

    # If no tracks were found with the main pattern, try a more flexible approach
    if not tracks:
        print("No tracks found with main pattern, trying alternative parsing...")
        
        # Alternative approach: split by <TRACKSEP> and parse each instrument
        if '<TRACKSEP>' in s:
            parts = s.split('<TRACKSEP>')
            for part in parts:
                part = part.strip()
                if ':' in part:
                    inst_part, notes_part = part.split(':', 1)
                    inst_key = inst_part.strip().lower()
                    raw_tokens = [tok for tok in notes_part.strip().split() if tok]
                    
                    # Apply the same filtering here
                    filtered_tokens = []
                    for token in raw_tokens:
                        if token == '<MASK>':
                            continue
                        if is_musical_token(token):
                            filtered_tokens.append(token)
                    
                    if filtered_tokens:
                        tracks[inst_key] = filtered_tokens
                        print(f"Alternative parsing added {inst_key}: {filtered_tokens}")

    # 3. Build final JSON structure
    final_output = {
        "name": metadata["name"],
        "bpm": metadata["bpm"],
        "duration_beats": metadata["duration_beats"],
        "duration_minutes": metadata["duration_minutes"],
        "tracks": tracks,
    }

    # auto-number files
    existing_files = [f for f in os.listdir(save_dir) if f.startswith("piece_") and f.endswith(".txt")]
    if existing_files:
        nums = [int(f.split("_")[1].split(".")[0]) for f in existing_files]
        next_num = max(nums) + 1
    else:
        next_num = 1
    
    # Save the final output
    filepath = os.path.join(save_dir, f"piece_{next_num}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"Saved piece to {filepath} with {len(tracks)} tracks")
    
    return filepath