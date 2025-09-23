import os
import re
import json


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

    # Remove header like "wow>190>190>1.00>"
    header_match = re.search(r'>[A-Za-z].*?:', s)
    if header_match:
        s = s[header_match.start()+1:].strip()

    tracks = {}

    # Regex: find "Instrument: notes..." up until the next "OtherInstrument:" or end
    pattern = re.compile(r'([A-Za-z][A-Za-z0-9_#\- ]{0,40}?):\s*([^:]+?)(?=\s+[A-Za-z][A-Za-z0-9_#\- ]{0,40}?:|$)')
    matches = pattern.findall(s)

    for inst, notes in matches:
        inst_key = inst.strip().lower()
        tokens = [tok for tok in notes.strip().split() if tok]
        tracks[inst_key] = tokens
        


    # 3. Build final JSON structure
    final_output  = {
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
        json.dump(final_output , f, indent=2)


    print(f"Saved piece to {filepath}")
    
    return filepath
