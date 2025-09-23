import json
from pathlib import Path

DATA_DIR = Path("../dataset/text")
OUT = Path("../dataset/train.txt")

'''
This script prepares the dataset for training a language model.
It converts structured JSON/TXT files into a single flat text file,
where each line represents one musical piece in a specific format.

template:
<|startofpiece|><NAME=11><BPM=120><DURATION_BEATS=3530.0>
<TRACKS>
Pipe Organ: Rest_w G#4.F5.C5_q ...
Piano: ...
<|endofpiece|>
'''

def flatten_tracks(tracks:dict) -> str:
    """
    Converts the dictionary of tracks into a single string.
    Each track is formatted as "TrackName: token1 token2..."
    and tracks are separated by a special <TRACKSEP> token.
    
    Args:
        tracks (dict): A dictionary where keys are instrument names and
                        values are lists of musical tokens.
    Returns:
        str: A single-line string representation of all tracks.
    """
    lines = []
    
    for name, tokens in tracks.items():
        # Format: "InstrumentName:token token token..."
        lines.append(f"{name}:" + " ".join(tokens))
        
    # Join all track strings with a separator token.
    return " <TRACKSEP> ".join(lines)


def make_example(data):
    """
    Constructs a single training example string from the parsed JSON data.
    This string includes metadata and the flattened track data, enclosed
    in special tokens for the language model.
    
    Args:
        data (dict): The dictionary loaded from a JSON/TXT file.
        
    Returns:
        str: The fully formatted training example string.
    """
    # Safely get metadata, defaulting to empty strings if not found.
    name = data.get("name", "")
    bpm = data.get("bpm", "")
    
    dur_beats = data.get("duration_beats", "")
    dur_minutes = data.get("duration_minutes", "")
    
    # Flatten the 'tracks' dictionary into a single string.
    tracks = flatten_tracks(data.get("tracks", {}))
    
    # Assemble the final string using an f-string for clarity.
    ex = (
        "<|startofpiece|>"
        f"<NAME={name}>"
        f"<BPM={bpm}>"
        f"<DURATION_BEATS={dur_beats}>"
        f"<DURATION_MINUTES={dur_minutes}>"
        "<TRACKS>"
        f"{tracks}"
        "<|endofpiece|>"
    )
    return ex


def creat_training_data():
    # Find all .txt files in the data directory.
    files = list(DATA_DIR.glob("*.txt"))

    with OUT.open("w", encoding="utf-8") as fo:
        for p in files:
            try:
                txt = p.read_text(encoding="utf-8")
                data = json.loads(txt)
                fo.write(make_example(data).strip() + "\n")
            except Exception as e: # Catch potential JSON parsing errors or other issues.
                print("skip", p, e)