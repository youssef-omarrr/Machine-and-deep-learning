import json
from pathlib import Path

DATA_DIR = Path("../../dataset/text")
OUT = Path("../../dataset/FULL.txt")

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

MAX_TOKENS_PER_LINE = 50  # maximum number of note tokens per line

def split_tokens(tokens, chunk_size=MAX_TOKENS_PER_LINE):
    
    """Split a list of tokens into chunks of size <= chunk_size."""
    for i in range(0, len(tokens), chunk_size):
        yield tokens[i:i + chunk_size]
        

def flatten_tracks(tracks:dict) -> str:
    """
    Converts the dictionary of tracks into a list of track strings.
    Each track is formatted as "TrackName: token1 token2..."
    Tracks longer than MAX_TOKENS_PER_LINE are split into multiple lines.

    Args:
        tracks (dict): instrument -> list of tokens

    Returns:
        list[str]: A list of flattened track strings (chunks).
    """
    # First, split each instrument into chunks
    instrument_chunks = []
    
    for name, tokens in tracks.items():
        for chunk in split_tokens(tokens):
            token_str = " ".join(chunk)
            instrument_chunks.append(f"{name}: {token_str}")

    # Now join everything with <TRACKSEP>
    # But if this total string is still > MAX_TOKENS_PER_LINE, we need to split again
    joined = " <TRACKSEP> ".join(instrument_chunks)

    # Final safeguard split
    final_chunks = []
    words = joined.split()
    for i in range(0, len(words), MAX_TOKENS_PER_LINE):
        final_chunks.append(" ".join(words[i:i + MAX_TOKENS_PER_LINE]))

    return final_chunks


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
    
    # Flatten tracks (possibly split into chunks)
    track_chunks = flatten_tracks(data.get("tracks", {})) 
    
    # Group tracks back into examples, respecting <TRACKSEP>
    examples = []
    for chunk in track_chunks:
        ex = (
            "<|startofpiece|>"
            f"<NAME={name}>"
            f"<BPM={bpm}>"
            f"<DURATION_BEATS={dur_beats}>"
            f"<DURATION_MINUTES={dur_minutes}>"
            "<TRACKS>"
            f"{chunk}"
            "<|endofpiece|>"
        )
        examples.append(ex)

    return examples


def creat_training_data():
    # Find all .txt files in the data directory.
    files = list(DATA_DIR.glob("*.txt"))

    with OUT.open("w", encoding="utf-8") as fo:
        for p in files:
            try:
                txt = p.read_text(encoding="utf-8")
                data = json.loads(txt)

                examples = make_example(data)
                for ex in examples:
                    fo.write(ex.strip() + "\n")

            except Exception as e:
                print("skip", p, e)