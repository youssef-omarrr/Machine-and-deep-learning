import json
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

DATA_DIR = Path("../dataset/text")
TRAIN_OUT = Path("../dataset/train_file_2.txt")
TEST_OUT = Path("../dataset/test_file_2.txt")
MASK_PROBABILITY = 0.2   # Probability of masking an instrument
FULL_MASK_PROBABILITY = 0.3  # Probability of masking ALL instruments in a piece

def flatten_tracks(tracks: dict, mask_instruments: bool = False, force_full_mask: bool = False) -> str:
    """
    Converts the dictionary of tracks into a single string with optional masking.
    
    Args:
        tracks (dict): A dictionary where keys are instrument names and
                        values are lists of musical tokens.
        mask_instruments (bool): Whether to randomly mask some instruments.
        force_full_mask (bool): Whether to mask ALL instruments (overrides mask_instruments).
    Returns:
        str: A single-line string representation of all tracks.
    """
    lines = []
    
    for name, tokens in tracks.items():
        if force_full_mask:
            # Mask everything
            lines.append(f"{name}: <MASK>")
        elif mask_instruments and random.random() < MASK_PROBABILITY:
            # Random mask
            lines.append(f"{name}: <MASK>")
        else:
            # Keep tokens
            lines.append(f"{name}:" + " ".join(tokens))
    
    return " <TRACKSEP> ".join(lines)

def make_example(data, mask_instruments=False, force_full_mask=False):
    """
    Constructs a single training example string with optional instrument masking.
    
    Args:
        data (dict): The dictionary loaded from a JSON/TXT file.
        mask_instruments (bool): Whether to mask some instruments in this example.
        force_full_mask (bool): Whether to mask ALL instruments.
        
    Returns:
        str: The fully formatted training example string.
    """
    name = data.get("name", "")
    bpm = data.get("bpm", "")
    dur_beats = data.get("duration_beats", "")
    dur_minutes = data.get("duration_minutes", "")
    
    tracks = flatten_tracks(data.get("tracks", {}), mask_instruments, force_full_mask)
    
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

def create_training_data():
    files = list(DATA_DIR.glob("*.txt"))
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    
    print(f"Found {len(files)} files, splitting into {len(train_files)} train and {len(test_files)} test files")
    
    with TRAIN_OUT.open("w", encoding="utf-8") as train_fo:
        for p in train_files:
            try:
                txt = p.read_text(encoding="utf-8")
                data = json.loads(txt)
                
                for _ in range(3):  # Create 3 versions per piece
                    # With probability, fully mask all instruments
                    if random.random() < FULL_MASK_PROBABILITY:
                        masked_example = make_example(data, force_full_mask=True)
                    else:
                        masked_example = make_example(data, mask_instruments=True)
                    
                    train_fo.write(masked_example.strip() + "\n")
                    
            except Exception as e:
                print("skip training file", p, e)
    
    # For test set → always FULLY masked (to match inference time)
    with TEST_OUT.open("w", encoding="utf-8") as test_fo:
        for p in test_files:
            try:
                txt = p.read_text(encoding="utf-8")
                data = json.loads(txt)
                
                masked_example = make_example(data, force_full_mask=True)
                test_fo.write(masked_example.strip() + "\n")
                    
            except Exception as e:
                print("skip test file", p, e)

special_tokens = {
    "bos_token": "<|startofpiece|>",
    "eos_token": "<|endofpiece|>",
    "pad_token": "<pad>",
    "additional_special_tokens": [
        "<TRACKS>", "<TRACKSEP>", "<NAME=", "<BPM=", 
        "<DURATION_BEATS=", "<DURATION_MINUTES=", "<MASK>"
    ],
}

if __name__ == "__main__":
    create_training_data()
    print(f"Created training data with partial + full masking at {TRAIN_OUT}")
    print(f"Created test data with full masking at {TEST_OUT}")
