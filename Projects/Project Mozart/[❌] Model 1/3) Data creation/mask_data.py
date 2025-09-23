from pathlib import Path
from sklearn.model_selection import train_test_split
import re

DATA_FILE = Path("../../dataset/FULL.txt")   # full sequences (already chunked)

TRAIN_INPUTS = Path("../../dataset/train_inputs.txt")
TRAIN_TARGETS = Path("../../dataset/train_targets.txt")
TEST_INPUTS = Path("../../dataset/test_inputs.txt")
TEST_TARGETS = Path("../../dataset/test_targets.txt")

TEST_SIZE = 0.2
RANDOM_STATE = 42

START_TRACKS_TAG = "<TRACKS>"
END_PIECE_TAG = "<|endofpiece|>"

# --- build masked version ---
def mask_line(line: str) -> str:
    """
    Mask the entire track section: every instrument's notes
    are replaced with <MASK>, including both sides of <TRACKSEP>.
    """
    line = line.rstrip("\n")
    if not line:
        return line

    start_idx = line.find(START_TRACKS_TAG)
    if start_idx == -1:
        return line

    prefix = line[: start_idx + len(START_TRACKS_TAG)]
    end_idx = line.find(END_PIECE_TAG, start_idx + len(START_TRACKS_TAG))
    if end_idx == -1:
        track_section = line[start_idx + len(START_TRACKS_TAG) :]
        suffix = ""
    else:
        track_section = line[start_idx + len(START_TRACKS_TAG) : end_idx]
        suffix = line[end_idx:]

    # split on <TRACKSEP> and mask each part
    instrument_parts = re.split(r"\s*<TRACKSEP>\s*", track_section.strip())

    masked_parts = []
    for part in instrument_parts:
        p = part.strip()
        if not p:
            continue
        if ":" in p:
            inst_name = p.split(":", 1)[0].strip()
            masked_parts.append(f"{inst_name}: <MASK>")
        else:
            masked_parts.append("<MASK>")

    masked_track_section = " <TRACKSEP> ".join(masked_parts)

    return prefix + masked_track_section + suffix


def mask_data():
    
    # --- load full dataset ---
    with DATA_FILE.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]


    # create inputs (masked) and targets (original)
    targets = lines[:]  # full unmasked lines
    inputs = [mask_line(ln) for ln in lines]
    
    # sanity check: ensure every input contains at least one <MASK>
    masked_counts = sum(1 for s in inputs if "<MASK>" in s)
    if masked_counts == 0:
        print("Warning: no <MASK> tokens were produced. Check input format and <TRACKS> tags.")

    # --- split train/test ---
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(
        inputs, targets, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # --- save ---
    TRAIN_INPUTS.write_text("\n".join(train_inputs), encoding="utf-8")
    TRAIN_TARGETS.write_text("\n".join(train_targets), encoding="utf-8")
    TEST_INPUTS.write_text("\n".join(test_inputs), encoding="utf-8")
    TEST_TARGETS.write_text("\n".join(test_targets), encoding="utf-8")

    
    # small report + samples for quick sanity-check
    print(f"Total examples: {len(lines)}")
    print(f"Train: {len(train_inputs)}  Test: {len(test_inputs)}")
    print(f"Masked examples in inputs: {masked_counts} / {len(lines)}")
    print("\nSample masked input (train):")
    print(train_inputs[0][:1000])
    print("\nSample target (train):")
    print(train_targets[0][:1000])