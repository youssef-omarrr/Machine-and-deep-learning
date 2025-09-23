from pathlib import Path

DATA_FILE = Path("../../dataset/FULL.txt")
NOTES_OUT = Path("../../dataset/unique_notes.txt")

START_TRACKS_TAG = "<TRACKS>"
END_PIECE_TAG = "<|endofpiece|>"
TRACKSEP = "<TRACKSEP>"

def is_full_note_token(tok: str) -> bool:
    """
    Return True if tok appears to be a full musical token:
    - contains '_' (duration separator)
    - not a metadata/special token (no '<', '>', '=', ':')
    - either starts with 'Rest_' OR the left side (before '_') contains at least one digit (octave)
        and at least one letter A-G (pitch)
    """
    if not tok or '_' not in tok:
        return False
    if '<' in tok or '>' in tok or '=' in tok or ':' in tok:
        return False

    left = tok.split('_', 1)[0]

    # Accept rests (Rest_...)
    if left.startswith("Rest"):
        return True

    # Left must contain a digit (octave number) and at least one letter A-G
    has_digit = any(ch.isdigit() for ch in left)
    has_pitch_letter = any(ch.upper() in "ABCDEFG" for ch in left)
    return has_digit and has_pitch_letter


def extract_unique_notes_from_full(data_path: Path = DATA_FILE,
                                    out_path: Path = NOTES_OUT):
    unique_notes = set()
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found")

    with data_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue

            # find every <TRACKS> ... (<|endofpiece|> or end of line)
            search_pos = 0
            while True:
                idx = line.find(START_TRACKS_TAG, search_pos)
                if idx == -1:
                    break

                content_start = idx + len(START_TRACKS_TAG)
                end_idx = line.find(END_PIECE_TAG, content_start)
                if end_idx == -1:
                    track_section = line[content_start:].strip()
                    search_pos = len(line)
                else:
                    track_section = line[content_start:end_idx].strip()
                    search_pos = end_idx + len(END_PIECE_TAG)

                if not track_section:
                    continue

                # split on literal TRACKSEP (tolerate spaces)
                parts = [p.strip() for p in track_section.split(TRACKSEP)]

                for part in parts:
                    if not part:
                        continue

                    # if instrument name present (InstrumentName: notes...), remove name
                    if ":" in part:
                        _, notes_str = part.split(":", 1)
                        notes_str = notes_str.strip()
                    else:
                        notes_str = part

                    if not notes_str:
                        continue

                    # split by whitespace and filter tokens
                    for tok in notes_str.split():
                        tok = tok.strip()
                        if not tok:
                            continue
                        if is_full_note_token(tok):
                            unique_notes.add(tok)

    sorted_notes = sorted(unique_notes)
    out_path.write_text("\n".join(sorted_notes), encoding="utf-8")
    print(f"Extracted {len(sorted_notes)} unique musical tokens → {out_path}")
    return sorted_notes
