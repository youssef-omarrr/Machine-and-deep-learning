# -----------------------------
# Imports
# -----------------------------
from music21 import stream, note, chord, tempo, metadata
import json
from fractions import Fraction
import os

# -----------------------------
# Duration mapping
# -----------------------------
# Map symbolic notation to quarterLength values
# s: sixteenth note (0.25), e: eighth note (0.5)
# q: quarter note (1.0), h: half note (2.0), w: whole note (4.0)
dur_map = {"s": 0.25, "e": 0.5, "q": 1.0, "h": 2.0, "w": 4.0}

# -----------------------------
# Token to music21 object
# -----------------------------
def token_to_note(token):
    """
    Convert a token string to a music21 Note, Chord, or Rest object.
    Token format: "NAME_DURATION" (e.g., "C4_q", "Rest_h", "C4.E4_q")
    """
    if "_" not in token:
        raise ValueError(f"Invalid token format: {token}")
    
    # Split token into name and duration
    name, dur_symbol = token.rsplit("_", 1)
    dur_symbol = dur_symbol.strip()
    
    # Convert duration symbol to quarterLength
    if dur_symbol in dur_map:
        duration = dur_map[dur_symbol]
    else:
        try:
            duration = float(dur_symbol)
        except ValueError:
            try:
                duration = float(Fraction(dur_symbol))
            except ValueError:
                raise ValueError(f"Unknown duration symbol: '{dur_symbol}' in token '{token}'")
    
    # Create music21 object
    if name.strip() == "Rest":
        rest_obj = note.Rest()
        rest_obj.duration.quarterLength = duration
        return rest_obj
    elif "." in name:
        chord_notes = [p.strip() for p in name.split(".")]
        chord_obj = chord.Chord(chord_notes)
        chord_obj.duration.quarterLength = duration
        return chord_obj
    else:
        note_obj = note.Note(name.strip())
        note_obj.duration.quarterLength = duration
        return note_obj

# -----------------------------
# Dictionary to music21 Score
# -----------------------------
def dict_to_score(data, bpm=120):
    """
    Convert a dictionary of token sequences into a music21 Score.
    Each key in the dictionary represents a part (instrument/track).
    """
    score = stream.Score()
    
    # Add metadata
    score.insert(0, metadata.Metadata())
    score.metadata.title = "Generated Composition"
    
    # Set tempo
    metronome_mark = tempo.MetronomeMark(number=bpm)
    score.insert(0, metronome_mark)
    
    # Add parts to score
    for track_name, tokens in data.items():
        part = stream.Part()
        part.id = track_name
        for token in tokens:
            part.append(token_to_note(token))
        score.append(part)
    
    return score

# -----------------------------
# Decode tokens to MIDI with auto-numbering
# -----------------------------
def decode_to_midi(data_or_path, output_dir="test_audio", base_name="output", bpm=120):
    """
    Decode a token dictionary or JSON file into a MIDI file.
    Automatically names the output as output_N.mid if files already exist.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Determine next available number
    existing_files = [f for f in os.listdir(output_dir) if f.startswith(base_name) and f.endswith(".mid")]
    numbers = []
    for f in existing_files:
        parts = f.rstrip(".mid").split("_")
        if len(parts) > 1 and parts[-1].isdigit():
            numbers.append(int(parts[-1]))
    next_number = max(numbers, default=0) + 1
    output_midi = os.path.join(output_dir, f"{base_name}_{next_number}.mid")

    # Load data
    if isinstance(data_or_path, str):
        with open(data_or_path, "r") as file:
            data = json.load(file)
    else:
        data = data_or_path

    # Convert dictionary to score
    score = dict_to_score(data, bpm)
    
    # Write to MIDI
    score.write('midi', fp=output_midi)
    print(f"Successfully decoded and saved MIDI as {output_midi} with bpm = {bpm}")
    
    return output_midi

