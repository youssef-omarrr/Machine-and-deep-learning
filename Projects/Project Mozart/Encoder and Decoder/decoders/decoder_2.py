# -----------------------------
# Imports
# -----------------------------
from music21 import stream, note, chord, tempo, metadata, instrument
import json
from fractions import Fraction
import os
from GM_PROGRAMS import *

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
# Instrument selection
# -----------------------------
def _choose_instrument_for_name(track_name):
    """
    Enhanced instrument selection with better pattern matching and fallbacks.
    """
    if not track_name:
        return instrument.Piano()  # Default fallback
    
    lower = track_name.lower().strip()
    
    # Remove common prefixes/suffixes that don't help identification
    lower = lower.replace("track_", "").replace("part_", "").replace("channel_", "")
    
    # Direct mapping first
    if lower in INSTRUMENT_MAPPING:
        return INSTRUMENT_MAPPING[lower]()
    
    # Partial matching for compound names
    for key, instr_class in INSTRUMENT_MAPPING.items():
        if key in lower:
            return instr_class()
    
    # Handle program numbers (e.g., "Program_0", "Program_40")
    if lower.startswith("program_"):
        try:
            prog_num = int(lower.split("_")[1])
            if prog_num in MIDI_PROGRAM_MAPPING:
                return MIDI_PROGRAM_MAPPING[prog_num]()
        except (IndexError, ValueError):
            pass
    
    # Handle track numbers - try to infer from position
    if lower.startswith("track_"):
        try:
            track_num = int(lower.split("_")[1])
            # Common conventions: track 1 is often piano, track 10 is drums, etc.
            if track_num == 10:
                return instrument.Percussion()
            elif track_num == 1:
                return instrument.Piano()
        except (IndexError, ValueError):
            pass
    
    # Handle "unknown" or generic names
    if lower in ["unknown", "unknowninstrument", "instrument"]:
        return instrument.Piano()  # Default to piano for unknown instruments
    
    # Final fallback: create a generic instrument but try to set a meaningful name
    generic_inst = instrument.Instrument()
    generic_inst.instrumentName = track_name
    return generic_inst

# -----------------------------
# Dict -> music21 Score (decoder)
# -----------------------------
def dict_to_score(data, bpm=120):
    """
    Convert {track_name: [tokens]} to a music21 Score.
    Enhanced version with better instrument handling.
    """
    score = stream.Score()
    score.insert(0, metadata.Metadata())
    score.metadata.title = "Generated Composition"
    score.insert(0, tempo.MetronomeMark(number=bpm))

    for track_name, tokens in data.items():
        print(f"Creating part for: {track_name}")
        
        part = stream.Part()
        part.id = track_name
        
        # Choose and insert appropriate instrument
        inst_obj = _choose_instrument_for_name(track_name)
        print(f"  -> Using instrument: {inst_obj.__class__.__name__}")
        
        # Set additional properties if available
        if hasattr(inst_obj, 'instrumentName') and not inst_obj.instrumentName:
            inst_obj.instrumentName = track_name
        
        part.insert(0, inst_obj)
        
        # Add notes to the part
        note_count = 0
        for tok in tokens:
            try:
                music_obj = token_to_note(tok)
                if music_obj:
                    part.append(music_obj)
                    note_count += 1
            except Exception as e:
                print(f"  Warning: Could not process token '{tok}': {e}")
        
        print(f"  -> Added {note_count} notes/rests")
        score.append(part)
    
    print(f"Score created with {len(score.parts)} parts")
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
        print(f"Loading data from: {data_or_path}")
        with open(data_or_path, "r") as file:
            data = json.load(file)
    else:
        data = data_or_path

    print(f"Input data contains {len(data)} instruments:")
    for track_name, tokens in data.items():
        print(f"  - {track_name}: {len(tokens)} tokens")

    # Convert dictionary to score
    score = dict_to_score(data, bpm)
    
    # Write to MIDI
    try:
        score.write('midi', fp=output_midi)
        print(f"Successfully decoded and saved MIDI as {output_midi} with bpm = {bpm}")
        return output_midi
    except Exception as e:
        print(f"Error writing MIDI file: {e}")
        raise