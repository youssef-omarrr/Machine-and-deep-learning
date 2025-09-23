# -----------------------------
# Imports
# -----------------------------
from music21 import converter, note, chord
from GM_PROGRAMS import GM_PROGRAMS
from mido import MidiFile
import json
import os

# -----------------------------
# Duration mapping
# -----------------------------
# Map music21 quarterLength to symbolic duration strings
dur_map = {
    0.25: "s",   # sixteenth note
    0.5:  "e",   # eighth note
    1.0:  "q",   # quarter note
    2.0:  "h",   # half note
    4.0:  "w"    # whole note
}

# -----------------------------
# Note/Chord tokenization
# -----------------------------
def note_to_token(n):
    """Convert a music21 note or chord to a token string with duration"""
    
    # Map the note/chord duration to its symbolic representation
    dur = dur_map.get(round(n.duration.quarterLength, 2), 
                        str(n.duration.quarterLength))  # fallback: use numeric duration
    
    if isinstance(n, note.Note):
        # Single note: "C4_q"
        return f"{n.nameWithOctave}_{dur}"
    
    elif isinstance(n, chord.Chord):
        # Chord: join all pitch names with dots, e.g., "C4.E4.G4_q"
        pitches = ".".join(p.nameWithOctave for p in n.pitches)
        return f"{pitches}_{dur}"
    
    return None  # Return None if input is neither Note nor Chord

# -----------------------------
# Instrument name retrieval
# -----------------------------
def get_instrument_name(part, idx):
    """Determine the name of an instrument for a given music21 part"""
    
    instr = part.getInstrument(returnDefault=True)  # Get the instrument object, use default if none set

    # Try using the part's explicit name
    if instr.partName:
        return instr.partName.strip()
    
    # Try using the instrument's general name
    if instr.instrumentName:
        return instr.instrumentName.strip()
    
    # Try using MIDI program number to get standard GM instrument name
    if hasattr(instr, "midiProgram") and instr.midiProgram is not None:
        return GM_PROGRAMS.get(instr.midiProgram, f"Program_{instr.midiProgram}")
    
    # Fallback: use generic track name based on index
    return f"Track_{idx+1}"

# -----------------------------
# BPM extraction from MIDI
# -----------------------------
def get_bpm_from_midi(midi_path):
    """
    Reads the first 'set_tempo' message from a MIDI file and returns BPM.
    Defaults to 120 BPM if no tempo message is found.
    """
    mid = MidiFile(midi_path)  # Load the MIDI file
    for track in mid.tracks:  # Iterate over all tracks
        for msg in track:  # Iterate over messages in the track
            if msg.type == 'set_tempo':  # Look for tempo messages
                # mido stores tempo as microseconds per beat
                tempo_us = msg.tempo
                bpm = 60_000_000 / tempo_us  # Convert to beats per minute
                return bpm
    return 120  # Return default BPM if no tempo message found

# -----------------------------
# Convert MIDI to token dictionary
# -----------------------------
def midi_to_dict(midi_path):
    """
    Convert a MIDI file into a dictionary mapping each instrument to a sequence of note/chord/rest tokens.
    """
    
    score = converter.parse(midi_path)  # Parse MIDI file into a music21 score
    data = {}  # Dictionary to store instrument-token mapping

    for i, part in enumerate(score.parts):  # Iterate over all parts (instruments)
        
        instr_name = get_instrument_name(part, i)  # Determine instrument name
        tokens = []  # List to store tokenized notes/chords/rests
        
        for n in part.recurse().notesAndRests:  # Iterate over all notes and rests in the part
            
            if isinstance(n, note.Rest):
                # Convert rest to token with duration
                dur = dur_map.get(round(n.duration.quarterLength, 2), str(n.duration.quarterLength))
                tokens.append(f"Rest_{dur}")
            else:
                # Convert note or chord to token
                tokens.append(note_to_token(n))
        
        data[instr_name] = tokens  # Store token sequence under instrument name
        
    return data  # Return the dictionary

# -----------------------------
# Save dictionary to TXT/JSON
# -----------------------------
def save_dict_to_txt(data, out_path="output.txt"):
    """Save a dictionary to a JSON-formatted text file"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2))

# -----------------------------
# Main function
# -----------------------------
def encode(midi_path, output_dir="test_txt", base_name="output"):
    """
    Convert MIDI to token dict and save to file.
    Automatically names the output as output_N.txt if files already exist.
    
    Return: output_path, 
            bpm
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List existing files and find the highest numbered output
    existing_files = [f for f in os.listdir(output_dir) if f.startswith(base_name) and f.endswith(".txt")]
    numbers = []
    for f in existing_files:
        parts = f.rstrip(".txt").split("_")
        if len(parts) > 1 and parts[-1].isdigit():
            numbers.append(int(parts[-1]))
    next_number = max(numbers, default=0) + 1

    # Build the new output file path
    output_path = os.path.join(output_dir, f"{base_name}_{next_number}.txt")

    # Process MIDI
    data = midi_to_dict(midi_path)
    bpm = get_bpm_from_midi(midi_path)
    print(f"BPM: {bpm}")
    save_dict_to_txt(data, output_path)
    print(f"Data saved to {output_path}")
    
    return output_path, bpm

# Example usage:
# main("example.mid")
