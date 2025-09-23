# -----------------------------
# Imports
# -----------------------------
from music21 import converter, note, chord, instrument
from GM_PROGRAMS import GM_PROGRAMS
from mido import MidiFile
import json
import os

# -----------------------------
# Duration mapping
# -----------------------------
# Map music21 quarterLength to symbolic duration strings
dur_map_to_symbol = {
    0.25: "s",   # sixteenth
    0.5:  "e",   # eighth
    1.0:  "q",   # quarter
    2.0:  "h",   # half
    4.0:  "w"    # whole
}
dur_map_from_symbol = {"s": 0.25, "e": 0.5, "q": 1.0, "h": 2.0, "w": 4.0}

# -----------------------------
# Note/Chord tokenization
# -----------------------------
def note_to_token(n):
    """Convert a music21 Note/Chord/Rest into token 'NAME_DURATION'."""
    
    # Map the note/chord duration to its symbolic representation
    dur = dur_map_to_symbol.get(round(n.duration.quarterLength, 2),
                                str(n.duration.quarterLength))
    
    if isinstance(n, note.Note):
        # Single note: "C4_q"
        return f"{n.nameWithOctave}_{dur}"
    
    if isinstance(n, chord.Chord):
        # Chord: join all pitch names with dots, e.g., "C4.E4.G4_q"
        pitches = ".".join(p.nameWithOctave for p in n.pitches)
        return f"{pitches}_{dur}"
    
    if isinstance(n, note.Rest):
        return f"Rest_{dur}"
    
    return None  # Return None if input is neither Note nor Chord

# -----------------------------
# Enhanced instrument name retrieval
# -----------------------------
def get_instrument_name(part, idx=None, score_title=None):
    """
    Try multiple ways to extract a readable instrument name from a music21 Part.
    Enhanced version with better MIDI program detection and fallback handling.
    """
    
    # Method 1: Look for explicit Instrument objects in the part
    insts = list(part.recurse().getElementsByClass(instrument.Instrument))
    if insts:
        inst = insts[0]
        
        # Check for instrumentName
        if hasattr(inst, 'instrumentName') and inst.instrumentName:
            name = inst.instrumentName.strip()
            if name and name != score_title:
                return name
        
        # Check for partName on instrument
        if hasattr(inst, 'partName') and inst.partName:
            name = inst.partName.strip()
            if name and name != score_title:
                return name
        
        # Check MIDI program number
        if hasattr(inst, 'midiProgram') and inst.midiProgram is not None:
            program = inst.midiProgram
            if program in GM_PROGRAMS:
                return GM_PROGRAMS[program]
            elif program in GM_PROGRAMS:
                return GM_PROGRAMS[program]
            else:
                return f"Program_{program}"
        
        # Use class name as fallback
        class_name = inst.__class__.__name__
        if class_name != "Instrument":  # Don't use generic "Instrument"
            return class_name
    
    # Method 2: Check part.getInstrument()
    try:
        instr = part.getInstrument(returnDefault=False)
        if instr:
            # Check instrumentName
            if hasattr(instr, 'instrumentName') and instr.instrumentName:
                name = instr.instrumentName.strip()
                if name and name != score_title:
                    return name
            
            # Check MIDI program
            if hasattr(instr, 'midiProgram') and instr.midiProgram is not None:
                program = instr.midiProgram
                if program in GM_PROGRAMS:
                    return GM_PROGRAMS[program]
                else:
                    return f"Program_{program}"
            
            # Use class name
            class_name = instr.__class__.__name__
            if class_name != "Instrument":
                return class_name
    except:
        pass
    
    # Method 3: Check part.partName (but avoid score title)
    if hasattr(part, 'partName') and part.partName:
        name = part.partName.strip()
        if name and name != score_title and name.lower() not in ['part', 'track']:
            return name
    
    # Method 4: Try to infer from MIDI channel/program via mido
    # This is more complex but can help with files that don't have proper instrument objects
    
    # Method 5: Final fallback
    if idx is not None:
        return f"Track_{idx+1}"
    
    return "Unknown"

# -----------------------------
# BPM extraction from MIDI
# -----------------------------
def get_bpm_from_midi(midi_path):
    """
    Reads the first 'set_tempo' message from a MIDI file and returns BPM.
    Defaults to 120 BPM if no tempo message is found.
    """
    try:
        mid = MidiFile(midi_path)  # Load the MIDI file
        for track in mid.tracks:  # Iterate over all tracks
            for msg in track:  # Iterate over messages in the track
                if msg.type == 'set_tempo':  # Look for tempo messages
                    # mido stores tempo as microseconds per beat
                    tempo_us = msg.tempo
                    bpm = 60_000_000 / tempo_us  # Convert to beats per minute
                    return round(bpm, 2)
    except Exception as e:
        print(f"Warning: Could not read tempo from MIDI file: {e}")
    
    return 120  # Return default BPM if no tempo message found

# -----------------------------
# Enhanced MIDI to dict conversion
# -----------------------------
def midi_to_dict(midi_path):
    """
    Convert a MIDI file into a dictionary mapping each instrument to a sequence of note/chord/rest tokens.
    Enhanced version with better instrument detection.
    """
    
    print(f"Processing MIDI file: {os.path.basename(midi_path)}")
    print("-"*55)
    score = converter.parse(midi_path)  # Parse MIDI file into a music21 score
    
    # Capture score title if present
    score_title = None
    if hasattr(score, 'metadata') and score.metadata and hasattr(score.metadata, 'title'):
        score_title = score.metadata.title
        # print(f"Score title: {score_title}")
        
    data = {}  # Dictionary to store instrument-token mapping

    for i, part in enumerate(score.parts):  # Iterate over all parts (instruments)
        
        instr_name = get_instrument_name(part, idx=i, score_title=score_title)
        
        # Handle duplicate instrument names by adding numbers
        original_name = instr_name
        counter = 1
        while instr_name in data:
            counter += 1
            instr_name = f"{original_name}_{counter}"
        
        # Debug print
        # print(f"Part #{i+1} -> '{instr_name}'")
        
        tokens = []  # List to store tokenized notes/chords/rests
        note_count = 0
        for el in part.recurse().notesAndRests:
            token = note_to_token(el)
            if token:
                tokens.append(token)
                note_count += 1
        
        # print(f"  -> {note_count} notes/rests converted to tokens")
        data[instr_name] = tokens
        
    print(f"Total instruments detected: {len(data)}")
    return data

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
    
    Return: output_path, bpm
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
    try:
        data = midi_to_dict(midi_path)
        bpm = get_bpm_from_midi(midi_path)
        print(f"BPM: {bpm}")
        save_dict_to_txt(data, output_path)
        print(f"Data saved to {output_path}")
        
        return output_path, bpm
    except Exception as e:
        print(f"Error processing MIDI file: {e}")
        raise

# Example usage:
# encode("example.mid")