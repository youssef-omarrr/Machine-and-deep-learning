# -----------------------------
# Imports
# -----------------------------
from music21 import converter, note, chord, instrument, tempo

from GM_PROGRAMS import (get_instrument_name as gm_program_name, GM_NAME_MAP, 
                        get_program_for_instrument, get_best_program_for_instrument_name,
                        name_file)

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
def get_instrument_name(part, idx=None, score_title=None, used_names=None):
    """
    Enhanced instrument name extraction with better diversity handling.
    Uses the enhanced GM_PROGRAMS system for better instrument identification.
    
    Args:
        part: music21 Part object
        idx: Part index (for fallback naming)
        score_title: Score title (to avoid conflicts)
        used_names: Set of already used names (for uniqueness)
        
    Returns:
        str: A unique, descriptive instrument name
    """
    if used_names is None:
        used_names = set()
    
    # Method 1: Look for explicit Instrument objects in the part (preferred)
    try:
        insts = list(part.recurse().getElementsByClass(instrument.Instrument))
    except Exception:
        insts = []

    base_name = None
    program_number = None
    
    if insts:
        inst = insts[0]

        # 1.a instrument.instrumentName (most authoritative)
        if hasattr(inst, "instrumentName") and inst.instrumentName:
            name = inst.instrumentName.strip()
            if name and name != score_title:
                base_name = name

        # 1.b instrument.partName (sometimes set on instrument)
        if not base_name and hasattr(inst, "partName") and inst.partName:
            name = inst.partName.strip()
            if name and name != score_title:
                base_name = name

        # 1.c Get program number for enhanced GM lookup
        if hasattr(inst, "midiProgram") and inst.midiProgram is not None:
            try:
                program_number = int(inst.midiProgram)
            except Exception:
                pass

        # 1.d Use the instrument object to get program via enhanced GM_PROGRAMS
        if not base_name or not program_number:
            try:
                program_number = get_program_for_instrument(inst)
                if program_number is not None:
                    gm_name = gm_program_name(program_number)
                    if gm_name and not base_name:
                        base_name = gm_name
            except Exception:
                pass

        # 1.e Fallback to instrument class name if informative
        if not base_name:
            try:
                class_name = inst.__class__.__name__
                if class_name and class_name != "Instrument":
                    base_name = class_name
            except Exception:
                pass

    # Method 2: Check part.getInstrument() (another path music21 exposes)
    if not base_name:
        try:
            instr = part.getInstrument(returnDefault=False)
            if instr:
                if hasattr(instr, "instrumentName") and instr.instrumentName:
                    name = instr.instrumentName.strip()
                    if name and name != score_title:
                        base_name = name

                if not program_number and hasattr(instr, "midiProgram") and instr.midiProgram is not None:
                    try:
                        program_number = int(instr.midiProgram)
                    except Exception:
                        pass

                if not base_name:
                    class_name = instr.__class__.__name__
                    if class_name != "Instrument":
                        base_name = class_name
        except Exception:
            pass

    # Method 3: Check part.partName
    if not base_name:
        try:
            if hasattr(part, "partName") and part.partName:
                name = part.partName.strip()
                if name and name != score_title and name.lower() not in ["part", "track"]:
                    base_name = name
        except Exception:
            pass

    # Method 4: Enhanced fallback using program number
    if not base_name and program_number is not None:
        try:
            base_name = gm_program_name(program_number)
        except Exception:
            pass

    # Method 5: Track-based fallback
    if not base_name and idx is not None:
        base_name = f"Track_{idx+1}"

    # Final fallback
    if not base_name:
        base_name = "Unknown"

    # Ensure uniqueness
    final_name = base_name
    counter = 1
    while final_name in used_names:
        counter += 1
        final_name = f"{base_name}_{counter}"
    
    used_names.add(final_name)
    return final_name

# -----------------------------
# BPM extraction from MIDI - FIXED VERSION
# -----------------------------
def get_bpm_from_midi(midi_path, default=120):
    """
    Try to extract a sensible BPM from a MIDI file.
    Strategy:
        1. Use mido 'set_tempo' messages, prioritizing the most common tempo.
        2. Fall back to music21's tempo marks if available.
        3. Sanity-check the result (30 <= BPM <= 240).
        4. Default to 120 BPM if no valid tempo found.
    """
    # -----------------------------
    # Step 1: Try mido tempo events with improved logic
    # -----------------------------
    try:
        mid = MidiFile(midi_path)
        bpms = []
        for track in mid.tracks:
            for msg in track:
                if msg.type == "set_tempo":
                    bpm = 60_000_000 / msg.tempo  # µs per beat → BPM
                    if 30 <= bpm <= 240:  # Only collect reasonable BPM values
                        bpms.append(bpm)
        
        if bpms:
            # Use the most common BPM instead of average
            from collections import Counter
            bpm_counter = Counter(bpms)
            most_common_bpm = bpm_counter.most_common(1)[0][0]
            return round(most_common_bpm, 2)
        
    except Exception as e:
        print(f"Warning: mido could not parse tempo: {e}")
        
    # -----------------------------
    # Step 2: Fall back to music21 tempo marks
    # -----------------------------
    try:
        score = converter.parse(midi_path)
        # Use .flatten() instead of .flat (avoids deprecation warning)
        marks = list(score.flatten().getElementsByClass(tempo.MetronomeMark))
        bpms = [m.number for m in marks if m.number is not None and 30 <= m.number <= 240]
        if bpms:
            # Use the most common BPM instead of average
            from collections import Counter
            bpm_counter = Counter(bpms)
            most_common_bpm = bpm_counter.most_common(1)[0][0]
            return round(most_common_bpm, 2)
    except Exception as e:
        print(f"Warning: music21 could not parse tempo: {e}")

    # -----------------------------
    # Step 3: Calculate BPM from duration if available
    # -----------------------------
    try:
        mid = MidiFile(midi_path)
        duration_seconds = mid.length
        score = converter.parse(midi_path)
        duration_beats = score.duration.quarterLength
        
        if duration_seconds > 0 and duration_beats > 0:
            calculated_bpm = (duration_beats / duration_seconds) * 60
            if 30 <= calculated_bpm <= 240:
                return round(calculated_bpm, 2)
    except Exception as e:
        print(f"Warning: Could not calculate BPM from duration: {e}")

    # -----------------------------
    # Step 4: Fallback
    # -----------------------------
    print(f"Warning: Using default BPM of {default}")
    return default

# -----------------------------
# Get accurate duration calculation - FIXED
# -----------------------------
def get_accurate_duration(score, midi_path):
    """
    Get accurate duration in beats and seconds for a MIDI file.
    """
    try:
        # Get duration in seconds from MIDI file
        mid = MidiFile(midi_path)
        duration_seconds = mid.length
        
        # Calculate duration in beats by finding the maximum offset + duration
        duration_beats = 0
        for part in score.parts:
            # Get all notes, chords, and rests
            for el in part.recurse().notesAndRests:
                end_time = el.offset + el.duration.quarterLength
                if end_time > duration_beats:
                    duration_beats = end_time
        
        # If we still have a ridiculously low beat count, try a different approach
        if duration_beats < 10 and duration_seconds > 60:  # Less than 10 beats but more than 60 seconds            
            # Alternative approach: count all notes and estimate beats
            total_notes = 0
            for part in score.parts:
                total_notes += len(list(part.recurse().notesAndRests))
            
            # Estimate based on average note density (this is a rough estimate)
            if total_notes > 0:
                # Assuming average note duration of 0.5 beats (eighth notes)
                duration_beats = total_notes * 0.5
                # print(f"Estimated {duration_beats} beats based on {total_notes} notes")
        
        return duration_beats, round(duration_seconds/60, 2)
    
    except Exception as e:
        print(f"Warning: Could not calculate accurate duration: {e}")
        # Fallback: use score duration
        return score.duration.quarterLength, None

# -----------------------------
# Enhanced MIDI to dict conversion
# -----------------------------
def midi_to_dict(midi_path, print_details = True):
    """
    Convert a MIDI file into a dictionary mapping each instrument to a sequence of note/chord/rest tokens.
    Enhanced version with better instrument detection and diversity.
    """
    
    print(f"Processing MIDI file: {os.path.basename(midi_path)}")
    print("-" * 55)
    score = converter.parse(midi_path)  # Parse MIDI file into a music21 score
    
    # Capture score title if present
    score_title = None
    if hasattr(score, 'metadata') and score.metadata and hasattr(score.metadata, 'title'):
        score_title = score.metadata.title
        print(f"Score title: {score_title}")
        
    data = {}  # Dictionary to store instrument-token mapping
    used_names = set()  # Track used names for uniqueness
    used_programs = set()  # Track used programs for diversity info

    for i, part in enumerate(score.parts):  # Iterate over all parts (instruments)
        
        # Get unique, descriptive instrument name
        instr_name = get_instrument_name(part, idx=i, score_title=score_title, used_names=used_names)
        
        # Get program number for reporting
        program_num = None
        try:
            insts = list(part.recurse().getElementsByClass(instrument.Instrument))
            if insts:
                inst = insts[0]
                program_num = get_program_for_instrument(inst)
            else:
                # Fallback: try to guess program from name
                program_num = get_best_program_for_instrument_name(instr_name, used_programs)
            
            if program_num is not None:
                used_programs.add(program_num)
        except Exception:
            pass
        
        # Debug print with program info
        if print_details:
            prog_info = f" (Program {program_num}: {gm_program_name(program_num)})" if program_num is not None else ""
            print(f"Part #{i+1} -> '{instr_name}'{prog_info}")
            
        tokens = []  # List to store tokenized notes/chords/rests
        note_count = 0
        for el in part.recurse().notesAndRests:
            token = note_to_token(el)
            if token:
                tokens.append(token)
                note_count += 1
        
        if print_details:
            print(f"  |__ {note_count} notes/rests converted to tokens")
            
        data[instr_name] = tokens
        
    print(f"Total instruments detected: {len(data)}")
    if len(used_programs) < len(data):
        print(f"Note: {len(data) - len(used_programs)} instruments may use duplicate programs")
    else:
        print("All instruments have unique program assignments")
        
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
def encode(midi_path, output_dir="../dataset/text", print_details = True, numerate = True):
    """
    Convert MIDI to token dict and save to file.
    Automatically names the output using the song name + incremented suffix.
    Enhanced with better instrument diversity tracking.
    
    Return: output_path
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract song name (prefer metadata, fallback to filename)
    score = converter.parse(midi_path)
    name = None
    
    if hasattr(score, 'metadata') and score.metadata and score.metadata.title:
        name = score.metadata.title
    if not name:
        name = os.path.splitext(os.path.basename(midi_path))[0]
        
    safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)

    # Use helper to generate unique file path
    output_path = name_file(output_dir=output_dir, 
                            safe_name=safe_name, 
                            extension=".txt",
                            numerate= numerate)

    # Process MIDI
    try:
        # Get data and bpm
        data = midi_to_dict(midi_path, print_details)
        bpm = get_bpm_from_midi(midi_path)
        
        # Get accurate duration - FIXED
        duration_beats, duration_minutes = get_accurate_duration(score, midi_path)
        
        # If MIDI duration calculation failed, estimate from beats and BPM
        if duration_minutes is None:
            duration_minutes = duration_beats / bpm
            print(f"Warning: Using estimated duration from BPM and beats")


        # Wrap into one dictionary
        full_data = {
            "name": name,
            "bpm": bpm,
            "duration_beats": round(duration_beats, 2),
            "duration_minutes": round(duration_minutes, 2),
            "tracks": data   # put all instrument parts under "tracks"
        }
        
        
        # Debug print
        print(f"Estimated length: {duration_minutes} minutes")


        # Save the text file
        save_dict_to_txt(full_data, output_path)
        
        print("="*55)
        print(f"\033[92mData saved to {output_path} \nwith bpm = {bpm}\033[0m")
        print("="*55)
        print('\n')
        
        return output_path

    except Exception as e:
        print(f"Error processing MIDI file: {e}")
        raise

# Example usage:
# encode("example.mid")