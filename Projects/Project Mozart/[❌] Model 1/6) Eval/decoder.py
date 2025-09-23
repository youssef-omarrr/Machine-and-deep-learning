# -----------------------------
# Imports
# -----------------------------
from music21 import stream, note, chord, tempo, metadata, instrument
import json
from fractions import Fraction
import os

from GM_PROGRAMS import (GM_CLASS_MAP, GM_NAME_MAP, make_instrument_instance, 
                        get_instrument_name, name_file, get_best_program_for_instrument_name,
                        get_program_for_instrument)

from pydub import AudioSegment 
import subprocess

import os
os.environ["PATH"] += r";C:\tools\fluidsynth\bin"

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
# Enhanced instrument selection
# -----------------------------
def choose_instrument_for_name(track_name, used_programs=None):
    """
    Enhanced instrument selection using the new GM_PROGRAMS system.
    Returns (music21 instrument instance, program_number).
    
    Args:
        track_name: Name of the track/instrument
        used_programs: Set of already used program numbers to avoid
    
    Returns:
        tuple: (instrument_object, program_number)
    """
    if used_programs is None:
        used_programs = set()

    if not track_name:
        return instrument.Piano(), 0

    # Clean up track name
    clean_name = track_name.lower().strip()
    clean_name = clean_name.replace("track_", "").replace("part_", "").replace("channel_", "")

    # Handle explicit program specification
    if clean_name.startswith("program_") or clean_name.startswith("program "):
        digits = "".join(ch for ch in clean_name if ch.isdigit())
        try:
            prog_num = int(digits)
            inst = make_instrument_instance(prog_num)
            if inst is not None:
                inst.instrumentName = track_name
                return inst, prog_num
            # Fallback: generic instrument with readable name
            generic = instrument.Instrument()
            generic.instrumentName = get_instrument_name(prog_num)
            return generic, prog_num
        except Exception:
            pass

    # Use the smart assignment function from enhanced GM_PROGRAMS
    prog_num = get_best_program_for_instrument_name(track_name, used_programs)
    
    # Try to create the instrument instance
    inst = make_instrument_instance(prog_num)
    if inst is not None:
        # Set the name to match the track
        if hasattr(inst, 'instrumentName'):
            inst.instrumentName = track_name
        return inst, prog_num
    
    # Fallback: generic instrument
    generic = instrument.Instrument()
    generic.instrumentName = track_name
    return generic, prog_num

# -----------------------------
# Dict -> music21 Score (decoder)
# -----------------------------
def dict_to_score(data, bpm=120, print_details=True):
    """
    Convert {track_name: [tokens]} to a music21 Score.
    Enhanced version with better instrument diversity and no program conflicts.
    Returns both the score and a dict of {track_name: prog_num} for FluidSynth.
    """
    score = stream.Score()
    score.insert(0, metadata.Metadata())
    score.metadata.title = "Generated Composition"
    score.insert(0, tempo.MetronomeMark(number=bpm))

    program_map = {}  # Store MIDI program numbers for each track
    used_programs = set()  # Track used programs to ensure diversity

    for track_name, tokens in data.items():
        if print_details:
            print(f"Creating part for: {track_name}")
        
        part = stream.Part()
        part.id = track_name
        
        # Choose instrument with smart assignment
        inst_obj, prog_num = choose_instrument_for_name(track_name, used_programs)
        used_programs.add(prog_num)  # Mark this program as used
        
        if print_details:
            print(f"  -> Using instrument: {inst_obj.__class__.__name__} (Program {prog_num}: {get_instrument_name(prog_num)})")
        
        # Insert the music21 instrument object into the part
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
        
        if print_details:
            print(f"  -> Added {note_count} notes/rests")
        score.append(part)

        # Save program number for FluidSynth
        program_map[track_name] = prog_num
    
    print(f"Score created with {len(score.parts)} parts")
    return score, program_map

# -----------------------------
# Decode tokens to Audio (FluidSynth via midi2audio)
# -----------------------------
def decode_to_audio(data_or_path, soundfont_path = "../soundfonts/AegeanSymphonicOrchestra-SND.sf2",
                    output_dir="../example_outputs/model_output", bpm=None, print_details = True, numerate = True):
    """
    Decode a token dictionary or JSON file into audio (WAV).
    If JSON includes "bpm" and "tracks", use them automatically.
    Uses FluidSynth (via midi2audio) with a high-quality SoundFont for playback.
    Ensures each part is assigned the correct GM program from program_map.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    if isinstance(data_or_path, str):
        print(f"Loading data from: {data_or_path}")
        print("-"*55)
        with open(data_or_path, "r") as file:
            data = json.load(file)
    else:
        data = data_or_path
        
    # Detect new format with metadata
    if "tracks" in data:
        bpm = data.get("bpm", bpm if bpm else 120)
        name = data.get("name", "Generated_Composition")
        
        duration_beats = data.get("duration_beats")
        duration_seconds = data.get("duration_seconds")
        
        tracks = data["tracks"]
        
        print(f"Song name: {name}, BPM: {bpm}")
        if duration_beats and duration_seconds:
            print(f"Estimated duration: {duration_beats} beats (~{duration_seconds:.2f} seconds)")
        print("-"*55)
        
    else:
        # old format fallback
        tracks = data
        bpm = bpm if bpm else 120
        name = "Generated_Composition"
        print(f"No metadata found, defaulting BPM={bpm}")
        print("-"*55)
    
    # Sanitize name for filename
    safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)
    
    # Use helper to generate unique file path
    output_wav = name_file (output_dir= output_dir,
                        safe_name= safe_name,
                        extension= '.wav',
                        numerate = numerate)
    if print_details:
        print(f"Input data contains {len(tracks)} instruments:")
        for track_name, tokens in tracks.items():
            print(f"  - {track_name}: {len(tokens)} tokens")

        print("-"*55)
    
    # Convert dictionary to score and get program numbers
    score, program_map = dict_to_score(tracks, bpm, print_details)

    # Write score to temporary MIDI
    tmp_midi = os.path.join(output_dir, "temp.mid")
    score.write('midi', fp=tmp_midi)
    

    # Render MIDI to WAV with fluidsynth (no live playback, file only)
    try:
        cmd = [
            r"C:\tools\fluidsynth\bin\fluidsynth.exe",
            "-a", "file",           # write to file only
            "-F", output_wav,       # output wav
            "-ni", soundfont_path,  # load soundfont
            "-r", "44100",          # sample rate
            "-g", "2.0",            # gain boost
            tmp_midi                # MIDI file (last!)
        ]
        subprocess.run(cmd, check=True)
        
        # Delete temp midi
        if os.path.exists(tmp_midi):
            os.remove(tmp_midi)
        
        # Normalize the audio afterwards
        sound = AudioSegment.from_wav(output_wav)
        normalized = sound.normalize()
        normalized.export(output_wav, format="wav")

        
        if print_details:
            print("-"*55)
            print("Instrument program mapping used:")
            for track, prog in program_map.items():
                print(f"  {track}: Program {prog}")

        print("="*55)
        print(f"\033[92mSuccessfully decoded and saved audio as {output_wav} \nwith bpm = {bpm}\033[0m")
        print("="*55)
        print('\n')
        
        return output_wav
    except Exception as e:
        print(f"Error rendering audio with FluidSynth: {e}")
        raise
