# -----------------------------
# Imports
# -----------------------------
from music21 import stream, note, chord, tempo, metadata, instrument
import json
from fractions import Fraction
import os
from GM_PROGRAMS import *
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
# Instrument selection (music21 + program number)
# -----------------------------
def _choose_instrument_for_name(track_name):
    """
    Enhanced instrument selection with both music21 object and MIDI program number.
    Returns a tuple: (music21_instrument, midi_program_number).
    """
    if not track_name:
        return instrument.Piano(), 0  # Default to Acoustic Grand Piano

    lower = track_name.lower().strip()
    lower = lower.replace("track_", "").replace("part_", "").replace("channel_", "")

    # Direct mapping via INSTRUMENT_MAPPING
    if lower in INSTRUMENT_MAPPING:
        inst_class = INSTRUMENT_MAPPING[lower]
        # Find closest GM program number match
        for prog_num, name in GM_PROGRAMS.items():
            if lower in name.lower():
                return inst_class(), prog_num
        return inst_class(), 0  # fallback to piano if not found

    # Partial matching for compound names
    for key, inst_class in INSTRUMENT_MAPPING.items():
        if key in lower:
            for prog_num, name in GM_PROGRAMS.items():
                if key in name.lower():
                    return inst_class(), prog_num
            return inst_class(), 0

    # Handle program numbers in name (e.g., "Program_40")
    if lower.startswith("program_"):
        try:
            prog_num = int(lower.split("_")[1])
            if prog_num in MIDI_PROGRAM_MAPPING:
                return MIDI_PROGRAM_MAPPING[prog_num](), prog_num
            else:
                return instrument.Piano(), 0
        except (IndexError, ValueError):
            return instrument.Piano(), 0

    # Handle track numbers (common conventions)
    if lower.startswith("track_"):
        try:
            track_num = int(lower.split("_")[1])
            if track_num == 10:
                return instrument.Percussion(), 0  # Channel 10 percussion
            elif track_num == 1:
                return instrument.Piano(), 0
        except (IndexError, ValueError):
            pass

    # Handle unknown/generic names
    if lower in ["unknown", "unknowninstrument", "instrument"]:
        return instrument.Piano(), 0

    # Final fallback
    generic_inst = instrument.Instrument()
    generic_inst.instrumentName = track_name
    return generic_inst, 0

# -----------------------------
# Dict -> music21 Score (decoder)
# -----------------------------
def dict_to_score(data, bpm=120):
    """
    Convert {track_name: [tokens]} to a music21 Score.
    Enhanced version with better instrument handling.
    Returns both the score and a dict of {track_name: prog_num} for FluidSynth.
    """
    score = stream.Score()
    score.insert(0, metadata.Metadata())
    score.metadata.title = "Generated Composition"
    score.insert(0, tempo.MetronomeMark(number=bpm))

    program_map = {}  # Store MIDI program numbers for each track

    for track_name, tokens in data.items():
        # print(f"Creating part for: {track_name}")
        
        part = stream.Part()
        part.id = track_name
        
        # Choose instrument -> returns (music21 object, program number)
        inst_obj, prog_num = _choose_instrument_for_name(track_name)
        # print(f"  -> Using instrument: {inst_obj.__class__.__name__} (Program {prog_num})")
        
        # Set additional properties if available
        if hasattr(inst_obj, 'instrumentName') and not inst_obj.instrumentName:
            inst_obj.instrumentName = track_name
        
        # Insert only the music21 instrument object into the part
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
        
        # print(f"  -> Added {note_count} notes/rests")
        score.append(part)

        # Save program number for FluidSynth
        program_map[track_name] = prog_num
    
    print(f"Score created with {len(score.parts)} parts")
    return score, program_map

# -----------------------------
# Decode tokens to Audio (FluidSynth via midi2audio)
# -----------------------------
def decode_to_audio(data_or_path, soundfont_path = "soundfonts/AegeanSymphonicOrchestra-SND.sf2",
                    output_dir="test_audio", base_name="output", bpm=120):
    """
    Decode a token dictionary or JSON file into audio (WAV).
    Uses FluidSynth (via midi2audio) with a high-quality SoundFont for playback.
    Ensures each part is assigned the correct GM program from program_map.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Determine next available number
    existing_files = [f for f in os.listdir(output_dir) if f.startswith(base_name) and f.endswith(".wav")]
    numbers = []
    for f in existing_files:
        parts = f.rstrip(".wav").split("_")
        if len(parts) > 1 and parts[-1].isdigit():
            numbers.append(int(parts[-1]))
    next_number = max(numbers, default=0) + 1
    output_wav = os.path.join(output_dir, f"{base_name}_{next_number}.wav")

    # Load data
    if isinstance(data_or_path, str):
        print(f"Loading data from: {data_or_path}")
        print("-"*55)
        with open(data_or_path, "r") as file:
            data = json.load(file)
    else:
        data = data_or_path

    print(f"Input data contains {len(data)} instruments:")
    for track_name, tokens in data.items():
        print(f"  - {track_name}: {len(tokens)} tokens")

    print("-"*55)
    
    
    # Convert dictionary to score and get program numbers
    score, program_map = dict_to_score(data, bpm)

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
        
        # Normalize the audio afterwards
        sound = AudioSegment.from_wav(output_wav)
        normalized = sound.normalize()
        normalized.export(output_wav, format="wav")

        print("-"*55)
        print("Instrument program mapping used:")
        for track, prog in program_map.items():
            print(f"  {track}: Program {prog}")
            
        print("-"*55)

        print(f"Successfully decoded and saved audio as {output_wav} with bpm = {bpm}")
        return output_wav
    except Exception as e:
        print(f"Error rendering audio with FluidSynth: {e}")
        raise
