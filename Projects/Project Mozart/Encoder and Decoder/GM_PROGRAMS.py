# GM_PROGRAMS.py
"""
Enhanced General MIDI program mapping with music21 instrument-class links.
Extended beyond standard 128 GM programs to include more diverse instruments.

Each entry in GM_PROGRAMS maps:
    program_number -> (music21_instrument_class_or_None, "Human readable name")

Use:
    from GM_PROGRAMS import GM_PROGRAMS, get_instrument_class, get_instrument_name

Notes:
- Standard GM programs 0-127 are preserved for compatibility
- Extended programs 128+ provide more instrument diversity
- If the class is None, music21 doesn't supply a matching Instrument class;
    still pass the program number to FluidSynth — it will select the proper SoundFont patch.
"""

# ================================================================== #
# ========== HELPER FUNCTION ===============
# ================================================================== #

import os
def name_file(output_dir, safe_name, extension, numerate = True):
    """
    Generate a unique filename in output_dir using safe_name and extension.
    Ensures incrementing suffix (_1, _2, ...) if duplicates exist.
    
    Args:
        output_dir (str): Directory where file will be saved.
        safe_name (str): Base name (already sanitized).
        extension (str): File extension, with or without leading dot (e.g., "txt" or ".wav").
    
    Returns:
        str: Full path of the next available filename.
    """
    # Normalize extension to include dot
    if not extension.startswith("."):
        extension = "." + extension

    os.makedirs(output_dir, exist_ok=True)

    # Find existing files with this pattern
    if numerate:
        existing_files = [f for f in os.listdir(output_dir) if f.startswith(safe_name) and f.endswith(extension)]
        numbers = []
        for f in existing_files:
            parts = f[:-len(extension)].split("_")  # remove extension safely
            if len(parts) > 1 and parts[-1].isdigit():
                numbers.append(int(parts[-1]))
        next_number = max(numbers, default=0) + 1

        return os.path.join(output_dir, f"{safe_name}_{next_number}{extension}")
    
    else:
        return os.path.join(output_dir, f"{safe_name}{extension}")


from typing import Optional, Tuple, Dict
from music21 import instrument as m21inst

# --- Standard GM names (0-127) ---
GM_NAME_TABLE: Dict[int, str] = {
    0: "Acoustic Grand Piano",
    1: "Bright Acoustic Piano",
    2: "Electric Grand Piano",
    3: "Honky-tonk Piano",
    4: "Electric Piano 1",
    5: "Electric Piano 2",
    6: "Harpsichord",
    7: "Clavi",
    8: "Celesta",
    9: "Glockenspiel",
    10: "Music Box",
    11: "Vibraphone",
    12: "Marimba",
    13: "Xylophone",
    14: "Tubular Bells",
    15: "Dulcimer",
    16: "Drawbar Organ",
    17: "Percussive Organ",
    18: "Rock Organ",
    19: "Church Organ",
    20: "Reed Organ",
    21: "Accordion",
    22: "Harmonica",
    23: "Tango Accordion",
    24: "Acoustic Guitar (nylon)",
    25: "Acoustic Guitar (steel)",
    26: "Electric Guitar (jazz)",
    27: "Electric Guitar (clean)",
    28: "Electric Guitar (muted)",
    29: "Overdriven Guitar",
    30: "Distortion Guitar",
    31: "Guitar Harmonics",
    32: "Acoustic Bass",
    33: "Electric Bass (finger)",
    34: "Electric Bass (pick)",
    35: "Fretless Bass",
    36: "Slap Bass 1",
    37: "Slap Bass 2",
    38: "Synth Bass 1",
    39: "Synth Bass 2",
    40: "Violin",
    41: "Viola",
    42: "Cello",
    43: "Contrabass",
    44: "Tremolo Strings",
    45: "Pizzicato Strings",
    46: "Orchestral Harp",
    47: "Timpani",
    48: "String Ensemble 1",
    49: "String Ensemble 2",
    50: "Synth Strings 1",
    51: "Synth Strings 2",
    52: "Choir Aahs",
    53: "Voice Oohs",
    54: "Synth Choir",
    55: "Orchestra Hit",
    56: "Trumpet",
    57: "Trombone",
    58: "Tuba",
    59: "Muted Trumpet",
    60: "French Horn",
    61: "Brass Section",
    62: "Synth Brass 1",
    63: "Synth Brass 2",
    64: "Soprano Sax",
    65: "Alto Sax",
    66: "Tenor Sax",
    67: "Baritone Sax",
    68: "Oboe",
    69: "English Horn",
    70: "Bassoon",
    71: "Clarinet",
    72: "Piccolo",
    73: "Flute",
    74: "Recorder",
    75: "Pan Flute",
    76: "Blown Bottle",
    77: "Shakuhachi",
    78: "Whistle",
    79: "Ocarina",
    80: "Lead 1 (square)",
    81: "Lead 2 (sawtooth)",
    82: "Lead 3 (calliope)",
    83: "Lead 4 (chiff)",
    84: "Lead 5 (charang)",
    85: "Lead 6 (voice)",
    86: "Lead 7 (fifths)",
    87: "Lead 8 (bass + lead)",
    88: "Pad 1 (new age)",
    89: "Pad 2 (warm)",
    90: "Pad 3 (polysynth)",
    91: "Pad 4 (choir)",
    92: "Pad 5 (bowed)",
    93: "Pad 6 (metallic)",
    94: "Pad 7 (halo)",
    95: "Pad 8 (sweep)",
    96: "FX 1 (rain)",
    97: "FX 2 (soundtrack)",
    98: "FX 3 (crystal)",
    99: "FX 4 (atmosphere)",
    100: "FX 5 (brightness)",
    101: "FX 6 (goblins)",
    102: "FX 7 (echoes)",
    103: "FX 8 (sci-fi)",
    104: "Sitar",
    105: "Banjo",
    106: "Shamisen",
    107: "Koto",
    108: "Kalimba",
    109: "Bagpipe",
    110: "Fiddle",
    111: "Shanai",
    112: "Tinkle Bell",
    113: "Agogo",
    114: "Steel Drums",
    115: "Woodblock",
    116: "Taiko Drum",
    117: "Melodic Tom",
    118: "Synth Drum",
    119: "Reverse Cymbal",
    120: "Guitar Fret Noise",
    121: "Breath Noise",
    122: "Seashore",
    123: "Bird Tweet",
    124: "Telephone Ring",
    125: "Helicopter",
    126: "Applause",
    127: "Gunshot",
    
    # Extended instruments (128+) for more diversity
    128: "Clavichord",
    129: "Sampler",
    130: "Electric Piano",
    131: "Pipe Organ",
    132: "Electric Organ",
    133: "Mandolin",
    134: "Ukulele",
    135: "Lute",
    136: "Bass Clarinet",
    137: "Contrabassoon",
    138: "Soprano Saxophone",
    139: "Bass Trombone",
    140: "Horn",
    141: "Church Bells",
    142: "Handbells",
    143: "Gong",
    144: "Steel Drum",
    145: "Snare Drum",
    146: "Bass Drum",
    147: "Tom-Tom",
    148: "Cymbals",
    149: "Triangle",
    150: "Cowbell",
    151: "Tambourine",
    152: "Bongo Drums",
    153: "Conga Drum",
    154: "Timbales",
    155: "Maracas",
    156: "Castanets",
    157: "Woodblock (High)",
    158: "Temple Block",
    159: "Vibraslap",
    160: "Whip",
    161: "Ratchet",
    162: "Siren",
    163: "Wind Machine",
    164: "Soprano Voice",
    165: "Mezzo-Soprano Voice",
    166: "Alto Voice", 
    167: "Tenor Voice",
    168: "Baritone Voice",
    169: "Bass Voice",
    170: "Choir",
    171: "Hi-Hat Cymbal",
    172: "Crash Cymbals",
    173: "Ride Cymbals",
    174: "Splash Cymbals",
    175: "Suspended Cymbal",
    176: "Sizzle Cymbal",
    177: "Finger Cymbals",
    178: "Tam-Tam",
    179: "Sleigh Bells",
    180: "Tenor Drum",
    181: "Sandpaper Blocks",
}

# --- Try to get music21's built-in mapping if present ---
MIDI_PROGRAM_TO_INSTRUMENT = getattr(m21inst, "MIDI_PROGRAM_TO_INSTRUMENT", {})

# --- Extended instrument class mappings ---
EXTENDED_INSTRUMENT_CLASSES = {
    128: m21inst.Clavichord,
    129: m21inst.Sampler,
    130: m21inst.ElectricPiano,
    131: m21inst.PipeOrgan,
    132: m21inst.ElectricOrgan,
    133: m21inst.Mandolin,
    134: m21inst.Ukulele,
    135: m21inst.Lute,
    136: m21inst.BassClarinet,
    137: m21inst.Contrabassoon,
    138: m21inst.SopranoSaxophone,
    139: m21inst.BassTrombone,
    140: m21inst.Horn,
    141: m21inst.ChurchBells,
    142: m21inst.Handbells,
    143: m21inst.Gong,
    144: m21inst.SteelDrum,
    145: m21inst.SnareDrum,
    146: m21inst.BassDrum,
    147: m21inst.TomTom,
    148: m21inst.Cymbals,
    149: m21inst.Triangle,
    150: m21inst.Cowbell,
    151: m21inst.Tambourine,
    152: m21inst.BongoDrums,
    153: m21inst.CongaDrum,
    154: m21inst.Timbales,
    155: m21inst.Maracas,
    156: m21inst.Castanets,
    157: m21inst.Woodblock,
    158: m21inst.TempleBlock,
    159: m21inst.Vibraslap,
    160: m21inst.Whip,
    161: m21inst.Ratchet,
    162: m21inst.Siren,
    163: m21inst.WindMachine,
    164: m21inst.Soprano,
    165: m21inst.MezzoSoprano,
    166: m21inst.Alto,
    167: m21inst.Tenor,
    168: m21inst.Baritone,
    169: m21inst.Bass,
    170: m21inst.Choir,
    171: m21inst.HiHatCymbal,
    172: m21inst.CrashCymbals,
    173: m21inst.RideCymbals,
    174: m21inst.SplashCymbals,
    175: m21inst.SuspendedCymbal,
    176: m21inst.SizzleCymbal,
    177: m21inst.FingerCymbals,
    178: m21inst.TamTam,
    179: m21inst.SleighBells,
    180: m21inst.TenorDrum,
    181: m21inst.SandpaperBlocks,
}

# --- Build final mapping: program -> (class_or_None, name) ---
GM_PROGRAMS: Dict[int, Tuple[Optional[type], str]] = {}

# Standard GM programs (0-127)
for prog in range(128):
    cls = MIDI_PROGRAM_TO_INSTRUMENT.get(prog)  # may be None
    GM_PROGRAMS[prog] = (cls, GM_NAME_TABLE.get(prog, f"Program_{prog}"))

# Extended programs (128+)
for prog, name in GM_NAME_TABLE.items():
    if prog >= 128:
        cls = EXTENDED_INSTRUMENT_CLASSES.get(prog)
        GM_PROGRAMS[prog] = (cls, name)

# --- Convenience lookups ---
GM_NAME_MAP: Dict[int, str] = {k: v[1] for k, v in GM_PROGRAMS.items()}
GM_CLASS_MAP: Dict[int, Optional[type]] = {k: v[0] for k, v in GM_PROGRAMS.items()}

# --- Reverse lookup: instrument class to program number ---
CLASS_TO_PROGRAM: Dict[type, int] = {}
for prog, (cls, name) in GM_PROGRAMS.items():
    if cls is not None and cls not in CLASS_TO_PROGRAM:
        CLASS_TO_PROGRAM[cls] = prog

# --- Smart instrument assignment for similar instruments ---
def get_best_program_for_instrument_name(instrument_name: str, used_programs: set = None) -> int:
    """
    Get the best program number for an instrument name, trying to avoid duplicates.
    
    Args:
        instrument_name: Name of the instrument (case-insensitive)
        used_programs: Set of already used program numbers to avoid
    
    Returns:
        int: Best program number for the instrument
    """
    if used_programs is None:
        used_programs = set()
    
    instrument_name = instrument_name.lower()
    
    # Define instrument families and their program variations
    guitar_programs = [24, 25, 26, 27, 28, 29, 30, 31]  # Various guitar types
    bass_programs = [32, 33, 34, 35, 36, 37, 38, 39]    # Various bass types
    piano_programs = [0, 1, 2, 3, 4, 5, 128, 130]      # Piano variations + extended
    organ_programs = [16, 17, 18, 19, 20, 131, 132]    # Organ variations
    drum_programs = [145, 146, 147, 180]                # Drum variations
    cymbal_programs = [148, 171, 172, 173, 174, 175, 176, 177]  # Cymbal variations
    
    # Guitar family
    if any(word in instrument_name for word in ['guitar', 'gtr']):
        if 'electric' in instrument_name:
            candidates = [26, 27, 28, 29, 30, 31]
        elif 'acoustic' in instrument_name:
            candidates = [24, 25]
        else:
            candidates = guitar_programs
        
        for prog in candidates:
            if prog not in used_programs:
                return prog
        return guitar_programs[len(used_programs) % len(guitar_programs)]
    
    # Bass family
    elif any(word in instrument_name for word in ['bass', 'b dr']):
        if 'drum' in instrument_name:
            return 146  # Bass Drum
        elif 'electric' in instrument_name:
            candidates = [33, 34]
        elif 'acoustic' in instrument_name:
            candidates = [32]
        elif 'fretless' in instrument_name:
            candidates = [35]
        else:
            candidates = bass_programs
            
        for prog in candidates:
            if prog not in used_programs:
                return prog
        return bass_programs[len(used_programs) % len(bass_programs)]
    
    # Piano family
    elif any(word in instrument_name for word in ['piano', 'pno']):
        if 'electric' in instrument_name:
            candidates = [2, 4, 5, 130]
        else:
            candidates = [0, 1, 3]
            
        for prog in candidates:
            if prog not in used_programs:
                return prog
        return piano_programs[len(used_programs) % len(piano_programs)]
    
    # Drum family
    elif any(word in instrument_name for word in ['drum', 'snare', 'tom']):
        if 'snare' in instrument_name:
            return 145
        elif 'bass' in instrument_name:
            return 146
        elif 'tom' in instrument_name:
            return 147
        else:
            candidates = drum_programs
            for prog in candidates:
                if prog not in used_programs:
                    return prog
            return drum_programs[len(used_programs) % len(drum_programs)]
    
    # Cymbal family
    elif any(word in instrument_name for word in ['cymbal', 'cym']):
        for prog in cymbal_programs:
            if prog not in used_programs:
                return prog
        return cymbal_programs[len(used_programs) % len(cymbal_programs)]
    
    # Default fallback - find exact or partial match
    for prog, name in GM_NAME_MAP.items():
        if instrument_name in name.lower() and prog not in used_programs:
            return prog
    
    # Last resort - return a reasonable default based on common instruments
    if any(word in instrument_name for word in ['flute', 'fl']):
        return 73
    elif any(word in instrument_name for word in ['trumpet', 'tpt']):
        return 56
    elif any(word in instrument_name for word in ['violin', 'vln']):
        return 40
    elif any(word in instrument_name for word in ['cello', 'vc']):
        return 42
    else:
        return 0  # Default to piano

# --- Helper functions ---
def get_instrument_class(program: int) -> Optional[type]:
    """Return the music21 Instrument class for the given GM program (or None)."""
    return GM_CLASS_MAP.get(int(program))

def get_instrument_name(program: int) -> str:
    """Return the human readable name for the program (falls back to 'Program_N')."""
    return GM_NAME_MAP.get(int(program), f"Program_{int(program)}")

def make_instrument_instance(program: int, *args, **kwargs):
    """
    Instantiate and return a music21 instrument object for program if available.
    Returns None if music21 doesn't provide a class for that program.
    """
    cls = get_instrument_class(program)
    if cls is None:
        return None
    try:
        return cls(*args, **kwargs)
    except Exception:
        # Defensive: if instantiation fails for some reason, return None
        return None

def get_program_for_instrument(instrument_obj) -> int:
    """
    Get the best program number for a music21 instrument object.
    Returns the program number from the reverse lookup, or a smart assignment.
    """
    if hasattr(instrument_obj, 'midiProgram') and instrument_obj.midiProgram is not None:
        return instrument_obj.midiProgram
    
    instrument_class = type(instrument_obj)
    if instrument_class in CLASS_TO_PROGRAM:
        return CLASS_TO_PROGRAM[instrument_class]
    
    # Fallback to name-based assignment
    instrument_name = getattr(instrument_obj, 'instrumentName', '') or str(instrument_obj)
    return get_best_program_for_instrument_name(instrument_name)

# Export names
__all__ = [
    "GM_PROGRAMS", "GM_NAME_MAP", "GM_CLASS_MAP", "CLASS_TO_PROGRAM",
    "get_instrument_class", "get_instrument_name", "make_instrument_instance",
    "get_best_program_for_instrument_name", "get_program_for_instrument"
]