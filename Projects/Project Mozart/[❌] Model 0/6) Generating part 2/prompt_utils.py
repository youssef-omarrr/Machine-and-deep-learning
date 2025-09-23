import re
import random

def parse_user_instruction(text: str):
    """
    Enhanced parsing of user instructions with better handling of instruments.
    If 'other instruments' is specified, we add random instruments in addition
    to any explicitly mentioned ones.
    """
    t = text.lower()
    
    # --- Handle piece name ---
    name = None
    match_name = re.search(r"(?:named|title[d]?|called)\s+([a-zA-Z0-9_\- ]+)", t)
    
    if match_name:
        candidate = match_name.group(1).strip()
        
        # Stop at keywords like bpm, minutes, instruments
        stop_words = [
            "bpm", "minute", "minutes", "second", "seconds",
            "instrument", "instruments", "that", "which", "using", "with", "around"
        ]
        
        for sw in stop_words:
            if sw in candidate:
                candidate = candidate.split(sw, 1)[0].strip()
                
        # Clean trailing punctuation
        candidate = re.sub(r"[.,;:!?]+$", "", candidate).strip()
        
        name = candidate if candidate else None

    # Explicit "with any name" -> force None
    if re.search(r"\bwith any name\b", t):
        name = None
    
    # -------------------------------------------------------------------------------------------------------- #
    
    # --- BPM parsing with better context awareness ---
    bpm_match = re.search(r"\b(\d{2,4})\s*bpm\b", t, flags=re.IGNORECASE)
    bpm = None
    strict_bpm = False
    if bpm_match:
        bpm = int(bpm_match.group(1))
        span = bpm_match.span()
        window = t[max(0, span[0]-30): span[1]+30]
        strict_bpm = not bool(re.search(r"\baround\b|\bapprox\b|\bapproximately\b|\b~\b|\babout\b", window, flags=re.IGNORECASE))
        
    # -------------------------------------------------------------------------------------------------------- #

    # --- Duration parsing (minutes and seconds) ---
    duration_minutes = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:minutes|minute|mins|min)\b", t)
    if m:
        duration_minutes = float(m.group(1))
    else:
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds|second|secs|sec)\b", t)
        if m2:
            duration_minutes = float(m2.group(1)) / 60.0

    # -------------------------------------------------------------------------------------------------------- #
    
    # --- Instrument detection ---
    instruments = []
    instrument_patterns = {
        "piano": r"\bpiano\b",
        "violin": r"\bviolin\b",
        "cello": r"\bcello\b",
        "flute": r"\bflute\b|\bflutes\b",
        "clarinet": r"\bclarinet\b",
        "guitar": r"\bguitar\b",
        "drums": r"\bdrum\b|\bdrums\b|\bpercussion\b",
        "saxophone": r"\bsaxophone\b|\bsax\b",
        "trumpet": r"\btrumpet\b",
        "organ": r"\borgan\b",
        "bass": r"\bbass\b",
        "harp": r"\bharp\b",
        "synth": r"\bsynth\b|\bsynthesizer\b",
        "strings": r"\bstrings\b|\bstring section\b",
        "brass": r"\bbrass\b|\bbrass section\b",
        "woodwinds": r"\bwoodwinds\b|\bwoodwind\b",
        "orchestra": r"\borchestra\b|\bfull orchestra\b",
    }
    
    for instrument, pattern in instrument_patterns.items():
        if re.search(pattern, t) and instrument not in instruments:
            instruments.append(instrument)

    # Full instrument pool for random selection
    instrument_pool = list(instrument_patterns.keys())
    
    # -------------------------------------------------------------------------------------------------------- #
    
    # --- Instrument count and special cases ---
    # Handle "N other instruments"
    match_other = re.search(r"(\d+)\s+other instruments?", t)
    if match_other:
        num_other = int(match_other.group(1))
        already = set(instruments)
        available = [inst for inst in instrument_pool if inst not in already]
        additional = random.sample(available, k=min(num_other, len(available)))
        instruments.extend(additional)

    # Handle "only X" or "solo X"
    elif re.search(r"\bsolo\b|\bonly\b", t):
        solo_match = re.search(
            r"(piano|violin|guitar|flute|cello|trumpet|drums|saxophone|clarinet|organ|bass|harp|synth)\b",
            t,
        )
        if solo_match:
            instruments = [solo_match.group(1)]

    # Handle "at least N instruments"
    elif re.search(r"at least (\d+)\s+instruments?", t):
        num = int(re.search(r"at least (\d+)\s+instruments?", t).group(1))
        if len(instruments) < num:
            additional = random.sample(
                [i for i in instrument_pool if i not in instruments],
                k=min(num - len(instruments), len(instrument_pool) - len(instruments))
            )
            instruments.extend(additional)

    # Handle "up to N instruments"
    elif re.search(r"up to (\d+)\s+instruments?", t):
        num = int(re.search(r"up to (\d+)\s+instruments?", t).group(1))
        total = random.randint(1, num)
        if len(instruments) < total:
            additional = random.sample(
                [i for i in instrument_pool if i not in instruments],
                k=min(total - len(instruments), len(instrument_pool) - len(instruments))
            )
            instruments.extend(additional)

    # Handle "between N and M instruments"
    elif re.search(r"between (\d+)\s+and\s+(\d+)\s+instruments?", t):
        n, m = map(int, re.search(r"between (\d+)\s+and\s+(\d+)\s+instruments?", t).groups())
        total = random.randint(n, m)
        if len(instruments) < total:
            additional = random.sample(
                [i for i in instrument_pool if i not in instruments],
                k=min(total - len(instruments), len(instrument_pool) - len(instruments))
            )
            instruments.extend(additional)

    # Handle "many instruments" / "multiple instruments" / "orchestral"
    elif re.search(r"\bmany instruments\b|\bmultiple instruments\b|\borchestral\b|\bfull orchestra\b", t):
        additional = random.sample(
            [i for i in instrument_pool if i not in instruments],
            k=min(random.randint(5, 8), len(instrument_pool) - len(instruments))
        )
        instruments.extend(additional)

    # Handle generic "other instruments"
    elif re.search(r"\bother instruments\b|\bmore than one instrument\b", t):
        additional = random.sample(
            [i for i in instrument_pool if i not in instruments],
            k=min(random.choice([1, 2]), len(instrument_pool) - len(instruments))
        )
        instruments.extend(additional)

    # Handle "N instruments" (explicit total)
    else:
        num_match = re.search(r"\b(\d+)\s+instruments?\b", t)
        if num_match:
            num = int(num_match.group(1))
            if num > 0 and len(instruments) < num:
                additional = random.sample(
                    [i for i in instrument_pool if i not in instruments],
                    k=min(num - len(instruments), len(instrument_pool) - len(instruments))
                )
                instruments.extend(additional)

    # Ensure uniqueness
    instruments = list(dict.fromkeys(instruments))
    
    output_dict = {
        "name": name,
        "bpm": bpm,
        "strict_bpm": strict_bpm,
        "duration_minutes": duration_minutes,
        "instruments": instruments if instruments else None,
        "raw_text": text,
    }
    
    
    # Fill any missing data by random data
    output_dict = fill_missing_metadata(output_dict)

    return output_dict

    # -------------------------------------------------------------------------------------------------------- #

    # -------------------------------------------------------------------------------------------------------- #
    
def fill_missing_metadata(parsed: dict) -> dict:
    """
    Fill in missing fields from parse_user_input with random values.
    """
    name = parsed.get("name") or f"Generated_Symphony_{random.randint(0,500)}"

    bpm = parsed.get("bpm")
    if bpm is None:
        bpm = float(random.randint(60, 200))  # typical musical tempo range

    duration_minutes = parsed.get("duration_minutes")
    duration_beats = parsed.get("duration_beats")

    # if missing, compute or randomize
    if duration_minutes is None and duration_beats is None:
        duration_minutes = round(random.uniform(0.5, 5.0), 2)  # 30 sec – 5 min
        duration_beats = int(duration_minutes * bpm)
    elif duration_minutes is None:
        duration_minutes = round(duration_beats / bpm, 2)
    elif duration_beats is None:
        duration_beats = int(duration_minutes * bpm)

    instruments = parsed.get("instruments")
    if not instruments:
        candidate_instruments = [
            "Piano", "Violin", "Flute", "Clarinet", "Guitar"
        ]
        instruments = random.sample(candidate_instruments, k=random.randint(1, 5))

    return {
        "name": name,
        "bpm": bpm,
        "duration_beats": duration_beats,
        "duration_minutes": duration_minutes,
        "instruments": instruments,
    }

    
    # -------------------------------------------------------------------------------------------------------- #

    # -------------------------------------------------------------------------------------------------------- #

def build_structured_prompt(user_parsed: dict, mask_instruments: bool = True):
    """
    Build a structured music generation prompt from parsed user inputs.
    Optionally mask instruments for the model to generate.
    
    Args:
        user_parsed (dict): dictionary containing keys like:
            - name (str | None)
            - bpm (float | None)
            - duration_beats (int | None)
            - duration_minutes (float | None)
            - instruments (list[str] | None)
            - prompt_tracks (str | None)
        mask_instruments (bool): Whether to mask instruments with <MASK> token
    
    Returns:
        prompt (str): structured string for the model
        meta (dict): metadata dict with final values
    """

    # Extract fields with defaults
    name = user_parsed.get("name")
    bpm = user_parsed.get("bpm")
    duration_beats = user_parsed.get("duration_beats")
    duration_minutes = user_parsed.get("duration_minutes")
    instruments = user_parsed.get("instruments")
    prompt_tracks = user_parsed.get("prompt_tracks")

    # Compute beats or minutes if possible
    computed_beats = duration_beats
    computed_minutes = duration_minutes


    if bpm is not None:
        if duration_minutes is not None and duration_beats is None:
            computed_beats = int(round(duration_minutes * bpm))
        elif duration_beats is not None and duration_minutes is None:
            computed_minutes = round(duration_beats / bpm, 3)
            

    # Build track snippet
    if prompt_tracks:
        tracks_snippet = prompt_tracks
    elif instruments:
        if mask_instruments:
            # Mask all instruments with <MASK> token
            lines = [f"{inst.capitalize()}: <MASK>" for inst in instruments]
        else:
            # Leave instruments empty for model to generate
            lines = [f"{inst.capitalize()}: " for inst in instruments]
        tracks_snippet = " <TRACKSEP> ".join(lines)
    else:
        tracks_snippet = None


    # Build structured string
    parts = ["<|startofpiece|>"]
    parts.append(f"<NAME={name}>")

    if bpm is not None:
        parts.append(f"<BPM={int(bpm)}>")  
    else:
        parts.append("<BPM=120>")  # Added default BPM

    if computed_beats is not None:
        parts.append(f"<DURATION_BEATS={int(computed_beats)}>")  
    else:
        parts.append("<DURATION_BEATS=>")

    if computed_minutes is not None:
        parts.append(f"<DURATION_MINUTES={float(computed_minutes):.2f}>")
    else:
        parts.append("<DURATION_MINUTES=>")

    parts.append("<TRACKS>")
    if tracks_snippet:
        parts.append(tracks_snippet)

    prompt = "".join(parts)

    meta = {
        "name": name,
        "bpm": float(bpm) if bpm is not None else 120.0,  # Added default
        "duration_beats": computed_beats,
        "duration_minutes": computed_minutes,
        "instruments": instruments,
    }

    return prompt, meta