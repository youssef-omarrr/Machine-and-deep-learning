import re
import os
import time
import random
from typing import Optional, Tuple, Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ------------------------------------------------------------
# CONFIG (change these paths if needed)
# ------------------------------------------------------------
DEFAULT_BASE_MODEL = "../MODELS/gpt2-medium-local/"            # local base model folder (saved_pretrained)
SAVED_MODEL_DIR = "../MODELS/Project_Mozart_gpt2-medium"        # where LoRA adapters live
OUTPUT_GENERATIONS_DIR = "../example_outputs/model_output"      # where to save generated .txt files
DEFAULT_BPM = 120                                               # fallback if we must pick a BPM
CACHE_DIR = "../MODELS/"                                        # force HF cache here to avoid C: usage

os.makedirs(OUTPUT_GENERATIONS_DIR, exist_ok=True)

# ------------------------------------------------------------
# Helpers: parse user instruction
# ------------------------------------------------------------
def _parse_user_instruction(text: str) -> Dict[str, Any]:
    t = text.lower()
    bpm_match = re.search(r"(\d{2,3})(?:\s*)(?:bpm\b)?", t)
    bpm = None
    strict_bpm = False
    if bpm_match:
        span = bpm_match.span()
        window = t[max(0, span[0]-12): span[1]+12]
        bpm = int(bpm_match.group(1))
        strict_bpm = not bool(re.search(r"\baround\b|\bapprox\b|\bapproximately\b|\b~\b", window))

    duration_minutes = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:minutes|minute|mins|min)\b", t)
    if m:
        duration_minutes = float(m.group(1))
    else:
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds|second|secs|sec)\b", t)
        if m2:
            duration_minutes = float(m2.group(1)) / 60.0

    instruments = []
    instrument_list = [
        "piano", "violin", "cello", "flute", "clarinet", "guitar",
        "drum", "drums", "saxophone", "trumpet", "organ", "strings",
        "synth", "bass", "harp", "orchestra", "many instruments"
    ]
    for inst in instrument_list:
        if re.search(r"\b" + re.escape(inst) + r"\b", t):
            instruments.append(inst)
    if instruments == []:
        instruments = None

    return {
        "bpm": bpm,
        "strict_bpm": strict_bpm,
        "duration_minutes": duration_minutes,
        "duration_beats": None,
        "instruments": instruments,
        "raw_text": text,
    }

# ------------------------------------------------------------
# Helpers: durations <-> beats conversion
# The dataset uses tokens like _q, _e, _s, fractions and decimals.
# We interpret (common music notation):
#   w = whole (4 beats), h = half (2), q = quarter (1), e = eighth (0.5), s = sixteenth (0.25)
# For numeric suffix like _1.25 or _3/4 we parse numerically.
# ------------------------------------------------------------
DUR_SYMBOL_TO_BEATS = {
    "w": 4.0,
    "h": 2.0,
    "q": 1.0,
    "e": 0.5,
    "s": 0.25,
}

_fraction_re = re.compile(r"^(\d+)\s*/\s*(\d+)$")
_decimal_re = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")

def parse_duration_token_to_beats(s: str) -> Optional[float]:
    """
    parse strings like: '_q', '_e', '_s', '_1.25', '_3/4', '_0.75'
    returns beats (float) or None if unknown.
    """
    if s is None:
        return None
    s = s.strip()
    if s.startswith("_"):
        s = s[1:]
    if not s:
        return None
    # symbol
    if s in DUR_SYMBOL_TO_BEATS:
        return DUR_SYMBOL_TO_BEATS[s]
    # decimal
    if _decimal_re.match(s):
        return float(s)
    # fraction
    m = _fraction_re.match(s)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den != 0:
            return num / den
    # sometimes token like '1/3' or '2/3' used — parsed above
    return None

# token is something like 'C5_1.25' or 'Rest_1/3'
_token_duration_re = re.compile(r"_(?P<dur>[0-9./a-zA-Z_-]+)$")

def token_beats_estimate(token: str) -> float:
    """
    Estimate beats for a single token string.
    If token contains an explicit duration suffix (e.g., '_q', '_1.25', '_1/3') parse it.
    If no duration suffix found, fallback to 1.0 beat (conservative).
    """
    if token is None:
        return 0.0
    m = _token_duration_re.search(token)
    if m:
        dur_str = m.group("dur")
        b = parse_duration_token_to_beats("_" + dur_str)  # reuse parser
        if b is not None:
            return b
    # often notes include no suffix (treated as quarter)
    # conservative fallback: treat unspecified as 1 beat
    return 1.0

# ------------------------------------------------------------
# STEP 1: Build structured prompt (same token format as training)
# ------------------------------------------------------------
def build_structured_prompt(
    tokenizer,
    *,
    name: Optional[str] = None,
    bpm: Optional[float] = None,
    duration_beats: Optional[int] = None,
    duration_minutes: Optional[float] = None,
    instruments: Optional[List[str]] = None,
    prompt_tracks: Optional[str] = None,
) -> Tuple[str, Dict[str,Any]]:
    if name is None:
        name = f"Generated_{random.randint(1000,9999)}"

    computed_beats = duration_beats
    if duration_minutes is not None and bpm is not None and computed_beats is None:
        computed_beats = int(round(duration_minutes * bpm))

    computed_minutes = duration_minutes
    if computed_beats is not None and duration_minutes is None and bpm is not None:
        computed_minutes = round(computed_beats / bpm, 3)

    tracks_snippet = None
    if prompt_tracks:
        tracks_snippet = prompt_tracks
    elif instruments:
        lines = []
        for inst in instruments:
            lines.append(f"{inst.capitalize()}: Rest_q Rest_q Rest_q")
        tracks_snippet = " <TRACKSEP> ".join(lines)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    parts = ["<|startofpiece|>"]
    parts.append(f"<NAME={name}>")
    if bpm is not None:
        parts.append(f"<BPM={int(bpm)}>")
        final_bpm = float(bpm)
    else:
        parts.append("<BPM=>")
        final_bpm = None

    if computed_beats is not None:
        parts.append(f"<DURATION_BEATS={int(computed_beats)}>")
    if computed_minutes is not None:
        parts.append(f"<DURATION_MINUTES={float(computed_minutes):.2f}>")

    parts.append("<TRACKS>")
    if tracks_snippet:
        parts.append(tracks_snippet)

    prompt = "".join(parts)
    meta = {
        "name": name,
        "bpm": final_bpm,
        "duration_beats": computed_beats,
        "duration_minutes": computed_minutes,
        "instruments": instruments,
    }
    return prompt, meta

# ------------------------------------------------------------
# STEP 2: Load model + tokenizer (local-only, resize embeddings, attach PEFT)
# ------------------------------------------------------------
def load_model_and_tokenizer(
    saved_model_dir: str = SAVED_MODEL_DIR,
    base_model_name: str = DEFAULT_BASE_MODEL,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenizer: prefer saved dir, else base
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            saved_model_dir,
            use_fast=True,
            local_files_only=True,
            cache_dir=CACHE_DIR,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            use_fast=True,
            local_files_only=True,
            cache_dir=CACHE_DIR,
        )

    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    elif device.startswith("cuda"):
        kwargs["dtype"] = torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        local_files_only=True,
        cache_dir=CACHE_DIR,
        **kwargs
    )

    # important: resize embeddings to match tokenizer (fixes mismatch with adapter snapshots)
    base_model.resize_token_embeddings(len(tokenizer))

    # attach LoRA adapters if available
    model = base_model
    if os.path.isdir(saved_model_dir) and os.listdir(saved_model_dir):
        try:
            model = PeftModel.from_pretrained(base_model, saved_model_dir, local_files_only=True)
        except Exception as e:
            print("Warning: failed to load PEFT adapters, continuing with base model:", e)

    model.to(device)
    model.eval()
    return model, tokenizer

# ------------------------------------------------------------
# STEP 3: Strict generation logic
# - We iteratively generate chunks, parse out musical tokens only,
#   estimate cumulative beats using token suffixes and stop when target reached.
# - We aggressively strip non-musical prose and anything after <|endofpiece|>.
# ------------------------------------------------------------
# regex to extract the piece between start and end
_piece_between_re = re.compile(r"<\|startofpiece\|>(.*?)(?:<\|endofpiece\|>|$)", re.DOTALL)

# token-splitting: split by whitespace and commas; keep punctuation that belongs to token
_split_token_re = re.compile(r"[,\n\r]+|\s+")

# allowed token patterns roughly matching dataset tokens:
_allowed_token_re = re.compile(
    r"^("                                   # start
    r"<\|startofpiece\|>|<\|endofpiece\|>"  # special markers
    r"|<NAME=[^>]+>|<BPM=[0-9.]*>|<DURATION_BEATS=[0-9.]*>|<DURATION_MINUTES=[0-9.]*>"
    r"|<TRACKS>|<TRACKSEP>"
    r"|[A-G][#b]?[0-9]?(\.[A-G0-9#b]+)?(?:_[0-9./a-zA-Z-]+)?"
    r"|Rest(?:_[0-9./a-zA-Z-]+)?"
    r"|[0-9./]+"
    r")$"
)

def _filter_and_extract_piece_text(raw_text: str) -> str:
    """
    Extract the first <|startofpiece|> ... <|endofpiece|> block and remove trailing prose.
    Also attempt to remove any obviously non-musical lines (english sentences).
    Returns cleaned piece text (may be partial).
    """
    # take piece block if present
    m = _piece_between_re.search(raw_text)
    if m:
        block = m.group(0)  # include markers
    else:
        # fallback: use everything before any newline that starts english prose.
        block = raw_text

    # split into tokens, keep only allowed tokens and separators
    parts = _split_token_re.split(block)
    kept = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if _allowed_token_re.match(p):
            kept.append(p)
        else:
            # sometimes the model outputs punctuation or fragments: try to salvage numeric durations
            if re.match(r"^[0-9./]+$", p):
                kept.append(p)
            # otherwise skip
    # join using single space and re-insert markers if missing
    text = " ".join(kept)
    if "<|startofpiece|>" not in text:
        text = "<|startofpiece|> " + text
    if "<|endofpiece|>" not in text:
        text = text + " <|endofpiece|>"
    return text

def _tokens_from_piece_text(text: str) -> List[str]:
    # remove markers then split on whitespace; keep markers as tokens
    toks = []
    for p in _split_token_re.split(text):
        p = p.strip()
        if not p:
            continue
        toks.append(p)
    return toks

def _estimate_total_beats_from_tokens(tokens: List[str]) -> float:
    beats = 0.0
    # only consider tokens after <TRACKS> mark
    try:
        track_idx = tokens.index("<TRACKS>")
    except ValueError:
        track_idx = None
    idx_start = (track_idx + 1) if track_idx is not None else 0
    for t in tokens[idx_start:]:
        if t.startswith("<"):
            continue
        beats += token_beats_estimate(t)
    return beats

# generation core
def generate_piece_strict(
    model,
    tokenizer,
    *,
    prompt: str,
    target_beats: Optional[float] = None,
    chunk_max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.05,
    device: Optional[str] = None,
) -> str:
    """
    Incremental generation: generate chunks and accumulate musical tokens until
    estimated beats >= target_beats (if provided). Returns cleaned piece text.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenize prompt with attention_mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=False)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    generated_ids = input_ids  # will append to this
    accumulated_piece_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    # if no target_beats -> generate single chunk and then clean strictly
    if target_beats is None:
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=chunk_max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(out[0], skip_special_tokens=False)
        cleaned = _filter_and_extract_piece_text(raw)
        return cleaned

    # otherwise iterative loop: generate in chunks until beats >= target
    max_rounds = 800  # safety cap to prevent runaway loops
    rounds = 0
    last_len = generated_ids.size(1)

    while rounds < max_rounds:
        rounds += 1
        with torch.no_grad():
            out = model.generate(
                input_ids=generated_ids,
                attention_mask=None,   # not needed on subsequent calls
                max_new_tokens=chunk_max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        # we only want the newly generated tokens part
        new_ids = out[0][last_len:]
        if new_ids.numel() == 0:
            break
        # append
        generated_ids = torch.cat([generated_ids, new_ids.unsqueeze(0)], dim=1)
        last_len = generated_ids.size(1)
        raw = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        cleaned = _filter_and_extract_piece_text(raw)
        toks = _tokens_from_piece_text(cleaned)

        est_beats = _estimate_total_beats_from_tokens(toks)
        # if we've reached or exceeded target -> stop
        if est_beats >= target_beats:
            # ensure <|endofpiece|> exists
            if "<|endofpiece|>" not in cleaned:
                cleaned = cleaned + " <|endofpiece|>"
            return cleaned
        # else continue; keep loop but break if model repeats or produces nothing
        # safety: if model stalls producing same string, break
        rounds += 0  # (we already incremented)
    # fallback: return current cleaned text
    return _filter_and_extract_piece_text(tokenizer.decode(generated_ids[0], skip_special_tokens=False))

# ------------------------------------------------------------
# STEP 4: User-facing function: parse, build prompt, compute target beats, load model, generate, save
# ------------------------------------------------------------
def generate_from_text(
    user_instruction: str,
    *,
    saved_model_dir: str = SAVED_MODEL_DIR,
    base_model_name: str = DEFAULT_BASE_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.05,
    chunk_max_new_tokens: int = 64,
    enforce_user_bpm: bool = True,
    filename_prefix: Optional[str] = None,
) -> Tuple[str, float, str]:
    """
    Main entrypoint:
    - parse user instruction
    - build structured prompt
    - load model/tokenizer
    - generate until target beats satisfied
    - enforce BPM/duration tokens
    - save to .txt
    Returns (generated_text, final_bpm, filepath)
    """
    parsed = _parse_user_instruction(user_instruction)
    bpm = parsed.get("bpm") or DEFAULT_BPM
    duration_minutes = parsed.get("duration_minutes")
    instruments = parsed.get("instruments")

    duration_beats = None
    if duration_minutes is not None and bpm is not None:
        duration_beats = int(round(duration_minutes * bpm))

    # load model / tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model, tokenizer = load_model_and_tokenizer(
        saved_model_dir=saved_model_dir,
        base_model_name=base_model_name,
        device=device,
        dtype=dtype,
    )

    # Build prompt: we supply BPM and duration tokens to bias the model
    prompt, meta = build_structured_prompt(
        tokenizer,
        bpm=bpm,
        duration_beats=duration_beats,
        duration_minutes=duration_minutes,
        instruments=instruments,
    )

    # Compute required beats. If duration not provided, we generate a default length (e.g., 1 minute)
    if duration_beats is None:
        # if duration not specified, default to 1 minute at BPM
        target_beats = int(round(1 * bpm))
    else:
        target_beats = float(duration_beats)

    # Generate strictly — this returns piece text that contains only allowed tokens
    piece_text = generate_piece_strict(
        model,
        tokenizer,
        prompt=prompt,
        target_beats=target_beats,
        chunk_max_new_tokens=chunk_max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        device=device,
    )

    # Post-enforce BPM token if the user asked for exact BPM
    if parsed.get("bpm") is not None and enforce_user_bpm:
        user_bpm = int(parsed.get("bpm"))
        # replace any BPM token or insert if missing
        if re.search(r"<BPM=[0-9.]+>", piece_text):
            piece_text = re.sub(r"<BPM=[0-9.]+>", f"<BPM={user_bpm}>", piece_text)
        else:
            # try to insert after NAME
            piece_text = re.sub(r"(<NAME=[^>]+>)", r"\1" + f"<BPM={user_bpm}>", piece_text, count=1)

    # ensure DURATION_BEATS and DURATION_MINUTES are consistent (if user asked for minutes)
    if duration_minutes is not None and bpm is not None:
        computed_beats = int(round(duration_minutes * bpm))
        if re.search(r"<DURATION_BEATS=[0-9]+>", piece_text):
            piece_text = re.sub(r"<DURATION_BEATS=[0-9]+>", f"<DURATION_BEATS={computed_beats}>", piece_text)
        else:
            if "<BPM=" in piece_text:
                piece_text = re.sub(r"(<BPM=[0-9.]+>)", r"\1" + f"<DURATION_BEATS={computed_beats}>", piece_text, count=1)
            else:
                piece_text = re.sub(r"(<NAME=[^>]+>)", r"\1" + f"<DURATION_BEATS={computed_beats}>", piece_text, count=1)
        # also add minutes token
        piece_text = re.sub(r"<DURATION_MINUTES=[0-9.]+>", "", piece_text)  # remove existing if any
        piece_text = piece_text.replace("<TRACKS>", f"<DURATION_MINUTES={float(duration_minutes):.2f}><TRACKS>")

    # final ensure markers present
    if "<|startofpiece|>" not in piece_text:
        piece_text = "<|startofpiece|> " + piece_text
    if "<|endofpiece|>" not in piece_text:
        piece_text = piece_text + " <|endofpiece|>"

    # Save to file
    ts = int(time.time())
    prefix = filename_prefix or "gen"
    filename = f"{prefix}_{ts}.txt"
    filepath = os.path.join(OUTPUT_GENERATIONS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(piece_text)

    # attempt to extract bpm from piece text (fallback to the one we used)
    bpm_found = None
    m = re.search(r"<BPM=([0-9.]+)>", piece_text)
    if m:
        bpm_found = float(m.group(1))
    else:
        bpm_found = float(bpm)

    return piece_text, bpm_found, filepath
