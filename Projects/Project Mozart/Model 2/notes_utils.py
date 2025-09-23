# ------------------------------------------------------------
# NOTE → MIDI NUMBER
# ------------------------------------------------------------
NOTE_TO_SEMITONE = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
    'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}


def note_name_to_midi(note):
    import re
    m = re.match(r'^([A-G][#b]?)(-?\d+)$', note)
    if not m:
        raise ValueError(f"Invalid note name: {note}")
    name, octave = m.group(1), int(m.group(2))
    return 12 * (octave + 1) + NOTE_TO_SEMITONE[name]


# ------------------------------------------------------------
# VOCAB HELPERS
# ------------------------------------------------------------
def build_vocab_maps(tokenizer):
    vocab = tokenizer.vocab
    id_to_token = {int(v): k for k, v in vocab.items()}
    return vocab, id_to_token


def find_token_key_starting_with(vocab, prefix):
    if prefix in vocab:
        return prefix
    for k in vocab:
        if k.startswith(prefix):
            return k
    return None


def choose_velocity_token(vocab, target_vel=90):
    vel_tokens = []
    for k in vocab:
        if k.startswith("Velocity_"):
            try:
                n = int(k.split("_", 1)[1])
                vel_tokens.append((n, k))
            except Exception:
                pass
    if not vel_tokens:
        return None
    vel_tokens.sort(key=lambda x: abs(x[0] - target_vel))
    return vel_tokens[0][1]


# ------------------------------------------------------------
# NOTE SYMBOL → TOKENS
# ------------------------------------------------------------
def note_token_ids_from_symbolic(tokenizer, note_symbol, default_velocity=90):
    vocab, _ = build_vocab_maps(tokenizer)

    note_part, dur_code = note_symbol.split("_", 1)
    midi = note_name_to_midi(note_part)
    pitch_key = find_token_key_starting_with(vocab, f"Pitch_{midi}")

    dur_map = {'w': '4.0', 'h': '2.0', 'q': '1.0', 'e': '0.5', 's': '0.25'}
    duration_prefix = f"Duration_{dur_map.get(dur_code, dur_code)}"
    duration_token = find_token_key_starting_with(vocab, duration_prefix)

    velocity_token = choose_velocity_token(vocab, target_vel=default_velocity)

    return [int(vocab[pitch_key]), int(vocab[velocity_token]), int(vocab[duration_token])]
