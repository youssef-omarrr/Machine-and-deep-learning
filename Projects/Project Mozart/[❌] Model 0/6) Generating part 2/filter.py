import re

def is_musical_token(token):
    """Check if token is a valid full musical token"""

    # Allowed duration symbols
    duration_symbols = [
        "q", "e", "h", "w", "s", "t",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "3.0", "4.0",
        "1/4", "1/8", "1/16", "3/4", "3/8", "1/3", "2/3", "1/6"
    ]

    # Build regex dynamically from allowed durations
    duration_pattern = "|".join(map(re.escape, duration_symbols))

    # Full note (letter + optional accidental + octave + duration)
    note_pattern = rf'^[A-G](?:[#b♯♭])?[0-9]+_(?:{duration_pattern})$'
    # Full rest (Rest_ + duration)
    rest_pattern = rf'^(?:Rest|rest|REST|r)_(?:{duration_pattern})$'

    # Special tokens
    special_tokens = [
        '<|startofpiece|>', '<|endofpiece|>', '<TRACKS>', '<TRACKSEP>',
        '<NAME=', '<BPM=', '<DURATION_BEATS=', '<DURATION_MINUTES=', '<MASK>'
    ]

    # ✅ Valid if it matches a note or a rest
    if re.match(note_pattern, token) or re.match(rest_pattern, token):
        return True

    # ✅ Valid if it’s an exact special token or one of the "prefix=" types
    if (token in special_tokens or
        any(token.startswith(prefix) for prefix in [
            '<NAME=', '<BPM=', '<DURATION_BEATS=', '<DURATION_MINUTES='
        ])):
        return True

    # ✅ Valid if it’s an instrument name like "piano:" or "violin:"
    if token.endswith(':') and len(token) > 1:
        return True

    return False

