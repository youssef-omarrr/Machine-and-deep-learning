from pathlib import Path
from functools import lru_cache

UNIQUE_NOTES_FILE = Path("../../dataset/unique_notes.txt")


def load_unique_tokens(path: Path = UNIQUE_NOTES_FILE):
    """Load unique tokens from file, one-per-line, ignoring blanks."""
    if not path.exists():
        print(f"Warning: unique tokens file not found at {path}. No extra tokens will be added.")
        return []
    
    toks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                toks.append(t)

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    return uniq


@lru_cache(maxsize=1)
def get_unique_tokens():
    """Cached loader to avoid re-reading the file repeatedly."""
    return load_unique_tokens()


def is_music_token(token: str) -> bool:
    """Check if a token is in the unique music token list."""
    return token in get_unique_tokens()
