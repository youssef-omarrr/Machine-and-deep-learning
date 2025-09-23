import re
from datasets import load_dataset

TRAIN_FILE = "../dataset/train_file_2.txt"
TEST_FILE = "../dataset/test_file_2.txt"

# Enhance the MUSIC_NOTES list
MUSIC_NOTES = [
    # Basic notes with octaves
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    "c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b",
    
    # Rests
    "r", "R", "Rest",
    
    # Note durations
    "_w", "_h", "_q", "_e", "_s", "_t",
    "_0.25", "_0.5", "_0.75", "_1.0", "_1.5", "_2.0", "_3.0", "_4.0",
    "_1/4", "_1/8", "_1/16", "_3/4", "_3/8", "_1/3", "_2/3", "_1/6",
    
    # Special tokens
    "<MASK>",
    "<|startofpiece|>", "<|endofpiece|>", 
    "<NAME=", "><BPM=", "><DURATION_BEATS=", "><DURATION_MINUTES=",
    "<TRACKS>", "<TRACKSEP>",
    
    # Instrument names
    "Piano:", "Guitar:", "Violin:", "Cello:", "Flute:", "Trumpet:", "Drum:",
    
    # Music symbols
    "♯", "♭", "♮", "𝄞", "𝄢", "🎵", "🎶"
]

# Enhance the pre-tokenization function
def pre_tokenize_musical_text(text):
    """Pre-tokenize musical text to preserve musical tokens"""
    # Protect musical tokens by adding a SINGLE space around them
    # Musical note pattern: NoteOctave_Duration (G2_0.75, C4_q, Rest_s)
    text = re.sub(r'([A-G][#b]?[0-9]+_[a-z0-9/\.]+)', r' \1 ', text)
    
    # Rest pattern
    text = re.sub(r'(Rest_[a-z0-9/\.]+)', r' \1 ', text)
    
    # Single rests
    text = re.sub(r'\b(r|R|Rest)\b', r' \1 ', text)
    
    # Instrument names with colons
    text = re.sub(r'([A-Za-z][A-Za-z0-9_#\- ]{0,40}?):', r' \1: ', text)
    
    # Special tokens
    text = re.sub(r'(<\|startofpiece\|>|<\|endofpiece\|>|<TRACKS>|<TRACKSEP>|<MASK>)', r' \1 ', text)
    text = re.sub(r'(<NAME=|<BPM=|<DURATION_BEATS=|<DURATION_MINUTES=)', r' \1', text)
    text = re.sub(r'(><)', r' >< ', text)
    
    # CRITICAL: Clean up multiple spaces to single spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing spaces
    text = text.strip()
    
    return text

def enhance_training_data(dataset):
    """Pre-tokenize the dataset before GPT-2 tokenization"""
    def pre_tokenize_examples(examples):
        pre_tokenized_texts = []
        for text in examples["text"]:
            pre_tokenized_texts.append(pre_tokenize_musical_text(text))
        return {"text": pre_tokenized_texts}
    
    return dataset.map(
        pre_tokenize_examples,
        batched=True,
        desc="Pre-tokenizing musical data"
    )
    
def prepare_dataset(tokenizer):
    """Enhanced dataset preparation with pre-tokenization"""
    
    dataset = load_dataset("text", data_files={
        "train": TRAIN_FILE,
        "validation": TEST_FILE
    })

    # Pre-tokenize to preserve musical tokens
    dataset = enhance_training_data(dataset)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing pre-tokenized dataset"
    )

    return tokenized
