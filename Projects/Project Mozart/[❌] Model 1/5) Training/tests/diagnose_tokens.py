from transformers import AutoTokenizer
from filter import get_unique_tokens

from load_model_tokenizer import load_model_and_tokenizer

def test_tokens():
    # Load your tokenizer
    _, tokenizer = load_model_and_tokenizer()

    # Check special tokens
    print("Special tokens:")
    for token in ["<MASK>", "<|startofpiece|>", "<|endofpiece|>", "<TRACKS>", "<TRACKSEP>"]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {token}: {token_id}")

    # Check music tokens
    print("\nMusic tokens (first 20):")
    music_notes = get_unique_tokens()
    for note in music_notes[:20]:
        token_id = tokenizer.convert_tokens_to_ids(note)
        print(f"  {note}: {token_id}")

    # Check vocabulary sizes
    print(f"\nTokenizer vocab_size: {tokenizer.vocab_size}")
    print(f"Added vocab size: {len(tokenizer.get_added_vocab())}")
    print(f"Total vocab size: {tokenizer.vocab_size + len(tokenizer.get_added_vocab())}")

    # Test encoding
    test_text = "<MASK> C4_q D4_e"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"\nTest encoding: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")