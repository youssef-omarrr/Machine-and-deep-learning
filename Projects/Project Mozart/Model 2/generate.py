import os
import subprocess
from pydub import AudioSegment
import torch
from miditok import TokSequence

from notes_utils import build_vocab_maps, note_token_ids_from_symbolic
from train_model import build_structural_constraints 

# ------------------------------------------------------------
# FILE NAMING HELPER
# ------------------------------------------------------------
def name_file(output_dir, safe_name, extension, numerate=True):
    """
    Generate a unique filename in output_dir using safe_name and extension.
    Ensures incrementing suffix (_1, _2, ...) if duplicates exist.
    """
    if not extension.startswith("."):
        extension = "." + extension

    os.makedirs(output_dir, exist_ok=True)

    if numerate:
        existing_files = [
            f for f in os.listdir(output_dir) 
            if f.startswith(safe_name) and f.endswith(extension)
        ]
        numbers = []
        for f in existing_files:
            parts = f[:-len(extension)].split("_")  # strip extension, split
            if len(parts) > 1 and parts[-1].isdigit():
                numbers.append(int(parts[-1]))
        next_number = max(numbers, default=0) + 1
        return os.path.join(output_dir, f"{safe_name}_{next_number}{extension}")
    else:
        return os.path.join(output_dir, f"{safe_name}{extension}")


# ------------------------------------------------------------
# MIDI → WAV CONVERSION
# ------------------------------------------------------------
def midi_to_wav(
    midi_path,
    soundfont_path="../soundfonts/AegeanSymphonicOrchestra-SND.sf2",
    output_dir="model_outputs/wav_files/",
    output_name=None,
    sample_rate=44100,
    gain=3.0,
    normalize=True,
    fluidsynth_path=r"C:\tools\fluidsynth\bin\fluidsynth.exe",
    print_details=True
):
    """
    Convert a MIDI file to WAV using FluidSynth and a specified SoundFont.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Use helper to pick name
    if output_name is None:
        base = os.path.splitext(os.path.basename(midi_path))[0]
    else:
        base = output_name
    output_wav = name_file(output_dir, base, ".wav")

    if print_details:
        print(f"Rendering '{midi_path}' to WAV...")
        print(f"Using SoundFont: {soundfont_path}")
        print(f"Output path: {output_wav}")

    # Run FluidSynth
    cmd = [
        fluidsynth_path,
        "-a", "file",
        "-F", output_wav,
        "-ni", soundfont_path,
        "-r", str(sample_rate),
        "-g", str(gain),
        midi_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Error running FluidSynth: {e}")
        raise

    # Normalize audio if requested
    if normalize:
        sound = AudioSegment.from_wav(output_wav)
        normalized = sound.normalize()
        normalized.export(output_wav, format="wav")

    if print_details:
        print(f"✅ Successfully converted MIDI to WAV: {output_wav}")

    return output_wav


# ------------------------------------------------------------
# GENERATION LOOP
# ------------------------------------------------------------
def generate(
    model,
    tokenizer,
    start_symbols=["Bar_None", "Position_0", "C4_q"],
    max_len=2000,
    output_dir="model_outputs/midi_files",
    base_name="generated",
    temperature=0.8,
    max_input_len_window=512,
    soundfont_path="../soundfonts/AegeanSymphonicOrchestra-SND.sf2",
):
    """
    Autoregressive generation with EOS filtering and allowed token order
    (Bar -> Position -> Pitch -> Velocity -> Duration -> Position/Bar).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    vocab, _ = build_vocab_maps(tokenizer)
    _, token_type_arr, allowed_next_mask = build_structural_constraints(tokenizer, device)


    # Start tokens
    ids = []
    for sym in start_symbols:
        if sym in vocab:
            ids.append(vocab[sym])
        else:
            ids.extend(note_token_ids_from_symbolic(tokenizer, sym))

    eos_id = vocab.get("EOS_None", None)

    with torch.inference_mode():
        for _ in range(max_len):
            window = ids[-max_input_len_window:]
            x = torch.tensor([window], dtype=torch.long, device=device)
            logits = model(x)
            logits = logits[0, -1] / temperature

            # Structural filtering
            if len(ids) > 0:
                prev_id = ids[-1]
                prev_type = token_type_arr[prev_id].item()
                if prev_type >= 0:
                    mask = allowed_next_mask[prev_type].to(device)
                    logits = logits.masked_fill(~mask, float("-inf"))

            probs = torch.softmax(logits, dim=0)
            next_id = torch.multinomial(probs, 1).item()
            ids.append(next_id)

            if eos_id is not None and next_id == eos_id:
                break

    # Trim at EOS
    if eos_id is not None and eos_id in ids:
        ids = ids[: ids.index(eos_id)]

    # Save MIDI
    out_midi = name_file(output_dir, base_name, ".mid")
    seq = TokSequence(ids=ids)
    midi = tokenizer.decode([seq])
    midi.dump_midi(out_midi)

    # Convert to WAV
    out_wav = name_file(output_dir, base_name, ".wav")
    midi_to_wav(out_midi, soundfont_path=soundfont_path, output_name=base_name)

    print(f"✅ Saved: {out_midi} and {out_wav}")
    return ids

