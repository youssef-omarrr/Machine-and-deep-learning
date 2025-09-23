# Audio Processing Module

This part of the project handles the **encoding** and **decoding** of musical data. It enables converting between raw MIDI files, symbolic text representations, and high-quality audio using SoundFonts.

---

## 🔹 Encoder

The **Encoder** converts a `.mid` file into a structured `.txt` file.

* Each instrument track is represented as a **dictionary** entry.
* Notes and durations are encoded as symbolic tokens (e.g., `"C4_q"` for a quarter-note C4, `"Rest_h"` for a half-note rest).
* This textual format is compact, human-readable, and suitable for training language models (LLMs) to learn musical sequences.

**Final encoder implementation:** `encoder.py`

---

## 🔹 Decoder

The **Decoder** performs the reverse transformation:

1. Reads the text dictionary representation.
2. Converts it back into a `.mid` file.
3. Renders the `.mid` to `.wav` using **FluidSynth** and a high-quality SoundFont.

This step makes it possible to **audibly evaluate** the generated sequences while retaining fine control over instruments and timbre.

**Final decoder implementation:** `decoder.py`

---

## 📌 Notes

* Large `.wav` files are **not included** in the repository to save space.
* Default SoundFont used: **AegeanSymphonicOrchestra-SND.sf2** (chosen for a more realistic orchestral sound).
* **Next step:** Gather a diverse dataset of `.mid` files → encode them into text format → fine-tune the LLM for improved sequence generation.

---
