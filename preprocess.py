"""
CodeAlpha - Task 3: Music Generation with AI
---------------------------------------------
Step 1: Preprocessing.

Reads a folder of MIDI files, extracts notes and chords using music21,
and saves the resulting sequence of "note events" to disk (notes.pkl)
so train.py can build training sequences from it.

Usage:
    python preprocess.py --midi_dir data/midi
"""

import argparse
import glob
import pickle
import os

from music21 import converter, instrument, note, chord


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess MIDI files into note sequences")
    parser.add_argument("--midi_dir", type=str, required=True,
                         help="Folder containing .mid / .midi files")
    parser.add_argument("--out", type=str, default="notes.pkl",
                         help="Output pickle file for the extracted note sequence")
    return parser.parse_args()


def extract_notes(midi_dir):
    notes = []
    midi_files = glob.glob(os.path.join(midi_dir, "**", "*.mid"), recursive=True) + \
                 glob.glob(os.path.join(midi_dir, "**", "*.midi"), recursive=True)

    if not midi_files:
        raise FileNotFoundError(
            f"No .mid/.midi files found under '{midi_dir}'. "
            "Download some MIDI files (e.g. classical piano pieces) into that folder first."
        )

    print(f"Found {len(midi_files)} MIDI files. Parsing...")

    for i, file in enumerate(midi_files):
        try:
            midi = converter.parse(file)
        except Exception as e:
            print(f"  Skipping {file}: {e}")
            continue

        parts = instrument.partitionByInstrument(midi)
        elements = parts.parts[0].recurse() if parts else midi.flat.notes

        for el in elements:
            if isinstance(el, note.Note):
                notes.append(str(el.pitch))
            elif isinstance(el, chord.Chord):
                # Encode a chord as its pitch class ids joined by '.', e.g. "4.7.11"
                notes.append(".".join(str(n) for n in el.normalOrder))

        print(f"  [{i + 1}/{len(midi_files)}] parsed {os.path.basename(file)} "
              f"({len(notes)} events so far)")

    return notes


def main():
    args = parse_args()
    notes = extract_notes(args.midi_dir)

    with open(args.out, "wb") as f:
        pickle.dump(notes, f)

    vocab = sorted(set(notes))
    print(f"\nDone. Extracted {len(notes)} note/chord events, vocabulary size = {len(vocab)}.")
    print(f"Saved to '{args.out}'.")


if __name__ == "__main__":
    main()
