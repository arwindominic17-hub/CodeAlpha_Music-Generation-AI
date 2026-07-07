"""
CodeAlpha - Task 3: Music Generation with AI
---------------------------------------------
Step 3: Generate a brand-new note sequence from the trained model and
convert it into a playable MIDI file.

Usage:
    python generate.py --model model.h5 --notes notes.pkl --out generated.mid
"""

import argparse
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from music21 import stream, note, chord, instrument


def parse_args():
    parser = argparse.ArgumentParser(description="Generate new music from a trained LSTM model")
    parser.add_argument("--model", type=str, default="model.h5")
    parser.add_argument("--notes", type=str, default="notes.pkl",
                         help="Original notes.pkl, used to seed generation")
    parser.add_argument("--vocab", type=str, default="vocab.pkl")
    parser.add_argument("--length", type=int, default=200, help="Number of notes to generate")
    parser.add_argument("--out", type=str, default="generated.mid")
    return parser.parse_args()


def generate_notes(model, network_input_seed, int_to_note, n_vocab, length):
    pattern = list(network_input_seed)
    prediction_output = []

    for _ in range(length):
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        idx = np.argmax(prediction)
        prediction_output.append(int_to_note[idx])

        pattern.append(idx)
        pattern = pattern[1:]

    return prediction_output


def to_midi(prediction_output, out_path):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ("." in pattern) or pattern.isdigit():
            # It's a chord (encoded as normalOrder pitch-class ids)
            chord_notes = [note.Note(int(n)) for n in pattern.split(".")]
            for n in chord_notes:
                n.storedInstrument = instrument.Piano()
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5  # fixed spacing; feel free to randomize for more human feel

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp=out_path)


def main():
    args = parse_args()

    with open(args.notes, "rb") as f:
        notes = pickle.load(f)

    with open(args.vocab, "rb") as f:
        vocab_data = pickle.load(f)

    note_to_int = vocab_data["note_to_int"]
    seq_len = vocab_data["seq_len"]
    int_to_note = {i: n for n, i in note_to_int.items()}
    n_vocab = len(note_to_int)

    model = load_model(args.model)

    max_start = max(1, len(notes) - seq_len - 1)
    start = np.random.randint(0, max_start)

    seed = [note_to_int[n] for n in notes[start:start + seq_len]]

    generated = generate_notes(model, seed, int_to_note, n_vocab, args.length)
    to_midi(generated, args.out)

    print(f"Generated {args.length} notes. Saved MIDI to '{args.out}'.")


if __name__ == "__main__":
    main()
