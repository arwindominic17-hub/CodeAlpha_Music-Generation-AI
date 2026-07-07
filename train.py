"""
CodeAlpha - Task 3: Music Generation with AI
---------------------------------------------
Step 2: Train an LSTM model to learn note patterns from the preprocessed
sequence (notes.pkl) and generate new music.

Usage:
    python train.py --notes notes.pkl --epochs 60
"""

import argparse
import pickle
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical


def parse_args():
    parser = argparse.ArgumentParser(description="Train an LSTM music generation model")
    parser.add_argument("--notes", type=str, default="notes.pkl")
    parser.add_argument("--seq_len", type=int, default=50, help="Length of each training sequence")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out", type=str, default="model.h5")
    return parser.parse_args()


def prepare_sequences(notes, seq_len):
    pitch_names = sorted(set(notes))
    note_to_int = {n: i for i, n in enumerate(pitch_names)}

    network_input, network_output = [], []
    for i in range(len(notes) - seq_len):
        seq_in = notes[i:i + seq_len]
        seq_out = notes[i + seq_len]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_vocab = len(pitch_names)
    X = np.reshape(network_input, (len(network_input), seq_len, 1)) / float(n_vocab)
    y = to_categorical(network_output, num_classes=n_vocab)
    return X, y, note_to_int, n_vocab


def build_model(seq_len, n_vocab):
    model = Sequential([
        LSTM(256, input_shape=(seq_len, 1), return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256),
        Activation("relu"),
        Dense(n_vocab),
        Activation("softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def main():
    args = parse_args()

    with open(args.notes, "rb") as f:
        notes = pickle.load(f)

    X, y, note_to_int, n_vocab = prepare_sequences(notes, args.seq_len)
    print(f"Training sequences: {X.shape[0]}, vocab size: {n_vocab}")

    model = build_model(args.seq_len, n_vocab)
    model.summary()

    checkpoint = ModelCheckpoint(args.out, monitor="loss", save_best_only=True, verbose=1)
    model.fit(X, y, epochs=args.epochs, batch_size=args.batch_size, callbacks=[checkpoint])

    # Save vocab mapping alongside the model so generate.py can decode predictions
    with open("vocab.pkl", "wb") as f:
        pickle.dump({"note_to_int": note_to_int, "seq_len": args.seq_len}, f)

    print(f"\nTraining complete. Model saved to '{args.out}', vocab saved to 'vocab.pkl'.")


if __name__ == "__main__":
    main()
