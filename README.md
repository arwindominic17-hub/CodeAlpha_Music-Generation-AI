# CodeAlpha - Music Generation with AI

An LSTM-based deep learning model that learns musical patterns from a
collection of MIDI files and generates original piano compositions.

## Overview

This project uses a Recurrent Neural Network (LSTM) trained on note and
chord sequences extracted from MIDI files. Once trained, the model can
generate new, original note sequences which are converted back into a
playable `.mid` file.

**Pipeline:**
1. `preprocess.py` — parses MIDI files and extracts notes/chords into a sequence
2. `train.py` — trains an LSTM model on the extracted sequence
3. `generate.py` — uses the trained model to generate new music and saves it as MIDI

## Project Structure

```
Music-Generation-Project/
├── midi_data/          # Folder of training MIDI files (not included in repo)
├── preprocess.py        # Step 1: Extract notes/chords from MIDI files
├── train.py              # Step 2: Train the LSTM model
├── generate.py           # Step 3: Generate new music from the trained model
├── notes.pkl              # Extracted note sequence (output of preprocess.py)
├── vocab.pkl               # Vocabulary mapping + sequence length (output of train.py)
├── model.h5                 # Trained model weights (output of train.py)
├── generated.mid              # Example generated output
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

## Usage

### 1. Preprocess MIDI files

Place your training MIDI files (e.g. classical piano pieces) in a folder,
then run:

```bash
python preprocess.py --midi_dir midi_data
```

This extracts all notes and chords and saves them to `notes.pkl`.

Optional arguments:
- `--out` — output pickle file (default: `notes.pkl`)

### 2. Train the model

```bash
python train.py --notes notes.pkl --epochs 60
```

This trains a 2-layer LSTM network on the note sequence and saves the
best-performing model to `model.h5`, along with `vocab.pkl` (the
note-to-integer mapping needed for generation).

Optional arguments:
- `--seq_len` — length of each training sequence (default: 50)
- `--batch_size` — training batch size (default: 64)
- `--out` — output model file (default: `model.h5`)

### 3. Generate new music

```bash
python generate.py --model model.h5 --notes notes.pkl --out generated.mid
```

This seeds generation with a random sequence from the training data, then
predicts new notes one at a time using the trained model, and writes the
result to a playable MIDI file.

Optional arguments:
- `--length` — number of notes to generate (default: 200)
- `--vocab` — vocabulary pickle file (default: `vocab.pkl`)

## Model Architecture

```
LSTM(256, return_sequences=True)
Dropout(0.3)
LSTM(256)
BatchNormalization()
Dropout(0.3)
Dense(256) + ReLU
Dense(n_vocab) + Softmax
```

Trained with categorical crossentropy loss and the Adam optimizer.

## Requirements

- Python 3.9+
- TensorFlow 2.15+
- music21
- NumPy

See `requirements.txt` for exact versions.

## Notes

- Chords are encoded as their pitch-class normal order (e.g. `4.7.11`) and
  decoded back into `music21` Chord objects during generation.
- Generated note spacing is currently fixed (0.5 offset per note); this can
  be randomized for a more human/expressive feel.
- Larger and more diverse MIDI training sets generally produce more
  musically coherent results.

## Part of the CodeAlpha Internship

This is Task 3 of the CodeAlpha AI internship program. The companion
project, **Object Detection and Tracking**, is available in a separate
repository: [CodeAlpha_Object-Detection-Tracking](https://github.com/arwindominic17-hub/CodeAlpha_Object-Detection-Tracking)
