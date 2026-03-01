import kagglehub

path = kagglehub.dataset_download("pranaykoppula/torgo-audio")

print("Path to dataset files:", path)

import os
import random
from pathlib import Path
import json

# Root dataset path
# DATA_ROOT = Path("/kaggle/input/torgo-audio")
DATA_ROOT = Path(path)

# Fixed seed for reproducibility
SEED = 42
random.seed(SEED)

# Subset folders
subsets = {
    "F_Con": ["FC01S01", "FC02S02"],
    "F_Dys": ["F01", "F03S01"],
    "M_Con": ["MC01S01", "MC01S02"],
    "M_Dys": ["M01S01", "M01S02"],
}

def get_train_test_files():
    train_files = []
    test_files = []

    for label, speakers in subsets.items():
        for speaker in speakers:
            speaker_path = DATA_ROOT / label / f"wav_arrayMic_{speaker}"
            print(speaker_path)

            # Get only .wav files and sort for determinism
            files = sorted([f for f in speaker_path.glob("*.wav")])

            # Use ONLY first 50 files
            files = files[:50]

            # Shuffle with fixed seed
            random.shuffle(files)

            # 80/20 split
            N = len(files)
            N_train = int(0.8 * N)

            train_split = files[:N_train]
            test_split = files[N_train:]

            # Store with label
            train_files += [(str(f), label) for f in train_split]
            test_files += [(str(f), label) for f in test_split]

    print(f"Train samples: {len(train_files)}")
    print(f"Test samples: {len(test_files)}")
    return train_files , test_files


from pathlib import Path
import torchaudio
import torch

NOISE_ROOT = Path("/teamspace/studios/this_studio/SPP_Lab/assignment_3/Noises&RIR")
NOISE_DIR = NOISE_ROOT / "Noises"
RIR_DIR = NOISE_ROOT / "RIR"


def load_wavs_from_dir(directory, sample_rate=16000):
    paths = sorted(directory.glob("*.wav"))
    data = {}

    for p in paths:
        wav, sr = torchaudio.load(p)

        # mono
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)

        # resample
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)

        data[p.stem] = wav.squeeze(0)

    return data


def load_noises(sample_rate=16000):
    return load_wavs_from_dir(NOISE_DIR, sample_rate)


def load_rirs(sample_rate=16000):
    return load_wavs_from_dir(RIR_DIR, sample_rate)


if __name__ == "__main__":
    noises = load_noises()
    rirs = load_rirs()

    print("Loaded noises:", list(noises.keys()))
    print("Loaded RIRs:", list(rirs.keys()))

