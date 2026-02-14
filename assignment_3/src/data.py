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
