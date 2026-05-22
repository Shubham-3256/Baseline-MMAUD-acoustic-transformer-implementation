import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

AUDIO_DIR = Path("audio/Pham4")
GT_DIR = Path("dataset/Pham4/ground_truth")
OUT_CSV = Path("features/Pham4/labels.csv")

SAMPLE_RATE = 41800
WINDOW_MS = 400
OVERLAP = 0.75


# ─────────────────────────────────────────────────────────────
# Load GT timestamps + positions
# ─────────────────────────────────────────────────────────────

gt_files = sorted(GT_DIR.glob("*.npy"))

gt_times = []
gt_positions = []

for file in gt_files:

    timestamp = float(file.stem)

    xyz = np.load(file)

    gt_times.append(timestamp)
    gt_positions.append(xyz)

gt_times = np.array(gt_times)
gt_positions = np.array(gt_positions)

print(f"Loaded {len(gt_times)} GT poses")


# ─────────────────────────────────────────────────────────────
# Determine number of audio windows
# ─────────────────────────────────────────────────────────────

wav_path = AUDIO_DIR / "ch1.wav"

audio, sr = sf.read(wav_path)

assert sr == SAMPLE_RATE

window_samples = int(WINDOW_MS * 1e-3 * sr)
hop_samples = int(window_samples * (1 - OVERLAP))

num_windows = (
    (len(audio) - window_samples) // hop_samples
) + 1

print(f"Audio windows: {num_windows}")


# ─────────────────────────────────────────────────────────────
# Build aligned labels
# ─────────────────────────────────────────────────────────────

rows = []

start_time = gt_times[0]

for idx in range(num_windows):

    # Window center time (seconds)
    center_sample = idx * hop_samples + window_samples / 2

    center_time = (
        start_time
        + center_sample / SAMPLE_RATE
    )

    # Find nearest GT timestamp
    nearest_idx = np.argmin(
        np.abs(gt_times - center_time)
    )

    xyz = gt_positions[nearest_idx]

    rows.append({
        "window_idx": idx,
        "timestamp": center_time,
        "x": float(xyz[0]),
        "y": float(xyz[1]),
        "z": float(xyz[2]),
    })

df = pd.DataFrame(rows)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUT_CSV, index=False)

print("\nSaved aligned labels:")
print(OUT_CSV)

print(df.head())
print(f"\nTotal aligned labels: {len(df)}")