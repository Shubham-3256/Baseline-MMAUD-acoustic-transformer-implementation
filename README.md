# Audio Array-Based 3D UAV Trajectory Estimation with Transformer Networks

A research-oriented deep learning system for estimating **3D UAV trajectories** using **multi-channel audio recordings** from microphone arrays.

This project implements an end-to-end pipeline using:
- ROS bag audio extraction
- GCC-PHAT spatial acoustic features
- CNN + Transformer trajectory modeling
- Ground-truth synchronization
- 3D localization evaluation and visualization

---

# Features

- Multi-channel ROS1 audio extraction
- MMAUD dataset support
- STFT + GCC-PHAT feature generation
- CNN encoder + Transformer trajectory decoder
- Temporal trajectory conditioning
- Accurate GT timestamp synchronization
- Full training / evaluation pipeline
- 3D trajectory visualization
- Inference on unseen audio

---

# Project Pipeline

```text
ROS Bags
   ↓
Audio Extraction
   ↓
Feature Extraction
   ↓
Ground Truth Alignment
   ↓
Transformer Training
   ↓
Evaluation
   ↓
Visualization / Inference
```

---

# Dataset Structure

```text
dataset/
├── bags/
│   ├── Pham4.bag
│   └── 2023-08-24-11-30-56_phantom4.bag
│
└── Pham4/
    └── ground_truth/
        ├── 1692847902.611685.npy
        ├── 1692847902.663781.npy
        └── ...
```

Each `.npy` file contains:

```python
[x, y, z]
```

The filename itself is the timestamp.

---

# Project Structure

```text
.
├── config.yaml
├── dataset.py
├── evaluate.py
├── extract_audio.py
├── extract_pham4.py
├── feature_extraction.py
├── generate_aligned_labels.py
├── inference.py
├── model.py
├── train.py
├── visualize.py
├── requirements.txt
├── README.md
│
├── audio/
├── checkpoints/
├── dataset/
├── features/
├── logs/
└── outputs/
```

---

# Installation

## 1. Clone Repository

```bash
git clone <your_repo_url>
cd <repo_name>
```

---

## 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Required Packages

Main dependencies:

- torch
- torchvision
- numpy
- pandas
- scipy
- librosa
- soundfile
- matplotlib
- tqdm
- rosbags

---

# Usage

# 1. Extract Audio from ROS Bags

```bash
python extract_audio.py
```

Output:

```text
audio/
└── Pham4/
    ├── ch1.wav
    ├── ch2.wav
    ├── ch3.wav
    └── ch4.wav
```

---

# 2. Extract Acoustic Features

```bash
python feature_extraction.py
```

Output:

```text
features/
└── Pham4/
    ├── window_00000.npy
    ├── window_00001.npy
    └── ...
```

Feature tensor shape:

```text
(6, 513, 64)
```

---

# 3. Generate Time-Aligned Labels

```bash
python generate_aligned_labels.py
```

This synchronizes:
- audio windows
- GT timestamps

and creates:

```text
features/Pham4/labels.csv
```

---

# 4. Train Model

```bash
python train.py
```

Model checkpoints:

```text
checkpoints/
├── best_model.pth
└── last_model.pth
```

---

# 5. Evaluate Model

```bash
python evaluate.py
```

Outputs:

```text
outputs/
├── metrics.csv
└── predictions.csv
```

---

# 6. Visualize Results

```bash
python visualize.py --eval_csv outputs/predictions.csv
```

Generated plots:

```text
outputs/
├── trajectory_3d.png
├── trajectory_axes.png
├── trajectory_side_by_side.png
└── error_histogram.png
```

---

# 7. Run Inference on New Audio

```bash
python inference.py --audio_dir audio/Pham4
```

---

# Model Architecture

## Acoustic Encoder
- Multi-channel log spectrograms
- GCC-PHAT spatial features
- CNN feature extraction

## Temporal Modeling
- Transformer encoder
- Positional encoding
- Trajectory-conditioned decoding

## Output
- 3D UAV position regression
- `(x, y, z)` coordinates

---

# Results

## Final Evaluation Metrics

| Metric | Value |
|---|---|
| APE | 0.186 m |
| Dx | 0.142 m |
| Dy | 0.105 m |
| Dz | 0.013 m |

---

# Baseline Comparison

| Method | APE (m) |
|---|---|
| AudioNet | 2.80 |
| DroneChase | 2.64 |
| TAME | 0.55 |
| Ours | 0.19 |

---

# Visualization Examples

The system generates:
- 3D trajectory overlays
- XYZ temporal plots
- error histograms
- GT vs prediction comparisons

---

# Configuration

Main configuration file:

```text
config.yaml
```

Controls:
- audio settings
- feature parameters
- model architecture
- training hyperparameters
- dataset paths

---

# Current Status

Implemented:
- End-to-end acoustic trajectory estimation
- MMAUD integration
- GT synchronization
- Training / evaluation / inference
- Visualization pipeline

Planned enhancements:
- Full GCC microphone-pair expansion
- Multi-sequence training
- Temporal smoothing
- LiDAR pseudo-labeling
- Real-time streaming inference

---

# Citation

If you use this project in research, please cite the original MMAUD / UAV localization papers accordingly.

---

# License

MIT License

---

# Author

Shubham Sharma

Research Project — Audio-Based UAV 3D Trajectory Estimation