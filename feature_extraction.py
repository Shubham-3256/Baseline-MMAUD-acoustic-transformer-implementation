"""
feature_extraction.py
─────────────────────
Converts 4-channel WAV audio → 6-channel feature tensors used by the model.

Feature layout (axis-0 = channel index):
  [0]  LogMag ch1         ← log(|STFT ch1|)
  [1]  LogMag ch2
  [2]  LogMag ch3
  [3]  LogMag ch4
  [4]  GCC-PHAT ch1–ch2   ← cross-correlation between mic pair (0,1)
  [5]  GCC-PHAT ch1–ch3   ← cross-correlation between mic pair (0,2)

Shape per window: (10, freq_bins, time_frames)

Usage
-----
    python feature_extraction.py --audio_dir audio --out_dir features
    python feature_extraction.py --demo          # build features from demo audio
"""

import os
import argparse
import numpy as np
import soundfile as sf
import yaml
from pathlib import Path
from scipy.signal import stft as scipy_stft
from scipy.ndimage import zoom
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Core DSP
# ──────────────────────────────────────────────────────────────────────────────

def compute_stft(signal: np.ndarray, n_fft: int, hop_length: int, sr: int) -> np.ndarray:
    """Returns complex STFT: shape (freq_bins, time_frames)."""
    _, _, Z = scipy_stft(
        signal, fs=sr, nperseg=n_fft,
        noverlap=n_fft - hop_length, window="hann", padded=True,
    )
    return Z  # complex (freq_bins, time_frames)


def log_magnitude(Z: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Log-magnitude spectrogram: shape (freq_bins, time_frames)."""
    return np.log(np.abs(Z) + eps)


def gcc_phat(
    sig_ref: np.ndarray,
    sig_sec: np.ndarray,
    n_fft: int,
    hop_length: int,
    sr: int,
    max_delay: int = 50,
) -> np.ndarray:
    """
    GCC-PHAT between two channels, computed frame-by-frame.

    Returns shape: (2 * max_delay + 1, time_frames)
    """
    _, _, Z_ref = scipy_stft(sig_ref, fs=sr, nperseg=n_fft,
                              noverlap=n_fft - hop_length, window="hann", padded=True)
    _, _, Z_sec = scipy_stft(sig_sec, fs=sr, nperseg=n_fft,
                              noverlap=n_fft - hop_length, window="hann", padded=True)

    # Phase-only cross-power spectrum (PHAT weighting)
    cpsd      = Z_ref * np.conj(Z_sec)
    cpsd_norm = cpsd / (np.abs(cpsd) + 1e-8)

    # scipy_stft returns the one-sided spectrum: shape (n_fft//2+1, frames).
    # np.fft.ifft must be told the full transform size (n_fft) so it
    # zero-pads the one-sided input back to the full lag axis before computing
    # the IFFT.  Without n=n_fft the result has only 513 rows and the
    # negative-lag indexing  cc[n_fft - max_delay : n_fft]  is an empty slice.
    cc       = np.real(np.fft.ifft(cpsd_norm, n=n_fft, axis=0))   # (n_fft, time_frames)
    n_frames = cc.shape[1]

    if n_frames == 0:
        return np.zeros((2 * max_delay + 1, 0), dtype=np.float32)

    out = np.zeros((2 * max_delay + 1, n_frames), dtype=np.float32)
    out[:max_delay]      = cc[n_fft - max_delay : n_fft, :]
    out[max_delay]       = cc[0, :]
    out[max_delay + 1:]  = cc[1 : max_delay + 1, :]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Shape helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pad_or_crop(arr: np.ndarray, target_F: int, target_T: int) -> np.ndarray:
    """Ensure spectrogram is exactly (target_F, target_T)."""
    F, T = arr.shape
    if F > target_F:
        arr = arr[:target_F, :]
    elif F < target_F:
        arr = np.pad(arr, ((0, target_F - F), (0, 0)))
    if T > target_T:
        arr = arr[:, :target_T]
    elif T < target_T:
        arr = np.pad(arr, ((0, 0), (0, target_T - T)))
    return arr


def _resize_gcc(gcc: np.ndarray, target_F: int, target_T: int) -> np.ndarray:
    """Resize GCC array (lag_bins, T) → (target_F, target_T) via bilinear zoom."""
    if gcc.shape[1] == 0:
        return np.zeros((target_F, target_T), dtype=np.float32)

    scale_F  = target_F / gcc.shape[0]
    scale_T  = target_T / gcc.shape[1]
    resized  = zoom(gcc.astype(np.float64), (scale_F, scale_T), order=1)
    # Clip to exact target in case of rounding
    return resized[:target_F, :target_T].astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Window-level feature builder
# ──────────────────────────────────────────────────────────────────────────────

def build_feature_tensor(
    channels: list[np.ndarray],
    cfg_audio: dict,
    cfg_feat: dict,
) -> np.ndarray:
    """
    Build the (6, F, T) feature tensor from 4 mono audio arrays.

    Parameters
    ----------
    channels : list of 4 float32 arrays, each (n_samples,)

    Returns
    -------
    tensor : np.ndarray, shape (6, freq_bins, time_frames), dtype float32
    """
    sr         = cfg_audio["sample_rate"]
    n_fft      = cfg_audio["n_fft"]
    hop_length = cfg_audio["hop_length"]
    max_delay  = cfg_feat["gcc_max_delay"]
    T          = cfg_feat["time_frames"]
    F          = cfg_feat["freq_bins"]

    # ── 4 log-magnitude spectrograms ──────────────────────────────────────────
    log_mags = []
    for ch_signal in channels:
        Z  = compute_stft(ch_signal, n_fft, hop_length, sr)
        lm = log_magnitude(Z)
        log_mags.append(_pad_or_crop(lm, F, T))

    # ── GCC-PHAT features for all microphone pairs ─────────────────────────────

    gcc_pairs = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
        ]
        #Converts 4-channel WAV audio → 10-channel feature tensors
    gcc_features = []

    for i, j in gcc_pairs:

        gcc = gcc_phat(
        channels[i],
        channels[j],
        n_fft,
        hop_length,
        sr,
        max_delay
        )

        gcc_r = _resize_gcc(gcc, F, T)

        gcc_features.append(gcc_r)

    # Final tensor:
    # 4 spectrogram channels + 6 GCC channels = 10 channels

    tensor = np.stack(
        log_mags + gcc_features,
        axis=0
    )
    return tensor.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Sliding-window segmentation
# ──────────────────────────────────────────────────────────────────────────────

def sliding_windows(
    channels: list[np.ndarray],
    window_samples: int,
    hop_samples: int,
) -> list[list[np.ndarray]]:
    """
    Slice 4-channel audio into overlapping windows.

    Returns list of windows; each window is a list of 4 arrays (one per channel).
    """
    n_samples = channels[0].shape[0]
    windows   = []
    start     = 0
    while start + window_samples <= n_samples:
        windows.append([ch[start : start + window_samples] for ch in channels])
        start += hop_samples
    return windows


# ──────────────────────────────────────────────────────────────────────────────
# File-level pipeline
# ──────────────────────────────────────────────────────────────────────────────

def process_sequence(seq_dir: Path, out_dir: Path, cfg: dict) -> int:
    """
    Load ch1-4.wav from seq_dir, extract features per sliding window,
    and save each as a .npy tensor to out_dir.

    Returns
    -------
    Number of windows written.
    """
    sr     = cfg["audio"]["sample_rate"]
    win_ms = cfg["audio"]["window_ms"]
    ovlp   = cfg["audio"]["overlap"]

    window_samples = int(win_ms * 1e-3 * sr)
    hop_samples    = int(window_samples * (1 - ovlp))

    # ── Load channels ─────────────────────────────────────────────────────────
    wav_paths = [seq_dir / f"ch{i}.wav" for i in range(1, 5)]
    missing   = [str(p) for p in wav_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing WAV files: {missing}")

    channels = []
    for wp in wav_paths:
        data, file_sr = sf.read(str(wp))
        if file_sr != sr:
            try:
                import librosa
                data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
            except ImportError:
                raise RuntimeError(
                    f"WAV sample rate {file_sr} ≠ config {sr} Hz. "
                    "Install librosa to auto-resample:  pip install librosa"
                )
        channels.append(data.astype(np.float32))

    # Trim all channels to the shortest length
    min_len  = min(len(c) for c in channels)
    channels = [c[:min_len] for c in channels]

    # ── Sliding window → features → disk ─────────────────────────────────────
    windows = sliding_windows(channels, window_samples, hop_samples)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, win_chs in enumerate(windows):
        feat = build_feature_tensor(win_chs, cfg["audio"], cfg["features"])
        np.save(str(out_dir / f"window_{idx:05d}.npy"), feat)

    return len(windows)


# ──────────────────────────────────────────────────────────────────────────────
# Demo mode
# ──────────────────────────────────────────────────────────────────────────────

def demo_extraction(cfg: dict) -> None:
    """Build features from synthetic demo audio (generated by extract_audio.py --demo)."""
    audio_dir = Path(cfg["paths"]["audio_dir"])  / "demo"
    out_dir   = Path(cfg["paths"]["features_dir"]) / "demo"

    if not audio_dir.exists():
        print(f"Demo audio not found at {audio_dir}.")
        print("Run first:  python extract_audio.py --demo")
        return

    n = process_sequence(audio_dir, out_dir, cfg)
    feat_shape = (
        cfg["features"]["num_channels_out"],
        cfg["features"]["freq_bins"],
        cfg["features"]["time_frames"],
    )
    print(f"Demo features ready: {n} windows saved to {out_dir}")
    print(f"Feature shape per window: {feat_shape}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract features from 4-channel UAV audio")
    p.add_argument("--audio_dir", type=str, default=None,
                   help="Root audio directory (overrides config)")
    p.add_argument("--out_dir",   type=str, default=None,
                   help="Root features directory (overrides config)")
    p.add_argument("--config",    type=str, default="config.yaml")
    p.add_argument("--demo",      action="store_true",
                   help="Use demo audio (must run extract_audio.py --demo first)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    if args.demo:
        demo_extraction(cfg)
        return

    audio_root = Path(args.audio_dir or cfg["paths"]["audio_dir"])
    feat_root  = Path(args.out_dir   or cfg["paths"]["features_dir"])

    if not audio_root.exists():
        print(f"Audio directory not found: {audio_root}")
        print("Tip: run --demo or python extract_audio.py first.")
        return

    seq_dirs = sorted([d for d in audio_root.iterdir() if d.is_dir()])
    if not seq_dirs:
        print(f"No sequence subdirectories found in {audio_root}")
        return

    total_windows = 0
    for seq_dir in tqdm(seq_dirs, desc="Sequences"):
        out_seq = feat_root / seq_dir.name
        try:
            n = process_sequence(seq_dir, out_seq, cfg)
            total_windows += n
            tqdm.write(f"  {seq_dir.name}: {n} windows → {out_seq}")
        except Exception as exc:
            tqdm.write(f"  SKIP {seq_dir.name}: {exc}")

    feat_shape = (
        cfg["features"]["num_channels_out"],
        cfg["features"]["freq_bins"],
        cfg["features"]["time_frames"],
    )
    print(f"\nDone. {total_windows:,} total windows extracted.")
    print(f"Feature shape per window: {feat_shape}")


if __name__ == "__main__":
    main()
