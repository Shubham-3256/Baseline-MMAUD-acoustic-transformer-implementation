"""
dataset.py
──────────
PyTorch Dataset classes for the UAV 3D trajectory estimation task.

Expected on-disk layout
───────────────────────
features/
    <seq_name>/
        window_00000.npy      ← (6, F, T) feature tensor
        window_00001.npy
        ...
        labels.csv            ← columns: window_idx, x, y, z

OR use the SyntheticUAVDataset for end-to-end testing without real data.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Optional
import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ──────────────────────────────────────────────────────────────────────────────

class FeatureNormalizer:
    """Online mean/std normalisation computed once over the training set."""

    def __init__(self):
        self.mean: Optional[torch.Tensor] = None
        self.std:  Optional[torch.Tensor] = None

    def fit(self, tensors: list[torch.Tensor]):
        """tensors: list of (6, F, T) tensors."""
        all_data = torch.stack(tensors, dim=0).float()       # (N, 6, F, T)
        self.mean = all_data.mean(dim=(0, 2, 3), keepdim=True).squeeze(0)  # (6,1,1)
        self.std  = all_data.std(dim=(0, 2, 3), keepdim=True).squeeze(0) + 1e-8

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None:
            return x
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def save(self, path: str):
        torch.save({"mean": self.mean, "std": self.std}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.mean = ckpt["mean"]
        self.std  = ckpt["std"]


# ──────────────────────────────────────────────────────────────────────────────
# Main Dataset
# ──────────────────────────────────────────────────────────────────────────────

class UAVTrajectoryDataset(Dataset):
    """
    Loads pre-extracted feature tensors and associated ground-truth 3D positions.

    Each sample = (feature_tensor, traj_history, label)
      feature_tensor : (6, F, T)     — current window features
      traj_history   : (K, 3)        — last K ground-truth positions
      label          : (3,)          — current ground-truth (x, y, z)
    """

    def __init__(self, features_dir: str,
                 traj_seq_len: int = 10,
                 normalizer: Optional[FeatureNormalizer] = None,
                 augment: bool = False):
        """
        Parameters
        ----------
        features_dir : root features directory; subdirs = sequences
        traj_seq_len : number of past positions fed to the decoder
        normalizer   : fitted FeatureNormalizer (None = no normalisation)
        augment      : apply mild data augmentation during training
        """
        self.traj_seq_len = traj_seq_len
        self.normalizer   = normalizer
        self.augment      = augment
        self.samples: list[dict] = []  # each entry = {feat_path, traj_history, label}

        feat_root = Path(features_dir)
        seq_dirs  = sorted([d for d in feat_root.iterdir() if d.is_dir()])

        for seq_dir in seq_dirs:
            labels_path = seq_dir / "labels.csv"
            if not labels_path.exists():
                print(f"  SKIP {seq_dir.name}: labels.csv not found at {labels_path}")
                print(f"    → Pose extraction likely failed. Re-run:")
                print(f"      python extract_pham4.py --bag <your.bag> --list_topics")
                print(f"      python extract_pham4.py --bag <your.bag> --pose_topic <topic>")
                continue

            labels_df = pd.read_csv(str(labels_path))
            # Expected columns: window_idx, x, y, z
            required = {"window_idx", "x", "y", "z"}
            if not required.issubset(labels_df.columns):
                missing_cols = required - set(labels_df.columns)
                print(f"  SKIP {seq_dir.name}: labels.csv missing columns {missing_cols}")
                print(f"    Found columns: {list(labels_df.columns)}")
                continue

            labels_df = labels_df.sort_values("window_idx").reset_index(drop=True)
            positions  = labels_df[["x", "y", "z"]].values.astype(np.float32)  # (N, 3)

            npy_files = sorted(seq_dir.glob("window_*.npy"))

            if not npy_files:
                print(f"  SKIP {seq_dir.name}: no window_*.npy files found in {seq_dir}")
                print(f"    → Run: python feature_extraction.py")
                continue

            if len(npy_files) != len(positions):
                print(f"  WARNING {seq_dir.name}: {len(npy_files)} .npy files but "
                      f"{len(positions)} label rows — using min({len(npy_files)}, {len(positions)})")

            n_windows = min(len(npy_files), len(positions))

            if n_windows <= traj_seq_len:
                print(f"  SKIP {seq_dir.name}: only {n_windows} windows, need > {traj_seq_len} "
                      f"(traj_seq_len). Sequence too short.")
                continue

            for i in range(traj_seq_len, n_windows):
                feat_path    = npy_files[i]
                traj_history = positions[i - traj_seq_len : i]  # (K, 3)
                label        = positions[i]                      # (3,)

                self.samples.append({
                    "feat_path":    str(feat_path),
                    "traj_history": traj_history,
                    "label":        label,
                })

        print(f"Dataset loaded: {len(self.samples)} samples from {len(seq_dirs)} sequences.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s    = self.samples[idx]
        feat = torch.from_numpy(np.load(s["feat_path"])).float()   # (6, F, T)
        hist = torch.from_numpy(s["traj_history"]).float()         # (K, 3)
        lbl  = torch.from_numpy(s["label"]).float()                # (3,)

        if self.normalizer is not None:
            feat = self.normalizer.transform(feat)

        if self.augment:
            feat, hist, lbl = self._augment(feat, hist, lbl)

        return feat, hist, lbl

    # ── Augmentation ──────────────────────────────────────────────────────
    @staticmethod
    def _augment(feat: torch.Tensor,
                 hist: torch.Tensor,
                 lbl:  torch.Tensor):
        # Random additive Gaussian noise on features
        if torch.rand(1).item() < 0.5:
            feat = feat + 0.01 * torch.randn_like(feat)
        # Random frequency masking (SpecAugment-lite)
        if torch.rand(1).item() < 0.3:
            F = feat.shape[1]
            f0 = torch.randint(0, F - 10, (1,)).item()
            feat[:, f0 : f0 + 10, :] = 0.0
        # Random time masking
        if torch.rand(1).item() < 0.3:
            T = feat.shape[2]
            t0 = torch.randint(0, T - 5, (1,)).item()
            feat[:, :, t0 : t0 + 5] = 0.0
        return feat, hist, lbl


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic Dataset  (no real data required)
# ──────────────────────────────────────────────────────────────────────────────

class SyntheticUAVDataset(Dataset):
    """
    Generates random feature tensors + smooth 3D trajectories on-the-fly.
    Useful for debugging the full training pipeline without the MMAUD dataset.
    """

    def __init__(self, n_samples: int = 2000, traj_seq_len: int = 10,
                 freq_bins: int = 513, time_frames: int = 64):
        self.n        = n_samples
        self.K        = traj_seq_len
        self.F        = freq_bins
        self.T        = time_frames
        self.traj     = self._generate_trajectory(n_samples + traj_seq_len)

    def _generate_trajectory(self, n: int) -> np.ndarray:
        """Smooth random walk in 3D."""
        pos  = np.zeros((n, 3), dtype=np.float32)
        vel  = np.random.randn(3).astype(np.float32) * 0.05
        for i in range(1, n):
            vel  = vel * 0.95 + np.random.randn(3).astype(np.float32) * 0.02
            pos[i] = pos[i - 1] + vel
        return pos

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        # Random feature tensor
        feat = torch.randn(6, self.F, self.T).float()

        # Trajectory history and current label
        i    = idx + self.K
        hist = torch.from_numpy(self.traj[i - self.K : i]).float()
        lbl  = torch.from_numpy(self.traj[i]).float()

        return feat, hist, lbl


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict, synthetic: bool = False):
    """
    Returns (train_loader, val_loader, test_loader, normalizer).
    Set synthetic=True for a quick end-to-end test.
    """
    tr_cfg = cfg["training"]
    m_cfg  = cfg["model"]
    f_cfg  = cfg["features"]
    p_cfg  = cfg["paths"]

    K = m_cfg["traj_seq_len"]

    if synthetic:
        print("Using synthetic dataset.")
        full = SyntheticUAVDataset(
            n_samples   = 3000,
            traj_seq_len = K,
            freq_bins   = f_cfg["freq_bins"],
            time_frames = f_cfg["time_frames"],
        )
        # 70 / 15 / 15 split
        n_train = int(0.70 * len(full))
        n_val   = int(0.15 * len(full))
        n_test  = len(full) - n_train - n_val
        train_ds, val_ds, test_ds = random_split(
            full, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(cfg["training"]["seed"])
        )
        normalizer = None
    else:
        feat_root = p_cfg["features_dir"]
        normalizer = FeatureNormalizer()
        # Build full dataset without augmentation for normalisation stats
        full_ds = UAVTrajectoryDataset(feat_root, traj_seq_len=K,
                                       normalizer=None, augment=False)
        if len(full_ds) == 0:
            raise RuntimeError(
                f"No samples found in {feat_root}.\n"
                "Run feature_extraction.py first, or pass synthetic=True."
            )
        # Fit normalizer on training samples only (70%)
        n_train = int(0.70 * len(full_ds))
        n_val   = int(tr_cfg["val_split"] * len(full_ds))
        n_test  = len(full_ds) - n_train - n_val

        train_indices = list(range(n_train))
        sample_feats  = [torch.from_numpy(np.load(full_ds.samples[i]["feat_path"])).float()
                         for i in train_indices[:500]]   # fit on first 500 for speed
        normalizer.fit(sample_feats)
        norm_path = os.path.join(p_cfg["checkpoints_dir"], "normalizer.pt")
        os.makedirs(p_cfg["checkpoints_dir"], exist_ok=True)
        normalizer.save(norm_path)

        train_ds = UAVTrajectoryDataset(feat_root, K, normalizer, augment=True)
        val_ds   = UAVTrajectoryDataset(feat_root, K, normalizer, augment=False)
        test_ds  = UAVTrajectoryDataset(feat_root, K, normalizer, augment=False)

        # Simple sequential splits (sequences are already sorted)
        train_ds.samples = full_ds.samples[:n_train]
        val_ds.samples   = full_ds.samples[n_train : n_train + n_val]
        test_ds.samples  = full_ds.samples[n_train + n_val :]

    bsz        = tr_cfg["batch_size"]
    num_workers = tr_cfg["num_workers"]

    train_loader = DataLoader(train_ds, batch_size=bsz, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bsz, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bsz, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, normalizer


# ──────────────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    train_loader, val_loader, test_loader, norm = build_dataloaders(cfg, synthetic=True)

    feat, hist, lbl = next(iter(train_loader))
    print("Feature shape :", feat.shape)   # (B, 6, 513, 64)
    print("History shape :", hist.shape)   # (B, K, 3)
    print("Label shape   :", lbl.shape)    # (B, 3)
