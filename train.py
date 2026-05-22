"""
train.py
────────
Full training loop for the AcousticTransformer.

Features
────────
  • Smooth L1 loss  (paper default)
  • AdamW optimiser with cosine-annealing LR scheduler
  • Gradient clipping
  • Early stopping
  • TensorBoard logging
  • Best-model & last-model checkpointing

Usage
-----
    # Quick smoke-test with synthetic data (no real dataset needed)
    python train.py --synthetic

    # Full training on MMAUD features
    python train.py

    # Resume from checkpoint
    python train.py --resume checkpoints/last_model.pth
"""

import os
import argparse
import time
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from model   import build_model
from dataset import build_dataloaders


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def ape_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Absolute Position Error (Euclidean distance) averaged over batch."""
    return torch.norm(pred - target, dim=-1).mean()


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_val_ape", float("inf"))


# ──────────────────────────────────────────────────────────────────────────────
# Train / Validate one epoch
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(model: nn.Module,
              loader,
              criterion,
              optimizer,
              device: torch.device,
              grad_clip: float,
              is_train: bool,
              epoch: int,
              writer: SummaryWriter,
              global_step: list):

    model.train(is_train)
    total_loss = 0.0
    total_ape  = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    phase = "Train" if is_train else "Val"

    with ctx:
        for feat, hist, lbl in tqdm(loader, desc=f"  {phase} epoch {epoch}", leave=False):
            feat = feat.to(device, non_blocking=True)   # (B, 6, F, T)
            hist = hist.to(device, non_blocking=True)   # (B, K, 3)
            lbl  = lbl.to(device, non_blocking=True)    # (B, 3)

            pred = model(feat, hist)                    # (B, 3)
            loss = criterion(pred, lbl)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                global_step[0] += 1
                writer.add_scalar("Loss/train_step", loss.item(), global_step[0])

            with torch.no_grad():
                ape = ape_metric(pred, lbl).item()

            total_loss += loss.item()
            total_ape  += ape
            n_batches  += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_ape  = total_ape  / max(n_batches, 1)
    return avg_loss, avg_ape


# ──────────────────────────────────────────────────────────────────────────────
# Main training routine
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: dict,
          synthetic: bool = False,
          resume: str     = None):

    # ── Setup ─────────────────────────────────────────────────────────────
    torch.manual_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(cfg["paths"]["checkpoints_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["logs_dir"],        exist_ok=True)

    writer      = SummaryWriter(log_dir=cfg["paths"]["logs_dir"])
    global_step = [0]

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, _, _ = build_dataloaders(cfg, synthetic=synthetic)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)

    # ── Optimiser & Scheduler ─────────────────────────────────────────────
    tr   = cfg["training"]
    optim = AdamW(model.parameters(),
                  lr           = tr["learning_rate"],
                  weight_decay = tr["weight_decay"])

    scheduler = CosineAnnealingLR(optim,
                                  T_max  = tr["num_epochs"],
                                  eta_min = tr["learning_rate"] * 0.01)

    criterion = nn.SmoothL1Loss()

    # ── Optionally resume ─────────────────────────────────────────────────
    start_epoch    = 0
    best_val_ape   = float("inf")
    patience_count = 0

    if resume:
        start_epoch, best_val_ape = load_checkpoint(resume, model, optim, scheduler)
        print(f"Resumed from {resume} at epoch {start_epoch}, best APE={best_val_ape:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────
    patience = tr["early_stopping_patience"]
    print(f"\nTraining for up to {tr['num_epochs']} epochs …\n")

    for epoch in range(start_epoch + 1, tr["num_epochs"] + 1):
        t0 = time.time()

        train_loss, train_ape = run_epoch(
            model, train_loader, criterion, optim, device,
            tr["grad_clip"], is_train=True,
            epoch=epoch, writer=writer, global_step=global_step
        )
        val_loss, val_ape = run_epoch(
            model, val_loader, criterion, optim, device,
            tr["grad_clip"], is_train=False,
            epoch=epoch, writer=writer, global_step=global_step
        )

        scheduler.step()
        elapsed = time.time() - t0

        # ── Log ───────────────────────────────────────────────────────────
        writer.add_scalar("Loss/train_epoch",   train_loss, epoch)
        writer.add_scalar("Loss/val_epoch",     val_loss,   epoch)
        writer.add_scalar("APE/train",          train_ape,  epoch)
        writer.add_scalar("APE/val",            val_ape,    epoch)
        writer.add_scalar("LR",                 scheduler.get_last_lr()[0], epoch)

        print(f"Epoch {epoch:4d}/{tr['num_epochs']} | "
              f"Train loss={train_loss:.4f} APE={train_ape:.4f} | "
              f"Val loss={val_loss:.4f} APE={val_ape:.4f} | "
              f"LR={scheduler.get_last_lr()[0]:.2e} | "
              f"{elapsed:.1f}s")

        # ── Checkpointing ─────────────────────────────────────────────────
        state = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optim.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_ape":    best_val_ape,
            "config":          cfg,
        }
        save_checkpoint(state, cfg["paths"]["last_model"])

        if val_ape < best_val_ape:
            best_val_ape   = val_ape
            patience_count = 0
            save_checkpoint(state, cfg["paths"]["best_model"])
            print(f"  ✓ New best model saved (APE={best_val_ape:.4f})")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
                break

    writer.close()
    print(f"\nTraining complete.  Best Val APE: {best_val_ape:.4f}")
    print(f"Best model saved at: {cfg['paths']['best_model']}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train AcousticTransformer")
    p.add_argument("--config",    type=str, default="config.yaml")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data (no dataset needed)")
    p.add_argument("--resume",    type=str, default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    train(cfg, synthetic=args.synthetic, resume=args.resume)
