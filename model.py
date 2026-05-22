"""
model.py
────────
Acoustic Transformer for UAV 3D Trajectory Estimation.

Architecture
────────────
  Input  : (B, 6, F, T)   — 6-channel spectrogram tensor
  Encoder: CNN  → flatten → Transformer (self-attention)
  Decoder: Cross-attention over encoded features, conditioned on
           the last `traj_seq_len` predicted positions
  Output : (B, 3)          — predicted (x, y, z) for the current window
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 1.  CNN Encoder  —  local spectro-temporal feature extractor
# ──────────────────────────────────────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    Three conv-bn-relu-pool blocks that progressively reduce (F, T) while
    increasing channel depth.

    Output: (B, d_model, H', W')  where H' = F//8, W' = T//8
    """

    def __init__(self, in_channels: int = 6,
                 cnn_channels: list = (32, 64, 128),
                 kernel_size: int = 3,
                 dropout: float = 0.1,
                 d_model: int = 256):
        super().__init__()

        layers = []
        ch_in = in_channels
        for ch_out in cnn_channels:
            layers += [
                nn.Conv2d(ch_in, ch_out, kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch_out, ch_out, kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout),
            ]
            ch_in = ch_out

        self.cnn = nn.Sequential(*layers)
        # 1×1 projection to d_model
        self.proj = nn.Conv2d(ch_in, d_model, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 6, F, T)  →  (B, d_model, H', W')"""
        x = self.cnn(x)
        x = self.proj(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Sinusoidal 2D Positional Encoding
# ──────────────────────────────────────────────────────────────────────────────

class PositionalEncoding2D(nn.Module):
    """Additive sinusoidal positional encoding for 2-D feature maps."""

    def __init__(self, d_model: int, max_h: int = 64, max_w: int = 64):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D pos enc"
        d_half = d_model // 2

        # Frequency encoding along height
        pe_h = torch.zeros(max_h, d_half)
        pos  = torch.arange(max_h).unsqueeze(1).float()
        div  = torch.exp(torch.arange(0, d_half, 2).float() * (-math.log(10000.0) / d_half))
        pe_h[:, 0::2] = torch.sin(pos * div)
        pe_h[:, 1::2] = torch.cos(pos * div)

        # Frequency encoding along width
        pe_w = torch.zeros(max_w, d_half)
        pos  = torch.arange(max_w).unsqueeze(1).float()
        pe_w[:, 0::2] = torch.sin(pos * div)
        pe_w[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe_h", pe_h)  # (max_h, d_half)
        self.register_buffer("pe_w", pe_w)  # (max_w, d_half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_model, H, W)  →  same shape with positional encoding added."""
        B, D, H, W = x.shape
        d_half = D // 2

        # Build (H, W, D) PE map
        h_enc = self.pe_h[:H, :d_half].unsqueeze(1).expand(H, W, d_half)   # (H, W, d_half)
        w_enc = self.pe_w[:W, :d_half].unsqueeze(0).expand(H, W, d_half)   # (H, W, d_half)
        pe    = torch.cat([h_enc, w_enc], dim=-1)                            # (H, W, D)
        pe    = pe.permute(2, 0, 1).unsqueeze(0)                             # (1, D, H, W)

        return x + pe


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Transformer Encoder  —  global spatio-temporal reasoning
# ──────────────────────────────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """
    Standard multi-head self-attention transformer operating on the flattened
    spatial tokens from the CNN.
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # (B, S, D)
            norm_first=True,    # pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers,
                                             norm=nn.LayerNorm(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, d_model)  →  (B, S, d_model)"""
        return self.encoder(x)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Cross-Attention Trajectory Decoder
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryDecoder(nn.Module):
    """
    Decodes the current 3-D position by attending to:
      - the encoder memory (acoustic context)
      - the trajectory history (past positions)

    Input
    -----
    memory        : (B, S, d_model)  — transformer encoder output
    traj_history  : (B, K, 3)        — last K (x,y,z) positions

    Output
    ------
    pos_pred : (B, 3)
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float,
                 traj_seq_len: int, output_dim: int = 3):
        super().__init__()

        self.traj_embed = nn.Linear(output_dim, d_model)
        pos_enc = self._build_1d_pe(traj_seq_len, d_model)
        self.register_buffer("pos_enc", pos_enc)   # (K, d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers,
                                             norm=nn.LayerNorm(d_model))
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

    @staticmethod
    def _build_1d_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe  # (max_len, d_model)

    def forward(self, memory: torch.Tensor,
                traj_history: torch.Tensor) -> torch.Tensor:
        """
        memory       : (B, S, d_model)
        traj_history : (B, K, 3)   — K = traj_seq_len
        """
        B, K, _ = traj_history.shape

        # Embed trajectory history → query tokens
        q = self.traj_embed(traj_history)               # (B, K, d_model)
        q = q + self.pos_enc[:K].unsqueeze(0)           # add positional encoding

        # Cross-attend to acoustic memory
        out = self.decoder(tgt=q, memory=memory)        # (B, K, d_model)

        # Use the last token as the representative feature
        last = out[:, -1, :]                            # (B, d_model)
        return self.out_proj(last)                       # (B, 3)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Full Model
# ──────────────────────────────────────────────────────────────────────────────

class AcousticTransformer(nn.Module):
    """
    End-to-end UAV 3D trajectory estimator.

    Parameters (from config.yaml → model section)
    ──────────────────────────────────────────────
    in_channels     : int  = 6
    cnn_channels    : list = [32, 64, 128]
    cnn_kernel_size : int  = 3
    cnn_dropout     : float
    d_model         : int  = 256
    nhead           : int  = 8
    num_encoder_layers : int = 4
    num_decoder_layers : int = 4
    dim_feedforward : int  = 512
    transformer_dropout : float
    traj_seq_len    : int  = 10
    output_dim      : int  = 3
    freq_bins       : int  (from features config)
    time_frames     : int  (from features config)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        f = cfg["features"]

        d_model     = m["d_model"]
        cnn_ch      = m["cnn_channels"]
        dropout_cnn = m["cnn_dropout"]
        dropout_tr  = m["transformer_dropout"]

        # ── CNN encoder ────────────────────────────────────────────────────
        self.cnn_encoder = CNNEncoder(
            in_channels  = f["num_channels_out"],
            cnn_channels = cnn_ch,
            kernel_size  = m["cnn_kernel_size"],
            dropout      = dropout_cnn,
            d_model      = d_model,
        )

        # Spatial dims after 3× MaxPool2d(2)
        H = f["freq_bins"]  // (2 ** len(cnn_ch))
        W = f["time_frames"] // (2 ** len(cnn_ch))

        self.pos_enc_2d = PositionalEncoding2D(d_model, max_h=H + 4, max_w=W + 4)

        # ── Transformer encoder ────────────────────────────────────────────
        self.tr_encoder = TransformerEncoder(
            d_model         = d_model,
            nhead           = m["nhead"],
            num_layers      = m["num_encoder_layers"],
            dim_feedforward = m["dim_feedforward"],
            dropout         = dropout_tr,
        )

        # ── Trajectory decoder ─────────────────────────────────────────────
        self.decoder = TrajectoryDecoder(
            d_model         = d_model,
            nhead           = m["nhead"],
            num_layers      = m["num_decoder_layers"],
            dim_feedforward = m["dim_feedforward"],
            dropout         = dropout_tr,
            traj_seq_len    = m["traj_seq_len"],
            output_dim      = m["output_dim"],
        )

        self._init_weights()

    # ── Weight initialisation ──────────────────────────────────────────────
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ── Forward ────────────────────────────────────────────────────────────
    def forward(self, spec: torch.Tensor,
                traj_history: torch.Tensor) -> torch.Tensor:
        """
        spec         : (B, 6, F, T)
        traj_history : (B, K, 3)

        Returns
        -------
        pred_pos : (B, 3)   — predicted (x, y, z)
        """
        # CNN: local spectro-temporal features
        feat = self.cnn_encoder(spec)                   # (B, d_model, H', W')

        # 2D positional encoding
        feat = self.pos_enc_2d(feat)                    # (B, d_model, H', W')

        # Flatten spatial → sequence
        B, D, H, W = feat.shape
        tokens = feat.flatten(2).permute(0, 2, 1)       # (B, H'*W', d_model)

        # Transformer self-attention
        memory = self.tr_encoder(tokens)                # (B, S, d_model)

        # Cross-attention decoder
        pred_pos = self.decoder(memory, traj_history)   # (B, 3)
        return pred_pos


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: dict) -> AcousticTransformer:
    model = AcousticTransformer(cfg)
    n = count_parameters(model)
    print(f"AcousticTransformer  |  trainable params: {n:,}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml, torch

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg)

    B   = 4
    F   = cfg["features"]["freq_bins"]
    T   = cfg["features"]["time_frames"]
    K   = cfg["model"]["traj_seq_len"]

    spec    = torch.randn(B, 6, F, T)
    history = torch.randn(B, K, 3)

    out = model(spec, history)
    print(f"Input spec    : {spec.shape}")
    print(f"Input history : {history.shape}")
    print(f"Output (pred) : {out.shape}")   # Expected: (4, 3)
