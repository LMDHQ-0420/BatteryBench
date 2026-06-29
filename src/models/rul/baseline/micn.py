"""
rul/micn.py — MICN (Multi-scale Isometric Convolution Network) for RUL prediction.
Reference: Wang et al., AAAI 2023.
Input:  batch['curves'] (B, S, 3, L) → per-cycle token (B, S, 3*L)
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _IsometricConv(nn.Module):
    def __init__(self, d: int, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(d, d, kernel_size=kernel, padding=0, groups=d)
        self.norm = nn.BatchNorm1d(d)

    def forward(self, x):   # x: (B, d, S)
        x_pad = F.pad(x, (self.kernel - 1, 0), mode='circular')
        return F.relu(self.norm(self.conv(x_pad)))


class MICN(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        S       = m.get('n_cycles', 100)
        L       = cfg.get('data', {}).get('charge_discharge_length', 300)
        d_model = m.get('micn_d_model', 64)
        scales  = m.get('micn_scales', [3, 7, 13])
        dropout = m.get('dropout', 0.1)

        self.input_proj = nn.Linear(3 * L, d_model)
        self.convs = nn.ModuleList([_IsometricConv(d_model, k) for k in scales])
        self.merge = nn.Sequential(
            nn.Linear(d_model * len(scales), d_model), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, batch: dict):
        x = batch['curves']                   # (B, S, 3, L)
        B, S, C, L = x.shape
        x = x.reshape(B, S, C * L)           # (B, S, 3*L)
        h = self.input_proj(x)               # (B, S, d)
        h = h.permute(0, 2, 1)              # (B, d, S)
        feats  = [conv(h) for conv in self.convs]
        pooled = [f.mean(dim=-1) for f in feats]
        fused  = self.merge(torch.cat(pooled, dim=-1))
        pred   = self.head(fused)
        return pred, None
