"""
soh_traj/micn.py — MICN for SOH degradation trajectory prediction.
Reference: Wang et al., AAAI 2023.
Input:  batch['cycle_curve_data'] (B, S, 3, L) + batch['curve_attn_mask'] (B, S)
        未观测圈已由 dataset 置零。每圈拼成 token (3*L)。
Output: (pred:(B, n_future), None)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._masking import get_inputs, flatten_cycles


class _IsometricConv(nn.Module):
    def __init__(self, d: int, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(d, d, kernel_size=kernel, padding=0, groups=d)
        self.norm = nn.BatchNorm1d(d)

    def forward(self, x):
        x_pad = F.pad(x, (self.kernel - 1, 0), mode='circular')
        return F.relu(self.norm(self.conv(x_pad)))


class MICN(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        L        = cfg.get('data', {}).get('charge_discharge_length', 300)
        n_future = cfg.get('data', {}).get('n_future', 5000)
        d_model  = m.get('micn_d_model', 64)
        scales   = m.get('micn_scales', [3, 7, 13])
        dropout  = m.get('dropout', 0.1)

        self.input_proj = nn.Linear(3 * L, d_model)
        self.convs = nn.ModuleList([_IsometricConv(d_model, k) for k in scales])
        self.merge = nn.Sequential(
            nn.Linear(d_model * len(scales), d_model), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_future),
        )

    def forward(self, batch: dict):
        x, _ = get_inputs(batch)             # (B, S, 3, L)
        x = flatten_cycles(x)                # (B, S, 3*L)  未观测圈已是0
        h = self.input_proj(x).permute(0, 2, 1)  # (B, d, S)
        pooled = [conv(h).mean(dim=-1) for conv in self.convs]
        fused  = self.merge(torch.cat(pooled, dim=-1))
        pred   = self.head(fused)             # (B, n_future)
        return pred, None
