"""
soh_traj/dlinear.py — DLinear for SOH degradation trajectory prediction.
Reference: Zeng et al., AAAI 2023.
Input:  batch['curves'] (B, S, 3, L) → per-cycle token (B, S, 3*L)
Output: (pred:(B, n_future), None)

Channel-independent: each feature channel maps S→n_future independently,
then mean-pool over F channels. Params: 2*(S*n_future) only.
"""

import torch
import torch.nn as nn


class DLinear(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        S        = m.get('n_cycles', 100)
        L        = cfg.get('data', {}).get('charge_discharge_length', 300)
        n_future = cfg.get('data', {}).get('n_future', 100)
        kernel   = m.get('dlinear_kernel', 25)

        pad = kernel // 2
        self.avg        = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=pad)
        # channel-independent: Linear(S, n_future) applied to each of F channels
        self.w_trend    = nn.Linear(S, n_future)
        self.w_seasonal = nn.Linear(S, n_future)

    def forward(self, batch: dict):
        x = batch['curves']                   # (B, S, 3, L)
        B, S, C, L = x.shape
        x = x.reshape(B, S, C * L)           # (B, S, F)
        xT = x.permute(0, 2, 1)             # (B, F, S)
        trend = self.avg(xT)
        if trend.shape[-1] != S:
            trend = trend[:, :, :S]
        seasonal = xT - trend                 # (B, F, S)
        # per-channel: (B, F, S) → (B, F, n_future)
        out = self.w_trend(trend) + self.w_seasonal(seasonal)
        return out.mean(dim=1), None          # mean over F → (B, n_future)
