"""
soh_point/dlinear.py — DLinear for SOH single-point estimation.
Reference: Zeng et al., AAAI 2023.
Input:  batch['curves'] (B, S, 3, L) → per-cycle token (B, S, 3*L)
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn


class DLinear(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        S      = m.get('n_cycles', 100)
        L      = cfg.get('data', {}).get('charge_discharge_length', 300)
        kernel = m.get('dlinear_kernel', 25)
        F_dim  = 3 * L

        pad = kernel // 2
        self.avg        = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=pad)
        self.w_trend    = nn.Linear(S * F_dim, 1)
        self.w_seasonal = nn.Linear(S * F_dim, 1)

    def forward(self, batch: dict):
        x = batch['curves']                   # (B, S, 3, L)
        B, S, C, L = x.shape
        x = x.reshape(B, S, C * L)           # (B, S, F)
        xT = x.permute(0, 2, 1)             # (B, F, S)
        trend = self.avg(xT)
        if trend.shape[-1] != S:
            trend = trend[:, :, :S]
        seasonal = xT - trend
        trend    = trend.permute(0, 2, 1).reshape(B, -1)
        seasonal = seasonal.permute(0, 2, 1).reshape(B, -1)
        pred = self.w_trend(trend) + self.w_seasonal(seasonal)
        return pred, None
