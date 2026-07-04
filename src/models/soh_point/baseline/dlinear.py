"""
soh_point/dlinear.py — DLinear for SOH single-point estimation.
Reference: Zeng et al., AAAI 2023.
Input:  batch['curves'] (B, 3, L) — channel-independent: Linear(L, 1) per channel
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn


class DLinear(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        L      = cfg.get('data', {}).get('charge_discharge_length', 300)
        kernel = m.get('dlinear_kernel', 25)

        pad = kernel // 2
        self.avg        = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=pad)
        self.w_trend    = nn.Linear(L, 1)
        self.w_seasonal = nn.Linear(L, 1)

    def forward(self, batch: dict):
        x = batch['curves']              # (B, 3, L)
        trend = self.avg(x)
        if trend.shape[-1] != x.shape[-1]:
            trend = trend[:, :, :x.shape[-1]]
        seasonal = x - trend             # (B, 3, L)
        # channel-independent: predict per channel, mean over channels
        out = self.w_trend(trend) + self.w_seasonal(seasonal)  # (B, 3, 1)
        pred = out.mean(dim=1)           # (B, 1)
        return pred, None
