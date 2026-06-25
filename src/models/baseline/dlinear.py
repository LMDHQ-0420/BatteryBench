"""
dlinear.py — DLinear baseline for RUL prediction.
Reference: Zeng et al., AAAI 2023.

Input:  batch['Q'] (B, S, N)  → capacity_seq = Q.max(dim=-1) → (B, S)
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn


class DLinear(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        S      = m.get('n_cycles', 100)
        kernel = m.get('dlinear_kernel', 25)

        pad = kernel // 2
        self.avg        = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=pad)
        self.w_trend    = nn.Linear(S, 1)
        self.w_seasonal = nn.Linear(S, 1)

    def forward(self, batch: dict):
        x = batch['Q'].max(dim=-1).values        # (B, S)
        trend = self.avg(x.unsqueeze(1)).squeeze(1)
        if trend.shape[-1] != x.shape[-1]:
            trend = trend[:, :x.shape[-1]]
        seasonal = x - trend
        pred = self.w_trend(trend) + self.w_seasonal(seasonal)
        return pred, None
