"""
soh_point/dlinear.py — DLinear for SOH single-point estimation.
Reference: Zeng et al., AAAI 2023.
Input:  batch['cycle_curve_data'] (B, S, 3, L) + batch['curve_attn_mask'] (B, S)
        每圈拼成 feature (B, S, 3*L)，未观测圈已由 dataset 置零。
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn

from src.models._masking import get_inputs, flatten_cycles


class DLinear(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        S      = m.get('n_cycles', cfg.get('data', {}).get('early_cycle', 100))
        L      = cfg.get('data', {}).get('charge_discharge_length', 300)
        kernel = m.get('dlinear_kernel', 25)
        F_dim  = 3 * L

        pad = kernel // 2
        self.avg        = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=pad)
        self.w_trend    = nn.Linear(S * F_dim, 1)
        self.w_seasonal = nn.Linear(S * F_dim, 1)

    def forward(self, batch: dict):
        x, _ = get_inputs(batch)              # (B, S, 3, L)
        x = flatten_cycles(x)                 # (B, S, 3*L)
        B, S = x.shape[0], x.shape[1]
        # decompose along cycle axis for each feature independently
        xT = x.permute(0, 2, 1)              # (B, F, S)
        trend = self.avg(xT)                  # (B, F, S)
        if trend.shape[-1] != S:
            trend = trend[:, :, :S]
        seasonal = xT - trend
        trend    = trend.permute(0, 2, 1).reshape(B, -1)    # (B, S*F)
        seasonal = seasonal.permute(0, 2, 1).reshape(B, -1) # (B, S*F)
        pred = self.w_trend(trend) + self.w_seasonal(seasonal)
        return pred, None
