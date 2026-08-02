"""
soh_point/mlp.py — MLP for SOH single-point estimation.
Input:  batch['cycle_curve_data'] (B, S, 3, L) + batch['curve_attn_mask'] (B, S)
        未观测圈已由 dataset 置零。
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn

from src.models._masking import get_inputs, flatten_cycles


_FIXED_S = 128  # AdaptiveAvgPool1d target — normalizes variable S to a fixed token count


class MLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        L       = cfg.get('data', {}).get('charge_discharge_length', 300)
        dropout = m.get('dropout', 0.1)

        self.pool = nn.AdaptiveAvgPool1d(_FIXED_S)
        in_dim = _FIXED_S * 3 * L
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),    nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, batch: dict):
        x, _ = get_inputs(batch)              # (B, S, 3, L)
        B, S, C, L = x.shape
        x = flatten_cycles(x)                 # (B, S, 3*L)
        # pool S → _FIXED_S so the Linear head sees a fixed input size
        x = self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, _FIXED_S, 3*L)
        pred = self.net(x.reshape(B, -1))
        return pred, None
