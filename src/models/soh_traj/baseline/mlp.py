"""
soh_traj/mlp.py — MLP for SOH degradation trajectory prediction.
Input:  batch['cycle_curve_data'] (B, S, 3, L) + batch['curve_attn_mask'] (B, S)
        未观测圈已由 dataset 置零。
Output: (pred:(B, n_future), None)
"""

import torch
import torch.nn as nn

from src.models._masking import get_inputs, flatten_cycles


class MLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        n_cycles = m.get('n_cycles', cfg.get('data', {}).get('early_cycle', 100))
        L        = cfg.get('data', {}).get('charge_discharge_length', 300)
        n_future = cfg.get('data', {}).get('n_future', 5000)
        dropout  = m.get('dropout', 0.1)

        in_dim = n_cycles * 3 * L
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),    nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_future),
        )

    def forward(self, batch: dict):
        x, _ = get_inputs(batch)              # (B, S, 3, L)
        x = flatten_cycles(x)                 # (B, S, 3*L)
        pred = self.net(x.reshape(x.shape[0], -1))  # (B, n_future)
        return pred, None
