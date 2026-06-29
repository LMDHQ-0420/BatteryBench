"""
soh_traj/mlp.py — MLP for SOH degradation trajectory prediction.
Input:  batch['curves'] (B, S, 3, L) → flattened to (B, S*3*L)
Output: (pred:(B, n_future), None)
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        n_cycles = m.get('n_cycles', 100)
        L        = cfg.get('data', {}).get('charge_discharge_length', 300)
        n_future = cfg.get('data', {}).get('n_future', 100)
        dropout  = m.get('dropout', 0.1)

        in_dim = n_cycles * 3 * L
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),    nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_future),
        )

    def forward(self, batch: dict):
        x = batch['curves']                   # (B, S, 3, L)
        pred = self.net(x.reshape(x.shape[0], -1))  # (B, n_future)
        return pred, None
