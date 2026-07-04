"""
soh_point/mlp.py — MLP for SOH single-point estimation.
Input:  batch['curves'] (B, 3, L) → flatten (B, 3*L)
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        L       = cfg.get('data', {}).get('charge_discharge_length', 300)
        dropout = m.get('dropout', 0.1)

        in_dim = 3 * L
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),    nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, batch: dict):
        x = batch['curves']              # (B, 3, L)
        pred = self.net(x.reshape(x.shape[0], -1))
        return pred, None
