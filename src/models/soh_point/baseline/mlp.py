"""
soh_point/mlp.py — MLP for SOH single-point estimation.
Input:  batch['Q'] (B, S, N)
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        n_grid   = m.get('n_grid', 200)
        n_cycles = m.get('n_cycles', 100)
        dropout  = m.get('dropout', 0.1)

        in_dim = n_cycles * n_grid
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),    nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, batch: dict):
        Q = batch['Q']
        pred = self.net(Q.reshape(Q.shape[0], -1))
        return pred, None
