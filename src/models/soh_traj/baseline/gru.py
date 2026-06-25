"""
soh_traj/gru.py — GRU for SOH degradation trajectory prediction.
Input:  batch['Q'] (B, S, N)
Output: (pred:(B, n_future), None)
"""

import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        n_grid   = m.get('n_grid', 200)
        n_future = cfg.get('data', {}).get('n_future', 100)
        dropout  = m.get('dropout', 0.1)

        self.gru = nn.GRU(
            input_size=n_grid, hidden_size=128,
            num_layers=2, batch_first=True, dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_future),
        )

    def forward(self, batch: dict):
        Q = batch['Q']
        _, h = self.gru(Q)
        pred = self.head(h[-1])  # (B, n_future)
        return pred, None
