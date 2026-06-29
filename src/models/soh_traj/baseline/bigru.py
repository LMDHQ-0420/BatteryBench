"""
soh_traj/bigru.py — BiGRU for SOH degradation trajectory prediction.
Input:  batch['curves'] (B, S, 3, L) → per-cycle token (B, S, 3*L)
Output: (pred:(B, n_future), None)
"""

import torch
import torch.nn as nn


class BiGRU(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        L        = cfg.get('data', {}).get('charge_discharge_length', 300)
        n_future = cfg.get('data', {}).get('n_future', 100)
        dropout  = m.get('dropout', 0.1)

        self.gru = nn.GRU(
            input_size=3 * L, hidden_size=128,
            num_layers=2, batch_first=True,
            dropout=dropout, bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_future),
        )

    def forward(self, batch: dict):
        x = batch['curves']                   # (B, S, 3, L)
        B, S, C, L = x.shape
        x = x.reshape(B, S, C * L)
        _, h = self.gru(x)
        pred = self.head(torch.cat([h[-2], h[-1]], dim=-1))  # (B, n_future)
        return pred, None
