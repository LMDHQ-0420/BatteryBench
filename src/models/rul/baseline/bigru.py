"""
rul/bigru.py — BiGRU for RUL prediction.
Input:  batch['curves'] (B, S, 3, L) → per-cycle token (B, S, 3*L)
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn


class BiGRU(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        L       = cfg.get('data', {}).get('charge_discharge_length', 300)
        dropout = m.get('dropout', 0.1)

        self.gru = nn.GRU(
            input_size=3 * L, hidden_size=128,
            num_layers=2, batch_first=True,
            dropout=dropout, bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, batch: dict):
        x = batch['curves']                   # (B, S, 3, L)
        B, S, C, L = x.shape
        x = x.reshape(B, S, C * L)           # (B, S, 3*L)
        _, h = self.gru(x)                    # h: (4, B, 128)
        fwd = h[-2]; bwd = h[-1]
        pred = self.head(torch.cat([fwd, bwd], dim=-1))
        return pred, None
