"""
rul/bigru.py — BiGRU for RUL prediction.
Input:  batch['Q'] (B, S, N)
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn


class BiGRU(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        n_grid  = m.get('n_grid', 200)
        dropout = m.get('dropout', 0.1)

        self.gru = nn.GRU(
            input_size=n_grid, hidden_size=128,
            num_layers=2, batch_first=True,
            dropout=dropout, bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, batch: dict):
        Q = batch['Q']               # (B, S, N)
        _, h = self.gru(Q)           # h: (4, B, 128)  [2 layers * 2 directions]
        # concat last-layer forward and backward hidden states
        fwd = h[-2]                  # (B, 128)
        bwd = h[-1]                  # (B, 128)
        pred = self.head(torch.cat([fwd, bwd], dim=-1))  # (B, 1)
        return pred, None
