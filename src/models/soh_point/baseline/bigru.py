"""
soh_point/bigru.py — BiGRU for SOH single-point estimation.
Input:  batch['curves'] (B, 3, L) → (B, L, 3) time-step sequence
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn


class BiGRU(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        dropout = m.get('dropout', 0.1)

        self.gru = nn.GRU(
            input_size=3, hidden_size=128,
            num_layers=2, batch_first=True,
            dropout=dropout, bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, batch: dict):
        x = batch['curves']              # (B, 3, L)
        x = x.permute(0, 2, 1)          # (B, L, 3)
        _, h = self.gru(x)
        pred = self.head(torch.cat([h[-2], h[-1]], dim=-1))
        return pred, None
