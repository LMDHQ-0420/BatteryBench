"""
soh_point/bilstm.py — BiLSTM for SOH single-point estimation.
Input:  batch['curves'] (B, S, 3, L) → per-cycle token (B, S, 3*L)
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        L       = cfg.get('data', {}).get('charge_discharge_length', 300)
        dropout = m.get('dropout', 0.1)

        self.lstm = nn.LSTM(
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
        x = x.reshape(B, S, C * L)
        _, (h, _) = self.lstm(x)
        pred = self.head(torch.cat([h[-2], h[-1]], dim=-1))
        return pred, None
