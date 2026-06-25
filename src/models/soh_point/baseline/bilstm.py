"""
soh_point/bilstm.py — BiLSTM for SOH single-point estimation.
Input:  batch['Q'] (B, S, N)
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        n_grid  = m.get('n_grid', 200)
        dropout = m.get('dropout', 0.1)

        self.lstm = nn.LSTM(
            input_size=n_grid, hidden_size=128,
            num_layers=2, batch_first=True,
            dropout=dropout, bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, batch: dict):
        Q = batch['Q']
        _, (h, _) = self.lstm(Q)
        pred = self.head(torch.cat([h[-2], h[-1]], dim=-1))
        return pred, None
