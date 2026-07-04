"""
soh_point/cnn.py — 1D CNN for SOH single-point estimation.
Input:  batch['curves'] (B, 3, L) — 3-channel 1D signal
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        dropout = m.get('dropout', 0.1)

        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 8, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, batch: dict):
        x = batch['curves']              # (B, 3, L)
        h = self.conv(x).reshape(x.shape[0], -1)
        pred = self.head(h)
        return pred, None
