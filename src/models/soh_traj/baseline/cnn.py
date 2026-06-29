"""
soh_traj/cnn.py — 2D CNN for SOH degradation trajectory prediction.
Input:  batch['curves'] (B, S, 3, L) → treated as 3-channel image (B, 3, S, L)
Output: (pred:(B, n_future), None)
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        n_future = cfg.get('data', {}).get('n_future', 100)
        dropout  = m.get('dropout', 0.1)

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 7), padding=(1, 3)), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 3)), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_future),
        )

    def forward(self, batch: dict):
        x = batch['curves']                   # (B, S, 3, L)
        x = x.permute(0, 2, 1, 3)            # (B, 3, S, L)
        h = self.conv(x).reshape(x.shape[0], -1)
        pred = self.head(h)                   # (B, n_future)
        return pred, None
