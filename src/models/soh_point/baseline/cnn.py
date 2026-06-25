"""
soh_point/cnn.py — 2D CNN for SOH single-point estimation.
Input:  batch['Q'] (B, S, N)  treated as single-channel image
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
            nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3)), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 3)), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, batch: dict):
        Q = batch['Q'].unsqueeze(1)
        h = self.conv(Q).reshape(Q.shape[0], -1)
        pred = self.head(h)
        return pred, None
