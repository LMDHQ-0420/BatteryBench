"""
rul/cnn.py — 2D CNN for RUL prediction.
Input:  batch['cycle_curve_data'] (B, S, 3, L) + batch['curve_attn_mask'] (B, S)
        未观测圈已由 dataset 置零。permute → (B, 3, S, L) 作为3通道图像。
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn

from src.models._masking import get_inputs


class CNN(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        dropout = m.get('dropout', 0.1)

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 7), padding=(1, 3)), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 3)), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, batch: dict):
        x, _ = get_inputs(batch)              # (B, S, 3, L)
        x = x.permute(0, 2, 1, 3)             # (B, 3, S, L)
        h = self.conv(x).reshape(x.shape[0], -1)
        pred = self.head(h)
        return pred, None
