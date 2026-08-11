"""
soh_point/mlp.py — MLP for SOH single-point estimation.
Input:  batch['cycle_curve_data'] (B, S=1, 3, L) — S 恒为 1（每样本仅当前观测圈）。
        直接将该圈的 3*L 曲线展平作为特征向量。
Output: (pred:(B,1), None)
"""

import torch.nn as nn

from src.models._masking import get_curve_seq


class MLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        L       = cfg.get('data', {}).get('charge_discharge_length', 300)
        dropout = m.get('dropout', 0.1)

        in_dim = 3 * L
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),    nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, batch: dict):
        x = get_curve_seq(batch)              # (B, L, 3)
        B = x.shape[0]
        pred = self.net(x.reshape(B, -1))
        return pred, None
