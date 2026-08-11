"""
soh_point/lstm.py — LSTM for SOH single-point estimation.
Input:  batch['cycle_curve_data'] (B, S=1, 3, L) — S 恒为 1（每样本仅当前观测圈）。
        真实时序轴是圈内曲线 L（定长，无需 padding），逐时间步的 3 通道向量作为输入。
Output: (pred:(B,1), None)
"""

import torch.nn as nn

from src.models._masking import get_curve_seq


class LSTM(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        dropout = m.get('dropout', 0.1)

        self.lstm = nn.LSTM(
            input_size=3, hidden_size=128,
            num_layers=2, batch_first=True, dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, batch: dict):
        x = get_curve_seq(batch)              # (B, L, 3)
        _, (h, _) = self.lstm(x)              # h: (2, B, 128)
        pred = self.head(h[-1])               # (B, 1)
        return pred, None
