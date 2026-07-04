"""
soh_point/ic2ml.py — IC²ML for SOH single-point estimation.
Reference: Huang et al., Journal of Power Sources 666 (2026) 239148
Input: batch['Q_single'] (B, N) — single-cycle Q(V) curve
Output: (pred:(B,1), None)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class IC2ML(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        n_grid  = m.get('n_grid', 200)
        d_model = m.get('ic2ml_d_model', 64)
        dropout = m.get('dropout', 0.1)

        self.head = nn.Sequential(
            nn.Linear(n_grid, d_model), nn.LayerNorm(d_model), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, batch):
        q = batch['Q_single']    # (B, N)
        pred = self.head(q)
        return pred, None
