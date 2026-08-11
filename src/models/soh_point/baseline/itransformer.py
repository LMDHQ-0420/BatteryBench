"""
soh_point/itransformer.py — iTransformer for SOH single-point estimation.
Reference: Liu et al., ICLR 2024.
Input:  batch['cycle_curve_data'] (B, S=1, 3, L) — S 恒为 1（每样本仅当前观测圈）。
        倒置为每通道一个 token：每个 variate token 由该通道整条长度 L 的圈内曲线
        embedding 而来，在 3 个 variate 间做自注意力（iTransformer 的核心设计）。
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn

from src.models._masking import get_curve_seq


class iTransformer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        L        = cfg.get('data', {}).get('charge_discharge_length', 300)
        d_model  = m.get('itransformer_d_model', 64)
        n_heads  = m.get('itransformer_n_heads', 4)
        n_layers = m.get('itransformer_n_layers', 2)
        dropout  = m.get('dropout', 0.1)

        self.var_proj = nn.Linear(L, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, batch: dict):
        x = get_curve_seq(batch)              # (B, L, 3)
        xc = x.permute(0, 2, 1)               # (B, C=3, L) — 每通道一条长度 L 的曲线
        h = self.var_proj(xc)                 # (B, 3, d) — 每 variate 一个 token
        h = self.encoder(h)
        pred = self.head(h.mean(dim=1))
        return pred, None
