"""
soh_point/itransformer.py — iTransformer for SOH single-point estimation.
Reference: Liu et al., ICLR 2024.
Input:  batch['curves'] (B, 3, L) → each channel as a token (B, 3, L)
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn


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
        x = batch['curves']              # (B, 3, L)
        h = self.var_proj(x)             # (B, 3, d)
        h = self.encoder(h)
        pred = self.head(h.mean(dim=1))
        return pred, None
