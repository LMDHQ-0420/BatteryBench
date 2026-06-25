"""
soh_point/itransformer.py — iTransformer for SOH single-point estimation.
Reference: Liu et al., ICLR 2024.
Input:  batch['Q'] (B, S, N)
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn


class iTransformer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        S        = m.get('n_cycles', 100)
        d_model  = m.get('itransformer_d_model', 64)
        n_heads  = m.get('itransformer_n_heads', 4)
        n_layers = m.get('itransformer_n_layers', 2)
        dropout  = m.get('dropout', 0.1)

        self.var_proj = nn.Linear(S, d_model)
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
        Q = batch['Q']
        x = Q.permute(0, 2, 1)     # (B, N, S)
        h = self.var_proj(x)        # (B, N, d)
        h = self.encoder(h)
        pred = self.head(h.mean(dim=1))
        return pred, None
