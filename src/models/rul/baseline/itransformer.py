"""
rul/itransformer.py — iTransformer for RUL prediction.
Reference: Liu et al., ICLR 2024.
Key idea: transpose the Transformer to attend over variables (grid points)
          rather than time steps.
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
        n_grid   = m.get('n_grid', 200)
        d_model  = m.get('itransformer_d_model', 64)
        n_heads  = m.get('itransformer_n_heads', 4)
        n_layers = m.get('itransformer_n_layers', 2)
        dropout  = m.get('dropout', 0.1)

        # project each variable (grid point) from S time steps to d_model
        self.var_proj = nn.Linear(S, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # pool over N variables, then predict scalar
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, batch: dict):
        Q = batch['Q']               # (B, S, N)
        # treat each grid point as a token: (B, N, S) → (B, N, d)
        x = Q.permute(0, 2, 1)      # (B, N, S)
        h = self.var_proj(x)         # (B, N, d)
        h = self.encoder(h)          # (B, N, d)
        pred = self.head(h.mean(dim=1))  # (B, 1)
        return pred, None
