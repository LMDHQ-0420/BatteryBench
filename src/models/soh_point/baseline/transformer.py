"""
soh_point/transformer.py — Vanilla Transformer for SOH single-point estimation.
Input:  batch['curves'] (B, 3, L) → (B, L, 3) time-step sequence
Output: (pred:(B,1), None)
"""

import math
import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        L        = cfg.get('data', {}).get('charge_discharge_length', 300)
        d_model  = m.get('transformer_d_model', 64)
        n_heads  = m.get('transformer_n_heads', 4)
        n_layers = m.get('transformer_n_layers', 2)
        dropout  = m.get('dropout', 0.1)

        self.input_proj = nn.Linear(3, d_model)

        pe = torch.zeros(L, d_model)
        pos = torch.arange(L).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

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
        x = x.permute(0, 2, 1)          # (B, L, 3)
        h = self.input_proj(x) + self.pe
        h = self.encoder(h)
        pred = self.head(h.mean(dim=1))
        return pred, None
