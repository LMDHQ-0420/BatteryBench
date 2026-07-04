"""
soh_point/transformer.py — Vanilla Transformer for SOH single-point estimation.
Input:  batch['cycle_curve_data'] (B, S, 3, L) + batch['curve_attn_mask'] (B, S)
        未观测圈已由 dataset 置零。每圈拼成 token (3*L)。
Output: (pred:(B,1), None)
"""

import math
import torch
import torch.nn as nn

from src.models._masking import get_inputs, flatten_cycles, key_padding_mask


class Transformer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        S        = m.get('n_cycles', cfg.get('data', {}).get('early_cycle', 100))
        L        = cfg.get('data', {}).get('charge_discharge_length', 300)
        d_model  = m.get('transformer_d_model', 64)
        n_heads  = m.get('transformer_n_heads', 4)
        n_layers = m.get('transformer_n_layers', 2)
        dropout  = m.get('dropout', 0.1)

        self.input_proj = nn.Linear(3 * L, d_model)

        pe = torch.zeros(S, d_model)
        pos = torch.arange(S).unsqueeze(1).float()
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
        x, mask = get_inputs(batch)           # (B, S, 3, L), (B, S)
        x = flatten_cycles(x)                 # (B, S, 3*L)
        h = self.input_proj(x) + self.pe[:, :x.shape[1], :]
        kpm = key_padding_mask(mask)
        h = self.encoder(h, src_key_padding_mask=kpm)
        m = mask.unsqueeze(-1)
        feat = (h * m).sum(1) / m.sum(1).clamp(min=1)
        pred = self.head(feat)
        return pred, None
