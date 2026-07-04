"""
rul/itransformer.py — iTransformer for RUL prediction.
Reference: Liu et al., ICLR 2024.
Key idea: transpose the Transformer to attend over channels (3 curves)
          rather than time steps.
Input:  batch['cycle_curve_data'] (B, S, 3, L) + batch['curve_attn_mask'] (B, S)
        未观测圈已由 dataset 置零。倒置为每通道一个 token (B, 3, S*L)。
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn

from src.models._masking import get_inputs


class iTransformer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        S        = m.get('n_cycles', cfg.get('data', {}).get('early_cycle', 100))
        L        = cfg.get('data', {}).get('charge_discharge_length', 300)
        d_model  = m.get('itransformer_d_model', 64)
        n_heads  = m.get('itransformer_n_heads', 4)
        n_layers = m.get('itransformer_n_layers', 2)
        dropout  = m.get('dropout', 0.1)

        # project each channel from S*L time steps to d_model
        self.var_proj = nn.Linear(S * L, d_model)

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
        x, _ = get_inputs(batch)              # (B, S, 3, L)  未观测圈已置零
        B, S, C, L = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, C, S * L)  # (B, 3, S*L)
        h = self.var_proj(x)                  # (B, 3, d)
        h = self.encoder(h)                   # (B, 3, d)
        pred = self.head(h.mean(dim=1))       # (B, 1)
        return pred, None
