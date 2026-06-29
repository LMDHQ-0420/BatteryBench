"""
rul/patchtst.py — PatchTST for RUL prediction.
Reference: Nie et al., ICLR 2023.
Input:  batch['curves'] (B, S, 3, L) → per-cycle token (B, S, 3*L)
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn


class PatchTST(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        S         = m.get('n_cycles', 100)
        L         = cfg.get('data', {}).get('charge_discharge_length', 300)
        patch_len = m.get('patchtst_patch_len', 16)
        stride    = m.get('patchtst_stride', 8)
        d_model   = m.get('patchtst_d_model', 64)
        n_heads   = m.get('patchtst_n_heads', 4)
        n_layers  = m.get('patchtst_n_layers', 2)
        dropout   = m.get('dropout', 0.1)

        self.patch_len = patch_len
        self.stride    = stride
        F_dim          = 3 * L
        n_patches      = max(1, (S - patch_len) // stride + 1)

        self.patch_proj = nn.Linear(patch_len * F_dim, d_model)
        self.pos_emb    = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_patches * d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, batch: dict):
        x = batch['curves']                   # (B, S, 3, L)
        B, S, C, L = x.shape
        x = x.reshape(B, S, C * L)           # (B, S, F)
        # create patches along cycle axis
        patches = x.unfold(1, self.patch_len, self.stride)  # (B, n_patches, F, patch_len)
        B2, P, F, PL = patches.shape
        patches = patches.reshape(B2, P, F * PL)            # (B, n_patches, F*patch_len)
        h = self.patch_proj(patches) + self.pos_emb[:, :P, :]
        h = self.encoder(h)
        pred = self.head(h)
        return pred, None
