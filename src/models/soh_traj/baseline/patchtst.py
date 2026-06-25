"""
soh_traj/patchtst.py — PatchTST for SOH degradation trajectory prediction.
Reference: Nie et al., ICLR 2023.
Input:  batch['Q'] (B, S, N) → capacity_seq = Q.max(dim=-1) → (B, S)
Output: (pred:(B, n_future), None)
"""

import torch
import torch.nn as nn


class PatchTST(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        S         = m.get('n_cycles', 100)
        n_future  = cfg.get('data', {}).get('n_future', 100)
        patch_len = m.get('patchtst_patch_len', 16)
        stride    = m.get('patchtst_stride', 8)
        d_model   = m.get('patchtst_d_model', 64)
        n_heads   = m.get('patchtst_n_heads', 4)
        n_layers  = m.get('patchtst_n_layers', 2)
        dropout   = m.get('dropout', 0.1)

        self.patch_len = patch_len
        self.stride    = stride
        n_patches      = max(1, (S - patch_len) // stride + 1)

        self.patch_proj = nn.Linear(patch_len, d_model)
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
            nn.Linear(d_model, n_future),
        )

    def forward(self, batch: dict):
        x = batch['Q'].max(dim=-1).values
        patches = x.unfold(1, self.patch_len, self.stride)
        h = self.patch_proj(patches) + self.pos_emb[:, :patches.shape[1], :]
        h = self.encoder(h)
        pred = self.head(h)  # (B, n_future)
        return pred, None
