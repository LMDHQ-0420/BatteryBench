"""
rul/patchtst.py — PatchTST for RUL prediction.
Reference: Nie et al., ICLR 2023.
Input:  batch['cycle_curve_data'] (B, S, 3, L) + batch['curve_attn_mask'] (B, S)
        未观测圈已由 dataset 置零。每圈拼成 token (3*L)，沿 cycle 轴切 patch。
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn

from src.models._masking import get_inputs, flatten_cycles


class PatchTST(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        S         = m.get('n_cycles', cfg.get('data', {}).get('early_cycle', 100))
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
        x, mask = get_inputs(batch)           # (B, S, 3, L), (B, S)
        x = flatten_cycles(x)                 # (B, S, F)
        # create patches along cycle axis
        patches = x.unfold(1, self.patch_len, self.stride)  # (B, P, F, patch_len)
        B2, P, F, PL = patches.shape
        patches = patches.reshape(B2, P, F * PL)            # (B, P, F*patch_len)
        h = self.patch_proj(patches) + self.pos_emb[:, :P, :]
        # patch is padded only if all its cycles are unobserved
        mpatch = mask.unfold(1, self.patch_len, self.stride)  # (B, P, patch_len)
        kpm = mpatch.sum(-1) <= 0                              # (B, P) bool
        h = self.encoder(h, src_key_padding_mask=kpm)
        pred = self.head(h)
        return pred, None
