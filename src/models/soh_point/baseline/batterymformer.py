"""
batterymformer.py — BatteryMFormer for SOH point estimation.

Simplified for single-cycle input (B, 3, L):
  - PatchTST-style patch tokenization over the L dimension
  - Transformer encoder
  - MDPM memory module
  - Linear head → SOH scalar

Input batch keys:
  'curves': (B, 3, L)  — V/max_V, I/C-rate, Q/Q_nom
Output: (pred: (B,1), None)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MDPM(nn.Module):
    """Memory-augmented Degradation Pattern Module."""
    def __init__(self, d: int, N_mem: int):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(N_mem, d) * 0.02)
        self.gate  = nn.Linear(d * 2, d)
        self.ln    = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q_n = F.normalize(h, dim=-1)
        s_n = F.normalize(self.slots, dim=-1)
        sim = q_n @ s_n.t()
        top2_vals, top2_idx = sim.topk(2, dim=-1)
        alpha = F.softmax(top2_vals, dim=-1)
        h_mem = (alpha.unsqueeze(-1) * self.slots[top2_idx]).sum(dim=1)
        gate = torch.sigmoid(self.gate(torch.cat([h, h_mem], dim=-1)))
        return self.ln(gate * h + (1 - gate) * h_mem)


class BatteryMFormer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m         = cfg.get('model', {})
        d         = m.get('bmf_d_model', 64)
        n_heads   = m.get('bmf_n_heads', 4)
        dropout   = m.get('dropout', 0.1)
        L         = m.get('charge_discharge_length', 300)
        patch_len = m.get('patch_len', 16)
        n_layers  = m.get('bmf_n_dec_layers', 2)
        N_mem     = m.get('bmf_N_mem', 16)

        stride    = patch_len // 2
        n_patches = (L - patch_len) // stride + 1

        self.patch_len = patch_len
        self.stride    = stride

        self.patch_proj = nn.Linear(patch_len * 3, d)
        self.pos_emb    = nn.Parameter(torch.zeros(1, n_patches, d))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads,
            dim_feedforward=d * 4,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.mdpm    = MDPM(d, N_mem)
        self.head    = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d * 2, 1),
        )

    def forward(self, batch: dict):
        x = batch['curves']                          # (B, 3, L)
        x = x.permute(0, 2, 1)                       # (B, L, 3)
        # extract patches
        patches = x.unfold(1, self.patch_len, self.stride)  # (B, P, 3, patch_len)
        B, P, C, PL = patches.shape
        patches = patches.reshape(B, P, C * PL)      # (B, P, 3*patch_len)
        z = self.patch_proj(patches) + self.pos_emb  # (B, P, d)
        z = self.encoder(z)                           # (B, P, d)
        h = z.mean(dim=1)                             # (B, d)
        h = self.mdpm(h)                              # (B, d)
        pred = self.head(h)                           # (B, 1)
        return pred, None
