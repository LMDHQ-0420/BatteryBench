"""
soh_point/patchtst.py — PatchTST for SOH single-point estimation.
Reference: Nie et al., ICLR 2023.
Input:  batch['cycle_curve_data'] (B, S=1, 3, L) — S 恒为 1（每样本仅当前观测圈）。
        沿圈内曲线 L 轴切 patch，3 个曲线通道独立 patch
        （channel-independent，共享同一套 patch 投影/编码器权重，仅在输出头处混合）。
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn

from src.models._masking import get_curve_seq


class PatchTST(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        L         = cfg.get('data', {}).get('charge_discharge_length', 300)
        patch_len = m.get('patchtst_patch_len', 16)
        stride    = m.get('patchtst_stride', 8)
        d_model   = m.get('patchtst_d_model', 64)
        n_heads   = m.get('patchtst_n_heads', 4)
        n_layers  = m.get('patchtst_n_layers', 2)
        dropout   = m.get('dropout', 0.1)

        self.n_channels = 3
        self.L          = L
        self.patch_len  = min(patch_len, L)
        self.stride     = stride

        n_patches = max(1, (L - self.patch_len) // stride + 1)

        self.patch_proj = nn.Linear(self.patch_len, d_model)
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
            nn.Linear(self.n_channels * n_patches * d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _revin_normalize(self, x):
        """RevIN 风格的实例归一化：按 (B, channel) 在 L 轴上求 mean/std。"""
        mean = x.mean(dim=-1, keepdim=True)                   # (B, C, 1)
        std = torch.sqrt(x.var(dim=-1, keepdim=True) + 1e-5)
        return (x - mean) / std

    def forward(self, batch: dict):
        x = get_curve_seq(batch)               # (B, L, 3)
        xc = x.permute(0, 2, 1)                # (B, C=3, L)
        xc = self._revin_normalize(xc)         # (B, C, L)

        B, C, L = xc.shape
        patches = xc.unfold(-1, self.patch_len, self.stride)   # (B, C, P, patch_len)
        patches = patches.reshape(B * C, -1, self.patch_len)   # (B*C, P, patch_len)

        h = self.patch_proj(patches) + self.pos_emb            # (B*C, P, d_model)
        h = self.encoder(h)                                    # (B*C, P, d_model)
        h = h.reshape(B, C, h.shape[1], -1)                     # (B, C, P, d_model)
        pred = self.head(h)                                     # (B, 1)
        return pred, None
