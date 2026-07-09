"""
rul/patchtst.py — PatchTST for RUL prediction.
Reference: Nie et al., ICLR 2023.
Input:  batch['cycle_curve_data'] (B, S, 3, L) + batch['curve_attn_mask'] (B, S)
        未观测圈已由 dataset 置零。沿 cycle 轴切 patch，3 个曲线通道独立 patch
        （channel-independent，共享同一套 patch 投影/编码器权重，仅在输出头处混合）。
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn

from src.models._masking import get_inputs


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

        self.n_channels = 3
        self.L          = L
        self.patch_len  = patch_len
        self.stride     = stride
        n_patches       = max(1, (S - patch_len) // stride + 1)
        self.n_patches  = n_patches

        # channel-independent patch embedding：同一套权重独立处理每个通道
        self.patch_proj = nn.Linear(patch_len * L, d_model)
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

    def _revin_normalize(self, x, mask):
        """RevIN 风格的实例归一化：按 (B, channel) 在已观测圈上求 mean/std，
        未观测圈保持置零，避免归一化后泄漏非零值。"""
        m = mask.unsqueeze(-1).unsqueeze(-1)                    # (B, S, 1, 1)
        count = (mask.sum(dim=1) * self.L).clamp(min=1).view(-1, 1, 1, 1)  # (B,1,1,1)
        mean = (x * m).sum(dim=1, keepdim=True).sum(dim=3, keepdim=True) / count  # (B,1,C,1)
        var = ((x - mean) ** 2 * m).sum(dim=1, keepdim=True).sum(dim=3, keepdim=True) / count
        std = torch.sqrt(var + 1e-5)
        x_norm = (x - mean) / std
        return x_norm * m

    def forward(self, batch: dict):
        x, mask = get_inputs(batch)           # (B, S, 3, L), (B, S)
        B, S, C, L = x.shape
        x = self._revin_normalize(x, mask)    # (B, S, 3, L)

        xc = x.permute(0, 2, 1, 3).reshape(B * C, S, L)         # (B*C, S, L)
        patches = xc.unfold(1, self.patch_len, self.stride)     # (B*C, P, L, patch_len)
        Bc, P, _, PL = patches.shape
        patches = patches.permute(0, 1, 3, 2).reshape(Bc, P, PL * L)  # (B*C, P, patch_len*L)

        h = self.patch_proj(patches) + self.pos_emb[:, :P, :]

        mpatch = mask.unfold(1, self.patch_len, self.stride)    # (B, P, patch_len)
        kpm = mpatch.sum(-1) <= 0                                # (B, P) bool
        kpm = kpm.repeat_interleave(C, dim=0)                    # (B*C, P)
        full_mask_rows = kpm.all(dim=1)
        if full_mask_rows.any():
            kpm = kpm.clone()
            kpm[full_mask_rows] = False

        h = self.encoder(h, src_key_padding_mask=kpm)            # (B*C, P, d_model)
        h = h.reshape(B, C, P, -1)                                # (B, C, P, d_model)
        pred = self.head(h)                                      # (B, 1)
        return pred, None
