"""
soh_point/timemixer.py — TimeMixer for SOH single-point estimation.
Reference: Wang et al., ICLR 2024 (simplified adaptation).
Input:  batch['cycle_curve_data'] (B, S, 3, L) + batch['curve_attn_mask'] (B, S)
        未观测圈已由 dataset 置零。每圈拼成 token (3*L)，沿 cycle 轴多尺度混合。
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._masking import get_inputs, flatten_cycles


class TimeMixer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        L       = cfg.get('data', {}).get('charge_discharge_length', 300)
        d_model = m.get('timemixer_d_model', 64)
        dropout = m.get('dropout', 0.1)
        scales  = m.get('timemixer_scales', [1, 4, 8, 16])
        F_dim   = 3 * L

        # Fixed token counts per scale (ratio-based, independent of input S)
        _base = 64
        self.fixed_lens = [max(1, _base // k) for k in scales]

        self.input_proj = nn.Linear(F_dim, d_model)
        self.pools      = nn.ModuleList()
        self.mixers     = nn.ModuleList()

        for fixed_len in self.fixed_lens:
            self.pools.append(nn.AdaptiveAvgPool1d(fixed_len))
            self.mixers.append(nn.Sequential(
                nn.Linear(fixed_len * d_model, d_model), nn.ReLU(), nn.Dropout(dropout)
            ))

        self.gate = nn.Linear(len(scales) * d_model, len(scales))
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, 1),
        )

    def forward(self, batch: dict):
        x, _ = get_inputs(batch)              # (B, S, 3, L)  未观测圈已置零
        B, S = x.shape[0], x.shape[1]
        x = flatten_cycles(x)                 # (B, S, F)
        h  = self.input_proj(x)              # (B, S, d)
        hT = h.permute(0, 2, 1)             # (B, d, S)
        # coarse downsample when S is very large to avoid CUDA AdaptiveAvgPool limits
        if S > 2048:
            stride = S // 1024
            hT = F.avg_pool1d(hT, kernel_size=stride, stride=stride)
        scale_feats = []
        for pool, mixer, fixed_len in zip(self.pools, self.mixers, self.fixed_lens):
            hs = pool(hT)                        # (B, d, fixed_len)
            hs = hs.permute(0, 2, 1).reshape(B, -1)
            scale_feats.append(mixer(hs))
        stacked = torch.stack(scale_feats, dim=1)
        weights = F.softmax(self.gate(torch.cat(scale_feats, dim=-1)), dim=-1)
        fused   = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        pred    = self.head(fused)
        return pred, None
