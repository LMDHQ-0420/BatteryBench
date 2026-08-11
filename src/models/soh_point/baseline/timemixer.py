"""
soh_point/timemixer.py — TimeMixer for SOH single-point estimation.
Reference: Wang et al., ICLR 2024 (simplified adaptation).
Input:  batch['cycle_curve_data'] (B, S=1, 3, L) — S 恒为 1（每样本仅当前观测圈）。
        真实时序轴是圈内曲线 L，逐时间步的 3 通道向量视作该步的 token，沿 L 轴多尺度混合。
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._masking import get_curve_seq


class TimeMixer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        L       = cfg.get('data', {}).get('charge_discharge_length', 300)
        d_model = m.get('timemixer_d_model', 64)
        dropout = m.get('dropout', 0.1)
        scales  = m.get('timemixer_scales', [1, 4, 8, 16])

        self.fixed_lens = [max(1, L // k) for k in scales]

        self.input_proj = nn.Linear(3, d_model)
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
        x = get_curve_seq(batch)              # (B, L, 3)
        B = x.shape[0]
        h  = self.input_proj(x)               # (B, L, d)
        hT = h.permute(0, 2, 1)               # (B, d, L)
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
