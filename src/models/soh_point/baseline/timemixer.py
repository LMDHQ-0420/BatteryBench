"""
soh_point/timemixer.py — TimeMixer for SOH single-point estimation.
Reference: Wang et al., ICLR 2024 (simplified adaptation).
Input:  batch['curves'] (B, 3, L) → (B, L, 3) multi-scale mixing
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeMixer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        L       = cfg.get('data', {}).get('charge_discharge_length', 300)
        d_model = m.get('timemixer_d_model', 64)
        dropout = m.get('dropout', 0.1)
        scales  = m.get('timemixer_scales', [1, 4, 8, 16])

        self.input_proj = nn.Linear(3, d_model)
        self.pools      = nn.ModuleList()
        self.mixers     = nn.ModuleList()
        self.scale_lens = []

        for k in scales:
            if k == 1:
                self.pools.append(nn.Identity())
                self.scale_lens.append(L)
            else:
                self.pools.append(nn.AvgPool1d(kernel_size=k, stride=k))
                self.scale_lens.append(L // k)
            self.mixers.append(nn.Sequential(
                nn.Linear(self.scale_lens[-1] * d_model, d_model), nn.ReLU(), nn.Dropout(dropout)
            ))

        self.gate = nn.Linear(len(scales) * d_model, len(scales))
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, 1),
        )

    def forward(self, batch: dict):
        x = batch['curves']              # (B, 3, L)
        x = x.permute(0, 2, 1)          # (B, L, 3)
        B = x.shape[0]
        h  = self.input_proj(x)          # (B, L, d)
        hT = h.permute(0, 2, 1)         # (B, d, L)
        scale_feats = []
        for pool, mixer, slen in zip(self.pools, self.mixers, self.scale_lens):
            hs = hT if isinstance(pool, nn.Identity) else pool(hT)
            hs = hs[:, :, :slen].permute(0, 2, 1).reshape(B, -1)
            scale_feats.append(mixer(hs))
        stacked = torch.stack(scale_feats, dim=1)
        weights = F.softmax(self.gate(torch.cat(scale_feats, dim=-1)), dim=-1)
        fused   = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        pred    = self.head(fused)
        return pred, None
