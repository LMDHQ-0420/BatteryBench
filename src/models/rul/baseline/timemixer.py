"""
rul/timemixer.py — TimeMixer for RUL prediction.
Reference: Wang et al., ICLR 2024 (simplified adaptation).
Input:  batch['Q'] (B, S, N)  → capacity_seq = Q.max(dim=-1) → (B, S)
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeMixer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        S       = m.get('n_cycles', 100)
        d_model = m.get('timemixer_d_model', 64)
        dropout = m.get('dropout', 0.1)
        scales  = m.get('timemixer_scales', [1, 4, 8, 16])

        self.pools      = nn.ModuleList()
        self.mixers     = nn.ModuleList()
        self.scale_lens = []

        for k in scales:
            if k == 1:
                self.pools.append(nn.Identity())
                self.scale_lens.append(S)
            else:
                self.pools.append(nn.AvgPool1d(kernel_size=k, stride=k))
                self.scale_lens.append(S // k)
            self.mixers.append(nn.Sequential(
                nn.Linear(self.scale_lens[-1], d_model), nn.ReLU(), nn.Dropout(dropout)
            ))

        self.gate = nn.Linear(len(scales) * d_model, len(scales))
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, 1),
        )

    def forward(self, batch: dict):
        x = batch['Q'].max(dim=-1).values   # (B, S)
        scale_feats = []
        for pool, mixer, slen in zip(self.pools, self.mixers, self.scale_lens):
            xs = x if isinstance(pool, nn.Identity) else pool(x.unsqueeze(1)).squeeze(1)
            scale_feats.append(mixer(xs[:, :slen]))
        stacked = torch.stack(scale_feats, dim=1)
        weights = F.softmax(self.gate(torch.cat(scale_feats, dim=-1)), dim=-1)
        fused   = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        pred    = self.head(fused)
        return pred, None
