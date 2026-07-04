"""
soh_point/ic2ml.py — IC²ML for SOH single-point estimation.
Reference: Huang et al., Journal of Power Sources 666 (2026) 239148

Input: batch['Q']  (B, S, N) — 未观测圈已由 dataset 置零
       batch['curve_attn_mask'] (B, S)
Output: (pred:(B,1), None)  — 预测最后观测圈的 SOH
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        branch_ch = out_ch // 3
        extra = out_ch - branch_ch * 3
        self.b1 = nn.Conv2d(in_ch, branch_ch, kernel_size=(1, 3), padding=(0, 1))
        self.b2 = nn.Conv2d(in_ch, branch_ch, kernel_size=(3, 1), padding=(1, 0))
        self.b3 = nn.Conv2d(in_ch, branch_ch + extra, kernel_size=(3, 3), padding=(1, 1))
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return F.relu(self.bn(out))


class CrossAttention(nn.Module):
    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x_q, x_kv):
        B, d = x_q.shape
        q = self.q_proj(x_q).unsqueeze(1)
        k = self.k_proj(x_kv).unsqueeze(1)
        v = self.v_proj(x_kv).unsqueeze(1)
        def split_heads(t):
            return t.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, 1, d).squeeze(1)
        return self.out_proj(out)


class IC2ML(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        n_cycles = m.get('n_cycles', cfg.get('data', {}).get('early_cycle', 100))
        n_grid   = m.get('n_grid', 200)
        d_model  = m.get('ic2ml_d_model', 64)
        n_heads  = m.get('ic2ml_n_heads', 4)
        dropout  = m.get('dropout', 0.1)

        self.stride = 20
        pts = n_grid // self.stride

        self.cycle_fcn = nn.Sequential(
            nn.Linear(pts, d_model), nn.LayerNorm(d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        pe = torch.zeros(n_cycles, d_model)
        pos = torch.arange(n_cycles).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.inter_attn = nn.TransformerEncoder(enc_layer, num_layers=1)

        self.inception = InceptionBlock2D(in_ch=1, out_ch=d_model)
        self.pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.proj2d = nn.Linear(d_model, d_model)

        self.cross_attn = CrossAttention(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, 1),
        )

    def forward(self, batch):
        Q = batch['Q']                          # (B, S, N)
        B, S, N = Q.shape
        mask = batch.get('curve_attn_mask')
        if mask is None:
            mask = torch.ones(B, S, device=Q.device, dtype=Q.dtype)
        kpm = mask <= 0

        q1d = Q[:, :, ::self.stride]
        h1d = self.cycle_fcn(q1d)
        h1d = h1d + self.pe[:, :S, :]
        h1d = self.inter_attn(h1d, src_key_padding_mask=kpm)
        mm = mask.unsqueeze(-1)
        feat1d = (h1d * mm).sum(dim=1) / mm.sum(dim=1).clamp(min=1)

        q2d = Q.unsqueeze(1)
        feat2d = self.inception(q2d)
        feat2d = self.pool2d(feat2d).flatten(1)
        feat2d = self.proj2d(feat2d)

        fused = self.cross_attn(feat2d, feat1d)
        fused = self.norm(fused + feat2d)

        pred = self.head(fused)                 # (B, 1)
        return pred, None
