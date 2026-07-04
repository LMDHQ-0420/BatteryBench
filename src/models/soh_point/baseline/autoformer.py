"""
soh_point/autoformer.py — Autoformer for SOH single-point estimation.
Reference: Wu et al., NeurIPS 2021 (simplified adaptation).
Input:  batch['cycle_curve_data'] (B, S, 3, L) + batch['curve_attn_mask'] (B, S)
        未观测圈已由 dataset 置零。每圈拼成 token (3*L)。
Output: (pred:(B,1), None)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._masking import get_inputs, flatten_cycles


class _AutoCorrelation(nn.Module):
    def __init__(self, d_model: int, n_heads: int, top_k: int = 3):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.top_k    = top_k
        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, S, d = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        q_fft = torch.fft.rfft(q, dim=2)
        k_fft = torch.fft.rfft(k, dim=2)
        corr  = torch.fft.irfft(q_fft * k_fft.conj(), n=S, dim=2)
        k_val = min(self.top_k, S)
        weights, delays = corr.topk(k_val, dim=2)
        weights = F.softmax(weights, dim=2)
        out = torch.zeros_like(v)
        for i in range(k_val):
            d_shift  = delays[:, :, i:i+1, :]
            v_rolled = torch.roll(v, shifts=-int(d_shift.float().mean().item()), dims=2)
            out += weights[:, :, i:i+1, :] * v_rolled
        out = out.permute(0, 2, 1, 3).contiguous().view(B, S, d)
        return self.out_proj(out)


class _AutoformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, kernel: int, dropout: float):
        super().__init__()
        self.autocorr = _AutoCorrelation(d_model, n_heads)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x + self.drop(self.autocorr(x)))
        return self.norm2(h + self.drop(self.ff(h)))


class Autoformer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        S        = m.get('n_cycles', cfg.get('data', {}).get('early_cycle', 100))
        L        = cfg.get('data', {}).get('charge_discharge_length', 300)
        d_model  = m.get('autoformer_d_model', 64)
        n_heads  = m.get('autoformer_n_heads', 4)
        n_layers = m.get('autoformer_n_layers', 2)
        kernel   = m.get('autoformer_kernel', 13)
        dropout  = m.get('dropout', 0.1)

        self.input_proj = nn.Linear(3 * L, d_model)
        pe = torch.zeros(S, d_model)
        pos = torch.arange(S).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

        self.layers = nn.ModuleList([
            _AutoformerLayer(d_model, n_heads, kernel, dropout) for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, batch: dict):
        x, mask = get_inputs(batch)           # (B, S, 3, L), (B, S)
        x = flatten_cycles(x)                 # (B, S, 3*L)
        h = self.input_proj(x) + self.pe[:, :x.shape[1], :]
        for layer in self.layers:
            h = layer(h)
        m = mask.unsqueeze(-1)
        feat = (h * m).sum(1) / m.sum(1).clamp(min=1)
        pred = self.head(feat)
        return pred, None
