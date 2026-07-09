"""
soh_point/autoformer.py — Autoformer for SOH single-point estimation.
Reference: Wu et al., NeurIPS 2021.
Input:  batch['cycle_curve_data'] (B, S, 3, L) + batch['curve_attn_mask'] (B, S)
        未观测圈已由 dataset 置零。每圈拼成 token (3*L)。
Output: (pred:(B,1), None)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._masking import get_inputs, flatten_cycles


class _MyLayernorm(nn.Module):
    """季节项专用 LayerNorm：去除逐时间步的均值偏置，对齐原始 Autoformer 实现。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):          # x: (B, S, d)
        x_hat = self.layernorm(x)
        bias = x_hat.mean(dim=1, keepdim=True)
        return x_hat - bias


class _SeriesDecomp(nn.Module):
    """边缘重复填充的滑动平均分解，对齐原始 Autoformer 实现（而非零填充）。"""

    def __init__(self, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.avg = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=0)

    def forward(self, x):          # x: (B, S, d)
        pad_l = (self.kernel - 1) // 2
        pad_r = self.kernel - 1 - pad_l
        front = x[:, :1, :].expand(-1, pad_l, -1)
        end = x[:, -1:, :].expand(-1, pad_r, -1)
        x_pad = torch.cat([front, x, end], dim=1)
        trend = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        return x - trend, trend


class _AutoCorrelation(nn.Module):
    """FFT 周期检测 + 逐 (batch,head,channel) 保留分辨率的 time-delay 聚合
    （通过 torch.gather 实现，对齐原始实现；而非将 delay 折叠成单一全局标量位移）。
    """

    def __init__(self, d_model: int, n_heads: int, factor: int = 1):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.factor   = factor
        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):          # x: (B, S, d)
        B, S, d = x.shape
        H, E = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, S, H, E).permute(0, 2, 3, 1)  # (B,H,E,S)
        k = self.k_proj(x).view(B, S, H, E).permute(0, 2, 3, 1)
        v = self.v_proj(x).view(B, S, H, E).permute(0, 2, 3, 1)

        q_fft = torch.fft.rfft(q.contiguous(), dim=-1)
        k_fft = torch.fft.rfft(k.contiguous(), dim=-1)
        corr  = torch.fft.irfft(q_fft * k_fft.conj(), n=S, dim=-1)  # (B,H,E,S)

        top_k = max(1, min(S, int(self.factor * math.log(max(S, 2)))))
        weights, delay = corr.topk(top_k, dim=-1)          # (B,H,E,top_k)
        tmp_corr = F.softmax(weights, dim=-1)

        v_rep = v.repeat(1, 1, 1, 2)                        # (B,H,E,2S) 供环绕索引
        init_index = torch.arange(S, device=x.device).view(1, 1, 1, S).expand(B, H, E, S)
        out = torch.zeros_like(v)
        for i in range(top_k):
            idx = init_index + delay[..., i:i + 1]          # (B,H,E,S)
            pattern = torch.gather(v_rep, dim=-1, index=idx)
            out = out + pattern * tmp_corr[..., i:i + 1]

        out = out.permute(0, 3, 1, 2).reshape(B, S, d)       # (B,S,d)
        return self.out_proj(out)


class _AutoformerLayer(nn.Module):
    """渐进式分解结构：AutoCorrelation → decomp1 丢弃趋势 → FFN → decomp2 丢弃趋势
    （对齐原始实现：用分解代替 LayerNorm 作为逐层归一化机制）。
    """

    def __init__(self, d_model: int, n_heads: int, kernel: int, dropout: float):
        super().__init__()
        self.autocorr = _AutoCorrelation(d_model, n_heads)
        self.decomp1  = _SeriesDecomp(kernel)
        self.decomp2  = _SeriesDecomp(kernel)
        self.conv1 = nn.Conv1d(d_model, d_model * 4, kernel_size=1)
        self.conv2 = nn.Conv1d(d_model * 4, d_model, kernel_size=1)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):          # x: (B, S, d)
        x = x + self.drop(self.autocorr(x))
        x, _ = self.decomp1(x)
        y = self.drop(F.gelu(self.conv1(x.transpose(1, 2))))
        y = self.drop(self.conv2(y).transpose(1, 2))
        res, _ = self.decomp2(x + y)
        return res


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
        self.norm = _MyLayernorm(d_model)
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
        h = self.norm(h)
        m = mask.unsqueeze(-1)                # (B, S, 1)
        feat = (h * m).sum(1) / m.sum(1).clamp(min=1)        # masked mean
        pred = self.head(feat)                # (B, 1)
        return pred, None
