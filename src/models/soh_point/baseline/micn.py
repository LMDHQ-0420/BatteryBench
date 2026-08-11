"""
soh_point/micn.py — MICN for SOH single-point estimation.
Reference: Wang et al., AAAI 2023.
Input:  batch['cycle_curve_data'] (B, S=1, 3, L) — S 恒为 1（每样本仅当前观测圈），
        真实时序轴是圈内曲线 L，逐时间步的 3 通道向量作为该步输入。
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._masking import get_curve_seq


class _SeriesDecomp(nn.Module):
    """边缘重复填充的滑动平均分解（同 Autoformer 约定）。"""

    def __init__(self, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.avg = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=0)

    def forward(self, x):          # x: (B, L, d)
        pad_l = (self.kernel - 1) // 2
        pad_r = self.kernel - 1 - pad_l
        front = x[:, :1, :].expand(-1, pad_l, -1)
        end = x[:, -1:, :].expand(-1, pad_r, -1)
        x_pad = torch.cat([front, x, end], dim=1)
        trend = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        return x - trend, trend


class _SeriesDecompMulti(nn.Module):
    """多核滑动平均分解（对齐 FEDformer 的 series_decomp_multi）。"""

    def __init__(self, kernels):
        super().__init__()
        self.decomps = nn.ModuleList([_SeriesDecomp(k) for k in kernels])

    def forward(self, x):
        seas, trend = [], []
        for d in self.decomps:
            s, t = d(x)
            seas.append(s)
            trend.append(t)
        return sum(seas) / len(seas), sum(trend) / len(trend)


class _MICBlock(nn.Module):
    """单一尺度的 downsample-conv → adaptive-pool (全局聚合) → upsample-transconv 三段式管线。
    用 AdaptiveAvgPool1d(1) 替换 isometric conv，使该 block 与输入序列长度无关。
    """

    def __init__(self, d_model: int, down_kernel: int, dropout: float):
        super().__init__()
        pad = down_kernel // 2
        self.conv_down = nn.Conv1d(d_model, d_model, kernel_size=down_kernel,
                                    stride=down_kernel, padding=pad)
        self.pool_iso  = nn.AdaptiveAvgPool1d(1)
        self.conv_up = nn.ConvTranspose1d(d_model, d_model, kernel_size=down_kernel,
                                           stride=down_kernel)
        self.norm_iso = nn.LayerNorm(d_model)
        self.norm_up = nn.LayerNorm(d_model)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):          # x: (B, L, d)
        B, S, d = x.shape
        xt = x.transpose(1, 2)                              # (B, d, L)
        x1 = self.drop(self.act(self.conv_down(xt)))         # (B, d, L1)
        g = self.pool_iso(x1)                                # (B, d, 1)  全局特征
        h = self.norm_iso((g + x1).transpose(1, 2)).transpose(1, 2)  # 广播相加, (B, d, L1)
        up = self.drop(self.act(self.conv_up(h)))            # (B, d, ~L)
        up = up[:, :, :S]
        out = self.norm_up(up.transpose(1, 2) + x)           # (B, L, d)
        return out


class MICN(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        d_model = m.get('micn_d_model', 64)
        scales  = m.get('micn_scales', [3, 7, 13])
        dropout = m.get('dropout', 0.1)

        decomp_kernels = [k if k % 2 == 1 else k + 1 for k in scales]

        self.input_proj = nn.Linear(3, d_model)
        self.trend_proj = nn.Linear(3, d_model)
        self.decomp_multi = _SeriesDecompMulti(decomp_kernels)
        self.blocks = nn.ModuleList([_MICBlock(d_model, k, dropout) for k in scales])
        self.merge = nn.Conv2d(d_model, d_model, kernel_size=(len(scales), 1))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Conv1d(d_model, d_model * 4, kernel_size=1)
        self.ff2 = nn.Conv1d(d_model * 4, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, batch: dict):
        x = get_curve_seq(batch)             # (B, L, 3)
        seasonal, trend = self.decomp_multi(x)             # (B, L, 3) each

        h = self.input_proj(seasonal)                      # (B, L, d)
        multi = [block(h) for block in self.blocks]         # list of (B, L, d)
        mg = torch.stack(multi, dim=1)                      # (B, n_scales, L, d)
        mg = self.merge(mg.permute(0, 3, 1, 2)).squeeze(2).permute(0, 2, 1)  # (B, L, d)

        y = self.norm1(mg)
        y = self.drop(self.ff2(F.relu(self.ff1(y.transpose(1, 2))))).transpose(1, 2)
        fused = self.norm2(mg + y)                          # (B, L, d)

        fused = fused + self.trend_proj(trend)              # 趋势项在最后加回

        feat = fused.mean(dim=1)                             # (B, d)
        pred = self.head(feat)                               # (B, 1)
        return pred, None
