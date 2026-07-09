"""
rul/micn.py — MICN for RUL prediction.
Reference: Wang et al., AAAI 2023.
Input:  batch['cycle_curve_data'] (B, S, 3, L) + batch['curve_attn_mask'] (B, S)
        未观测圈已由 dataset 置零。每圈拼成 token (3*L)。
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._masking import get_inputs, flatten_cycles


def _conv_out_len(L: int, kernel: int, stride: int, padding: int) -> int:
    return max(1, (L + 2 * padding - kernel) // stride + 1)


class _SeriesDecomp(nn.Module):
    """边缘重复填充的滑动平均分解（同 Autoformer 约定）。"""

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
    """单一尺度的 downsample-conv → isometric-conv → upsample-transconv 三段式管线。
    isometric conv 的 kernel 恰好等于下采样后的长度，输出长度收缩为 1，
    通过广播把“全局”特征重新注入下采样序列的每个位置（对齐原始 MICN 设计）。
    """

    def __init__(self, d_model: int, seq_len: int, down_kernel: int, dropout: float):
        super().__init__()
        pad = down_kernel // 2
        self.conv_down = nn.Conv1d(d_model, d_model, kernel_size=down_kernel,
                                    stride=down_kernel, padding=pad)
        s1 = _conv_out_len(seq_len, down_kernel, down_kernel, pad)
        self.conv_iso = nn.Conv1d(d_model, d_model, kernel_size=s1, padding=0)
        self.conv_up = nn.ConvTranspose1d(d_model, d_model, kernel_size=down_kernel,
                                           stride=down_kernel)
        self.seq_len = seq_len
        self.norm_iso = nn.LayerNorm(d_model)
        self.norm_up = nn.LayerNorm(d_model)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):          # x: (B, S, d)
        xt = x.transpose(1, 2)                              # (B, d, S)
        x1 = self.drop(self.act(self.conv_down(xt)))         # (B, d, S1)
        g = self.drop(self.act(self.conv_iso(x1)))           # (B, d, 1)  全局特征
        h = self.norm_iso((g + x1).transpose(1, 2)).transpose(1, 2)  # 广播相加, (B, d, S1)
        up = self.drop(self.act(self.conv_up(h)))            # (B, d, ~S)
        up = up[:, :, :self.seq_len]
        out = self.norm_up(up.transpose(1, 2) + x)           # (B, S, d)
        return out


class MICN(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        S       = m.get('n_cycles', cfg.get('data', {}).get('early_cycle', 100))
        L       = cfg.get('data', {}).get('charge_discharge_length', 300)
        d_model = m.get('micn_d_model', 64)
        scales  = m.get('micn_scales', [3, 7, 13])
        dropout = m.get('dropout', 0.1)

        decomp_kernels = [k if k % 2 == 1 else k + 1 for k in scales]

        self.input_proj = nn.Linear(3 * L, d_model)
        self.trend_proj = nn.Linear(3 * L, d_model)
        self.decomp_multi = _SeriesDecompMulti(decomp_kernels)
        self.blocks = nn.ModuleList([_MICBlock(d_model, S, k, dropout) for k in scales])
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
        x, mask = get_inputs(batch)          # (B, S, 3, L), (B, S)
        x = flatten_cycles(x)                # (B, S, 3*L)  未观测圈已是0
        seasonal, trend = self.decomp_multi(x)             # (B, S, 3*L) each

        h = self.input_proj(seasonal)                      # (B, S, d)
        multi = [block(h) for block in self.blocks]         # list of (B, S, d)
        mg = torch.stack(multi, dim=1)                      # (B, n_scales, S, d)
        mg = self.merge(mg.permute(0, 3, 1, 2)).squeeze(2).permute(0, 2, 1)  # (B, S, d)

        y = self.norm1(mg)
        y = self.drop(self.ff2(F.relu(self.ff1(y.transpose(1, 2))))).transpose(1, 2)
        fused = self.norm2(mg + y)                          # (B, S, d)

        fused = fused + self.trend_proj(trend)              # 趋势项在最后加回

        m = mask.unsqueeze(-1)                               # (B, S, 1)
        feat = (fused * m).sum(1) / m.sum(1).clamp(min=1)    # masked mean
        pred = self.head(feat)                               # (B, 1)
        return pred, None
