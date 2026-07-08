"""
rul/dlinear.py — DLinear for RUL prediction.
Reference: Zeng et al., AAAI 2023 (individual=False: 权重按 S 轴映射，跨 channel 共享，
           以 1/seq_len 均匀平均初始化 —— 论文设计的核心，而非随机初始化)。

Input:  batch['cycle_curve_data'] (B, S, 3, L) + batch['curve_attn_mask'] (B, S)
        S(圈数) 视作时间轴；每圈的 3*L 特征展开为并行 channel，跨 channel 共享同一组权重。
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._masking import get_inputs, flatten_cycles


class _MovingAvg(nn.Module):
    """边缘重复填充的滑动平均，对齐 Zeng et al. 原始实现（而非零填充）。"""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):  # x: (B, C, S)
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size - 1 - pad_l
        front = x[:, :, :1].expand(-1, -1, pad_l)
        end = x[:, :, -1:].expand(-1, -1, pad_r)
        x = torch.cat([front, x, end], dim=-1)
        return F.avg_pool1d(x, kernel_size=self.kernel_size, stride=1)


class DLinear(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        S = m.get('n_cycles', cfg.get('data', {}).get('early_cycle', 100))
        kernel = m.get('dlinear_kernel', 25)
        pred_len = 1

        self.decompose = _MovingAvg(kernel)
        self.w_trend = nn.Linear(S, pred_len)
        self.w_seasonal = nn.Linear(S, pred_len)
        with torch.no_grad():
            self.w_trend.weight.fill_(1.0 / S)
            self.w_seasonal.weight.fill_(1.0 / S)
            self.w_trend.bias.zero_()
            self.w_seasonal.bias.zero_()

    def forward(self, batch: dict):
        x, _ = get_inputs(batch)              # (B, S, 3, L)
        x = flatten_cycles(x)                 # (B, S, F=3*L)
        xT = x.permute(0, 2, 1)               # (B, F, S) — F 个 channel 共享同一组权重
        trend = self.decompose(xT)            # (B, F, S)
        seasonal = xT - trend
        out = self.w_trend(trend) + self.w_seasonal(seasonal)  # (B, F, pred_len)
        return out.mean(dim=1), None          # 跨 channel 平均 → (B, pred_len)
