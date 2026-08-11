"""
soh_point/dlinear.py — DLinear for SOH single-point estimation.
Reference: Zeng et al., AAAI 2023 (individual=False: 权重按时间轴映射，跨 channel 共享，
           以 1/seq_len 均匀平均初始化 —— 论文设计的核心，而非随机初始化)。

Input:  batch['cycle_curve_data'] (B, S=1, 3, L) — S 恒为 1（每样本仅当前观测圈），
        真实时间轴是圈内曲线 L（充放电采样点，定长），3 个通道跨 channel 共享同一组权重。
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._masking import get_curve_seq


class _MovingAvg(nn.Module):
    """边缘重复填充的滑动平均，对齐 Zeng et al. 原始实现（而非零填充）。"""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):  # x: (B, C, L)
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
        L        = cfg.get('data', {}).get('charge_discharge_length', 300)
        kernel   = m.get('dlinear_kernel', 25)
        pred_len = 1

        self.decompose = _MovingAvg(kernel)
        self.w_trend    = nn.Linear(L, pred_len)
        self.w_seasonal = nn.Linear(L, pred_len)
        with torch.no_grad():
            self.w_trend.weight.fill_(1.0 / L)
            self.w_seasonal.weight.fill_(1.0 / L)
            self.w_trend.bias.zero_()
            self.w_seasonal.bias.zero_()

    def forward(self, batch: dict):
        x = get_curve_seq(batch)              # (B, L, 3) — 圈内曲线，L 为真实时序轴
        xT = x.permute(0, 2, 1)               # (B, C=3, L)
        trend = self.decompose(xT)            # (B, C, L)
        seasonal = xT - trend
        out = self.w_trend(trend) + self.w_seasonal(seasonal)  # (B, C, pred_len)
        return out.mean(dim=1), None          # 跨 channel 平均 → (B, pred_len)
