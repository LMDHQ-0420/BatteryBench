"""
_masking.py — 多样本 + attention mask 架构的共享工具

soh_point 为 FULL_CYCLE_MODE：每个样本只含"当前观测到的这一圈" (S 恒为 1)，
真正有意义的时序轴是圈内曲线本身 (cycle_curve_data 的 L=300 采样点，或 Q 的
n_grid=200 电压网格点)，S 轴不再承载任何时间语义。

    batch['cycle_curve_data'] : (B, 1, 3, L)
    batch['Q']                 : (B, 1, N)
    batch['curve_attn_mask']  : (B, 1)         恒为 1（S=1 时不存在未观测圈）

提供 (圈内曲线视角，S=1 场景下应优先使用):
    get_curve_seq(batch)           → (B, L, 3)     squeeze S，L 为真实时序轴
    get_q_seq(batch)                → (B, N, 1)     squeeze S，N(n_grid) 为真实时序轴

历史遗留 (S 轴视角，仅 early_cycle 定长模式下仍有意义，S=1 时已退化):
    get_inputs(batch)              → (x, mask)
    flatten_cycles(x)              → (B, S, 3*L)   每圈拼成一个 token
    seq_lengths(mask)              → (B,) long     每样本已观测圈数（≥1）
    key_padding_mask(mask)         → (B, S) bool   True=需屏蔽（未观测），供 nn.Transformer 用
"""

import torch


def get_curve_seq(batch: dict) -> torch.Tensor:
    """(B, S=1, 3, L) → (B, L, 3)。S 恒为 1，squeeze 后以 L 为时序轴。"""
    x = batch['cycle_curve_data']            # (B, 1, 3, L)
    x = x.squeeze(1)                         # (B, 3, L)
    return x.permute(0, 2, 1).contiguous()   # (B, L, 3)


def get_q_seq(batch: dict) -> torch.Tensor:
    """(B, S=1, N) → (B, N, 1)。S 恒为 1，squeeze 后以 n_grid 为时序轴。"""
    q = batch['Q']                           # (B, 1, N)
    q = q.squeeze(1)                         # (B, N)
    return q.unsqueeze(-1)                   # (B, N, 1)


def get_inputs(batch: dict):
    x = batch['cycle_curve_data']            # (B, S, 3, L)
    mask = batch.get('curve_attn_mask')
    if mask is None:
        B, S = x.shape[0], x.shape[1]
        mask = torch.ones(B, S, device=x.device, dtype=x.dtype)
    return x, mask


def flatten_cycles(x: torch.Tensor) -> torch.Tensor:
    """(B, S, 3, L) → (B, S, 3*L)。"""
    B, S = x.shape[0], x.shape[1]
    return x.reshape(B, S, -1)


def seq_lengths(mask: torch.Tensor) -> torch.Tensor:
    """(B, S) → (B,) long，每样本已观测圈数，至少为 1（防 pack 报错）。"""
    lengths = mask.sum(dim=1).long()
    return lengths.clamp(min=1)


def key_padding_mask(mask: torch.Tensor) -> torch.Tensor:
    """(B, S) → (B, S) bool，True 表示该位置需被 attention 屏蔽（未观测圈）。"""
    return mask <= 0
