"""
_masking.py — 多样本 + attention mask 架构的共享工具

所有模型统一消费:
    batch['cycle_curve_data'] : (B, S, 3, L)   S=early_cycle(100)，未观测圈已由 dataset 置零
    batch['curve_attn_mask']  : (B, S)         1=已观测, 0=未观测

提供:
    get_inputs(batch)              → (x, mask)
    flatten_cycles(x)              → (B, S, 3*L)   每圈拼成一个 token
    seq_lengths(mask)              → (B,) long     每样本已观测圈数（≥1）
    key_padding_mask(mask)         → (B, S) bool   True=需屏蔽（未观测），供 nn.Transformer 用
"""

import torch


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
