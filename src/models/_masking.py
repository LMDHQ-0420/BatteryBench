"""Shared helpers for charge-curve inputs and observed-cycle masks."""

import torch


def get_curve_seq(batch: dict) -> torch.Tensor:
    """Return the single current-cycle curve as (B, L, C)."""
    return batch['cycle_curve_data'].squeeze(1).permute(0, 2, 1).contiguous()


def get_inputs(batch: dict):
    return batch['cycle_curve_data'], batch['curve_attn_mask']


def flatten_cycles(curves: torch.Tensor) -> torch.Tensor:
    return curves.flatten(start_dim=2)


def seq_lengths(mask: torch.Tensor) -> torch.Tensor:
    return mask.sum(dim=1).long().clamp_min(1)


def key_padding_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask <= 0
