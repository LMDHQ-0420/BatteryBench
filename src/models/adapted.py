"""Capacity-safe adaptations of battery-specific baselines."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._masking import seq_lengths


def input_channels(cfg: dict) -> int:
    return 2 if cfg.get('data', {}).get('task') == 'soh_point' else 3


def output_size(cfg: dict) -> int:
    return cfg.get('data', {}).get('n_future', 5000) if cfg.get('data', {}).get('task') == 'soh_traj' else 1


class _Inception1D(nn.Module):
    def __init__(self, channels: int, width: int):
        super().__init__()
        part = width // 3
        self.branches = nn.ModuleList([
            nn.Conv1d(channels, part, 3, padding=1),
            nn.Conv1d(channels, part, 7, padding=3),
            nn.Conv1d(channels, width - 2 * part, 15, padding=7),
        ])
        self.norm = nn.BatchNorm1d(width)

    def forward(self, curves):
        return F.relu(self.norm(torch.cat([branch(curves) for branch in self.branches], dim=1)))


class CurveIC2ML(nn.Module):
    """IC2ML-style token attention and multiscale convolution over charge curves."""

    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg.get('model', {})
        channels = input_channels(cfg)
        width = model_cfg.get('ic2ml_d_model', 64)
        heads = model_cfg.get('ic2ml_n_heads', 4)
        dropout = model_cfg.get('dropout', 0.1)
        self.patch = 16
        self.tokenizer = nn.Conv1d(channels, width, self.patch, stride=8)
        layer = nn.TransformerEncoderLayer(
            width, heads, width * 4, dropout, batch_first=True,
        )
        self.attention = nn.TransformerEncoder(layer, 1)
        self.local = _Inception1D(channels, width)
        self.local_pool = nn.AdaptiveAvgPool1d(1)
        self.history = nn.GRU(width * 2, width, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width, output_size(cfg)),
        )

    def forward(self, batch):
        curves = batch['cycle_curve_data']
        mask = batch['curve_attn_mask']
        batch_size, steps, channels, length = curves.shape
        flat = curves.reshape(batch_size * steps, channels, length)
        tokens = self.tokenizer(flat).transpose(1, 2)
        global_feature = self.attention(tokens).mean(dim=1)
        local_feature = self.local_pool(self.local(flat)).flatten(1)
        features = torch.cat([global_feature, local_feature], dim=1).reshape(batch_size, steps, -1)
        packed = nn.utils.rnn.pack_padded_sequence(
            features, seq_lengths(mask).cpu(), batch_first=True, enforce_sorted=False,
        )
        _, hidden = self.history(packed)
        return self.head(hidden[-1]), None


class _ChargeEncoder(nn.Module):
    def __init__(self, channels: int, width: int, dropout: float):
        super().__init__()
        self.curve = nn.Sequential(
            nn.Conv1d(channels, 32, 7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, 7, padding=3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(),
            nn.Linear(64 * 8, width), nn.LayerNorm(width), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, curves, mask):
        batch_size, steps, channels, length = curves.shape
        features = self.curve(curves.reshape(batch_size * steps, channels, length))
        features = features.reshape(batch_size, steps, -1)
        weights = mask.unsqueeze(-1).to(features.dtype)
        return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class CurveBatLiNet(nn.Module):
    """BatLiNet-style intra/inter-cell learning using only the public charge curves."""

    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg.get('model', {})
        width = model_cfg.get('batlinet_d_model', 64)
        dropout = model_cfg.get('dropout', 0.1)
        channels = input_channels(cfg)
        target_size = output_size(cfg)
        self.lam = model_cfg.get('batlinet_lambda', 1.0)
        self.alpha = model_cfg.get('batlinet_alpha', 0.5)
        self.n_ref = model_cfg.get('batlinet_n_ref', 64)
        self.intra_encoder = _ChargeEncoder(channels, width, dropout)
        self.inter_encoder = _ChargeEncoder(channels, width, dropout)
        self.intra_head = nn.Linear(width, target_size)
        self.inter_head = nn.Linear(width, target_size, bias=False)
        self._reference = None

    @staticmethod
    def _masked_mse(prediction, target, mask=None):
        if mask is None:
            return F.mse_loss(prediction, target)
        return F.mse_loss(prediction[mask], target[mask]) if mask.any() else prediction.sum() * 0.0

    @staticmethod
    def _target(batch):
        for key in ('labels', 'soh_point', 'soh_traj'):
            if key in batch:
                return batch[key]
        raise KeyError('No target found in batch')

    def compute_loss(self, batch, device):
        curves = batch['cycle_curve_data'].to(device)
        mask = batch['curve_attn_mask'].to(device)
        target = self._target(batch).to(device)
        intra = self.intra_head(self.intra_encoder(curves, mask))
        if 'trajectory_mask' in batch:
            target_mask = batch['trajectory_mask'].to(device)[:, :intra.shape[1]] > 0
            loss = self._masked_mse(intra, target[:, :intra.shape[1]], target_mask)
        else:
            loss = self._masked_mse(intra, target)
        if len(curves) < 2:
            return loss

        count = min(64, len(curves) * (len(curves) - 1))
        left = torch.randint(0, len(curves), (count,), device=device)
        right = torch.randint(0, len(curves), (count,), device=device)
        right = torch.where(left == right, (right + 1) % len(curves), right)
        pair_mask = mask[left] * mask[right]
        difference = curves[left] - curves[right]
        prediction = self.inter_head(self.inter_encoder(difference, pair_mask))
        difference_target = target[left] - target[right]
        if 'trajectory_mask' in batch:
            trajectory_mask = batch['trajectory_mask'].to(device)
            valid = (trajectory_mask[left] * trajectory_mask[right])[:, :prediction.shape[1]] > 0
            inter_loss = self._masked_mse(
                prediction, difference_target[:, :prediction.shape[1]], valid,
            )
        else:
            inter_loss = self._masked_mse(prediction, difference_target)
        return loss + self.lam * inter_loss

    def set_reference(self, curves, targets, masks=None):
        if masks is None:
            masks = curves.new_ones(curves.shape[:2])
        self._reference = (curves, targets, masks)

    def clear_reference(self):
        self._reference = None

    def forward(self, batch):
        curves = batch['cycle_curve_data']
        mask = batch['curve_attn_mask']
        feature = self.intra_encoder(curves, mask)
        intra = self.intra_head(feature)
        if self._reference is None:
            return intra, None

        reference_curves, reference_targets, reference_masks = self._reference
        reference_curves = reference_curves.to(curves.device)
        reference_targets = reference_targets.to(curves.device)
        reference_masks = reference_masks.to(curves.device)
        if len(reference_curves) > self.n_ref:
            indices = torch.arange(self.n_ref, device=curves.device)
            reference_curves = reference_curves[indices]
            reference_targets = reference_targets[indices]
            reference_masks = reference_masks[indices]
        inter_feature = self.inter_encoder(curves, mask)
        reference_feature = self.inter_encoder(reference_curves, reference_masks)
        differences = inter_feature[:, None, :] - reference_feature[None, :, :]
        inter = self.inter_head(differences) + reference_targets[None, :, :]
        prediction = self.alpha * intra + (1.0 - self.alpha) * inter.mean(dim=1)
        return prediction, None


def curve_features(curves: np.ndarray, useable: int) -> np.ndarray:
    """Fixed statistical features from observed public input curves only."""
    observed = np.asarray(curves[:useable], dtype=np.float64)
    first, last = observed[0], observed[-1]
    features = []
    for current, delta in zip(last, last - first):
        for values in (current, delta):
            features.extend([
                values.mean(), values.std(), values.min(), values.max(), values[-1] - values[0],
            ])
    return np.nan_to_num(np.asarray(features, dtype=np.float64))
