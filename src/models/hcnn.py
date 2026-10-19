"""V/I charge-curve adaptation of Yang et al.'s HCNN."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.adapted import output_size


class HCNN(nn.Module):
    """Charge/difference fusion with 3D/2D CNN, CFA, and multiscale cycle attention."""

    def __init__(self, cfg: dict):
        super().__init__()
        model = cfg.get('model', {})
        width = model.get('hcnn_d_model', 32)
        self.fusion = nn.Sequential(
            nn.Conv3d(2, width, (3, 2, 7), padding=(1, 0, 3)),
            nn.ReLU(),
        )
        self.curve1 = nn.Conv2d(width, width, (3, 5), padding=(1, 2))
        self.curve2 = nn.Conv2d(width, width, (3, 3), padding=1)
        self.cfa = nn.Sequential(
            nn.Linear(width, width // 4), nn.ReLU(),
            nn.Linear(width // 4, width), nn.Sigmoid(),
        )
        self.cycle_scales = nn.ModuleList(
            nn.Conv1d(width, width, kernel, padding=kernel // 2, groups=width)
            for kernel in (1, 3, 5)
        )
        self.cycle_score = nn.Conv1d(width * 3, 1, 1)
        self.head = nn.Sequential(
            nn.Linear(width, 128), nn.ReLU(),
            nn.Dropout(model.get('dropout', 0.1)),
            nn.Linear(128, output_size(cfg)),
        )

    def forward(self, batch):
        curves = batch['cycle_curve_data'][:, :, :2]
        mask = batch['curve_attn_mask'] > 0
        b, s, _, length = curves.shape
        # Internal curve pooling keeps the public 400-point input unchanged.
        curves = F.adaptive_avg_pool1d(curves.reshape(b * s, 2, length), 64)
        curves = curves.reshape(b, s, 2, 64) * mask[:, :, None, None]
        changes = torch.cat((torch.zeros_like(curves[:, :1]),
                             curves[:, 1:] - curves[:, :-1]), dim=1)
        changes = changes * mask[:, :, None, None]
        fused = self.fusion(torch.stack((curves, changes), dim=1)).squeeze(3)
        fused = fused * mask[:, None, :, None]
        features = F.relu(self.curve1(fused)) * mask[:, None, :, None]
        features = F.relu(self.curve2(features)) * mask[:, None, :, None]
        weights = mask[:, None, :, None].to(features.dtype)
        channel_mean = (features * weights).sum(dim=(2, 3)) / (
            weights.sum(dim=(2, 3)).clamp_min(1) * features.shape[-1]
        )
        features = features * self.cfa(channel_mean)[:, :, None, None]
        cycles = features.mean(dim=-1) * mask[:, None]
        scales = torch.cat([conv(cycles) for conv in self.cycle_scales], dim=1)
        scores = self.cycle_score(scales).squeeze(1).masked_fill(~mask, -1e4)
        attended = (cycles * scores.softmax(dim=-1)[:, None]).sum(dim=-1)
        return self.head(attended), None
