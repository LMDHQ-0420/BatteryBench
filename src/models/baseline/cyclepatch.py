"""
cyclepatch.py — CyclePatch-MLP
Reference: BatteryLife benchmark (BatteryLife paper, 2024)

Architecture:
  Intra-cycle encoder: per-cycle shared MLP (Linear→LN→ReLU) × L layers
  Inter-cycle encoder: mean pool over cycles → MLP head
  Optional V/I/Q 3-channel input via batch['vic'] (B, S, 3, N)

Input: batch['Q'] (B, S, N)  [fallback if no 'vic']
       batch['vic'] (B, S, 3, N)  [preferred, 3 channels]
Output: (pred:(B,1), None)
"""

import torch
import torch.nn as nn


class CyclePatch(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        n_grid  = m.get('n_grid', 200)
        D       = m.get('cp_d_model', 64)
        L       = m.get('cp_n_layers', 2)
        H       = m.get('cp_hidden', 128)
        dropout = m.get('dropout', 0.1)

        self.D = D
        self.n_grid = n_grid

        # default: single-channel Q input (N,) per cycle
        # if vic is provided at runtime (3*N), the first forward call
        # will rebuild with the correct in_features via _maybe_rebuild
        self._build(n_grid, D, L, H, dropout)

    def _build(self, in_features: int, D: int, L: int, H: int, dropout: float):
        self._in_features = in_features
        layers = []
        in_f = in_features
        for _ in range(L):
            layers += [
                nn.Linear(in_f, D),
                nn.LayerNorm(D),
                nn.ReLU(),
            ]
            in_f = D
        self.intra_encoder = nn.Sequential(*layers)
        self.inter_head = nn.Sequential(
            nn.Linear(D, H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, 1),
        )
        # store for potential rebuild
        self._D = D
        self._L = L
        self._H = H
        self._dropout = dropout

    def forward(self, batch):
        if 'vic' in batch:
            # (B, S, 3, N) → (B, S, 3*N)
            vic = batch['vic']
            B, S, C, N = vic.shape
            x = vic.view(B, S, C * N)
            in_f = C * N
        else:
            Q = batch['Q']                  # (B, S, N)
            B, S, N = Q.shape
            x = Q
            in_f = N

        # rebuild if input feature size changed (e.g. first vic call)
        if in_f != self._in_features:
            device = x.device
            self._build(in_f, self._D, self._L, self._H, self._dropout)
            self.intra_encoder = self.intra_encoder.to(device)
            self.inter_head = self.inter_head.to(device)

        # intra-cycle: apply per cycle (share weights)
        x_flat = x.reshape(B * S, in_f)    # (B*S, in_f)
        h = self.intra_encoder(x_flat)      # (B*S, D)
        h = h.view(B, S, self.D)            # (B, S, D)

        # inter-cycle: mean pool
        h_pool = h.mean(dim=1)              # (B, D)

        pred = self.inter_head(h_pool)      # (B, 1)
        return pred, None
