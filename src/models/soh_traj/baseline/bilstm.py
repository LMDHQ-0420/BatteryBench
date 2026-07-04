"""
soh_traj/bilstm.py — BiLSTM for SOH degradation trajectory prediction.
Input:  batch['cycle_curve_data'] (B, S, 3, L) + batch['curve_attn_mask'] (B, S)
Output: (pred:(B, n_future), None)
每圈拼成 token (3*L)，用 pack_padded_sequence 按已观测圈数取最后有效步。
"""

import torch
import torch.nn as nn

from src.models._masking import get_inputs, flatten_cycles, seq_lengths


class BiLSTM(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        L        = cfg.get('data', {}).get('charge_discharge_length', 300)
        n_future = cfg.get('data', {}).get('n_future', 5000)
        dropout  = m.get('dropout', 0.1)

        self.lstm = nn.LSTM(
            input_size=3 * L, hidden_size=128,
            num_layers=2, batch_first=True,
            dropout=dropout, bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_future),
        )

    def forward(self, batch: dict):
        x, mask = get_inputs(batch)           # (B, S, 3, L), (B, S)
        x = flatten_cycles(x)                 # (B, S, 3*L)
        lengths = seq_lengths(mask).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)         # h: (4, B, 128)
        pred = self.head(torch.cat([h[-2], h[-1]], dim=-1))  # (B, n_future)
        return pred, None
