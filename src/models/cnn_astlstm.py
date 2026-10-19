"""Charge-curve adaptation of Li et al.'s CNN-ASTLSTM."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.adapted import input_channels, output_size


class ASTLSTMCell(nn.Module):
    """Coupled forget/input gates, active cell tracking, and an output peephole."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.forget = nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size)
        self.active = nn.Linear(input_size, hidden_size, bias=False)
        self.output = nn.Linear(input_size + hidden_size, hidden_size)
        self.peephole = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, state):
        h, c = state
        joined = torch.cat((x, h), dim=-1)
        forget = torch.sigmoid(self.forget(joined))
        candidate = torch.tanh(self.candidate(joined) + self.active(x) * c)
        c = forget * c + (1 - forget) * candidate
        h = torch.sigmoid(self.output(joined) + self.peephole * c) * torch.tanh(c)
        return h, c


class ASTLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = ASTLSTMCell(input_size, hidden_size)

    def forward(self, sequence):
        h = sequence.new_zeros(sequence.size(0), self.cell.hidden_size)
        c = torch.zeros_like(h)
        outputs = []
        for x in sequence.unbind(dim=1):
            h, c = self.cell(x, (h, c))
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class CNNASTLSTM(nn.Module):
    """CNN plus two AST layers; historical cycle features use an ordered readout."""

    def __init__(self, cfg: dict):
        super().__init__()
        data = cfg.get('data', {})
        model = cfg.get('model', {})
        self.point = data.get('task') == 'soh_point'
        self.n_cycles = data.get('n_cycles', 100)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels(cfg), 46, 7, stride=4, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.ast1 = ASTLSTM(46, 24)
        self.ast2 = ASTLSTM(24, 28)
        readout_size = 28 if self.point else 28 * self.n_cycles
        self.head = nn.Sequential(
            nn.Dropout(model.get('cnn_astlstm_dropout', 0.0609)),
            nn.Linear(readout_size, output_size(cfg)),
        )

    def forward(self, batch):
        curves = batch['cycle_curve_data']
        mask = batch['curve_attn_mask'] > 0
        b, s, _, _ = curves.shape
        if not self.point and s > self.n_cycles:
            raise ValueError(f'Observed sequence has {s} cycles; expected at most {self.n_cycles}')
        valid_curves = curves[mask]
        features = self.ast2(self.ast1(self.cnn(valid_curves).transpose(1, 2)))[:, -1]
        history = curves.new_zeros(b, s, features.shape[-1])
        history[mask] = features
        if self.point:
            representation = history[:, 0]
        else:
            representation = F.pad(history.flatten(1), (0, 28 * (self.n_cycles - s)))
        return self.head(representation), None
