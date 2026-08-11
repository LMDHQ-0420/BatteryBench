"""
batlinet.py — BatLiNet adapted for SOH point estimation.
Reference: Zhang et al., Nature Machine Intelligence 7 (2025) 270-277.

Architecture (Eq. 8 of the paper): intra-cell 编码器 f_θ 和 inter-cell 编码器 g_φ
参数独立（不共享权重），仅共享最后的线性头 w。

Input: batch['Q'] (B, S, N) — 未观测圈已由 dataset 置零
       batch['curve_attn_mask'] (B, S)
Output: (pred:(B,1), None)  — 预测最后观测圈的 SOH
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CNNEncoder(nn.Module):
    """(B, S, N) Q-feature map → (B, d) embedding via Conv2d."""
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, d_model),
            nn.LayerNorm(d_model), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.fc(self.conv(x.unsqueeze(1)))


class BatLiNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        d_model    = m.get('batlinet_d_model', 64)
        dropout    = m.get('dropout', 0.1)
        self.lam   = m.get('batlinet_lambda', 1.0)
        self.alpha = m.get('batlinet_alpha', 0.5)
        self.n_ref = m.get('batlinet_n_ref', 64)   # 论文 Fig. 5b：64 个参考电池最稳

        # 独立参数的 intra/inter 编码器（论文 Eq. 8：f_θ ≠ g_φ），仅共享最后的线性头 w
        self.encoder_intra = _CNNEncoder(d_model, dropout)
        self.encoder_inter = _CNNEncoder(d_model, dropout)
        self.head = nn.Linear(d_model, 1, bias=True)

        self._ref_Q = None
        self._ref_y = None

    def _intra_pred(self, Q):
        return self.head(self.encoder_intra(Q))

    def _inter_pred(self, dQ):
        return self.head(self.encoder_inter(dQ))

    def compute_loss(self, batch, device):
        Q = batch['Q'].to(device)               # (B, S, N)
        y = batch['soh_point'].to(device)       # (B, 1)
        B = Q.shape[0]

        pred_intra = self._intra_pred(Q)
        loss_intra = F.mse_loss(pred_intra, y)

        if B < 2:
            return loss_intra

        # 直接随机采样非对角 (i,j) 对，避免物化全部 B*(B-1) 组合的纯 Python
        # 双重循环（该循环每 batch 都会执行，是 batlinet 训练严重慢于其它模型
        # 的根因，而非 GPU/CPU 资源争抢）。i==j 时把 j 偏移 1（mod B）即可保证
        # i != j，且不改变采样分布的均匀性。
        max_pairs = 64
        n_pairs = min(max_pairs, B * (B - 1))
        idx_i = torch.randint(0, B, (n_pairs,), device=device)
        idx_j = torch.randint(0, B, (n_pairs,), device=device)
        collide = idx_i == idx_j
        idx_j = torch.where(collide, (idx_j + 1) % B, idx_j)

        dQ = Q[idx_i] - Q[idx_j]
        dy = y[idx_i] - y[idx_j]
        pred_inter = self._inter_pred(dQ)
        loss_inter = F.mse_loss(pred_inter, dy)
        return loss_intra + self.lam * loss_inter

    def set_reference(self, Q_ref: torch.Tensor, y_ref: torch.Tensor):
        self._ref_Q = Q_ref
        self._ref_y = y_ref

    def clear_reference(self):
        self._ref_Q = None
        self._ref_y = None

    def forward(self, batch):
        Q = batch['Q']                          # (B, S, N)
        device = Q.device
        B = Q.shape[0]

        pred_intra = self._intra_pred(Q)

        Q_ref = batch.get('Q_ref', self._ref_Q)
        y_ref = batch.get('y_ref', self._ref_y)
        if Q_ref is None or y_ref is None:
            return pred_intra, None

        Q_ref = Q_ref.to(device)
        y_ref = y_ref.to(device)
        R = Q_ref.shape[0]
        if R > self.n_ref:
            idx = torch.randperm(R, device=device)[:self.n_ref]
            Q_ref = Q_ref[idx]; y_ref = y_ref[idx]; R = self.n_ref

        # cap S before the (B,R) broadcast below — full_cycle_mode sequences can
        # reach ~20000 cycles, which would blow up dQ=(B*R, S, N) memory
        s_cap = 2048
        Q_diff = Q[:, :s_cap] if Q.shape[1] > s_cap else Q

        # align S dimension — full_cycle_mode batches may have different S from ref pool
        S_q, S_r = Q_diff.shape[1], Q_ref.shape[1]
        if S_q != S_r:
            S_max = max(S_q, S_r)
            if S_q < S_max:
                Q_diff = torch.cat([Q_diff, Q_diff.new_zeros(B, S_max - S_q, Q_diff.shape[2])], dim=1)
            else:
                Q_ref = torch.cat([Q_ref, Q_ref.new_zeros(R, S_max - S_r, Q_ref.shape[2])], dim=1)

        # 所有 (b_idx, ref_idx) 组合一次性 batch 过 CNN，等价于逐样本 for 循环
        # （eval 模式下 BatchNorm 用 running stats，与 batch 组成无关，结果一致）
        dQ = (Q_diff.unsqueeze(1) - Q_ref.unsqueeze(0)).reshape(B * R, *Q_diff.shape[1:])  # (B*R, S, N)
        pred_diff = self._inter_pred(dQ).view(B, R, 1)
        pred_inter = (pred_diff + y_ref.unsqueeze(0)).mean(dim=1)  # 论文 Eq.10: mean
        pred = self.alpha * pred_intra + (1 - self.alpha) * pred_inter
        return pred, None
