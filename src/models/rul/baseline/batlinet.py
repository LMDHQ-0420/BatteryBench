"""
batlinet.py — BatLiNet: Battery Lifetime Prediction with Inter-Cell Deep Learning
Reference: Zhang et al., Nature Machine Intelligence 7 (2025) 270-277

Architecture (Eq. 8 of the paper):
  Intra-cell encoder f_θ(·)：CNN，独立参数，处理原始曲线 Q
  Inter-cell encoder g_φ(·)：CNN，独立参数（与 f_θ 不共享权重），处理差分曲线 ΔQ
  Shared linear head w：两条分支的嵌入都映射到同一个线性层输出预测值

  Intra-cell path: ŷ° = w^T f_θ(x)         → 直接预测寿命
  Inter-cell path: ŷᶜ = w^T g_φ(x - x')     → 预测寿命差值

  Training loss (Eq. 9):
    L = Σ ||w^T f(x_i) - y_i||² + λ Σ_i Σ_{j≠i} ||w^T g(x_i - x_j) - (y_i - y_j)||²

  Inference (Eq. 10):
    ŷ = α * ŷ° + (1-α) * mean_k[ŷᶜ_k + y'_k]
    参考电池 x'_k 从训练集中随机采样（论文 Fig. 5 显示 64 个参考电池效果最稳）

Input: batch['Q']     (B, S, N)  — target cell feature maps
       batch['Q_ref'] (R, S, N)  — reference cell feature maps (inference)
       batch['y_ref'] (R, 1)     — reference cell lifetimes (inference)
Output: (pred:(B,1), None)

Note: during training, reference pairs are sampled internally from the batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CNNEncoder(nn.Module):
    """
    CNN encoder：(B, S, N) Q-feature map → (B, d) embedding。
    Uses Conv2d treating Q as a 2D image (S×N), similar to BatLiNet Fig. 2c.
    """
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.conv = nn.Sequential(
            # (B, 1, S, N)
            nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)),   # → (B, 64, 4, 8)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, S, N)
        h = self.conv(x.unsqueeze(1))   # (B, 64, 4, 8)
        return self.fc(h)               # (B, d_model)


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
        self.head = nn.Linear(d_model, 1, bias=True)   # 共享线性头 w

        # store reference cells for inference
        self._ref_Q = None   # (R, S, N)
        self._ref_y = None   # (R, 1)

    # ── training helpers ────────────────────────────────────────────────────

    def _intra_pred(self, Q):
        """直接寿命预测：ŷ° = w^T f_θ(x)"""
        h = self.encoder_intra(Q)   # (B, d)
        return self.head(h)         # (B, 1)

    def _inter_pred(self, dQ):
        """寿命差值预测：ŷᶜ = w^T g_φ(Δx)"""
        h = self.encoder_inter(dQ)  # (B, d)
        return self.head(h)         # (B, 1)

    def compute_loss(self, batch, device):
        """
        Joint training loss (Eq. 9).
        Uses at most max_pairs random pairs to avoid OOM on large batches.
        """
        Q = batch['Q'].to(device)                       # (B, S, N)
        y = batch['labels'].to(device)                  # (B, 1) — EOL 绝对值
        B = Q.shape[0]

        # intra-cell loss
        pred_intra = self._intra_pred(Q)                    # (B, 1)
        loss_intra = F.mse_loss(pred_intra, y)

        if B < 2:
            return loss_intra

        # sample up to max_pairs ordered pairs (i, j), i≠j
        max_pairs = 64  # cap memory usage
        all_i, all_j = [], []
        for i in range(B):
            for j in range(B):
                if i != j:
                    all_i.append(i)
                    all_j.append(j)

        n_pairs = len(all_i)
        if n_pairs > max_pairs:
            idx = torch.randperm(n_pairs)[:max_pairs]
            all_i = [all_i[k] for k in idx.tolist()]
            all_j = [all_j[k] for k in idx.tolist()]

        idx_i = torch.tensor(all_i, device=device)
        idx_j = torch.tensor(all_j, device=device)

        dQ = Q[idx_i] - Q[idx_j]                           # (P, S, N)
        dy = y[idx_i] - y[idx_j]                           # (P, 1)

        pred_inter = self._inter_pred(dQ)                   # (P, 1)
        loss_inter = F.mse_loss(pred_inter, dy)

        return loss_intra + self.lam * loss_inter

    # ── reference cell management ────────────────────────────────────────────

    def set_reference(self, Q_ref: torch.Tensor, y_ref: torch.Tensor):
        """
        Cache reference cells for inference.
        Q_ref: (R, S, N), y_ref: (R, 1)
        """
        self._ref_Q = Q_ref
        self._ref_y = y_ref

    def clear_reference(self):
        self._ref_Q = None
        self._ref_y = None

    # ── inference forward ────────────────────────────────────────────────────

    def forward(self, batch):
        Q = batch['Q']                  # (B, S, N)
        device = Q.device
        B = Q.shape[0]

        # intra-cell prediction
        pred_intra = self._intra_pred(Q)    # (B, 1)

        # inter-cell prediction (if references available in batch or cached)
        Q_ref = batch.get('Q_ref', self._ref_Q)
        y_ref = batch.get('y_ref', self._ref_y)

        if Q_ref is None or y_ref is None:
            # no references → return intra prediction only
            return pred_intra, None

        Q_ref = Q_ref.to(device)            # (R, S, N)
        y_ref = y_ref.to(device)            # (R, 1)
        R = Q_ref.shape[0]

        # cap references to self.n_ref (论文 Fig. 5b：64 个参考电池最稳)
        if R > self.n_ref:
            idx = torch.randperm(R, device=device)[:self.n_ref]
            Q_ref = Q_ref[idx]
            y_ref = y_ref[idx]
            R = self.n_ref

        # 所有 (b_idx, ref_idx) 组合一次性 batch 过 CNN，等价于逐样本 for 循环
        # （eval 模式下 BatchNorm 用 running stats，与 batch 组成无关，结果一致）
        dQ = (Q.unsqueeze(1) - Q_ref.unsqueeze(0)).reshape(B * R, *Q.shape[1:])  # (B*R, S, N)
        pred_diff = self._inter_pred(dQ).view(B, R, 1)                          # (B, R, 1)
        pred_inter = (pred_diff + y_ref.unsqueeze(0)).mean(dim=1)   # (B, 1)  论文 Eq.10: mean

        # blend intra + inter
        pred = self.alpha * pred_intra + (1 - self.alpha) * pred_inter  # (B, 1)
        return pred, None
