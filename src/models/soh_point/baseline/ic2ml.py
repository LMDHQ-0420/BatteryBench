"""
soh_point/ic2ml.py — IC²ML for SOH single-point estimation.
Reference: Huang et al., Journal of Power Sources 666 (2026) 239148

Input: batch['Q'] (B, S=1, N) — S 恒为 1（每样本仅当前观测圈）。
       真实时序轴是 IC(V) 曲线的电压网格 N（n_grid），融合两条路径：
       1) token 化 + self-attention：沿 N 轴切 token，做全局自注意力，捕捉曲线整体形状；
       2) 1D Inception 多尺度卷积：沿 N 轴做局部多尺度卷积，捕捉曲线局部峰谷特征。
Output: (pred:(B,1), None)  — 预测当前观测圈的 SOH
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._masking import get_q_seq


class InceptionBlock1D(nn.Module):
    """沿 n_grid 轴的多尺度 1D 卷积（局部感受野捕捉 IC 曲线峰谷特征）。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        branch_ch = out_ch // 3
        extra = out_ch - branch_ch * 3
        self.b1 = nn.Conv1d(in_ch, branch_ch, kernel_size=3, padding=1)
        self.b2 = nn.Conv1d(in_ch, branch_ch, kernel_size=7, padding=3)
        self.b3 = nn.Conv1d(in_ch, branch_ch + extra, kernel_size=15, padding=7)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):          # x: (B, C_in, N)
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return F.relu(self.bn(out))


class CrossAttention(nn.Module):
    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x_q, x_kv):
        B, d = x_q.shape
        q = self.q_proj(x_q).unsqueeze(1)
        k = self.k_proj(x_kv).unsqueeze(1)
        v = self.v_proj(x_kv).unsqueeze(1)
        def split_heads(t):
            return t.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, 1, d).squeeze(1)
        return self.out_proj(out)


class IC2ML(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        n_grid   = m.get('n_grid', 200)
        d_model  = m.get('ic2ml_d_model', 64)
        n_heads  = m.get('ic2ml_n_heads', 4)
        dropout  = m.get('dropout', 0.1)

        self.stride = 5
        n_tokens = max(1, n_grid // self.stride)

        self.token_proj = nn.Linear(self.stride, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.token_attn = nn.TransformerEncoder(enc_layer, num_layers=1)

        self.inception = InceptionBlock1D(in_ch=1, out_ch=d_model)
        self.pool1d = nn.AdaptiveAvgPool1d(1)
        self.proj_local = nn.Linear(d_model, d_model)

        self.cross_attn = CrossAttention(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, 1),
        )

    def forward(self, batch):
        q = get_q_seq(batch).squeeze(-1)        # (B, N) — n_grid 为真实时序轴
        B, N = q.shape

        n_tokens = N // self.stride
        q_trim = q[:, :n_tokens * self.stride]
        tokens = q_trim.reshape(B, n_tokens, self.stride)   # (B, n_tokens, stride)
        h_global = self.token_proj(tokens) + self.pos_emb[:, :n_tokens, :]
        h_global = self.token_attn(h_global)                 # (B, n_tokens, d)
        feat_global = h_global.mean(dim=1)                    # (B, d)

        q_1d = q.unsqueeze(1)                    # (B, 1, N)
        feat_local = self.inception(q_1d)         # (B, d, N)
        feat_local = self.pool1d(feat_local).flatten(1)  # (B, d)
        feat_local = self.proj_local(feat_local)

        fused = self.cross_attn(feat_local, feat_global)
        fused = self.norm(fused + feat_local)

        pred = self.head(fused)                   # (B, 1)
        return pred, None
