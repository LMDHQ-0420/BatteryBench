"""
batterymformer.py — BatteryMFormer adapted for RUL point prediction

Original paper: Tan et al., KDD 2026, "BatteryMFormer: Multi-level Learning
for Battery Degradation Trajectory Forecasting"

Adaptations from original:
  1. Task: trajectory forecasting → RUL point prediction (scalar output)
  2. ACDecoder LLM: Qwen3-0.6B → lightweight hash-based text embedding
     (avoids 600M param dependency; aging condition text is still encoded)
  3. No TrajectoryDecoder / recovery loss (not needed for RUL)
  4. Input: X ∈ R^(S×L×4) [V,I,Q,SOC] from BatteryMFormerDataset

Architecture:
  DualViewEncoder:
    - SOC-view:  1D Conv along SOC axis → M SOC-interval tokens per cycle
                 → temporal encoder across cycles → T^soc ∈ R^(M×d)
    - Temporal-view: CyclePatch + intra-cycle encoder → T^temporal ∈ R^(S×d)
  ACDecoder:
    - Inject aging-condition text embedding as query modulation
    - L_dec layers of ACAttention (cross-attn over dual-view tokens)
  MDPM:
    - N_mem learnable memory slots
    - Top-2 retrieval by cosine similarity
    - Gated fusion with decoder output
  Head: Linear → RUL scalar
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────── text embedding ──────────────────────────────────

class TextHashEmbedder(nn.Module):
    """
    Lightweight aging-condition embedder.
    Hashes text to fixed vocab indices, then learns embeddings.
    No external LLM dependency.
    For each aging_text string, extracts key tokens (cathode, anode, form factor)
    and maps them to learned vectors that are summed.
    """
    VOCAB_SIZE = 256

    def __init__(self, d: int):
        super().__init__()
        self.emb = nn.Embedding(self.VOCAB_SIZE, d)
        self.proj = nn.Linear(d, d)
        self.ln   = nn.LayerNorm(d)

    def _tokenize(self, text: str) -> list:
        """Split on delimiters and hash each token to [0, VOCAB_SIZE)."""
        tokens = [t.strip() for t in text.replace(';', ' ').split() if t.strip()]
        return [int(hashlib.md5(t.encode()).hexdigest(), 16) % self.VOCAB_SIZE
                for t in tokens] if tokens else [0]

    def forward(self, texts: list) -> torch.Tensor:
        """texts: list of B strings → (B, d)"""
        device = self.emb.weight.device
        out = []
        for text in texts:
            idxs = self._tokenize(text)
            idx_t = torch.tensor(idxs, dtype=torch.long, device=device)
            out.append(self.emb(idx_t).mean(dim=0))  # mean-pool tokens
        z = torch.stack(out, dim=0)                   # (B, d)
        return self.ln(self.proj(z))


import hashlib  # noqa: E402 (needed by TextHashEmbedder above)


# ─────────────────────────── SOC-view encoder ────────────────────────────────

class SOCViewEncoder(nn.Module):
    """
    Captures SOC-localized degradation signatures.

    For each cycle i, X_i ∈ R^(L×4):
      1. Conv1D along SOC axis → M patch tokens Z̃_i ∈ R^(d×M)
      2. Stack across S cycles → Z̃ ∈ R^(S×M×d)
      3. For each SOC interval m, apply temporal encoder across cycles
         → t_m^soc ∈ R^d
      4. Concatenate → T^soc ∈ R^(M×d)

    Key insight: each SOC interval has its OWN temporal evolution,
    capturing localized electrochemical degradation signatures.
    """

    def __init__(self, L: int, d: int, M: int, n_heads: int, dropout: float):
        super().__init__()
        P = L // M  # patch length
        self.M = M
        self.P = P
        self.patch_conv = nn.Conv1d(
            in_channels=4,       # V, I, Q, SOC channels
            out_channels=d,
            kernel_size=P,
            stride=P,            # non-overlapping patches
        )
        self.patch_ln = nn.LayerNorm(d)

        # temporal encoder: shared across SOC intervals
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads,
            dim_feedforward=d * 4,
            dropout=dropout, batch_first=True,
            activation='gelu',
        )
        self.temporal_enc = nn.TransformerEncoder(enc_layer, num_layers=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (B, S, L, 4) → T_soc: (B, M, d)"""
        B, S, L, C = X.shape
        # reshape to (B*S, 4, L) for Conv1D
        x = X.view(B * S, L, C).permute(0, 2, 1)   # (B*S, 4, L)
        z = self.patch_conv(x)                        # (B*S, d, M)
        z = z.permute(0, 2, 1)                        # (B*S, M, d)
        z = self.patch_ln(z)
        z = z.view(B, S, self.M, -1)                  # (B, S, M, d)

        # for each SOC interval m, encode across S cycles
        z_perm = z.permute(0, 2, 1, 3)               # (B, M, S, d)
        z_flat = z_perm.reshape(B * self.M, S, -1)    # (B*M, S, d)
        t = self.temporal_enc(z_flat)                  # (B*M, S, d)
        t = t.mean(dim=1)                              # (B*M, d) — pool over S
        T_soc = t.view(B, self.M, -1)                 # (B, M, d)
        return T_soc


# ─────────────────────────── Temporal-view encoder ───────────────────────────

class TemporalViewEncoder(nn.Module):
    """
    Captures global temporal dynamics across cycles.

    For each cycle i:
      1. Flatten X_i ∈ R^(L×4) → project to d (CyclePatch-style)
      2. Intra-cycle encoder refines per-cycle token
      3. Stack S tokens → H^temporal ∈ R^(S×d)
      4. Add cycle-level descriptors (Coulombic efficiency proxy)
    """

    def __init__(self, L: int, d: int, n_heads: int, dropout: float):
        super().__init__()
        # project L*4 → d
        self.cycle_proj = nn.Linear(L * 4, d)
        self.cycle_ln   = nn.LayerNorm(d)

        # intra-cycle encoder (1-layer transformer)
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads,
            dim_feedforward=d * 4,
            dropout=dropout, batch_first=True,
            activation='gelu',
        )
        self.intra_enc = nn.TransformerEncoder(enc, num_layers=1)

        # positional encoding
        self.register_buffer('_pe_buf', torch.zeros(1, 1, d))

    def _pe(self, S: int, d: int, device) -> torch.Tensor:
        pe = torch.zeros(S, d, device=device)
        pos = torch.arange(S, device=device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2, device=device).float()
                        * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # (1, S, d)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (B, S, L, 4) → H_temporal: (B, S, d)"""
        B, S, L, C = X.shape
        x_flat = X.view(B, S, L * C)                  # (B, S, L*4)
        h = self.cycle_ln(self.cycle_proj(x_flat))     # (B, S, d)
        h = h + self._pe(S, h.shape[-1], h.device)
        h = self.intra_enc(h)                          # (B, S, d)
        return h


# ─────────────────────────── ACDecoder ───────────────────────────────────────

class ACAttention(nn.Module):
    """
    Aging-Condition-Aware Attention.
    Modulates queries with aging condition embedding e_ac:
      head_i = Attention((Q + ê_i^ac) W_Q, K W_K, V W_V)
    """

    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            d, n_heads, dropout=dropout, batch_first=True)
        # per-query AC prior: maps e_ac → s query-specific vectors
        self.ac_proj = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, Q: torch.Tensor, KV: torch.Tensor,
                e_ac: torch.Tensor) -> torch.Tensor:
        """
        Q:    (B, s, d)
        KV:   (B, t, d)
        e_ac: (B, d)
        """
        e_ac_exp = self.ac_proj(e_ac).unsqueeze(1)  # (B, 1, d)
        Q_mod = Q + e_ac_exp                         # broadcast over s
        out, _ = self.mha(Q_mod, KV, KV)
        return self.ln(out + Q)


class ACDecoderLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        # self-attention over query tokens
        self.self_attn = nn.MultiheadAttention(
            d, n_heads, dropout=dropout, batch_first=True)
        self.ln0 = nn.LayerNorm(d)
        # cross-attention over temporal tokens with AC modulation
        self.ac_attn1 = ACAttention(d, n_heads, dropout)
        # cross-attention over SOC tokens with AC modulation
        self.ac_attn2 = ACAttention(d, n_heads, dropout)
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d * 4, d))
        self.ln_ffn = nn.LayerNorm(d)

    def forward(self, H: torch.Tensor,
                T_temporal: torch.Tensor,
                T_soc: torch.Tensor,
                e_ac: torch.Tensor) -> torch.Tensor:
        # self-attention
        h2, _ = self.self_attn(H, H, H)
        H = self.ln0(H + h2)
        # cross-attend temporal view
        H = self.ac_attn1(H, T_temporal, e_ac)
        # cross-attend SOC view
        H = self.ac_attn2(H, T_soc, e_ac)
        # FFN
        H = self.ln_ffn(H + self.ffn(H))
        return H


class ACDecoder(nn.Module):
    def __init__(self, d: int, n_heads: int, n_layers: int,
                 n_queries: int, dropout: float):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, d) * 0.02)
        self.layers  = nn.ModuleList([
            ACDecoderLayer(d, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, T_temporal: torch.Tensor,
                T_soc: torch.Tensor,
                e_ac: torch.Tensor) -> torch.Tensor:
        """
        T_temporal: (B, S, d)
        T_soc:      (B, M, d)
        e_ac:       (B, d)
        → H: (B, n_queries, d)
        """
        B = T_temporal.shape[0]
        H = self.queries.expand(B, -1, -1)
        for layer in self.layers:
            H = layer(H, T_temporal, T_soc, e_ac)
        return H


# ─────────────────────────── MDPM ────────────────────────────────────────────

class MDPM(nn.Module):
    """
    Meta Degradation Pattern Memory.
    N_mem learnable memory slots, top-2 retrieval by cosine similarity.
    Gated fusion with decoder output.
    """

    def __init__(self, d: int, N_mem: int):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(N_mem, d) * 0.02)
        self.query_proj = nn.Sequential(
            nn.Linear(d, d), nn.GELU())
        self.gate = nn.Sequential(
            nn.Linear(d * 2, d), nn.Sigmoid())
        self.ln = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d) → (B, d)"""
        q = self.query_proj(h)                                   # (B, d)
        q_n = F.normalize(q, dim=-1)
        s_n = F.normalize(self.slots, dim=-1)                    # (N, d)
        sim = q_n @ s_n.t()                                      # (B, N)

        # top-2 retrieval
        top2_vals, top2_idx = sim.topk(2, dim=-1)                # (B, 2)
        alpha = F.softmax(top2_vals, dim=-1)                     # (B, 2)
        h_mem = (alpha.unsqueeze(-1) *
                 self.slots[top2_idx]).sum(dim=1)                 # (B, d)

        gate = self.gate(torch.cat([h, h_mem], dim=-1))
        return self.ln(gate * h + (1 - gate) * h_mem)


# ─────────────────────────── BatteryMFormer ──────────────────────────────────

class BatteryMFormer(nn.Module):
    """
    BatteryMFormer adapted for RUL point prediction.

    Input batch keys:
      'X':          (B, S, L, 4)  — V/I/Q/SOC
      'aging_text': list of B strings
      'soh':        (B, S)        — optional, not used in forward

    Output: (pred: (B,1), None)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        d         = m.get('bmf_d_model', 64)
        n_heads   = m.get('bmf_n_heads', 4)
        dropout   = m.get('dropout', 0.1)
        L         = m.get('bmf_L', 300)          # points per cycle
        M         = m.get('bmf_M', 10)           # SOC intervals
        n_dec     = m.get('bmf_n_dec_layers', 2)
        n_queries = m.get('bmf_n_queries', 8)
        N_mem     = m.get('bmf_N_mem', 16)

        self.soc_enc  = SOCViewEncoder(L, d, M, n_heads, dropout)
        self.temp_enc = TemporalViewEncoder(L, d, n_heads, dropout)
        self.ac_emb   = TextHashEmbedder(d)
        self.decoder  = ACDecoder(d, n_heads, n_dec, n_queries, dropout)
        self.mdpm     = MDPM(d, N_mem)

        self.head = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d * 2, 1))

    def forward(self, batch: dict):
        X    = batch['X']                    # (B, S, L, 4)
        texts = batch.get('aging_text', ['unknown'] * X.shape[0])

        T_soc      = self.soc_enc(X)         # (B, M, d)
        T_temporal = self.temp_enc(X)        # (B, S, d)
        e_ac       = self.ac_emb(texts)      # (B, d)

        H = self.decoder(T_temporal, T_soc, e_ac)   # (B, n_queries, d)
        h = H.mean(dim=1)                            # (B, d)
        h = self.mdpm(h)                             # (B, d)

        pred = self.head(h)                          # (B, 1)
        return pred, None
