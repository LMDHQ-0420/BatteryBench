"""
dataset_batterymformer.py — Multi-channel BatteryDataset for BatteryMFormer

Each cycle is represented as X ∈ R^(L×4) with channels:
  [voltage_in_V, current_in_A, discharge_capacity_in_Ah, SOC]

SOC is computed as Q / Q_max for the discharge segment.
Each cycle is resampled to L=300 points along the discharge segment.

Aging condition metadata (text fields) is also returned for ACDecoder.
"""

import os
import pickle
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
from scipy.interpolate import interp1d

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils import scan_pkl_dir, compute_soh_series, compute_rul, build_q_matrix

CACHE_DIR = 'data/cache_batterymformer'
L_DEFAULT = 300


def _resample_cycle(cyc: dict, L: int,
                    v_min: float = None,
                    v_max: float = None) -> Optional[np.ndarray]:
    """
    Extract discharge segment and resample to L points.
    Returns (L, 4) array [V, I, Q, SOC] or None if insufficient data.

    Strategy (mirrors get_discharge_qv for robustness):
      1. Try current-based discharge detection (I < 0)
      2. Fallback: use voltage range [v_min, v_max] + Q monotonicity
         This handles datasets where discharge current is positive.
    """
    V = np.array(cyc.get('voltage_in_V', []), dtype=np.float32)
    I = np.array(cyc.get('current_in_A', []), dtype=np.float32)
    Q = np.array(cyc.get('discharge_capacity_in_Ah', []), dtype=np.float32)

    if len(V) < 10 or len(Q) < 10:
        return None

    # Strategy 1: current-based (I < 0 = discharge)
    dis_mask = I < -1e-6
    if dis_mask.sum() >= 10:
        V_d = V[dis_mask]
        I_d = I[dis_mask]
        Q_d = Q[dis_mask]
    else:
        # Strategy 2: voltage-range based (same logic as get_discharge_qv)
        # infer voltage range from data if not provided
        vlo = v_min if v_min is not None else float(V.min())
        vhi = v_max if v_max is not None else float(V.max())
        mask = (V >= vlo) & (V <= vhi)
        if mask.sum() < 10:
            mask = np.ones(len(V), dtype=bool)
        V_d = V[mask]
        I_d = I[mask]
        Q_d = Q[mask]
        # sort by descending voltage (discharge direction)
        order = np.argsort(-V_d)
        V_d = V_d[order]; I_d = I_d[order]; Q_d = Q_d[order]

    if len(V_d) < 10:
        return None

    Q_max = Q_d.max()
    if Q_max < 1e-6:
        return None
    SOC_d = Q_d / Q_max

    n = len(V_d)
    idx_orig = np.linspace(0, n - 1, n)
    idx_new  = np.linspace(0, n - 1, L)

    try:
        V_r   = interp1d(idx_orig, V_d,   kind='linear')(idx_new)
        I_r   = interp1d(idx_orig, I_d,   kind='linear')(idx_new)
        Q_r   = interp1d(idx_orig, Q_d,   kind='linear')(idx_new)
        SOC_r = interp1d(idx_orig, SOC_d, kind='linear')(idx_new)
    except Exception:
        return None

    X = np.stack([V_r, I_r, Q_r, SOC_r], axis=-1)  # (L, 4)
    if np.isnan(X).any() or np.isinf(X).any():
        return None
    return X.astype(np.float32)


def _make_aging_text(obj: dict) -> str:
    """Build a short text description of aging condition from pkl metadata."""
    parts = []
    if obj.get('cathode_material'):
        parts.append(f"cathode: {obj['cathode_material']}")
    if obj.get('anode_material'):
        parts.append(f"anode: {obj['anode_material']}")
    if obj.get('form_factor'):
        parts.append(f"form: {obj['form_factor']}")
    cap = obj.get('nominal_capacity_in_Ah')
    if cap is not None:
        try:
            parts.append(f"capacity: {float(cap):.2f}Ah")
        except Exception:
            pass
    protos = obj.get('charge_protocol', [])
    if protos and isinstance(protos, list) and len(protos) > 0:
        p = protos[0]
        if isinstance(p, dict):
            rate = p.get('rate_in_C') or p.get('current_in_A')
            if rate:
                parts.append(f"charge_rate: {rate}C")
    return '; '.join(parts) if parts else 'unknown'


def _cache_key_mv(pkl_path: str, n_cycles: int, L: int, soh_threshold: float) -> str:
    stem = Path(pkl_path).stem
    return f'{stem}_mv_c{n_cycles}_l{L}_t{int(soh_threshold*100)}'


def _load_or_compute_mv(
    pkl_path: str,
    n_cycles: int,
    L: int,
    soh_threshold: float,
) -> Optional[tuple]:
    """Returns (X_seq, soh_seq, rul, cell_id, aging_text) or None."""
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key_mv(pkl_path, n_cycles, L, soh_threshold)
    cache_file = cache_dir / f'{key}.npz'

    if cache_file.exists():
        try:
            data = np.load(str(cache_file), allow_pickle=True)
            return (data['X_seq'], data['soh_seq'],
                    int(data['rul']), str(data['cell_id']),
                    str(data['aging_text']))
        except Exception:
            cache_file.unlink(missing_ok=True)

    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)

    cycles = obj.get('cycle_data', [])
    if len(cycles) < n_cycles:
        return None

    # compute SOH series to find EOL
    cap0 = obj.get('nominal_capacity_in_Ah')
    if cap0 is None:
        return None
    try:
        cap0 = float(cap0)
    except Exception:
        return None
    if cap0 < 1e-6:
        return None

    # build capacity + SOH series (reuse compute_soh_series if available)
    caps = []
    for c in cycles:
        Q = np.array(c.get('discharge_capacity_in_Ah', []))
        caps.append(float(Q.max()) if len(Q) > 0 else np.nan)
    caps = np.array(caps, dtype=np.float32)
    soh_series = caps / cap0

    # compute RUL using same fallback logic as BatteryDataset
    rul = compute_rul(soh_series, threshold=soh_threshold,
                      obs_cycle=n_cycles, fallback_total=True)
    if rul is None:
        return None

    # get voltage range from cell metadata
    v_min = obj.get('min_voltage_limit_in_V', None)
    v_max = obj.get('max_voltage_limit_in_V', None)

    # build multivariate sequence for first n_cycles
    # allow up to 20% failed cycles (fill with zeros), matching BatteryDataset
    X_list = []
    soh_list = []
    fail_count = 0
    for i in range(n_cycles):
        x = _resample_cycle(cycles[i], L, v_min=v_min, v_max=v_max)
        if x is None:
            fail_count += 1
            x = np.zeros((L, 4), dtype=np.float32)  # fill failed cycles
        X_list.append(x)
        soh_list.append(soh_series[i])
    if fail_count > n_cycles * 0.2:
        return None  # too many failed cycles

    X_seq   = np.stack(X_list,  axis=0).astype(np.float32)  # (S, L, 4)
    soh_seq = np.array(soh_list, dtype=np.float32)           # (S,)

    cell_id    = str(obj.get('cell_id', Path(pkl_path).stem))
    aging_text = _make_aging_text(obj)

    np.savez(str(cache_file),
             X_seq=X_seq, soh_seq=soh_seq,
             rul=np.array(rul), cell_id=np.array(cell_id),
             aging_text=np.array(aging_text))

    return X_seq, soh_seq, rul, cell_id, aging_text


class BatteryMFormerDataset(Dataset):
    """
    Multi-channel battery dataset for BatteryMFormer.

    Returns dict with:
      'X':          (S, L, 4)  — V/I/Q/SOC per cycle, resampled to L points
      'soh':        (S,)       — normalized SOH per cycle
      'rul':        scalar     — remaining useful life (cycles)
      'aging_text': str        — text description of aging condition
      'cell_id':    str
    """

    def __init__(
        self,
        pkl_dirs: Union[str, List[str]],
        n_cycles: int = 100,
        L: int = L_DEFAULT,
        soh_threshold: float = 0.80,
        **kwargs,   # absorb unused args passed by train.py (n_grid, task, etc.)
    ):
        if isinstance(pkl_dirs, str):
            pkl_dirs = [pkl_dirs]

        self.samples = []
        all_pkls = []
        for d in pkl_dirs:
            all_pkls.extend(scan_pkl_dir(d))

        for pkl_path in tqdm(all_pkls, desc='Loading batteries'):
            result = _load_or_compute_mv(pkl_path, n_cycles, L, soh_threshold)
            if result is None:
                continue
            X_seq, soh_seq, rul, cell_id, aging_text = result
            self.samples.append({
                'X':          torch.from_numpy(X_seq),
                'soh':        torch.from_numpy(soh_seq),
                'rul':        torch.tensor(rul, dtype=torch.float32),
                'cell_id':    cell_id,
                'aging_text': aging_text,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
