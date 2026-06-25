"""
dataset.py — BatteryDataset 与数据集划分
"""

import os
import random
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils import (
    load_pkl, scan_pkl_dir,
    build_q_matrix, compute_soh_series,
    compute_rul, compute_delta_q,
)

CACHE_DIR = 'data/cache'
SOH_TRAJ_LEN = 5000  # fixed output length for soh_traj; zeros beyond actual lifetime


def _cache_key(pkl_path: str, n_cycles: int, n_grid: int, soh_threshold: float) -> str:
    stem = Path(pkl_path).stem
    return f'{stem}_c{n_cycles}_g{n_grid}_t{int(soh_threshold*100)}'


def _load_or_compute(pkl_path: str, n_cycles: int, n_grid: int, soh_threshold: float) -> Optional[tuple]:
    """
    返回 (Q, delta_q, rul, soh_point, soh_traj, soh_traj_len, cell_id, cathode_material, dataset_name, lognf) 或 None。
    soh_point   : float — 第 n_cycles 个 cycle 的 SOH 值
    soh_traj    : np.ndarray shape (SOH_TRAJ_LEN,) — 完整退化轨迹，超出实际寿命处填0
    soh_traj_len: int   — 实际有效步数
    """
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(pkl_path, n_cycles, n_grid, soh_threshold)
    cache_file = cache_dir / f'{key}.npz'

    if cache_file.exists():
        try:
            data = np.load(str(cache_file), allow_pickle=True)
            return (
                data['Q'], data['delta_q'], int(data['rul']),
                float(data['soh_point'])          if 'soh_point'     in data else 0.0,
                data['soh_traj']                  if 'soh_traj'      in data else np.zeros(SOH_TRAJ_LEN, dtype=np.float32),
                int(data['soh_traj_len'])          if 'soh_traj_len'  in data else SOH_TRAJ_LEN,
                str(data['cell_id']),
                str(data['cathode_material']) if 'cathode_material' in data else 'unknown',
                str(data['dataset_name'])     if 'dataset_name'     in data else 'unknown',
                float(data['lognf'])          if 'lognf'            in data else 0.0,
            )
        except Exception:
            cache_file.unlink(missing_ok=True)

    try:
        cell = load_pkl(pkl_path)
    except Exception:
        return None

    Q = build_q_matrix(cell, n_cycles=n_cycles, n_grid=n_grid)
    if Q is None:
        return None

    soh_series = compute_soh_series(cell)
    rul = compute_rul(soh_series, threshold=soh_threshold, obs_cycle=n_cycles)
    if rul is None:
        return None

    # soh_point: SOH at observation cycle
    soh_point = float(np.clip(soh_series[n_cycles - 1], 0.0, 1.0)) if len(soh_series) >= n_cycles else float(soh_series[-1])

    # soh_traj: full degradation trajectory, zero-padded to SOH_TRAJ_LEN
    full_len = len(soh_series)
    soh_traj_len = min(full_len, SOH_TRAJ_LEN)
    soh_traj = np.zeros(SOH_TRAJ_LEN, dtype=np.float32)
    soh_traj[:soh_traj_len] = np.clip(soh_series[:soh_traj_len], 0.0, 1.0)

    delta_q      = compute_delta_q(Q, early_cycle=10, late_cycle=n_cycles)
    cell_id      = cell.get('cell_id', Path(pkl_path).stem)
    cathode      = str(cell.get('cathode_material', 'unknown'))
    dataset_name = Path(pkl_path).parent.name
    nom          = cell.get('nominal_capacity_in_Ah')
    lognf        = float(np.log(nom)) if (nom and nom > 0) else 0.0

    try:
        np.savez_compressed(
            str(cache_file),
            Q=Q, delta_q=delta_q, rul=np.array(rul),
            soh_point=np.array(soh_point),
            soh_traj=soh_traj,
            soh_traj_len=np.array(soh_traj_len),
            cell_id=np.array(cell_id),
            cathode_material=np.array(cathode),
            dataset_name=np.array(dataset_name),
            lognf=np.array(lognf),
        )
    except Exception:
        pass

    return Q, delta_q, rul, soh_point, soh_traj, soh_traj_len, cell_id, cathode, dataset_name, lognf


class BatteryDataset(Dataset):
    """
    加载一个或多个数据集目录的所有电池。

    每个样本 batch 包含:
        Q            : (n_cycles, n_grid)
        delta_q      : (n_grid,)
        rul          : scalar
        cell_id      : str
        lognf        : scalar
    """

    def __init__(
        self,
        pkl_dir: Union[str, List[str]],
        n_cycles: int = 100,
        n_grid: int = 200,
        soh_threshold: float = 0.80,
        split_indices: Optional[List[int]] = None,
        use_log_rul: bool = False,
        pkl_files: Optional[List[str]] = None,      # explicit file list, overrides pkl_dir
        exclude_pattern: Optional[str] = None,      # fnmatch pattern to exclude from scan
    ):
        self.use_log_rul = use_log_rul

        if pkl_files is not None:
            pass  # use provided list directly
        else:
            pkl_dirs = [pkl_dir] if isinstance(pkl_dir, str) else list(pkl_dir)
            pkl_files = []
            for d in pkl_dirs:
                pkl_files.extend(scan_pkl_dir(d))
        if exclude_pattern:
                import fnmatch
                patterns = [exclude_pattern] if isinstance(exclude_pattern, str) else exclude_pattern
                pkl_files = [f for f in pkl_files
                             if not any(fnmatch.fnmatch(os.path.basename(f), p) for p in patterns)]

        self._all_samples = []
        for pkl_path in tqdm(pkl_files, desc='Loading batteries', leave=False):
            result = _load_or_compute(str(pkl_path), n_cycles, n_grid, soh_threshold)
            if result is not None:
                self._all_samples.append(result)

        if split_indices is not None:
            self._samples = [self._all_samples[i] for i in split_indices
                             if i < len(self._all_samples)]
        else:
            self._samples = self._all_samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        Q, delta_q, rul, soh_point, soh_traj, soh_traj_len, cell_id, cathode, dataset_name, lognf = self._samples[idx]
        rul_val = float(np.log1p(rul)) if self.use_log_rul else float(rul)
        return {
            'Q':            torch.FloatTensor(Q),
            'delta_q':      torch.FloatTensor(delta_q),
            'rul':          torch.FloatTensor([rul_val]),
            'soh_point':    torch.FloatTensor([soh_point]),
            'soh_traj':     torch.FloatTensor(soh_traj),
            'soh_traj_len': torch.tensor(soh_traj_len, dtype=torch.long),
            'lognf':        torch.FloatTensor([lognf]),
            'cell_id':      cell_id,
        }

    def get_all_ruls(self) -> List[int]:
        return [s[2] for s in self._all_samples]

    def get_meta(self) -> List[Dict]:
        """返回每个样本的元信息，供 splits.py 分层划分使用。"""
        return [
            {'cathode_material': s[7], 'dataset_name': s[8]}
            for s in self._all_samples
        ]

    @property
    def n_valid(self) -> int:
        return len(self._all_samples)


def make_random_splits(
    n_batteries: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    n_splits: int = 1,
    seed: int = 42,
) -> List[Dict[str, List[int]]]:
    """简单随机划分，支持多次划分（不同 seed）。"""
    splits = []
    for i in range(n_splits):
        indices = list(range(n_batteries))
        rng = random.Random(seed + i)
        rng.shuffle(indices)
        n_test = max(1, round(n_batteries * test_ratio))
        n_val  = max(1, round(n_batteries * val_ratio))
        splits.append({
            'test':  indices[:n_test],
            'val':   indices[n_test:n_test + n_val],
            'train': indices[n_test + n_val:],
        })
    return splits
