"""
data/cycle_dataset.py — 多样本 + attention mask 数据集（对齐 BatteryLife / BatteryMFormer）

核心设计:
    每个电池只把前 early_cycle 圈的曲线张量 (early_cycle, 3, L) 存一份，
    样本按「观测圈数 useable ∈ [seq_len, early_cycle]」展开为 (battery_idx, useable)。
    __getitem__ 时按 useable 生成 curve_attn_mask 并把未观测圈置零。

    这样样本数 ~ Σ_battery min(early_cycle, eol-1, valid)，与参考实现一致，
    且不产生 100× 的存储膨胀。

三个任务共享 battery 加载，标签不同:
    rul       → labels = eol           (绝对总寿命，1-based)
    soh_point → labels = soh_raw[useable-1]   (最后观测圈的原始 SOH)
    soh_traj  → traj (归一化完整轨迹) + trajectory_mask[useable:traj_len]

电池级 split：同一电池的所有样本同属 train/val/test，防数据泄露。
"""

import os
import fnmatch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils import (
    load_pkl, scan_pkl_dir, compute_soh_series, compute_eol,
    build_cycle_curve_tensor, build_q_matrix,
)

CACHE_DIR = 'data/cache'
SOH_TRAJ_LEN = 5000  # soh_traj 固定输出长度，超出实际寿命部分为 0


def _cache_key(pkl_path: str, early_cycle: int, L: int, n_grid: int) -> str:
    stem = Path(pkl_path).stem
    return f'{stem}_cyc_e{early_cycle}_L{L}_g{n_grid}_v4'


def _load_battery(pkl_path: str, early_cycle: int, charge_discharge_length: int,
                  soh_threshold: float, n_grid: int = 200) -> Optional[dict]:
    """
    加载单个电池，返回 dict 或 None（无效/圈数不足）。
    dict 字段:
        curves       : (early_cycle, 3, L) float32   充放电曲线
        Q            : (early_cycle, n_grid) float32  Q(V) 矩阵，供 IC2ML/BatLiNet 用
        valid_cycles : int
        soh_raw      : (full_len,) float32   原始 SOH 全序列
        eol          : int                    首次 SOH<threshold 的圈 (1-based)；None→未退化
        full_len     : int                    记录总圈数
        cathode_material, dataset_name, cell_id
    """
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(pkl_path, early_cycle, charge_discharge_length, n_grid)
    cache_file = cache_dir / f'{key}.npz'

    if cache_file.exists():
        try:
            d = np.load(str(cache_file), allow_pickle=True)
            eol = d['eol'].item()
            return {
                'curves':           d['curves'],
                'Q':                d['Q'],
                'valid_cycles':     int(d['valid_cycles']),
                'soh_raw':          d['soh_raw'],
                'eol':              (int(eol) if eol is not None else None),
                'full_len':         int(d['full_len']),
                'cathode_material': str(d['cathode_material']),
                'dataset_name':     str(d['dataset_name']),
                'cell_id':          str(d['cell_id']),
            }
        except Exception:
            cache_file.unlink(missing_ok=True)

    try:
        cell = load_pkl(pkl_path)
    except Exception:
        return None

    cycle_data = cell.get('cycle_data', [])
    if len(cycle_data) == 0:
        return None

    soh_raw = compute_soh_series(cell).astype(np.float32)
    full_len = len(soh_raw)
    eol = compute_eol(soh_raw, threshold=soh_threshold, fallback_total=False)

    curves, valid_cycles = build_cycle_curve_tensor(
        cell, early_cycle=early_cycle,
        charge_discharge_length=charge_discharge_length, num_var=3,
    )

    # Q(V) 矩阵，供 IC2ML/BatLiNet。build_q_matrix 有效圈不足会返回 None，
    # 此处失败则用零矩阵占位（这些模型对该电池仍可前向，只是特征为零）。
    Q = build_q_matrix(cell, n_cycles=early_cycle, n_grid=n_grid)
    if Q is None:
        Q = np.zeros((early_cycle, n_grid), dtype=np.float32)
    Q = Q.astype(np.float32)

    cell_id      = cell.get('cell_id', Path(pkl_path).stem)
    cathode      = str(cell.get('cathode_material', 'unknown'))
    dataset_name = Path(pkl_path).parent.name

    try:
        np.savez_compressed(
            str(cache_file),
            curves=curves.astype(np.float32),
            Q=Q,
            valid_cycles=np.array(valid_cycles),
            soh_raw=soh_raw,
            eol=np.array(eol, dtype=object),
            full_len=np.array(full_len),
            cathode_material=np.array(cathode),
            dataset_name=np.array(dataset_name),
            cell_id=np.array(cell_id),
        )
    except Exception:
        pass

    return {
        'curves': curves.astype(np.float32), 'Q': Q, 'valid_cycles': valid_cycles,
        'soh_raw': soh_raw, 'eol': eol, 'full_len': full_len,
        'cathode_material': cathode, 'dataset_name': dataset_name, 'cell_id': cell_id,
    }


class _CycleDatasetBase(Dataset):
    """
    多样本 + mask 数据集基类。子类实现 _make_label(bat, useable)。

    样本索引 self._samples: list of (battery_idx, useable_cycle)。
    battery 级 split 通过 split_battery_indices 控制。
    """

    def __init__(
        self,
        pkl_dir: Union[str, List[str]],
        n_grid: int = 200,
        soh_threshold: float = 0.80,
        early_cycle: int = 100,
        seq_len: int = 1,
        charge_discharge_length: int = 300,
        eol_threshold: float = None,
        split_battery_indices: Optional[List[int]] = None,
        pkl_files: Optional[List[str]] = None,
        exclude_pattern: Optional[str] = None,
        use_log_rul: bool = False,               # 兼容旧签名
        **kwargs,
    ):
        self.early_cycle = early_cycle
        self.seq_len = seq_len
        self.n_grid = n_grid
        self.charge_discharge_length = charge_discharge_length
        self.soh_threshold = soh_threshold if eol_threshold is None else eol_threshold

        if pkl_files is None:
            pkl_dirs = [pkl_dir] if isinstance(pkl_dir, str) else list(pkl_dir)
            pkl_files = []
            for d in pkl_dirs:
                pkl_files.extend(scan_pkl_dir(d))
        if exclude_pattern:
            patterns = [exclude_pattern] if isinstance(exclude_pattern, str) else exclude_pattern
            pkl_files = [f for f in pkl_files
                         if not any(fnmatch.fnmatch(os.path.basename(f), p) for p in patterns)]

        # 加载所有电池，排除 eol<=early_cycle 或未退化的
        self._batteries: List[dict] = []
        desc = f'Loading batteries ({self.__class__.__name__})'
        for pkl_path in tqdm(pkl_files, desc=desc, leave=False):
            bat = _load_battery(str(pkl_path), early_cycle,
                                charge_discharge_length, self.soh_threshold,
                                n_grid=n_grid)
            if bat is None:
                continue
            if bat['eol'] is None or bat['eol'] <= early_cycle:
                continue  # 对齐 BatteryLife：排除未跑到退化终点 / 寿命过短的电池
            self._batteries.append(bat)

        self._active_battery_idx = (split_battery_indices
                                    if split_battery_indices is not None
                                    else list(range(len(self._batteries))))
        self._rebuild_samples()

    def _rebuild_samples(self):
        """展开 (battery_idx, useable) 样本对。"""
        self._samples = []
        for bidx in self._active_battery_idx:
            if bidx >= len(self._batteries):
                continue
            bat = self._batteries[bidx]
            # useable 从 seq_len 到 early_cycle，但不超过有效圈数、不到 eol
            upper = min(self.early_cycle, bat['valid_cycles'], bat['eol'] - 1)
            for useable in range(self.seq_len, upper + 1):
                self._samples.append((bidx, useable))

    def __len__(self) -> int:
        return len(self._samples)

    def _base_item(self, bidx: int, useable: int):
        """构造 cycle_curve_data (early,3,L)、Q (early,N)、curve_attn_mask (early,)，未观测圈置零。"""
        bat = self._batteries[bidx]
        curves = bat['curves'].copy()                       # (early, 3, L)
        Q = bat['Q'].copy()                                 # (early, N)
        mask = np.zeros(self.early_cycle, dtype=np.float32)
        mask[:useable] = 1.0
        curves[useable:] = 0.0                              # 未观测圈置零
        Q[useable:] = 0.0
        return bat, curves, Q, mask

    def __getitem__(self, idx: int) -> dict:
        bidx, useable = self._samples[idx]
        bat, curves, Q, mask = self._base_item(bidx, useable)
        item = {
            'cycle_curve_data': torch.from_numpy(curves),          # (early, 3, L)
            'Q':                torch.from_numpy(Q),               # (early, N)
            'curve_attn_mask':  torch.from_numpy(mask),            # (early,)
            'useable_cycle':    useable,
            'cell_id':          bat['cell_id'],
        }
        item.update(self._make_label(bat, useable))
        return item

    def _make_label(self, bat: dict, useable: int) -> dict:
        raise NotImplementedError

    # ── split 支持 ──────────────────────────────────────────────────────────
    def get_battery_meta(self) -> List[Dict]:
        return [{'cathode_material': b['cathode_material'],
                 'dataset_name': b['dataset_name'],
                 'cell_id': b['cell_id']} for b in self._batteries]

    # 兼容 splits.py 旧接口
    def get_meta(self) -> List[Dict]:
        return self.get_battery_meta()

    @property
    def n_batteries(self) -> int:
        return len(self._batteries)

    @property
    def n_valid(self) -> int:
        return len(self._batteries)

    def subset_by_battery(self, battery_indices: List[int]):
        import copy
        sub = copy.copy(self)
        sub._active_battery_idx = list(battery_indices)
        sub._rebuild_samples()
        return sub


class RULDataset(_CycleDatasetBase):
    """
    RUL/BLP 任务：标签 = eol（绝对总寿命，1-based），每样本相同。
    对齐 BatteryLife：预测总寿命而非剩余寿命。
    """

    def __init__(self, *args, use_log_rul: bool = False, **kwargs):
        self.use_log_rul = use_log_rul
        super().__init__(*args, use_log_rul=use_log_rul, **kwargs)

    def _make_label(self, bat: dict, useable: int) -> dict:
        eol = float(bat['eol'])
        label = float(np.log1p(eol)) if self.use_log_rul else eol
        return {
            'labels':        torch.FloatTensor([label]),
            'eol':           torch.FloatTensor([eol]),
            # rul 供旧评估口径兼容：eol - 当前观测圈数
            'rul':           torch.FloatTensor([eol - useable]),
        }

    def get_all_ruls(self) -> List[int]:
        return [b['eol'] for b in self._batteries]


class SOHPointDataset(_CycleDatasetBase):
    """
    SOH 单点估计：标签 = 最后观测圈的原始 SOH（soh_raw[useable-1]）。
    """

    def _make_label(self, bat: dict, useable: int) -> dict:
        soh_raw = bat['soh_raw']
        idx = min(useable - 1, len(soh_raw) - 1)
        soh = float(np.clip(soh_raw[idx], 0.0, 1.0))
        return {'soh_point': torch.FloatTensor([soh])}


class SOHTrajDataset(_CycleDatasetBase):
    """
    SOH 退化轨迹预测：
        soh_traj        : (SOH_TRAJ_LEN,)  归一化完整轨迹 (SOH-thr)/(1-thr)，zero-pad
        trajectory_mask : (SOH_TRAJ_LEN,)  只在 [useable, eol) 置 1（只评估未来段）
    对齐 BatteryMFormer。
    """

    def _make_label(self, bat: dict, useable: int) -> dict:
        thr = self.soh_threshold
        soh_raw = bat['soh_raw']
        eol = bat['eol']
        full_len = min(len(soh_raw), SOH_TRAJ_LEN)

        # 归一化: (SOH - thr) / (1 - thr)
        traj = np.zeros(SOH_TRAJ_LEN, dtype=np.float32)
        traj[:full_len] = (soh_raw[:full_len] - thr) / (1.0 - thr)

        # mask 只覆盖未来段 [useable, min(eol, full_len))
        tmask = np.zeros(SOH_TRAJ_LEN, dtype=np.float32)
        hi = min(eol, full_len)
        if hi > useable:
            tmask[useable:hi] = 1.0

        return {
            'soh_traj':        torch.from_numpy(traj),
            'trajectory_mask': torch.from_numpy(tmask),
            'soh_traj_len':    torch.tensor(full_len, dtype=torch.long),
        }
