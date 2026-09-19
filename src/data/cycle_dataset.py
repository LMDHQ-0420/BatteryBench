"""Datasets built from complete charge curves."""

import fnmatch
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils import build_charge_curve_tensor, compute_eol, compute_soh_series, load_pkl, scan_pkl_dir

CACHE_DIR = Path('data/cache')
SOH_TRAJ_LEN = 5000


def _load_battery(
    pkl_path: str,
    n_cycles: Optional[int],
    curve_length: int,
    soh_threshold: float,
    include_capacity: bool,
    cache_group: str,
) -> Optional[dict]:
    """Read source data, rebuild charge curves, and overwrite the cache."""
    try:
        cell = load_pkl(pkl_path)
    except Exception:
        return None
    if not cell.get('cycle_data'):
        return None

    soh_raw = compute_soh_series(cell).astype(np.float32)
    n_cycles = len(soh_raw) if n_cycles is None else n_cycles
    curves, valid_cycles = build_charge_curve_tensor(
        cell, n_cycles=n_cycles, curve_length=curve_length, include_capacity=include_capacity,
    )
    if valid_cycles == 0:
        return None

    full_len = len(soh_raw)
    eol = compute_eol(soh_raw, threshold=soh_threshold, fallback_total=False)
    source = Path(pkl_path)
    dataset_name = source.parent.name
    metadata = {
        'valid_cycles': valid_cycles,
        'soh_raw': soh_raw,
        'eol': eol,
        'full_len': full_len,
        'cathode_material': str(cell.get('cathode_material', 'unknown')),
        'dataset_name': dataset_name,
        'cell_id': str(cell.get('cell_id', source.stem)),
    }

    cache_file = CACHE_DIR / cache_group / dataset_name / f'{source.stem}.npz'
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_name(f'.{cache_file.stem}.{os.getpid()}.npz')
    np.savez_compressed(
        temporary,
        curves=curves,
        valid_cycles=np.array(valid_cycles),
        soh_raw=soh_raw,
        eol=np.array(eol, dtype=object),
        full_len=np.array(full_len),
        cathode_material=np.array(metadata['cathode_material']),
        dataset_name=np.array(dataset_name),
        cell_id=np.array(metadata['cell_id']),
    )
    os.replace(temporary, cache_file)
    return {'curves': curves, **metadata}


class _CycleDatasetBase(Dataset):
    REQUIRES_EOL = False
    FULL_CYCLE_MODE = False
    INCLUDE_CAPACITY = True

    def __init__(
        self,
        pkl_dir: Union[str, List[str]],
        soh_threshold: float = 0.80,
        early_cycle: int = 100,
        seq_len: int = 1,
        curve_length: int = 400,
        eol_threshold: float = None,
        split_battery_indices: Optional[List[int]] = None,
        pkl_files: Optional[List[str]] = None,
        exclude_pattern: Optional[str] = None,
        use_log_rul: bool = False,
        **kwargs,
    ):
        self.early_cycle = early_cycle
        self.seq_len = seq_len
        self.curve_length = curve_length
        self.soh_threshold = soh_threshold if eol_threshold is None else eol_threshold

        if pkl_files is None:
            directories = [pkl_dir] if isinstance(pkl_dir, str) else list(pkl_dir)
            pkl_files = [str(path) for directory in directories for path in scan_pkl_dir(directory)]
        if exclude_pattern:
            patterns = [exclude_pattern] if isinstance(exclude_pattern, str) else exclude_pattern
            pkl_files = [path for path in pkl_files if not any(
                fnmatch.fnmatch(os.path.basename(path), pattern) for pattern in patterns
            )]

        self._batteries = []
        cache_group = 'soh_point' if self.FULL_CYCLE_MODE else 'history'
        for path in tqdm(pkl_files, desc=f'Loading batteries ({self.__class__.__name__})', leave=False):
            battery = _load_battery(
                str(path), None if self.FULL_CYCLE_MODE else self.early_cycle,
                curve_length, self.soh_threshold,
                self.INCLUDE_CAPACITY, cache_group,
            )
            if battery is None:
                continue
            if self.REQUIRES_EOL and (battery['eol'] is None or battery['eol'] <= early_cycle):
                continue
            self._batteries.append(battery)

        self._active_battery_idx = (
            list(split_battery_indices) if split_battery_indices is not None
            else list(range(len(self._batteries)))
        )
        self._rebuild_samples()

    def _rebuild_samples(self):
        self._samples = []
        for battery_index in self._active_battery_idx:
            if battery_index >= len(self._batteries):
                continue
            battery = self._batteries[battery_index]
            if self.FULL_CYCLE_MODE:
                upper = battery['valid_cycles']
            else:
                eol_bound = battery['valid_cycles'] if battery['eol'] is None else battery['eol'] - 1
                upper = min(self.early_cycle, battery['valid_cycles'], eol_bound)
            self._samples.extend((battery_index, useable) for useable in range(self.seq_len, upper + 1))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        battery_index, useable = self._samples[index]
        battery = self._batteries[battery_index]
        if self.FULL_CYCLE_MODE:
            curves = battery['curves'][useable - 1:useable]
            mask = np.ones(1, dtype=np.float32)
        else:
            curves = battery['curves'].copy()
            curves[useable:] = 0.0
            mask = np.zeros(self.early_cycle, dtype=np.float32)
            mask[:useable] = 1.0

        item = {
            'cycle_curve_data': torch.from_numpy(curves),
            'curve_attn_mask': torch.from_numpy(mask),
            'useable_cycle': useable,
            'cell_id': battery['cell_id'],
        }
        item.update(self._make_label(battery, useable))
        return item

    def _make_label(self, battery, useable):
        raise NotImplementedError

    def get_battery_meta(self) -> List[Dict]:
        return [{
            'cathode_material': battery['cathode_material'],
            'dataset_name': battery['dataset_name'],
            'cell_id': battery['cell_id'],
        } for battery in self._batteries]

    def get_meta(self):
        return self.get_battery_meta()

    @property
    def n_batteries(self):
        return len(self._batteries)

    @property
    def n_valid(self):
        return len(self._batteries)

    def subset_by_battery(self, battery_indices):
        import copy
        subset = copy.copy(self)
        subset._active_battery_idx = list(battery_indices)
        subset._rebuild_samples()
        return subset


class RULDataset(_CycleDatasetBase):
    REQUIRES_EOL = True

    def __init__(self, *args, use_log_rul=False, **kwargs):
        self.use_log_rul = use_log_rul
        super().__init__(*args, use_log_rul=use_log_rul, **kwargs)

    def _make_label(self, battery, useable):
        eol = float(battery['eol'])
        label = float(np.log1p(eol)) if self.use_log_rul else eol
        return {
            'labels': torch.tensor([label], dtype=torch.float32),
            'eol': torch.tensor([eol], dtype=torch.float32),
            'rul': torch.tensor([eol - useable], dtype=torch.float32),
        }

    def get_all_ruls(self):
        return [battery['eol'] for battery in self._batteries]


class SOHPointDataset(_CycleDatasetBase):
    FULL_CYCLE_MODE = True
    INCLUDE_CAPACITY = False

    def _make_label(self, battery, useable):
        index = min(useable - 1, len(battery['soh_raw']) - 1)
        value = float(np.clip(battery['soh_raw'][index], 0.0, 1.0))
        return {'soh_point': torch.tensor([value], dtype=torch.float32)}


class SOHTrajDataset(_CycleDatasetBase):
    def _make_label(self, battery, useable):
        threshold = self.soh_threshold
        soh = battery['soh_raw']
        full_len = min(len(soh), SOH_TRAJ_LEN)
        trajectory = np.zeros(SOH_TRAJ_LEN, dtype=np.float32)
        trajectory[:full_len] = (soh[:full_len] - threshold) / (1.0 - threshold)

        mask = np.zeros(SOH_TRAJ_LEN, dtype=np.float32)
        end = full_len if battery['eol'] is None else min(battery['eol'], full_len)
        if end > useable:
            mask[useable:end] = 1.0
        return {
            'soh_traj': torch.from_numpy(trajectory),
            'trajectory_mask': torch.from_numpy(mask),
            'soh_traj_len': torch.tensor(full_len, dtype=torch.long),
        }


def soh_point_collate_fn(samples):
    curves = torch.stack([sample['cycle_curve_data'] for sample in samples])
    masks = torch.stack([sample['curve_attn_mask'] for sample in samples])
    batch = {'cycle_curve_data': curves, 'curve_attn_mask': masks}
    for key in samples[0]:
        if key in batch:
            continue
        values = [sample[key] for sample in samples]
        try:
            batch[key] = torch.stack(values)
        except (TypeError, RuntimeError):
            batch[key] = values
    return batch
