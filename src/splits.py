"""
splits.py — 数据集划分策略

三种策略通过 config['data']['split_strategy'] 控制：

  random     — 随机打乱后按比例分 train/val/test
  stratified — 按 stratify_by 字段分层抽样，保证每个组在 train/val/test 中均有代表
  four_level — 四层泛化阶梯划分：
                 train/val 来自固定 train_dirs（按 cathode_material 分层抽 val_ratio 作 val）
                 test 来自固定 test_sets（每个测试集独立，附带 level 标签 L1/L2/L3/L4）
                 返回 1 个 split，test 为 None（测试集独立加载）

stratify_by 可选值（stratified 专用）：
  cathode_material — 按正极材料分层（LFP / NMC / NCA / LCO 等）
  dataset_name     — 按数据集来源分层（MATR / CALCE / HUST 等，100% 可靠）

用法:
    from src.splits import make_battery_splits
    splits = make_battery_splits(ds, cfg, seed=42)
"""

import random
from collections import defaultdict
from typing import List, Dict


def _random_splits(
    n: int,
    val_ratio: float,
    test_ratio: float,
    n_splits: int,
    seed: int,
) -> List[Dict[str, List[int]]]:
    splits = []
    for i in range(n_splits):
        indices = list(range(n))
        rng = random.Random(seed + i)
        rng.shuffle(indices)
        n_test = max(1, round(n * test_ratio))
        n_val  = max(1, round(n * val_ratio))
        splits.append({
            'test':  indices[:n_test],
            'val':   indices[n_test:n_test + n_val],
            'train': indices[n_test + n_val:],
        })
    return splits


def _stratified_splits(
    meta: List[Dict],
    stratify_by: str,
    val_ratio: float,
    test_ratio: float,
    n_splits: int,
    seed: int,
) -> List[Dict[str, List[int]]]:
    """
    按 stratify_by 字段分组，每组内独立随机划分后合并。
    保证每个化学体系 / 数据集在 train/val/test 中均有代表。
    """
    by_group = defaultdict(list)
    for i, m in enumerate(meta):
        by_group[m.get(stratify_by, 'unknown')].append(i)

    splits = []
    for si in range(n_splits):
        rng = random.Random(seed + si)
        train_idx, val_idx, test_idx = [], [], []
        for group_indices in by_group.values():
            items = list(group_indices)
            rng.shuffle(items)
            n_test = max(1, round(len(items) * test_ratio)) if len(items) > 1 else 0
            n_val  = max(1, round(len(items) * val_ratio))  if len(items) > 1 else 0
            test_idx.extend(items[:n_test])
            val_idx.extend( items[n_test:n_test + n_val])
            train_idx.extend(items[n_test + n_val:])
        splits.append({'train': train_idx, 'val': val_idx, 'test': test_idx})
    return splits


def make_battery_splits(ds, cfg: dict, seed: int = 42) -> List[Dict]:
    """
    电池级划分，适用于所有多样本 dataset（RUL/SOHPoint/SOHTraj）。

    每个 split 返回:
        {'train': <dataset子集>, 'val': <dataset子集>, 'test': <dataset子集或None>}

    在电池级别 split 后展开到样本，保证同一电池所有样本同属 train/val/test（防泄露）。
    four_level 模式下 test 为 None（测试集独立加载）。
    """
    d_cfg      = cfg.get('data', {})
    strategy   = d_cfg.get('split_strategy', 'random')
    val_ratio  = d_cfg.get('val_ratio', 0.1)
    test_ratio = d_cfg.get('test_ratio', 0.2)
    n_splits   = d_cfg.get('n_splits', 3)
    n_batt     = ds.n_batteries

    if strategy == 'random':
        idx_splits = _random_splits(n_batt, val_ratio, test_ratio, n_splits, seed)
    elif strategy == 'stratified':
        stratify_by = d_cfg.get('stratify_by', 'cathode_material')
        idx_splits = _stratified_splits(ds.get_battery_meta(), stratify_by,
                                        val_ratio, test_ratio, n_splits, seed)
    elif strategy == 'four_level':
        # val/train from pool; test sets loaded separately in scripts
        meta = ds.get_battery_meta()
        by_chem = defaultdict(list)
        for i, m in enumerate(meta):
            by_chem[m.get('cathode_material', 'unknown')].append(i)
        rng = random.Random(seed)
        train_idx, val_idx = [], []
        for chem in sorted(by_chem):
            items = list(by_chem[chem])
            rng.shuffle(items)
            n_val = max(1, round(len(items) * d_cfg.get('val_ratio', 0.08))) if len(items) > 1 else 0
            val_idx.extend(items[:n_val])
            train_idx.extend(items[n_val:])
        idx_splits = [{'train': train_idx, 'val': val_idx, 'test': []}]
    else:
        raise ValueError(f"Unknown split_strategy '{strategy}'")

    result = []
    for sp in idx_splits:
        result.append({
            'train': ds.subset_by_battery(sp['train']),
            'val':   ds.subset_by_battery(sp['val']),
            'test':  ds.subset_by_battery(sp['test']) if sp['test'] else None,
        })
    return result
