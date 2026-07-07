"""
splits.py — 数据集划分策略

三种策略通过 config['data']['split_strategy'] 控制：

  random     — 随机打乱后按比例分 train/val/test
  stratified — 按 stratify_by 字段分层抽样，保证每个组在 train/val/test 中均有代表
  three_level — 三层泛化阶梯划分：
                 train/val 来自固定 train_dirs（按 cathode_material 分层抽 val_ratio 作 val）
                 test 来自固定 test_sets（每个测试集独立，附带 level 标签 L1/L2/L3）
                 返回 1 个 split，test 字段为 list of {dir, level, indices}

stratify_by 可选值（stratified 专用）：
  cathode_material — 按正极材料分层（LFP / NMC / NCA / LCO 等）
  dataset_name     — 按数据集来源分层（MATR / CALCE / HUST 等，100% 可靠）

用法:
    from src.splits import make_splits
    splits = make_splits(ds, cfg, seed=42)
"""

import random
from collections import defaultdict
from typing import List, Dict

from src.data.dataset import BatteryDataset


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


def _three_level_split(cfg: dict, seed: int, pool_ds=None) -> List[Dict]:
    """
    三层泛化阶梯划分。

    返回固定 1 个 split，结构为：
    {
        'train':    List[int],   # BatteryDataset(train_dirs) 里的索引
        'val':      List[int],   # 同上，按 cathode_material 分层抽取
        'test_sets': [           # 每个测试集独立
            {'dir': str, 'level': str, 'n_cells': int},
            ...
        ]
    }

    test_sets 里不存索引，因为每个测试集是独立加载的 BatteryDataset。
    scripts/train.py 和 scripts/evaluate.py 通过 cfg['data']['test_sets'] 直接读取。
    """
    d_cfg     = cfg.get('data', {})
    val_ratio = d_cfg.get('val_ratio', 0.08)
    train_dirs = d_cfg.get('train_dirs', [])

    if not train_dirs:
        raise ValueError("hyperbat 策略需要在 config 中设置 train_dirs")

    from src.config import get_pkl_dir as _get_pkl_dir

    # 加载 train pool（若外部已加载则复用）
    n_cycles      = d_cfg.get('n_cycles', cfg.get('model', {}).get('n_cycles', 100))
    n_grid        = d_cfg.get('n_grid',   cfg.get('model', {}).get('n_grid', 200))
    soh_threshold = d_cfg.get('soh_threshold', 0.80)

    if pool_ds is None:
        pool_ds = BatteryDataset(train_dirs, n_cycles=n_cycles, n_grid=n_grid,
                                 soh_threshold=soh_threshold)
    meta = pool_ds.get_meta()

    # 按 cathode_material 分层抽 val_ratio 作 val
    by_chem = defaultdict(list)
    for i, m in enumerate(meta):
        by_chem[m.get('cathode_material', 'unknown')].append(i)

    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for chem in sorted(by_chem):
        items = list(by_chem[chem])
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_ratio)) if len(items) > 1 else 0
        val_idx.extend(items[:n_val])
        train_idx.extend(items[n_val:])

    # test_sets 直接从 config 读取，不需要索引
    test_sets = d_cfg.get('test_sets', [])

    return [{
        'train':     sorted(train_idx),
        'val':       sorted(val_idx),
        'test_sets': test_sets,   # [{'dir': ..., 'level': ...}, ...]
    }]


def make_battery_splits(ds, cfg: dict, seed: int = 42) -> List[Dict]:
    """
    电池级划分，适用于所有多样本 dataset（RUL/SOHPoint/SOHTraj）。

    每个 split 返回:
        {'train': <dataset子集>, 'val': <dataset子集>, 'test': <dataset子集或None>}

    在电池级别 split 后展开到样本，保证同一电池所有样本同属 train/val/test（防泄露）。
    three_level 模式下 test 为 None（测试集独立加载）。
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
    elif strategy == 'three_level':
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


# 向后兼容别名
make_soh_point_splits = make_battery_splits


def make_splits(ds, cfg: dict, seed: int = 42) -> List[Dict]:
    """
    统一划分入口，根据 cfg['data']['split_strategy'] 选择策略。

    config 参数:
        split_strategy : random | stratified | three_level
        val_ratio      : 验证集比例
        test_ratio     : 测试集比例（random / stratified）
        n_splits       : 重复划分次数（random / stratified）
        stratify_by    : cathode_material | dataset_name（stratified 专用）
        train_dirs     : 训练池目录列表（three_level 专用）
        test_sets      : 固定测试集列表（three_level 专用）
    """
    d_cfg      = cfg.get('data', {})
    strategy   = d_cfg.get('split_strategy', 'random')
    val_ratio  = d_cfg.get('val_ratio', 0.1)
    test_ratio = d_cfg.get('test_ratio', 0.2)
    n_splits   = d_cfg.get('n_splits', 3)

    if strategy == 'random':
        return _random_splits(ds.n_valid, val_ratio, test_ratio, n_splits, seed)

    elif strategy == 'stratified':
        stratify_by = d_cfg.get('stratify_by', 'cathode_material')
        return _stratified_splits(ds.get_meta(), stratify_by, val_ratio, test_ratio, n_splits, seed)

    elif strategy == 'three_level':
        return _three_level_split(cfg, seed, pool_ds=ds)

    else:
        raise ValueError(f"Unknown split_strategy '{strategy}'. 可选: random | stratified | three_level")


