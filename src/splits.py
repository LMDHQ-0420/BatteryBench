"""Battery-level train/validation/test splitting."""

import random
from collections import defaultdict


def _partition(indices, val_ratio, test_ratio, rng):
    indices = list(indices)
    rng.shuffle(indices)
    n_test = max(1, round(len(indices) * test_ratio)) if len(indices) > 1 else 0
    n_val = max(1, round(len(indices) * val_ratio)) if len(indices) > 1 else 0
    return {
        'test': indices[:n_test],
        'val': indices[n_test:n_test + n_val],
        'train': indices[n_test + n_val:],
    }


def _stratified(meta, field, val_ratio, test_ratio, rng):
    groups = defaultdict(list)
    for index, values in enumerate(meta):
        groups[values.get(field, 'unknown')].append(index)

    split = {'train': [], 'val': [], 'test': []}
    for indices in groups.values():
        group = _partition(indices, val_ratio, test_ratio, rng)
        for name in split:
            split[name].extend(group[name])
    return split


def make_battery_splits(dataset, config, seed=1):
    """Return one fixed battery-level split; Four-Level keeps its fixed external tests."""
    data = config.get('data', {})
    strategy = data.get('split_strategy', 'random')
    rng = random.Random(seed)

    if strategy == 'random':
        indices = _partition(
            range(dataset.n_batteries), data.get('val_ratio', 0.1),
            data.get('test_ratio', 0.2), rng,
        )
    elif strategy == 'stratified':
        indices = _stratified(
            dataset.get_battery_meta(), data.get('stratify_by', 'cathode_material'),
            data.get('val_ratio', 0.1), data.get('test_ratio', 0.2), rng,
        )
    elif strategy == 'four_level':
        groups = defaultdict(list)
        for index, values in enumerate(dataset.get_battery_meta()):
            groups[values.get('cathode_material', 'unknown')].append(index)
        indices = {'train': [], 'val': [], 'test': []}
        for name in sorted(groups):
            group = list(groups[name])
            rng.shuffle(group)
            n_val = max(1, round(len(group) * data.get('val_ratio', 0.08))) if len(group) > 1 else 0
            indices['val'].extend(group[:n_val])
            indices['train'].extend(group[n_val:])
    else:
        raise ValueError(f"Unknown split_strategy '{strategy}'")

    return [{
        'train': dataset.subset_by_battery(indices['train']),
        'val': dataset.subset_by_battery(indices['val']),
        'test': dataset.subset_by_battery(indices['test']) if indices['test'] else None,
    }]
