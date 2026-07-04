"""
scripts/evaluate.py — 评估已训练的 checkpoint

用法:
    python scripts/evaluate.py --domain li_ion --model gru --task rul
    python scripts/evaluate.py --domain li_ion --model all --task soh_point
    python scripts/evaluate.py --domain three_level --model gru --task rul
    python scripts/evaluate.py --domain li_ion --model gru --task rul \
        --checkpoint results/li_ion/rul/gru/split1_best.pt
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.config import load_config, get_pkl_dir, DOMAIN_CFG
from src.data.dataset import BatteryDataset
from src.data.soh_point.dataset import SOHPointDataset
from src.splits import make_splits, make_soh_point_splits
from src.models.registry import get_spec, ALL_MODELS, ALL_TASKS
import src.evaluate.rul.evaluate       as eval_rul
import src.evaluate.soh_point.evaluate as eval_soh_point
import src.evaluate.soh_traj.evaluate  as eval_soh_traj


def _get_evaluate_fn(task: str):
    return {
        'rul':       eval_rul.evaluate,
        'soh_point': eval_soh_point.evaluate,
        'soh_traj':  eval_soh_traj.evaluate,
    }[task]


def _print_metrics(metrics: dict):
    parts = [f'{k.upper()}={v:.4f}' for k, v in metrics.items()]
    print('  ' + '  '.join(parts))


# ── three_level 评估：L1 / L2 / L3 加权平均 ─────────────────────────────────

def _eval_three_level(model_name: str, task: str, cfg: dict,
                      save_dir: str, device: str):
    spec = get_spec(model_name, task)

    d_cfg         = cfg['data']
    t_cfg         = cfg['train']
    n_cycles      = d_cfg.get('n_cycles', cfg['model'].get('n_cycles', 100))
    n_grid        = d_cfg.get('n_grid',   cfg['model'].get('n_grid', 200))
    soh_threshold = d_cfg.get('soh_threshold', 0.80)
    use_log_rul   = t_cfg.get('use_log_rul', False) and task == 'rul'
    n_future      = d_cfg.get('n_future', 5000)
    batch_size    = t_cfg.get('batch_size', 32)
    evaluate_fn   = _get_evaluate_fn(task)

    model_save_dir = os.path.join(save_dir, model_name)

    if task == 'soh_point':
        _eval_soh_point(model_name, 'three_level', spec, cfg, d_cfg, t_cfg,
                        n_grid, batch_size, evaluate_fn,
                        None, model_save_dir, device)
        return

    # sklearn 模型：加载 pickle
    if spec.build_fn is None:
        pkl_path = os.path.join(model_save_dir, 'best.pkl')
        if not os.path.exists(pkl_path):
            print(f'  No pickle found at {pkl_path}. Run scripts/train.py first.')
            return
        # 动态导入对应任务的 evaluate 函数
        if task == 'rul':
            from src.train.rul.train_severson      import evaluate as sklearn_eval
        elif task == 'soh_point':
            from src.train.soh_point.train_severson import evaluate as sklearn_eval
        else:
            from src.train.soh_traj.train_severson  import evaluate as sklearn_eval
        test_sets = d_cfg.get('test_sets', [])
        if not test_sets:
            print('  No test_sets defined in config.')
            return
        by_level: dict[str, list] = {}
        all_results = []
        for ts in test_sets:
            ts_dir  = ts['dir']
            level   = ts['level']
            pattern = ts.get('pattern', None)
            ds_name = os.path.basename(ts_dir)
            if pattern:
                import fnmatch
                all_files = sorted(glob.glob(os.path.join(ts_dir, '*.pkl')))
                matched   = [f for f in all_files if fnmatch.fnmatch(os.path.basename(f), pattern)]
                ds_name   = f'{os.path.basename(ts_dir)}[{pattern}]'
                test_ds   = spec.dataset_cls(ts_dir, n_cycles=n_cycles, n_grid=n_grid,
                                             soh_threshold=soh_threshold,
                                             pkl_files=matched)
            else:
                test_ds = spec.dataset_cls(ts_dir, n_cycles=n_cycles, n_grid=n_grid,
                                           soh_threshold=soh_threshold)
            if len(test_ds) == 0:
                print(f'  [{level}] {ds_name}: empty, skipping.')
                continue
            metrics = sklearn_eval(test_ds, pkl_path)
            line = f'  [{level}] {ds_name:20s} n={len(test_ds):3d} | '
            line += '  '.join(f'{k.upper()}={v:.4f}' for k, v in metrics.items())
            print(line)
            by_level.setdefault(level, []).append((metrics, len(test_ds)))
            all_results.append({'dataset': ds_name, 'level': level,
                                'n_cells': len(test_ds), **metrics})
        if not all_results:
            return
        print()
        level_summary = {}
        for level in sorted(by_level):
            pairs = by_level[level]
            total = sum(n for _, n in pairs)
            keys  = list(pairs[0][0].keys())
            def _wavg(k):
                valid = [(m[k], n) for m, n in pairs if m[k] == m[k]]  # exclude NaN
                if not valid: return float('nan')
                return sum(v * n for v, n in valid) / sum(n for _, n in valid)
            w_avg = {k: _wavg(k) for k in keys}
            level_summary[level] = {**w_avg, 'n_cells': total}
            print(f'  {level} (n={total}): ' +
                  '  '.join(f'{k.upper()}={w_avg[k]:.4f}' for k in keys))
        result_path = os.path.join(model_save_dir, 'results.json')
        os.makedirs(model_save_dir, exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump({'domain': 'three_level', 'task': task, 'model': model_name,
                       'test_sets': all_results, 'level_summary': level_summary}, f, indent=2)
        print(f'  Saved → {result_path}')
        return

    ckpt_path = os.path.join(model_save_dir, 'best.pt')
    if not os.path.exists(ckpt_path):
        print(f'  No checkpoint found at {ckpt_path}. Run scripts/train.py first.')
        return

    model = spec.build_fn(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    test_sets = d_cfg.get('test_sets', [])
    if not test_sets:
        print('  No test_sets defined in config.')
        return

    by_level: dict[str, list] = {}
    all_results = []

    for ts in test_sets:
        ts_dir   = ts['dir']
        level    = ts['level']
        pattern  = ts.get('pattern', None)
        ds_name  = os.path.basename(ts_dir)
        if pattern:
            import fnmatch
            all_files = sorted(glob.glob(os.path.join(ts_dir, '*.pkl')))
            matched   = [f for f in all_files if fnmatch.fnmatch(os.path.basename(f), pattern)]
            ds_name   = f'{os.path.basename(ts_dir)}[{pattern}]'
            test_ds   = spec.dataset_cls(ts_dir, n_cycles=n_cycles, n_grid=n_grid,
                                         soh_threshold=soh_threshold,
                                         use_log_rul=use_log_rul,
                                         pkl_files=matched)
        else:
            test_ds = spec.dataset_cls(ts_dir, n_cycles=n_cycles, n_grid=n_grid,
                                       soh_threshold=soh_threshold,
                                       use_log_rul=use_log_rul)
        n_cells = len(test_ds)
        if n_cells == 0:
            print(f'  [{level}] {ds_name}: empty, skipping.')
            continue

        test_loader = DataLoader(test_ds, batch_size=batch_size,
                                 shuffle=False, num_workers=0)
        if task == 'rul':
            ckpt_path = os.path.join(model_save_dir, 'best.pt')
            scaler_path = ckpt_path.replace('.pt', '_scaler.pkl')
            metrics = evaluate_fn(model, test_loader, device, scaler_path=scaler_path)
        elif task == 'soh_traj':
            metrics = evaluate_fn(model, test_loader, device, n_future=n_future)
        else:
            metrics = evaluate_fn(model, test_loader, device)

        line = f'  [{level}] {ds_name:20s} n={n_cells:3d} | '
        line += '  '.join(f'{k.upper()}={v:.4f}' for k, v in metrics.items())
        print(line)

        by_level.setdefault(level, []).append((metrics, n_cells))
        all_results.append({'dataset': ds_name, 'level': level,
                            'n_cells': n_cells, **metrics})

    if not all_results:
        return

    print()
    level_summary = {}
    for level in sorted(by_level):
        pairs = by_level[level]
        total = sum(n for _, n in pairs)
        keys  = list(pairs[0][0].keys())
        def _wavg(k):
            valid = [(m[k], n) for m, n in pairs if m[k] == m[k]]  # exclude NaN
            if not valid: return float('nan')
            return sum(v * n for v, n in valid) / sum(n for _, n in valid)
        w_avg = {k: _wavg(k) for k in keys}
        level_summary[level] = {**w_avg, 'n_cells': total}
        print(f'  {level} (n={total}): ' +
              '  '.join(f'{k.upper()}={w_avg[k]:.4f}' for k in keys))

    result_path = os.path.join(model_save_dir, 'results.json')
    os.makedirs(model_save_dir, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({
            'domain': 'three_level', 'task': task, 'model': model_name,
            'test_sets': all_results,
            'level_summary': level_summary,
        }, f, indent=2)
    print(f'  Saved → {result_path}')


# ── soh_point 专用评估（battery-level split）────────────────────────────────

def _eval_soh_point(model_name, domain, spec, cfg, d_cfg, t_cfg,
                    n_grid, batch_size, evaluate_fn,
                    checkpoint, model_save_dir, device):
    from src.train.soh_point.train_severson import evaluate as sklearn_eval
    strategy        = d_cfg.get('split_strategy', 'random')
    exclude_pattern = d_cfg.get('exclude_pattern', None)
    train_dirs      = d_cfg.get('train_dirs', []) if strategy == 'three_level' else None
    if train_dirs:
        dirs = train_dirs
    else:
        dirs = get_pkl_dir(d_cfg)

    full_ds    = SOHPointDataset(dirs, n_grid=n_grid, exclude_pattern=exclude_pattern)
    all_splits = make_soh_point_splits(full_ds, cfg, seed=42)

    if spec.build_fn is None:
        pkl_files = sorted(glob.glob(os.path.join(model_save_dir, 'split*.pkl')))
        if not pkl_files:
            print(f'  No pkl files found in {model_save_dir}. Run scripts/train.py first.')
            return
        all_metrics = []
        for pkl_path in pkl_files:
            basename = os.path.basename(pkl_path)
            try:
                si = int(basename.replace('split', '').replace('.pkl', '')) - 1
            except ValueError:
                si = 0
            if si >= len(all_splits):
                continue
            test_ds = all_splits[si]['test'] or all_splits[si]['val']
            if len(test_ds) == 0:
                continue
            metrics = sklearn_eval(test_ds, pkl_path)
            print(f'  Split {si+1}:', end='  ')
            _print_metrics(metrics)
            all_metrics.append(metrics)
        if not all_metrics:
            return
        keys = list(all_metrics[0].keys())
        print(f'\n  === {model_name.upper()} @ soh_point / {domain} ({len(all_metrics)} splits) ===')
        for k in keys:
            vals = [m[k] for m in all_metrics]
            print(f'    {k.upper():8s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')
        result_path = os.path.join(model_save_dir, 'results.json')
        os.makedirs(model_save_dir, exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump({'domain': domain, 'task': 'soh_point', 'model': model_name,
                       'splits': all_metrics,
                       'mean': {k: float(np.mean([m[k] for m in all_metrics])) for k in keys},
                       'std':  {k: float(np.std( [m[k] for m in all_metrics])) for k in keys},
                       }, f, indent=2)
        print(f'  Saved → {result_path}')
        return

    if checkpoint:
        ckpt_files = [checkpoint]
    else:
        ckpt_files = sorted(glob.glob(os.path.join(model_save_dir, 'split*.pt')))
        if not ckpt_files:
            print(f'  No checkpoints found in {model_save_dir}. Run scripts/train.py first.')
            return

    all_metrics = []
    for ckpt_path in ckpt_files:
        basename = os.path.basename(ckpt_path)
        try:
            si = int(basename.replace('split', '').replace('.pt', '')) - 1
        except ValueError:
            si = 0
        if si >= len(all_splits):
            print(f'  split index {si+1} out of range, skipping {basename}')
            continue
        test_ds = all_splits[si]['test'] or all_splits[si]['val']
        if len(test_ds) == 0:
            print(f'  Empty test set for split {si+1}, skipping.')
            continue
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        model = spec.build_fn(cfg).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        metrics = evaluate_fn(model, test_loader, device)
        print(f'  Split {si+1}:', end='  ')
        _print_metrics(metrics)
        all_metrics.append(metrics)

    if not all_metrics:
        return
    keys = list(all_metrics[0].keys())
    print(f'\n  === {model_name.upper()} @ soh_point / {domain} ({len(all_metrics)} splits) ===')
    for k in keys:
        vals = [m[k] for m in all_metrics]
        print(f'    {k.upper():8s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')
    result_path = os.path.join(model_save_dir, 'results.json')
    os.makedirs(model_save_dir, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({'domain': domain, 'task': 'soh_point', 'model': model_name,
                   'splits': all_metrics,
                   'mean': {k: float(np.mean([m[k] for m in all_metrics])) for k in keys},
                   'std':  {k: float(np.std( [m[k] for m in all_metrics])) for k in keys},
                   }, f, indent=2)
    print(f'  Saved → {result_path}')


# ── 普通域评估（random / stratified）────────────────────────────────────────

def _eval_standard(model_name: str, task: str, domain: str, cfg: dict,
                   checkpoint, save_dir: str, device: str):
    spec = get_spec(model_name, task)

    d_cfg         = cfg['data']
    t_cfg         = cfg['train']
    pkl_dir       = get_pkl_dir(d_cfg)
    n_cycles      = d_cfg.get('n_cycles', cfg['model'].get('n_cycles', 100))
    n_grid        = d_cfg.get('n_grid',   cfg['model'].get('n_grid', 200))
    soh_threshold = d_cfg.get('soh_threshold', 0.80)
    use_log_rul   = t_cfg.get('use_log_rul', False) and task == 'rul'
    n_future      = d_cfg.get('n_future', 5000)
    batch_size    = t_cfg.get('batch_size', 32)
    evaluate_fn   = _get_evaluate_fn(task)

    model_save_dir = os.path.join(save_dir, model_name)

    if task == 'soh_point':
        _eval_soh_point(model_name, domain, spec, cfg, d_cfg, t_cfg,
                        n_grid, batch_size, evaluate_fn,
                        checkpoint, model_save_dir, device)
        return

    # sklearn 模型：从 split{si}.pkl 加载推理
    if spec.build_fn is None:
        if task == 'rul':
            from src.train.rul.train_severson       import evaluate as sklearn_eval
        elif task == 'soh_point':
            from src.train.soh_point.train_severson  import evaluate as sklearn_eval
        else:
            from src.train.soh_traj.train_severson   import evaluate as sklearn_eval

        pkl_files = sorted(glob.glob(os.path.join(model_save_dir, 'split*.pkl')))
        if not pkl_files:
            print(f'  No pkl files found in {model_save_dir}. Run scripts/train.py first.')
            return

        ds_for_splits = BatteryDataset(pkl_dir, n_cycles=n_cycles, n_grid=n_grid,
                                       soh_threshold=soh_threshold)
        all_splits = make_splits(ds_for_splits, cfg)

        import copy
        all_metrics = []
        for pkl_path in pkl_files:
            basename = os.path.basename(pkl_path)
            try:
                si = int(basename.replace('split', '').replace('.pkl', '')) - 1
            except ValueError:
                si = 0
            if si >= len(all_splits):
                continue
            split = all_splits[si]
            test_ds = copy.copy(ds_for_splits)
            test_ds._samples = [ds_for_splits._all_samples[i] for i in split['test']
                                 if i < len(ds_for_splits._all_samples)]
            if len(test_ds) == 0:
                continue
            metrics = sklearn_eval(test_ds, pkl_path)
            print(f'  Split {si+1}:', end='  ')
            _print_metrics(metrics)
            all_metrics.append(metrics)

        if not all_metrics:
            return
        keys = list(all_metrics[0].keys())
        print(f'\n  === {model_name.upper()} @ {task} / {domain} ({len(all_metrics)} splits) ===')
        for k in keys:
            vals = [m[k] for m in all_metrics]
            print(f'    {k.upper():8s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')
        result_path = os.path.join(model_save_dir, 'results.json')
        with open(result_path, 'w') as f:
            json.dump({'domain': domain, 'task': task, 'model': model_name,
                       'splits': all_metrics,
                       'mean': {k: float(np.mean([m[k] for m in all_metrics])) for k in keys},
                       'std':  {k: float(np.std( [m[k] for m in all_metrics])) for k in keys},
                       }, f, indent=2)
        print(f'  Saved → {result_path}')
        return

    model_save_dir = os.path.join(save_dir, model_name)

    if checkpoint:
        ckpt_files = [checkpoint]
    else:
        pattern = os.path.join(model_save_dir, 'split*.pt')
        ckpt_files = sorted(glob.glob(pattern))
        if not ckpt_files:
            print(f'  No checkpoints found at {pattern}. Run scripts/train.py first.')
            return

    ds_for_splits = BatteryDataset(pkl_dir, n_cycles=n_cycles, n_grid=n_grid,
                                   soh_threshold=soh_threshold)
    all_splits = make_splits(ds_for_splits, cfg)

    import copy
    use_slice = (spec.dataset_cls is BatteryDataset)

    if not use_slice:
        full_ds = spec.dataset_cls(pkl_dir, n_cycles=n_cycles, n_grid=n_grid,
                                   soh_threshold=soh_threshold)
    else:
        full_ds = ds_for_splits

    def _make_test_ds(indices):
        ds = copy.copy(full_ds)
        sliced = [full_ds._all_samples[i] for i in indices
                  if i < len(full_ds._all_samples)]
        if hasattr(full_ds, '_samples'):
            ds._samples = sliced
        else:
            ds.samples = sliced
        if hasattr(ds, 'use_log_rul'):
            ds.use_log_rul = use_log_rul
        return ds

    all_metrics = []
    for ckpt_path in ckpt_files:
        basename = os.path.basename(ckpt_path)
        try:
            si = int(basename.replace('split', '').replace('.pt', '')) - 1
        except ValueError:
            si = 0
        if si >= len(all_splits):
            print(f'  split index {si+1} out of range, skipping {basename}')
            continue

        split   = all_splits[si]
        test_ds = _make_test_ds(split['test'])
        if len(test_ds) == 0:
            print(f'  Empty test set for split {si+1}, skipping.')
            continue

        test_loader = DataLoader(test_ds, batch_size=batch_size,
                                 shuffle=False, num_workers=0)
        model = spec.build_fn(cfg).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                         weights_only=True))

        if task == 'rul':
            scaler_path = ckpt_path.replace('.pt', '_scaler.pkl')
            metrics = evaluate_fn(model, test_loader, device, scaler_path=scaler_path)
        elif task == 'soh_traj':
            metrics = evaluate_fn(model, test_loader, device, n_future=n_future)
        else:
            metrics = evaluate_fn(model, test_loader, device)

        print(f'  Split {si+1}:', end='  ')
        _print_metrics(metrics)
        all_metrics.append(metrics)

    if not all_metrics:
        return

    keys = list(all_metrics[0].keys())
    print(f'\n  === {model_name.upper()} @ {task} / {domain} ({len(all_metrics)} splits) ===')
    for k in keys:
        vals = [m[k] for m in all_metrics]
        print(f'    {k.upper():8s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')

    result_path = os.path.join(model_save_dir, 'results.json')
    os.makedirs(model_save_dir, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({
            'domain': domain, 'task': task, 'model': model_name,
            'splits': all_metrics,
            'mean': {k: float(np.mean([m[k] for m in all_metrics])) for k in keys},
            'std':  {k: float(np.std( [m[k] for m in all_metrics])) for k in keys},
        }, f, indent=2)
    print(f'  Saved → {result_path}')


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain',     required=True, choices=list(DOMAIN_CFG.keys()))
    parser.add_argument('--model',      required=True,
                        help='Model name or "all".')
    parser.add_argument('--task',       default=None, choices=list(ALL_TASKS),
                        help='rul | soh_point | soh_traj. Defaults to config data.task.')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to a specific .pt file (standard domains only).')
    parser.add_argument('--gpu',        type=int, default=None,
                        help='GPU index to use (e.g. 0, 1). Defaults to cuda:0 if available.')
    parser.add_argument('--config',     default='configs/default.yaml')
    parser.add_argument('--save_dir',   default=None)
    args = parser.parse_args()

    if args.gpu is not None and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    cfg      = load_config(args.config, DOMAIN_CFG[args.domain])
    task     = args.task or cfg.get('data', {}).get('task', 'rul')
    if task not in ALL_TASKS:
        raise ValueError(f"Unknown task '{task}'. Choose from {sorted(ALL_TASKS)}")

    save_dir = args.save_dir or os.path.join('results', args.domain, task)
    strategy = cfg.get('data', {}).get('split_strategy', 'random')

    task_models = ALL_MODELS[task]
    models = sorted(task_models) if args.model == 'all' else [args.model.lower()]
    for m in models:
        if m not in task_models:
            print(f'Unknown model "{m}" for task "{task}", skipping.')
            continue
        print(f'\n{"="*60}\n  Model: {m.upper()}  |  Task: {task}  |  Domain: {args.domain}\n{"="*60}')
        if strategy == 'three_level':
            _eval_three_level(m, task, cfg, save_dir, device)
        else:
            _eval_standard(m, task, args.domain, cfg, args.checkpoint, save_dir, device)


if __name__ == '__main__':
    main()
