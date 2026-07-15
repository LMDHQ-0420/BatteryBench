"""
scripts/evaluate.py — 评估已训练的 checkpoint（多样本 + attention mask 架构）

用法:
    python scripts/evaluate.py --domain li_ion --model gru --task rul
    python scripts/evaluate.py --domain li_ion --model all --task soh_point
    python scripts/evaluate.py --domain four_level --model gru --task rul

random/stratified: 逐 split 评估其 test 子集，报告 mean/std。
four_level: 对每个 test_set（L1/L2/L3/L4）独立加载并评估，按 level 加权平均。
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
from src.splits import make_battery_splits
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
    print('  ' + '  '.join(f'{k.upper()}={v:.4f}' for k, v in metrics.items()))


def _build_full_dataset(spec, cfg, dirs, exclude_pattern):
    d_cfg = cfg['data']
    return spec.dataset_cls(
        dirs,
        n_grid=d_cfg.get('n_grid', 200),
        soh_threshold=d_cfg.get('soh_threshold', 0.80),
        eol_threshold=d_cfg.get('eol_threshold', d_cfg.get('soh_threshold', 0.80)),
        early_cycle=d_cfg.get('early_cycle', 100),
        seq_len=d_cfg.get('seq_len', 1),
        charge_discharge_length=d_cfg.get('charge_discharge_length', 300),
        exclude_pattern=exclude_pattern,
    )


def _eval_dl(spec, cfg, task, model, test_ds, batch_size, device, scaler_path=None):
    """跑一个 test 子集，返回 metrics dict。"""
    evaluate_fn = _get_evaluate_fn(task)
    eol_thr = cfg['data'].get('eol_threshold', cfg['data'].get('soh_threshold', 0.80))
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    if task == 'soh_traj':
        return evaluate_fn(model, loader, device,
                           n_future=cfg['data'].get('n_future', 5000),
                           eol_threshold=eol_thr)
    if task == 'rul' and scaler_path and os.path.exists(scaler_path):
        return evaluate_fn(model, loader, device, scaler_path=scaler_path)
    return evaluate_fn(model, loader, device)


def evaluate_one_model(model_name, task, domain, cfg, save_dir, device):
    spec       = get_spec(model_name, task)
    d_cfg      = cfg['data']
    t_cfg      = cfg['train']
    strategy   = d_cfg.get('split_strategy', 'random')
    batch_size = t_cfg.get('batch_size', 32)
    if spec.batch_size_cap:
        batch_size = min(batch_size, spec.batch_size_cap)

    model_save_dir = os.path.join(save_dir, model_name)
    exclude_pattern = d_cfg.get('exclude_pattern', None)
    train_dirs      = d_cfg.get('train_dirs', []) if strategy == 'four_level' else None
    dirs = train_dirs if train_dirs else get_pkl_dir(d_cfg)

    if strategy == 'four_level':
        _eval_four_level(model_name, task, domain, spec, cfg, dirs,
                          exclude_pattern, batch_size, model_save_dir, device)
        return

    full_ds = _build_full_dataset(spec, cfg, dirs, exclude_pattern)
    all_splits = make_battery_splits(full_ds, cfg, seed=42)

    # ── sklearn 模型 ─────────────────────────────────────────────────────────
    if spec.build_fn is None:
        all_metrics = []
        for i, split in enumerate(all_splits):
            si = i + 1
            pkl_path = os.path.join(model_save_dir, f'split{si}.pkl')
            if not os.path.exists(pkl_path):
                continue
            test_ds = split['test'] if split['test'] is not None else split['val']
            if len(test_ds) == 0:
                continue
            print(f'\n--- Split {si} (n={len(test_ds)}) ---')
            from importlib import import_module
            ev = import_module(f'src.train.{task}.train_severson').evaluate
            metrics = ev(test_ds, pkl_path)
            _print_metrics(metrics)
            all_metrics.append(metrics)
        _save_splits(all_metrics, domain, task, model_name, model_save_dir)
        return

    # ── 神经网络模型 ─────────────────────────────────────────────────────────
    all_metrics = []
    for i, split in enumerate(all_splits):
        si = i + 1
        ckpt = os.path.join(model_save_dir, f'split{si}.pt')
        if not os.path.exists(ckpt):
            continue
        test_ds = split['test'] if split['test'] is not None else split['val']
        if len(test_ds) == 0:
            continue
        model = spec.build_fn(cfg).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        scaler_path = ckpt.replace('.pt', '_scaler.pkl')
        print(f'\n--- Split {si} (n={len(test_ds)}) ---')
        if task == 'rul' and os.path.exists(scaler_path):
            loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
            metrics = eval_rul.evaluate(model, loader, device, scaler_path=scaler_path)
        else:
            metrics = _eval_dl(spec, cfg, task, model, test_ds, batch_size, device)
        _print_metrics(metrics)
        all_metrics.append(metrics)
    _save_splits(all_metrics, domain, task, model_name, model_save_dir)


def _save_splits(all_metrics, domain, task, model_name, model_save_dir):
    if not all_metrics:
        print('  No checkpoints found; nothing to evaluate.')
        return
    keys = list(all_metrics[0].keys())
    result_path = os.path.join(model_save_dir, 'results.json')
    with open(result_path, 'w') as f:
        json.dump({
            'domain': domain, 'task': task, 'model': model_name,
            'splits': all_metrics,
            'mean': {k: float(np.mean([m[k] for m in all_metrics])) for k in keys},
            'std':  {k: float(np.std( [m[k] for m in all_metrics])) for k in keys},
        }, f, indent=2)
    print(f'  Saved → {result_path}')


def _eval_four_level(model_name, task, domain, spec, cfg, train_dirs,
                      exclude_pattern, batch_size, model_save_dir, device):
    d_cfg = cfg['data']
    test_sets = d_cfg.get('test_sets', [])
    ckpt = os.path.join(model_save_dir, 'best.pt')
    pkl  = os.path.join(model_save_dir, 'best.pkl')
    is_sklearn = spec.build_fn is None

    model = None
    scaler_path = ckpt.replace('.pt', '_scaler.pkl')
    if not is_sklearn:
        if not os.path.exists(ckpt):
            print(f'  No checkpoint at {ckpt}.'); return
        model = spec.build_fn(cfg).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    else:
        if not os.path.exists(pkl):
            print(f'  No pkl at {pkl}.'); return
        from importlib import import_module
        sk_eval = import_module(f'src.train.{task}.train_severson').evaluate

    by_level, all_results = {}, []
    for ts in test_sets:
        ts_dir, level = ts['dir'], ts['level']
        pattern = ts.get('pattern', None)
        ds_name = os.path.basename(ts_dir) + (f'[{pattern}]' if pattern else '')
        pkl_files = None
        if pattern:
            import fnmatch
            allf = sorted(glob.glob(os.path.join(ts_dir, '*.pkl')))
            pkl_files = [f for f in allf if fnmatch.fnmatch(os.path.basename(f), pattern)]
        test_ds = _build_full_dataset(spec, cfg,
                                      [ts_dir] if pkl_files is None else ts_dir,
                                      None)
        if pkl_files is not None:
            # rebuild restricted to matched files
            test_ds = spec.dataset_cls(
                ts_dir, n_grid=d_cfg.get('n_grid', 200),
                soh_threshold=d_cfg.get('soh_threshold', 0.80),
                eol_threshold=d_cfg.get('eol_threshold', d_cfg.get('soh_threshold', 0.80)),
                early_cycle=d_cfg.get('early_cycle', 100),
                seq_len=d_cfg.get('seq_len', 1),
                charge_discharge_length=d_cfg.get('charge_discharge_length', 300),
                pkl_files=pkl_files,
            )
        n = len(test_ds)
        if n == 0:
            print(f'  [{level}] {ds_name}: empty, skipping.'); continue
        if is_sklearn:
            metrics = sk_eval(test_ds, pkl)
        else:
            metrics = _eval_dl(spec, cfg, task, model, test_ds, batch_size, device,
                               scaler_path=scaler_path)
        print(f'  [{level}] {ds_name:22s} n={n:4d} | ' +
              '  '.join(f'{k.upper()}={v:.4f}' for k, v in metrics.items()))
        by_level.setdefault(level, []).append((metrics, n))
        all_results.append({'dataset': ds_name, 'level': level, 'n_cells': n, **metrics})

    if not all_results:
        return
    level_summary = {}
    for level in sorted(by_level):
        pairs = by_level[level]
        keys = list(pairs[0][0].keys())
        def _wavg(k):
            valid = [(m[k], n) for m, n in pairs if m[k] == m[k]]
            return sum(v * n for v, n in valid) / sum(n for _, n in valid) if valid else float('nan')
        level_summary[level] = {**{k: _wavg(k) for k in keys},
                                'n_cells': sum(n for _, n in pairs)}
        print(f'  {level}: ' + '  '.join(f'{k.upper()}={level_summary[level][k]:.4f}' for k in keys))
    with open(os.path.join(model_save_dir, 'results.json'), 'w') as f:
        json.dump({'domain': 'four_level', 'task': task, 'model': model_name,
                   'test_sets': all_results, 'level_summary': level_summary}, f, indent=2)
    print(f'  Saved → {os.path.join(model_save_dir, "results.json")}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', required=True, choices=list(DOMAIN_CFG))
    parser.add_argument('--model',  required=True)
    parser.add_argument('--task',   default=None, choices=list(ALL_TASKS))
    parser.add_argument('--gpu',    type=int, default=None)
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--save_dir', default=None)
    args = parser.parse_args()

    if args.gpu is not None and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    cfg  = load_config(args.config, DOMAIN_CFG[args.domain])
    task = args.task or cfg.get('data', {}).get('task', 'rul')
    save_dir = args.save_dir or os.path.join('results', args.domain, task)

    task_models = ALL_MODELS[task]
    models = sorted(task_models) if args.model == 'all' else [args.model.lower()]
    for m in models:
        if m not in task_models:
            print(f'Unknown model "{m}" for task "{task}", skipping.'); continue
        print(f'\n{"="*60}\n  Model: {m.upper()}  |  Task: {task}  |  Domain: {args.domain}\n{"="*60}')
        evaluate_one_model(m, task, args.domain, cfg, save_dir, device)


if __name__ == '__main__':
    main()
