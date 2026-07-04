"""
scripts/train.py — 训练单个模型

用法:
    python scripts/train.py --domain li_ion --model gru --task rul
    python scripts/train.py --domain li_ion --model all --task soh_point
    python scripts/train.py --domain three_level --model gru --task rul
    python scripts/train.py --domain li_ion --model batlinet --task rul --split_idx 2

参数:
    --domain      数据域，见 configs/domains/
    --model       模型名称，或 all 跑全部
    --task        rul | soh_point | soh_traj  （默认读 config data.task）
    --split_idx   只跑第几个 split（1-based）。three_level 忽略此参数
    --seed        随机种子，默认 42
    --config      config 文件路径，默认 configs/default.yaml
    --save_dir    结果保存根目录，默认 results/<domain>/<task>
"""

import os
import sys
import json
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


def _build_full_dataset(spec, cfg, dirs, exclude_pattern):
    """构造统一多样本 dataset（RUL/SOHPoint/SOHTraj）。"""
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
        use_log_rul=(d_cfg.get('task') == 'rul' and cfg['train'].get('use_log_rul', False)),
    )


def train_one_model(model_name: str, task: str, domain: str, cfg: dict,
                    split_idx, save_dir: str, device: str):
    spec          = get_spec(model_name, task)
    d_cfg         = cfg['data']
    t_cfg         = cfg['train']
    strategy      = d_cfg.get('split_strategy', 'random')
    batch_size    = t_cfg.get('batch_size', 32)
    if spec.batch_size_cap:
        batch_size = min(batch_size, spec.batch_size_cap)

    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    exclude_pattern = d_cfg.get('exclude_pattern', None)
    train_dirs      = d_cfg.get('train_dirs', []) if strategy == 'three_level' else None
    dirs = train_dirs if train_dirs else get_pkl_dir(d_cfg)

    full_ds = _build_full_dataset(spec, cfg, dirs, exclude_pattern)
    all_splits = make_battery_splits(full_ds, cfg, seed=42)

    if strategy == 'three_level':
        splits, split_offset = all_splits, 0
    elif split_idx is not None:
        if split_idx < 1 or split_idx > len(all_splits):
            raise ValueError(f"--split_idx 范围 1..{len(all_splits)}")
        splits, split_offset = [all_splits[split_idx - 1]], split_idx - 1
    else:
        splits, split_offset = all_splits, 0

    # ── sklearn 模型（severson）：train_fn(train_ds, test_ds, save_path) ─────────
    if spec.build_fn is None:
        all_metrics = []
        for i, split in enumerate(splits):
            si = i + split_offset + 1
            label = 'best' if strategy == 'three_level' else f'split{si}'
            train_ds = split['train']
            test_ds  = split['test'] if split['test'] is not None else split['val']
            print(f'\n--- Split {si}/{len(all_splits)} | train={len(train_ds)} test={len(test_ds)} ---')
            if len(train_ds) == 0 or len(test_ds) == 0:
                print('  Skipping empty split.'); continue
            pkl_path = os.path.join(model_save_dir, f'{label}.pkl')
            metrics = spec.train_fn(train_ds, test_ds, save_path=pkl_path)
            _print_metrics(metrics)
            all_metrics.append(metrics)
        if all_metrics:
            keys = list(all_metrics[0].keys())
            with open(os.path.join(model_save_dir, 'results.json'), 'w') as f:
                json.dump({
                    'domain': domain, 'task': task, 'model': model_name,
                    'splits': all_metrics,
                    'mean': {k: float(np.mean([m[k] for m in all_metrics])) for k in keys},
                    'std':  {k: float(np.std( [m[k] for m in all_metrics])) for k in keys},
                }, f, indent=2)
            print(f'  Saved → {os.path.join(model_save_dir, "results.json")}')
        return

    # ── 神经网络模型：train_fn(model, train_loader, val_loader, cfg, save_path) ──
    for i, split in enumerate(splits):
        si = i + split_offset + 1
        label = 'best' if strategy == 'three_level' else f'split{si}'
        train_ds, val_ds = split['train'], split['val']
        print(f'\n--- Split {si}/{len(all_splits)} | train={len(train_ds)} val={len(val_ds)} ---')
        if len(train_ds) == 0 or len(val_ds) == 0:
            print('  Skipping empty split.'); continue
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
        model     = spec.build_fn(cfg).to(device)
        save_path = os.path.join(model_save_dir, f'{label}.pt')
        spec.train_fn(model, train_loader, val_loader, cfg, save_path, device)
        print(f'  Checkpoint → {save_path}')
    return


def _print_metrics(metrics: dict):
    parts = [f'{k.upper()}={v:.4f}' for k, v in metrics.items()]
    print('  ' + '  '.join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain',    required=True, choices=list(DOMAIN_CFG.keys()))
    parser.add_argument('--model',     required=True,
                        help='Model name or "all".')
    parser.add_argument('--task',      default=None, choices=list(ALL_TASKS),
                        help='rul | soh_point | soh_traj. Defaults to config data.task.')
    parser.add_argument('--split_idx', type=int, default=None,
                        help='Only run this split (1-based). Ignored for three_level.')
    parser.add_argument('--seed',      type=int, default=None,
                        help='Random seed. If not set, no seed is fixed.')
    parser.add_argument('--gpu',       type=int, default=None,
                        help='GPU index to use (e.g. 0, 1). Defaults to cuda:0 if available.')
    parser.add_argument('--config',    default='configs/default.yaml')
    parser.add_argument('--save_dir',  default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    if args.gpu is not None and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    cfg  = load_config(args.config, DOMAIN_CFG[args.domain])
    task = args.task or cfg.get('data', {}).get('task', 'rul')
    if task not in ALL_TASKS:
        raise ValueError(f"Unknown task '{task}'. Choose from {sorted(ALL_TASKS)}")

    save_dir = args.save_dir or os.path.join('results', args.domain, task)
    os.makedirs(save_dir, exist_ok=True)

    task_models = ALL_MODELS[task]
    models = sorted(task_models) if args.model == 'all' else [args.model.lower()]
    for m in models:
        if m not in task_models:
            print(f'Unknown model "{m}" for task "{task}", skipping.')
            continue
        print(f'\n{"="*60}\n  Model: {m.upper()}  |  Task: {task}  |  Domain: {args.domain}\n{"="*60}')
        train_one_model(m, task, args.domain, cfg, args.split_idx, save_dir, device)


if __name__ == '__main__':
    main()
