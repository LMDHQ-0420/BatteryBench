"""Train one model on the fixed BatteryBench data split."""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.config import DOMAIN_CFG, get_pkl_dir, load_config
from src.data.cycle_dataset import soh_point_collate_fn
from src.models.registry import ALL_MODELS, ALL_TASKS, get_spec
from src.splits import make_battery_splits


def _build_dataset(spec, config, directories, exclude_pattern):
    data = config['data']
    return spec.dataset_cls(
        directories,
        soh_threshold=data.get('soh_threshold', 0.80),
        eol_threshold=data.get('eol_threshold', data.get('soh_threshold', 0.80)),
        early_cycle=data.get('early_cycle', 100),
        seq_len=data.get('seq_len', 1),
        curve_length=data.get('curve_length', 400),
        exclude_pattern=exclude_pattern,
        use_log_rul=(data.get('task') == 'rul' and config['train'].get('use_log_rul', False)),
    )


def _print_metrics(metrics):
    print('  ' + '  '.join(f'{key.upper()}={value:.4f}' for key, value in metrics.items()))


def train_one_model(model_name, task, domain, config, save_dir, device, seed=1):
    spec = get_spec(model_name, task)
    data = config['data']
    train_config = config['train']
    strategy = data.get('split_strategy', 'random')
    batch_size = train_config.get(
        'soh_point_batch_size' if task == 'soh_point' else 'batch_size', 32,
    )
    if spec.batch_size_cap:
        batch_size = min(batch_size, spec.batch_size_cap)

    seed_dir = os.path.join(save_dir, model_name, f'seed{seed}')
    os.makedirs(seed_dir, exist_ok=True)
    directories = data.get('train_dirs') if strategy == 'four_level' else get_pkl_dir(data)
    dataset = _build_dataset(spec, config, directories, data.get('exclude_pattern'))
    split_seed = data.get('split_seed', 1)
    split = make_battery_splits(dataset, config, seed=split_seed)[0]

    if spec.build_fn is None:
        test_dataset = split['test'] if split['test'] is not None else split['val']
        if not len(split['train']) or not len(test_dataset):
            print('  Skipping empty split.')
            return
        save_path = os.path.join(seed_dir, 'best.pkl')
        print(f"  train={len(split['train'])} test={len(test_dataset)}")
        _print_metrics(spec.train_fn(split['train'], test_dataset, save_path=save_path))
        print(f'  Model → {save_path}')
        return

    if not len(split['train']) or not len(split['val']):
        print('  Skipping empty split.')
        return
    collate = soh_point_collate_fn if task == 'soh_point' else None
    train_loader = DataLoader(
        split['train'], batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate,
    )
    val_loader = DataLoader(
        split['val'], batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate,
    )
    model = spec.build_fn(config).to(device)
    save_path = os.path.join(seed_dir, 'best.pt')
    print(f"  train={len(split['train'])} val={len(split['val'])}")
    spec.train_fn(model, train_loader, val_loader, config, save_path, device)
    print(f'  Checkpoint → {save_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', required=True, choices=list(DOMAIN_CFG))
    parser.add_argument('--model', required=True, help='Model name or "all".')
    parser.add_argument('--task', choices=list(ALL_TASKS))
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--save_dir')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = f'cuda:{args.gpu}' if args.gpu is not None and torch.cuda.is_available() else (
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Device: {device}')

    config = load_config(args.config, DOMAIN_CFG[args.domain])
    task = args.task or config.get('data', {}).get('task', 'rul')
    if task not in ALL_TASKS:
        raise ValueError(f"Unknown task '{task}'. Choose from {sorted(ALL_TASKS)}")
    config['data']['task'] = task
    save_dir = args.save_dir or os.path.join('results', args.domain, task)
    os.makedirs(save_dir, exist_ok=True)

    available = ALL_MODELS[task]
    models = sorted(available) if args.model == 'all' else [args.model.lower()]
    for model in models:
        if model not in available:
            print(f'Unknown model "{model}" for task "{task}", skipping.')
            continue
        print(f'\n{"=" * 60}\n  Model: {model.upper()} | Task: {task} | '
              f'Domain: {args.domain} | Seed: {args.seed}\n{"=" * 60}')
        train_one_model(model, task, args.domain, config, save_dir, device, args.seed)


if __name__ == '__main__':
    main()
