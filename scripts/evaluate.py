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
import fnmatch
import fcntl
from importlib import import_module
from pathlib import Path
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
from src.data.cycle_dataset import soh_point_collate_fn
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
        soh_threshold=d_cfg.get('soh_threshold', 0.80),
        eol_threshold=d_cfg.get('eol_threshold', d_cfg.get('soh_threshold', 0.80)),
        early_cycle=d_cfg.get('early_cycle', 100),
        seq_len=d_cfg.get('seq_len', 1),
        curve_length=d_cfg.get('curve_length', 400),
        exclude_pattern=exclude_pattern,
    )


def _load_model(spec, cfg, checkpoint, device, train_ds=None):
    model = spec.build_fn(cfg).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    if train_ds is not None:
        # BatLiNet 的参考池来自训练集；固定抽样便于重复评估。
        indices = np.random.default_rng(42).choice(
            len(train_ds), min(model.n_ref, len(train_ds)), replace=False)
        samples = [train_ds[int(i)] for i in indices]
        label = {'rul': 'eol', 'soh_point': 'soh_point', 'soh_traj': 'soh_traj'}[cfg['data']['task']]
        model.set_reference(
            torch.stack([sample['cycle_curve_data'] for sample in samples]).to(device),
            torch.stack([sample[label] for sample in samples]).to(device),
            torch.stack([sample['curve_attn_mask'] for sample in samples]).to(device),
        )
    return model


def _eval_dl(cfg, task, model, test_ds, batch_size, device, checkpoint, output_dir):
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=soh_point_collate_fn if task == 'soh_point' else None)
    options = {'output_dir': output_dir}
    if task == 'soh_traj':
        options.update(n_future=cfg['data'].get('n_future', 5000),
                       eol_threshold=cfg['data'].get('eol_threshold', cfg['data'].get('soh_threshold', 0.80)))
    elif task == 'rul':
        scaler_path = checkpoint.replace('.pt', '_scaler.pkl')
        if os.path.exists(scaler_path):
            options['scaler_path'] = scaler_path
    return _get_evaluate_fn(task)(model, loader, device, **options)


def _write_json(path, result):
    with open(path, 'w') as file:
        json.dump(result, file, indent=2)


def _save_seed_summary(model_dir, domain, task, model_name):
    # 同一模型的多个 seed 可并发评估，汇总写入需串行。
    with open(Path(model_dir) / 'summary.json', 'a+') as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        seeds = {}
        for seed in [1, 7, 42, 123, 2024]:
            path = Path(model_dir) / f'seed{seed}' / 'results.json'
            if path.exists():
                with path.open() as seed_file:
                    result = json.load(seed_file)
                seeds[str(seed)] = result['level_summary'] if domain == 'four_level' else result['mean']
        def aggregate(metrics):
            keys = [key for key in metrics[0] if key not in ('n_samples', 'n_batteries')]
            return {stat: {key: float(function([m[key] for m in metrics])) for key in keys}
                    for stat, function in [('mean', np.mean), ('std', np.std)]}
        if domain == 'four_level':
            levels = sorted({level for summary in seeds.values() for level in summary})
            statistics = {level: aggregate([summary[level] for summary in seeds.values()
                                            if level in summary]) for level in levels}
        else:
            statistics = aggregate(list(seeds.values()))
        file.seek(0)
        file.truncate()
        json.dump({'domain': domain, 'task': task, 'model': model_name,
                   'n_seeds': len(seeds), 'seeds': seeds, 'statistics': statistics}, file, indent=2)


def evaluate_one_model(model_name, task, domain, cfg, save_dir, device, seed=42):
    spec = get_spec(model_name, task)
    cfg['data']['task'] = task
    batch_size = cfg['train'].get(
        'soh_point_batch_size' if task == 'soh_point' else 'batch_size', 32)
    if spec.batch_size_cap:
        batch_size = min(batch_size, spec.batch_size_cap)
    model_dir = os.path.join(save_dir, model_name)
    seed_dir = os.path.join(model_dir, f'seed{seed}')
    if cfg['data'].get('split_strategy') == 'four_level':
        _eval_four_level(model_name, task, spec, cfg, batch_size, seed_dir, device)
    else:
        full_ds = _build_full_dataset(spec, cfg, get_pkl_dir(cfg['data']), None)
        splits = make_battery_splits(full_ds, cfg, seed=42)
        metrics = []
        for index, split in enumerate(splits, 1):
            extension = 'pkl' if spec.build_fn is None else 'pt'
            checkpoint = os.path.join(seed_dir, f'split{index}.{extension}')
            output_dir = os.path.join(seed_dir, 'test', f'split{index}')
            test_ds = split['test']
            if spec.build_fn is None:
                evaluate = import_module(f'src.train.{task}.train_severson').evaluate
                result = evaluate(test_ds, checkpoint, output_dir=output_dir)
            else:
                model = _load_model(spec, cfg, checkpoint, device,
                                    split['train'] if model_name == 'batlinet' else None)
                result = _eval_dl(cfg, task, model, test_ds, batch_size, device,
                                  checkpoint, output_dir)
            _print_metrics(result)
            metrics.append(result)
        keys = list(metrics[0])
        _write_json(os.path.join(seed_dir, 'results.json'), {
            'domain': domain, 'task': task, 'model': model_name,
            'splits': [{'split_idx': index, 'n_samples': len(split['test']),
                         'n_batteries': len({bidx for bidx, _ in split['test']._samples}), **result}
                       for index, (split, result) in enumerate(zip(splits, metrics), 1)],
            'mean': {key: float(np.mean([m[key] for m in metrics])) for key in keys},
            'std': {key: float(np.std([m[key] for m in metrics])) for key in keys}})
    _save_seed_summary(model_dir, domain, task, model_name)


def _eval_four_level(model_name, task, spec, cfg, batch_size, seed_dir, device):
    data = cfg['data']
    extension = 'pkl' if spec.build_fn is None else 'pt'
    checkpoint = os.path.join(seed_dir, f'best.{extension}')
    if spec.build_fn is None:
        evaluate = import_module(f'src.train.{task}.train_severson').evaluate
    else:
        train_ds = None
        if model_name == 'batlinet':
            full_ds = _build_full_dataset(spec, cfg, data['train_dirs'], data.get('exclude_pattern'))
            train_ds = make_battery_splits(full_ds, cfg, seed=42)[0]['train']
        model = _load_model(spec, cfg, checkpoint, device, train_ds)
    results, by_level = [], {}
    for test_set in data['test_sets']:
        directory, level = test_set['dir'], test_set['level']
        files = sorted(glob.glob(os.path.join(directory, '*.pkl')))
        files = [file for file in files
                 if fnmatch.fnmatch(os.path.basename(file), test_set.get('pattern', '*.pkl'))]
        test_ds = spec.dataset_cls(
            directory, pkl_files=files,
            soh_threshold=data.get('soh_threshold', 0.80),
            eol_threshold=data.get('eol_threshold', data.get('soh_threshold', 0.80)),
            early_cycle=data.get('early_cycle', 100), seq_len=data.get('seq_len', 1),
            curve_length=data.get('curve_length', 400))
        name = os.path.basename(directory)
        counts = {'dataset': name, 'level': level, 'n_samples': len(test_ds),
                  'n_batteries': len({bidx for bidx, _ in test_ds._samples})}
        if not len(test_ds):
            results.append({**counts, 'status': 'empty'})
            continue
        output_dir = os.path.join(seed_dir, 'test', f'{level}_{name}')
        if spec.build_fn is None:
            metrics = evaluate(test_ds, checkpoint, output_dir=output_dir)
        else:
            metrics = _eval_dl(cfg, task, model, test_ds, batch_size, device,
                               checkpoint, output_dir)
        _print_metrics(metrics)
        results.append({**counts, **metrics})
        by_level.setdefault(level, []).append((metrics, counts))
    summary = {}
    for level, pairs in by_level.items():
        n = sum(counts['n_samples'] for _, counts in pairs)
        summary[level] = {
            **{key: sum(metrics[key] * counts['n_samples'] for metrics, counts in pairs) / n
               for key in pairs[0][0]},
            'n_samples': n,
            'n_batteries': sum(counts['n_batteries'] for _, counts in pairs)}
    _write_json(os.path.join(seed_dir, 'results.json'), {
        'domain': 'four_level', 'task': task, 'model': model_name,
        'test_sets': results, 'level_summary': summary})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', required=True, choices=list(DOMAIN_CFG))
    parser.add_argument('--model',  required=True)
    parser.add_argument('--task',   default=None, choices=list(ALL_TASKS))
    parser.add_argument('--gpu',    type=int, default=None)
    parser.add_argument('--seed',   type=int, default=42,
                        help='Selects the results/<model>/seed<N>/ subdir to evaluate.')
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
    cfg['data']['task'] = task
    save_dir = args.save_dir or os.path.join('results', args.domain, task)

    task_models = ALL_MODELS[task]
    models = sorted(task_models) if args.model == 'all' else [args.model.lower()]
    for m in models:
        if m not in task_models:
            print(f'Unknown model "{m}" for task "{task}", skipping.'); continue
        print(f'\n{"="*60}\n  Model: {m.upper()}  |  Task: {task}  |  Domain: {args.domain}  |  Seed: {args.seed}\n{"="*60}')
        evaluate_one_model(m, task, args.domain, cfg, save_dir, device, seed=args.seed)


if __name__ == '__main__':
    main()
