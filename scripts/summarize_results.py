"""Validate five-seed results and update the complete benchmark summaries."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
SEEDS = {'1', '2', '3', '4', '5'}
DOMAINS = ('li_ion', 'calb', 'na_ion', 'zn_ion')
TASKS = ('soh_traj', 'soh_point', 'rul')
LEVELS = ('L1', 'L2', 'L3', 'L4')
MODELS = (
    ('autoformer', 'Autoformer'), ('bigru', 'BiGRU'), ('bilstm', 'BiLSTM'),
    ('cnn', 'CNN'), ('dlinear', 'DLinear'), ('gru', 'GRU'), ('ic2ml', 'IC2ML'),
    ('itransformer', 'iTransformer'), ('lstm', 'LSTM'), ('micn', 'MICN'),
    ('mlp', 'MLP'), ('patchtst', 'PatchTST'), ('severson', 'Severson'),
    ('timemixer', 'TimeMixer'), ('transformer', 'Transformer'),
)
DISPLAY = dict(MODELS)
DOMAIN_NAMES = {
    'li_ion': ('Li-ion', '锂离子'),
    'calb': ('CALB', 'CALB'),
    'na_ion': ('Na-ion', '钠离子'),
    'zn_ion': ('Zn-ion', '锌离子'),
}
TASK_NAMES = {
    'soh_traj': ('SOH Trajectory', 'SOH 轨迹'),
    'soh_point': ('SOH Point', 'SOH 点估计'),
    'rul': ('RUL', 'RUL'),
}
METRICS = ('rmse', 'mae', 'mape')


def load_summary(domain: str, task: str, model: str) -> dict:
    path = RESULTS / domain / task / model / 'summary.json'
    data = json.loads(path.read_text())
    if data.get('domain') != domain or data.get('task') != task or data.get('model') != model:
        raise ValueError(f'identity mismatch: {path}')
    if data.get('n_seeds') != 5 or set(data.get('seeds', {})) != SEEDS:
        raise ValueError(f'incomplete five-seed summary: {path}')
    return data


def collect() -> dict:
    standard = {}
    for domain in DOMAINS:
        standard[domain] = {}
        for task in TASKS:
            standard[domain][task] = {
                model: load_summary(domain, task, model)['statistics']
                for model, _ in MODELS
            }
    four_level = {}
    for task in TASKS:
        four_level[task] = {
            model: load_summary('four_level', task, model)['statistics']
            for model, _ in MODELS
        }
    return {
        'seeds': [1, 2, 3, 4, 5],
        'n_experiments': len(MODELS) * 5 * (len(DOMAINS) + 1) * len(TASKS),
        'metric_order': list(METRICS),
        'standard_domains': standard,
        'four_level': four_level,
    }


def value(stats: dict, metric: str) -> tuple[float, float]:
    return stats['mean'][metric], stats['std'][metric]


def formatted(stats: dict, metric: str, best: float) -> str:
    mean, std = value(stats, metric)
    if metric == 'mape':
        text = f'{100 * mean:.2f} ± {100 * std:.2f}%'
    else:
        text = f'{mean:.4f} ± {std:.4f}'
    return f'<u>{text}</u>' if math.isclose(mean, best, rel_tol=1e-12, abs_tol=1e-15) else text


def standard_table(records: dict) -> str:
    best = {metric: min(value(records[model], metric)[0] for model, _ in MODELS)
            for metric in METRICS}
    rows = ['| Model | RMSE | MAE | MAPE |', '|---|---:|---:|---:|']
    for model, name in MODELS:
        cells = [formatted(records[model], metric, best[metric]) for metric in METRICS]
        rows.append(f"| {name} | " + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def four_level_table(records: dict) -> str:
    best = {
        level: {
            metric: min(value(records[model][level], metric)[0] for model, _ in MODELS
                        if level in records[model])
            for metric in METRICS
        }
        for level in LEVELS
        if any(level in records[model] for model, _ in MODELS)
    }
    lines = [
        '<table>', '<thead>', '<tr>', '<th rowspan="2">Model</th>',
        *[f'<th colspan="3">{level}</th>' for level in LEVELS],
        '</tr>', '<tr>',
        *['<th>RMSE</th><th>MAE</th><th>MAPE</th>' for _ in LEVELS],
        '</tr>', '</thead>', '<tbody>',
    ]
    for model, name in MODELS:
        lines += ['<tr>', f'<td>{name}</td>']
        for level in LEVELS:
            if level not in records[model]:
                lines += ['<td></td>', '<td></td>', '<td></td>']
            else:
                lines += [f'<td>{formatted(records[model][level], metric, best[level][metric])}</td>'
                          for metric in METRICS]
        lines += ['</tr>']
    lines += ['</tbody>', '</table>']
    return '\n'.join(lines)


def results_section(summary: dict, chinese: bool) -> str:
    if chinese:
        lines = [
            '## 结果', '',
            '以下汇总覆盖 5 个数据域、3 个任务、15 个模型和种子 1–5，共 1125 个完整实验。所有数值均为五个种子的**均值 ± 总体标准差**，下划线表示对应列的最优均值；RMSE 和 MAE 越低越好，MAPE 以百分数表示且越低越好。', '',
            '### Four-Level', '',
            'Four-Level 表使用两行表头：第一行是泛化层级，第二行依次为 RMSE、MAE、MAPE。RUL 的 L1 没有有效测试样本，因此留空。', '',
        ]
    else:
        lines = [
            '## Results', '',
            'The complete summary covers five domains, three tasks, 15 models, and seeds 1–5: 1,125 complete experiments. Values are **mean ± population standard deviation** over five seeds. Underlining marks the best mean in each metric column; lower is better for RMSE, MAE, and MAPE. MAPE is shown as a percentage.', '',
            '### Four-Level', '',
            'Four-Level tables use a two-row header: the first row gives the generalization level and the second gives RMSE, MAE, and MAPE. RUL L1 has no valid test samples and is left empty.', '',
        ]
    for task in TASKS:
        lines += [f"#### {TASK_NAMES[task][1 if chinese else 0]}", '',
                  four_level_table(summary['four_level'][task]), '']
    lines += ['### 标准数据域' if chinese else '### Standard Domains', '']
    for domain in DOMAINS:
        lines += [f"#### {DOMAIN_NAMES[domain][1 if chinese else 0]}", '']
        for task in TASKS:
            lines += [f"##### {TASK_NAMES[task][1 if chinese else 0]}", '',
                      standard_table(summary['standard_domains'][domain][task]), '']
    return '\n'.join(lines).rstrip() + '\n\n'


def update_readme(path: Path, section: str, chinese: bool) -> None:
    text = path.read_text()
    start_heading = '## 结果\n' if chinese else '## Results\n'
    end_heading = '## 项目结构\n' if chinese else '## Project Structure\n'
    start = text.index(start_heading)
    end = text.index(end_heading, start)
    path.write_text(text[:start] + section + text[end:])


def main() -> None:
    summary = collect()
    (RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    update_readme(ROOT / 'README.md', results_section(summary, False), False)
    update_readme(ROOT / 'README-ZH.md', results_section(summary, True), True)
    print(f"summarized {summary['n_experiments']} complete experiments")


if __name__ == '__main__':
    main()
