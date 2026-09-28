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
    'na_ion': ('Na-ion', 'Na-ion'),
    'zn_ion': ('Zn-ion', 'Zn-ion'),
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


def formatted(stats: dict, metric: str, best: float, nowrap: bool = False, html: bool = False) -> str:
    mean, std = value(stats, metric)
    if metric == 'mape':
        text = f'{100 * mean:.2f} ± {100 * std:.2f}%'
    else:
        text = f'{mean:.4f} ± {std:.4f}'
    if nowrap:
        text = text.replace(' ± ', '&nbsp;±&nbsp;')
    if not math.isclose(mean, best, rel_tol=1e-12, abs_tol=1e-15):
        return text
    return f'<strong>{text}</strong>' if html else f'**{text}**'


def standard_table(records: dict) -> str:
    best = {metric: min(value(records[model], metric)[0] for model, _ in MODELS)
            for metric in METRICS}
    rows = ['| Model | RMSE | MAE | MAPE |', '|:---:|:---:|:---:|:---:|']
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
        '<table align="center">', '<thead>', '<tr>',
        '<th align="center" nowrap rowspan="2">Model</th>',
        *[f'<th align="center" nowrap colspan="3">{level}</th>' for level in LEVELS],
        '</tr>', '<tr>',
        *['<th align="center" nowrap>RMSE</th><th align="center" nowrap>MAE</th><th align="center" nowrap>MAPE</th>'
          for _ in LEVELS],
        '</tr>', '</thead>', '<tbody>',
    ]
    for model, name in MODELS:
        lines += ['<tr>', f'<td align="center" nowrap>{name}</td>']
        for level in LEVELS:
            if level not in records[model]:
                lines += ['<td align="center" nowrap>-</td>'] * 3
            else:
                lines += [
                    f'<td align="center" nowrap>{formatted(records[model][level], metric, best[level][metric], nowrap=True, html=True)}</td>'
                    for metric in METRICS
                ]
        lines += ['</tr>']
    lines += ['</tbody>', '</table>']
    return '\n'.join(lines)


def results_section(summary: dict, chinese: bool) -> str:
    if chinese:
        lines = [
            '## 结果', '',
            '以下汇总覆盖 5 个数据域、3 个任务、15 个模型和种子 1–5，共 1125 个完整实验。所有数值均为五个种子的**均值 ± 总体标准差**；粗体表示对应列的最优均值。RMSE、MAE 和 MAPE 均越低越好，MAPE 以百分数表示。', '',
            '### 四个数据集', '',
        ]
    else:
        lines = [
            '## Results', '',
            'The complete summary covers five domains, three tasks, 15 models, and seeds 1–5: 1,125 complete experiments. Values are **mean ± population standard deviation** over five seeds. Bold marks the best mean in each metric column; lower is better for RMSE, MAE, and MAPE. MAPE is shown as a percentage.', '',
            '### Four Datasets', '',
        ]
    for domain in DOMAINS:
        lines += [f"#### {DOMAIN_NAMES[domain][1 if chinese else 0]}", '']
        for task in TASKS:
            lines += [f"##### {TASK_NAMES[task][1 if chinese else 0]}", '',
                      standard_table(summary['standard_domains'][domain][task]), '']
    if chinese:
        lines += [
            '### Four-Level', '',
            'Four-Level 表使用两行表头：第一行是泛化层级，第二行依次为 RMSE、MAE、MAPE。表格中的数值使用不换行格式。', '',
            '**为什么 RUL 没有 L1：** L1 测试集由 HUST batch 8–9 的 16 块电池组成。这些电池在现有记录中均未下降到 80% SOH 阈值，因此没有可作为真值的 EOL，也无法构造 RUL 测试样本。对应位置以 `-` 表示；这不是模型漏跑。', '',
        ]
    else:
        lines += [
            '### Four-Level', '',
            'Four-Level tables use a two-row header: the first row gives the generalization level and the second gives RMSE, MAE, and MAPE. Values within each table use nonbreaking spacing.', '',
            '**Why RUL has no L1 result:** the L1 test set contains 16 cells from HUST batches 8–9. None reaches the 80% SOH threshold within its recorded lifetime, so no ground-truth EOL or valid RUL test sample can be constructed. These entries are shown as `-`; the experiments were not skipped.', '',
        ]
    for task in TASKS:
        lines += [f"#### {TASK_NAMES[task][1 if chinese else 0]}", '',
                  four_level_table(summary['four_level'][task]), '']
    return '\n'.join(lines).rstrip() + '\n\n'


def update_readme(path: Path, section: str, chinese: bool) -> None:
    text = path.read_text()
    start_heading = '## 结果\n' if chinese else '## Results\n'
    end_heading = '## 安装\n' if chinese else '## Installation\n'
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
