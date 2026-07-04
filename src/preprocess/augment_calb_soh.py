"""
augment_calb_soh.py — 为 CALB pkl 注入全寿命 SOH 序列

问题：CALB 的 pkl 只含前 ~100 圈详细曲线（早期寿命），真实退化（1000-3000 圈，
SOH 掉到 0.65-0.89）只存在于 overall_CALB_cycling_data.xlsx 汇总表里，
导致 compute_soh_series 从截断的 cycle_data 算不出真实 EOL。

对齐 BatteryLife（Extract_life_labels.py）：曲线用早期100圈，SOH/EOL 用汇总表全序列。
本脚本从汇总表读每个电池的完整放电容量序列，计算 SOH = Qd / Qd[0]，
写入 pkl 的 'full_soh_series' 字段。curves（cycle_data）不动。

用法:
    python -m src.preprocess.augment_calb_soh \
        --summary data/raw/CALB/overall_CALB_cycling_data.xlsx \
        --pkl_dir data/preprocessed/CALB
"""

import os
import glob
import pickle
import argparse
import numpy as np
import pandas as pd


# sheet -> (列前缀, cell_id 前缀, 名称映射)
_SHEET_CFG = [
    ('0℃循环',   'A1',  'CALB_0_',  lambda n: n.replace('A', 'B')),
    ('25℃ 循环', 'T25', 'CALB_25_', lambda n: n),
    ('35℃ 循环', 'B',   'CALB_35_', lambda n: n),
    ('45℃循环',  'B',   'CALB_45_', lambda n: n),
]


def load_summary_capacity(summary_file: str) -> dict:
    """返回 {cell_id: discharge_capacity_series(np.ndarray)}。列偏移对齐 BatteryLife。"""
    out = {}
    for sheet, prefix, cell_pfx, namemap in _SHEET_CFG:
        df = pd.read_excel(summary_file, sheet_name=sheet)
        cols = df.columns.tolist()
        for c in cols:
            if not str(c).startswith(prefix):
                continue
            si = cols.index(c)
            name = cell_pfx + namemap(str(c))
            # BatteryLife 偏移: 放电容量 = start_col + 4
            dc = pd.to_numeric(df.iloc[:, si + 4], errors='coerce').dropna().values
            if len(dc) >= 2:
                out[name] = dc.astype(np.float64)
    return out


def augment(summary_file: str, pkl_dir: str):
    cap = load_summary_capacity(summary_file)
    print(f"汇总表电池数: {len(cap)}")

    pkls = sorted(glob.glob(os.path.join(pkl_dir, '*.pkl')))
    injected, skipped = 0, 0
    for p in pkls:
        name = os.path.basename(p)[:-4]
        if name not in cap:
            print(f"  跳过（汇总表无匹配）: {name}")
            skipped += 1
            continue
        dc = cap[name]
        nom = float(dc[0])  # 首圈放电容量作为标称（对齐 BatteryLife CALB）
        soh = np.clip(dc / nom, 0.0, 1.0).astype(np.float32)

        with open(p, 'rb') as f:
            cell = pickle.load(f)
        cell['full_soh_series'] = soh
        with open(p, 'wb') as f:
            pickle.dump(cell, f)
        injected += 1

    print(f"注入完成: {injected} 个 pkl, 跳过 {skipped} 个")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', default='data/raw/CALB/overall_CALB_cycling_data.xlsx')
    ap.add_argument('--pkl_dir', default='data/preprocessed/CALB')
    args = ap.parse_args()
    augment(args.summary, args.pkl_dir)
