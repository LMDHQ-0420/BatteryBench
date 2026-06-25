"""
utils.py — 数据处理工具函数
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple


# ── Q(V) 插值 ──────────────────────────────────────────────────────────────

def get_discharge_qv(
    cycle: dict,
    v_min: float,
    v_max: float,
    n_grid: int = 200,
) -> Optional[np.ndarray]:
    """
    从单圈原始时序插值得到 Q(V) 在 [v_max→v_min] 均匀网格上的值。
    返回 shape (n_grid,)，或 None（数据不足）。
    """
    v_raw = np.asarray(cycle.get('voltage_in_V', []), dtype=float)
    q_raw = np.asarray(cycle.get('discharge_capacity_in_Ah', []), dtype=float)

    if len(v_raw) < 10:
        return None

    # 过滤到 [v_min, v_max]
    mask = (v_raw >= v_min) & (v_raw <= v_max)
    v_f = v_raw[mask]
    q_f = q_raw[mask]

    if len(v_f) < 10:
        return None

    # 按电压降序排列
    sort_idx = np.argsort(-v_f)
    v_s = v_f[sort_idx]
    q_s = q_f[sort_idx]

    # 去重（相同 V 取均值）
    v_u, inv = np.unique(v_s, return_inverse=True)
    q_u = np.zeros_like(v_u)
    counts = np.zeros_like(v_u)
    for i, idx in enumerate(inv):
        q_u[idx] += q_s[i]
        counts[idx] += 1
    q_u /= counts

    if len(v_u) < 10:
        return None

    # 插值到均匀网格（v_max → v_min）
    V_grid = np.linspace(v_max, v_min, n_grid)
    try:
        Qdlin = np.interp(V_grid, v_u[::-1], q_u[::-1])
    except Exception:
        return None

    return Qdlin


def get_matr_qv(cycle: dict, n_grid: int = 200) -> Optional[np.ndarray]:
    """
    MATR 专用：从 Qdlin（1000点，V=3.5→2.0）重采样到 n_grid 点。
    """
    q = cycle.get('Qdlin')
    if q is None:
        return None
    q = np.asarray(q, dtype=float)
    if len(q) < 10:
        return None
    # 原始 1000 点网格
    V_orig = np.linspace(3.5, 2.0, len(q))
    V_new = np.linspace(3.5, 2.0, n_grid)
    return np.interp(V_new, V_orig[::-1], q[::-1])


def is_matr_cell(cell: dict) -> bool:
    """判断是否为 MATR 电池（有 Qdlin 字段且第一圈非 None）。"""
    cycle_data = cell.get('cycle_data', [])
    if not cycle_data:
        return False
    return cycle_data[0].get('Qdlin') is not None


# ── SOH / RUL ──────────────────────────────────────────────────────────────

def compute_soh_series(cell: dict) -> np.ndarray:
    """
    计算全寿命 SOH 序列，shape (n_cycles,)。
    SOH_i = Q_discharge_i / Q_nominal
    """
    q_nom = cell.get('nominal_capacity_in_Ah')
    cycle_data = cell['cycle_data']

    # 若 nominal_capacity 为 None，用第1圈最大放电容量
    if q_nom is None or q_nom <= 0:
        q0 = cycle_data[0].get('discharge_capacity_in_Ah', [])
        q_nom = float(max(q0)) if q0 else 1.0

    soh_list = []
    for cyc in cycle_data:
        q_dis = cyc.get('discharge_capacity_in_Ah', [])
        if q_dis:
            q_i = float(max(q_dis))
        else:
            q_i = 0.0
        soh_list.append(np.clip(q_i / q_nom, 0.0, 1.0))

    return np.array(soh_list, dtype=float)


def compute_rul(
    soh_series: np.ndarray,
    threshold: float = 0.80,
    obs_cycle: int = 100,
    fallback_total: bool = True,
) -> Optional[int]:
    """
    计算 RUL = t_EOL - obs_cycle。

    t_EOL 确定策略：
    1. 首先尝试找首次 SOH < threshold 的圈数（标准定义）
    2. 若找不到且 fallback_total=True，用 total_cycles 作为 t_EOL
       （适用于 MATR 等实验终止即寿命终止的数据集）
    若 RUL <= 0 返回 None。
    """
    below = np.where(soh_series < threshold)[0]
    if len(below) > 0:
        t_eol = int(below[0]) + 1  # 1-based
    elif fallback_total:
        t_eol = len(soh_series)
    else:
        return None

    rul = t_eol - obs_cycle
    if rul <= 0:
        return None
    return rul


# ── Q 矩阵构建 ─────────────────────────────────────────────────────────────

def build_q_matrix(
    cell: dict,
    n_cycles: int = 100,
    n_grid: int = 200,
) -> Optional[np.ndarray]:
    """
    构建早期观测矩阵 Q ∈ R^{n_cycles × n_grid}。
    若有效圈数 < n_cycles * 0.8 返回 None。
    """
    cycle_data = cell['cycle_data']
    v_min = cell.get('min_voltage_limit_in_V', 2.0)
    v_max = cell.get('max_voltage_limit_in_V', 3.5)
    use_matr = is_matr_cell(cell)

    Q = np.zeros((n_cycles, n_grid), dtype=float)
    valid_count = 0

    for i in range(n_cycles):
        if i >= len(cycle_data):
            break
        cyc = cycle_data[i]
        if use_matr:
            qv = get_matr_qv(cyc, n_grid=n_grid)
        else:
            qv = get_discharge_qv(cyc, v_min, v_max, n_grid=n_grid)

        if qv is not None:
            Q[i] = qv
            valid_count += 1

    if valid_count < n_cycles * 0.8:
        return None

    return Q


def compute_delta_q(
    Q: np.ndarray,
    early_cycle: int = 10,
    late_cycle: int = 100,
) -> np.ndarray:
    """
    ΔQ = Q[late_cycle-1] - Q[early_cycle-1]，shape (n_grid,)。
    """
    return Q[late_cycle - 1] - Q[early_cycle - 1]


# ── pkl 加载 ────────────────────────────────────────────────────────────────

def load_pkl(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def scan_pkl_dir(pkl_dir: str):
    """返回目录下所有 .pkl 文件路径列表，排序。"""
    return sorted(Path(pkl_dir).glob('*.pkl'))
