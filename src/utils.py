"""Shared battery data utilities."""

from pathlib import Path
from typing import Optional
import pickle

import numpy as np


def load_pkl(path: str) -> dict:
    with open(path, 'rb') as file:
        return pickle.load(file)


def scan_pkl_dir(pkl_dir: str):
    return sorted(Path(pkl_dir).glob('*.pkl'))


def get_soc_interval(cell: dict) -> float:
    soc = cell.get('SOC_interval')
    if not soc or len(soc) < 2:
        return 1.0
    interval = float(soc[1]) - float(soc[0])
    return interval if abs(interval) > 1e-6 else 1.0


def compute_soh_series(cell: dict) -> np.ndarray:
    """Compute labels from discharge capacity. Capacity never enters model inputs."""
    full = cell.get('full_soh_series')
    if full is not None and len(full) > 0:
        return np.clip(np.asarray(full, dtype=float), 0.0, 1.0)

    cycles = cell['cycle_data']
    nominal = cell.get('nominal_capacity_in_Ah')
    if nominal is None or nominal <= 0:
        first = cycles[0].get('discharge_capacity_in_Ah', [])
        nominal = float(max(first)) if len(first) else 1.0

    interval = get_soc_interval(cell)
    soh = []
    for cycle in cycles:
        capacity = cycle.get('discharge_capacity_in_Ah', [])
        value = float(max(capacity)) if len(capacity) else 0.0
        soh.append(np.clip(value / nominal / interval, 0.0, 1.0))
    return np.asarray(soh, dtype=float)


def compute_eol(
    soh_series: np.ndarray,
    threshold: float = 0.80,
    fallback_total: bool = False,
) -> Optional[int]:
    below = np.flatnonzero(soh_series < threshold)
    if len(below):
        return int(below[0]) + 1
    return len(soh_series) if fallback_total else None


def _resample(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.zeros(length, dtype=np.float32)
    values = values[finite]
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, length)
    return np.interp(target, source, values).astype(np.float32)


def get_charge_curve(
    cycle: dict,
    nominal_capacity: float,
    curve_length: int = 400,
    include_capacity: bool = True,
) -> Optional[np.ndarray]:
    """Return a complete positive-current charge curve as V/I[/Q], resampled together."""
    voltage = np.asarray(cycle.get('voltage_in_V', []), dtype=float)
    current = np.asarray(cycle.get('current_in_A', []), dtype=float)
    charge = np.asarray(cycle.get('charge_capacity_in_Ah', []), dtype=float)
    size = min(len(voltage), len(current), len(charge))
    if size < 5:
        return None

    nominal = float(nominal_capacity) if nominal_capacity and nominal_capacity > 0 else 1.0
    voltage, current, charge = voltage[:size], current[:size], charge[:size]
    mask = (current > 0.01 * nominal) & np.isfinite(voltage) & np.isfinite(current) & np.isfinite(charge)
    if mask.sum() < 5:
        return None

    voltage = _resample(voltage[mask], curve_length)
    current = _resample(current[mask], curve_length)
    charge = _resample(charge[mask], curve_length)
    max_voltage = float(np.max(voltage))
    if max_voltage > 1e-6:
        voltage /= max_voltage

    channels = [voltage, current / nominal]
    if include_capacity:
        channels.append(charge / nominal)
    return np.stack(channels).astype(np.float32)


def build_charge_curve_tensor(
    cell: dict,
    n_cycles: int,
    curve_length: int = 400,
    include_capacity: bool = True,
) -> tuple[np.ndarray, int]:
    """Build consecutive charge-only curves; stop at the first unusable cycle."""
    channels = 3 if include_capacity else 2
    curves = np.zeros((n_cycles, channels, curve_length), dtype=np.float32)
    nominal = cell.get('nominal_capacity_in_Ah') or 1.0
    valid_cycles = 0
    for index, cycle in enumerate(cell.get('cycle_data', [])[:n_cycles]):
        curve = get_charge_curve(cycle, nominal, curve_length, include_capacity)
        if curve is None:
            break
        curves[index] = curve
        valid_cycles += 1
    return curves, valid_cycles
