"""
evaluate/soh_traj/evaluate.py — SOH 退化轨迹评估
输出: mae, mse, rmse, mape

对齐 BatteryMFormer：
  - 只评估未来段（trajectory_mask=1，即观测窗口之后到 EOL），不含已观测圈
  - 轨迹在 dataset 中归一化为 (SOH-thr)/(1-thr)，评估前反归一化回真实 SOH 再算指标
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    n_future: int = 5000,
    eol_threshold: float = 0.80,
) -> Dict[str, float]:
    model.eval()
    all_preds, all_trues = [], []

    with torch.no_grad():
        for batch in loader:
            true_traj = batch['soh_traj'][:, :n_future]
            tmask     = batch['trajectory_mask'][:, :n_future]

            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
            out = model(b)
            pred = (out[0] if isinstance(out, (tuple, list)) else out).cpu()

            m = tmask > 0
            if m.any():
                all_preds.append(pred[m].numpy())
                all_trues.append(true_traj[m].numpy())

    if not all_preds:
        return {'mae': float('nan'), 'mse': float('nan'),
                'rmse': float('nan'), 'mape': float('nan')}

    preds = np.concatenate(all_preds).astype(np.float64)
    trues = np.concatenate(all_trues).astype(np.float64)

    # 反归一化回真实 SOH： SOH = norm*(1-thr) + thr
    scale = 1.0 - eol_threshold
    preds_soh = preds * scale + eol_threshold
    trues_soh = trues * scale + eol_threshold

    mae  = float(np.mean(np.abs(preds_soh - trues_soh)))
    mse  = float(np.mean((preds_soh - trues_soh) ** 2))
    rmse = float(np.sqrt(mse))

    mask = np.abs(trues_soh) > 1e-6
    rel_err = np.abs(preds_soh[mask] - trues_soh[mask]) / np.abs(trues_soh[mask])
    mape = float(np.mean(rel_err)) if mask.any() else float('nan')

    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape}
