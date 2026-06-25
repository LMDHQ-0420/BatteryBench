"""
evaluate/soh_traj.py — SOH 退化轨迹评估
输出: mae, mse, rmse, mape  (无 acc15)
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
    n_future: int = 100,
) -> Dict[str, float]:
    """
    评估 SOH 退化轨迹预测模型。
    预测输出 shape: (B, n_future)；真实值取 batch['soh_traj'][:, :n_future]。
    返回 {'mae', 'mse', 'rmse', 'mape'}。不含 acc15（轨迹任务不适用）。
    """
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            true_traj = batch['soh_traj'][:, :n_future].numpy()  # (B, n_future)
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
            out = model(b)
            pred = (out[0] if isinstance(out, (tuple, list)) else out).cpu().numpy()  # (B, n_future)

            preds.append(pred)
            trues.append(true_traj)

    preds = np.concatenate(preds, axis=0).flatten().astype(np.float64)
    trues = np.concatenate(trues, axis=0).flatten().astype(np.float64)

    mae  = float(np.mean(np.abs(preds - trues)))
    mse  = float(np.mean((preds - trues) ** 2))
    rmse = float(np.sqrt(mse))

    mask = trues > 1e-6
    rel_err = np.abs(preds[mask] - trues[mask]) / trues[mask]
    mape = float(np.mean(rel_err) * 100) if mask.any() else float('nan')

    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape}
