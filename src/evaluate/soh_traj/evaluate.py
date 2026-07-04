"""
evaluate/soh_traj/evaluate.py — SOH 退化轨迹评估
输出: mae, mse, rmse, mape
使用 soh_traj_len mask，只在有数据的位置计算误差。
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
) -> Dict[str, float]:
    model.eval()
    all_preds, all_trues = [], []

    with torch.no_grad():
        for batch in loader:
            lens = batch['soh_traj_len']
            true_traj = batch['soh_traj'][:, :n_future]

            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
            out = model(b)
            pred = (out[0] if isinstance(out, (tuple, list)) else out).cpu()

            B = pred.shape[0]
            for i in range(B):
                l = int(lens[i].item())
                l = min(l, n_future, pred.shape[1], true_traj.shape[1])
                if l > 0:
                    all_preds.append(pred[i, :l].numpy())
                    all_trues.append(true_traj[i, :l].numpy())

    if not all_preds:
        return {'mae': float('nan'), 'mse': float('nan'),
                'rmse': float('nan'), 'mape': float('nan')}

    preds = np.concatenate(all_preds).astype(np.float64)
    trues = np.concatenate(all_trues).astype(np.float64)

    mae  = float(np.mean(np.abs(preds - trues)))
    mse  = float(np.mean((preds - trues) ** 2))
    rmse = float(np.sqrt(mse))

    mask = trues > 1e-6
    rel_err = np.abs(preds[mask] - trues[mask]) / trues[mask]
    mape = float(np.mean(rel_err)) if mask.any() else float('nan')

    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape}
