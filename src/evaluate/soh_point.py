"""
evaluate/soh_point.py — SOH 单点估计评估
输出: mae, mse, rmse, mape, acc15
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
) -> Dict[str, float]:
    """
    评估 SOH 单点估计模型。
    返回 {'mae', 'mse', 'rmse', 'mape', 'acc15'}。
    acc15: 相对误差 ≤ 15% 的样本占比 (%)。
    SOH 值域 [0, 1]，mape 基于真实值计算。
    """
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            true_soh = batch['soh_point'].numpy().flatten()
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
            out = model(b)
            pred = (out[0] if isinstance(out, (tuple, list)) else out).cpu().numpy().flatten()

            preds.extend(pred.tolist())
            trues.extend(true_soh.tolist())

    preds = np.array(preds, dtype=np.float64)
    trues = np.array(trues, dtype=np.float64)

    mae  = float(np.mean(np.abs(preds - trues)))
    mse  = float(np.mean((preds - trues) ** 2))
    rmse = float(np.sqrt(mse))

    mask = trues > 1e-6
    rel_err = np.abs(preds[mask] - trues[mask]) / trues[mask]
    mape  = float(np.mean(rel_err) * 100) if mask.any() else float('nan')
    acc15 = float(np.mean(rel_err <= 0.15) * 100) if mask.any() else float('nan')

    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'acc15': acc15}
