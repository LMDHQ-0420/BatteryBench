"""
evaluate/soh_point/evaluate.py — SOH 单点估计评估
输出: mae, mse, rmse, mape  (0-1 范围)

每条样本是单圈输入，batch['soh_point'] shape (B, 1)。
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
    mape = float(np.mean(rel_err)) if mask.any() else float('nan')

    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape}
