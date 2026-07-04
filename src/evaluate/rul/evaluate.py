"""
evaluate/rul/evaluate.py — RUL 预测评估
输出: mae, mse, rmse, mape, acc15
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    scaler_path: Optional[str] = None,
    use_log_rul: bool = False,
) -> Dict[str, float]:
    scaler = None
    if scaler_path:
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        except Exception:
            pass

    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            true_rul = batch['rul'].numpy().flatten()
            true_eol = batch.get('eol', batch['rul']).numpy().flatten()
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
            out = model(b)
            pred = (out[0] if isinstance(out, (tuple, list)) else out).cpu().numpy().flatten()

            if scaler is not None:
                pred_eol = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
                n_cycles = true_eol - true_rul
                pred = pred_eol - n_cycles

            preds.extend(pred.tolist())
            trues.extend(true_rul.tolist())

    preds = np.array(preds, dtype=np.float64)
    trues = np.array(trues, dtype=np.float64)

    mae  = float(np.mean(np.abs(preds - trues)))
    mse  = float(np.mean((preds - trues) ** 2))
    rmse = float(np.sqrt(mse))

    mask = trues > 1e-6
    rel_err = np.abs(preds[mask] - trues[mask]) / trues[mask]
    mape  = float(np.mean(rel_err))         if mask.any() else float('nan')
    acc15 = float(np.mean(rel_err <= 0.15)) if mask.any() else float('nan')

    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'acc15': acc15}
