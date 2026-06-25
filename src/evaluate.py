"""
evaluate.py — 评估与指标计算
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    use_log_rul: bool = False,
) -> Dict[str, float]:
    """
    评估模型，返回 {'mae', 'rmse', 'mape'}。
    """
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            rul = batch['rul'].numpy().flatten()
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
            if 'X' in b:
                pred, _ = model(b)
            else:
                pred, _ = model(b)
            pred = pred.cpu().numpy().flatten()

            if use_log_rul:
                pred = np.expm1(pred)
                rul = np.expm1(rul)

            preds.extend(pred.tolist())
            trues.extend(rul.tolist())

    preds = np.array(preds)
    trues = np.array(trues)

    mae = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    mask = trues > 1e-6
    rel_err = np.abs(preds[mask] - trues[mask]) / trues[mask]
    mape = float(np.mean(rel_err) * 100)
    acc15 = float(np.mean(rel_err <= 0.15) * 100)

    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'acc15': acc15}


def collect_attention_weights(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, List[str]]:
    """
    收集所有样本的 attention 权重。
    返回：alphas (n_samples, N)，cell_ids List[str]
    """
    model.eval()
    all_alphas = []
    all_cell_ids = []

    with torch.no_grad():
        for batch in loader:
            Q = batch['Q'].to(device)
            delta_q = batch['delta_q'].to(device)
            cell_ids = batch['cell_id']

            _, alpha = model({'Q': Q, 'delta_q': delta_q})
            all_alphas.append(alpha.cpu().numpy())
            all_cell_ids.extend(cell_ids)

    alphas = np.concatenate(all_alphas, axis=0)  # (n_samples, N)
    return alphas, all_cell_ids
