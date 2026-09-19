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
from src.evaluate.output import PredictionWriter
from typing import Dict


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    n_future: int = 5000,
    eol_threshold: float = 0.80,
    output_dir=None,
) -> Dict[str, float]:
    writer = PredictionWriter(output_dir, loader.dataset, 'soh_traj', n_future) if output_dir else None
    scale = 1.0 - eol_threshold
    model.eval()
    absolute_sum = squared_sum = relative_sum = 0.0
    count = relative_count = 0

    with torch.no_grad():
        for batch in loader:
            true_traj = batch['soh_traj'][:, :n_future]
            tmask     = batch['trajectory_mask'][:, :n_future]

            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
            out = model(b)
            pred = (out[0] if isinstance(out, (tuple, list)) else out).cpu()

            if writer:
                # Match metric precision before storing float32 trajectories.
                writer.write(true_traj.numpy().astype(np.float64) * scale + eol_threshold,
                             pred.numpy().astype(np.float64) * scale + eol_threshold,
                             tmask.numpy() > 0)

            m = tmask > 0
            prediction = pred[m].numpy().astype(np.float64) * scale + eol_threshold
            truth = true_traj[m].numpy().astype(np.float64) * scale + eol_threshold
            error = np.abs(prediction - truth)
            absolute_sum += error.sum()
            squared_sum += np.square(error).sum()
            count += len(error)
            positive = np.abs(truth) > 1e-6
            relative_sum += (error[positive] / np.abs(truth[positive])).sum()
            relative_count += int(positive.sum())

    if writer:
        writer.close()
    mae = float(absolute_sum / count) if count else float('nan')
    mse = float(squared_sum / count) if count else float('nan')
    mape = float(relative_sum / relative_count) if relative_count else float('nan')
    return {'mae': mae, 'mse': mse, 'rmse': float(np.sqrt(mse)), 'mape': mape}
