"""
train/soh_traj/train_base.py — SOH 退化轨迹预测标准训练流程
适用: mlp, gru, bigru, lstm, bilstm, cnn, dlinear, patchtst,
      autoformer, itransformer, transformer, micn, batterymformer
目标: batch['soh_traj'] shape (B, n_future)，MSE 损失。
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def train_one_epoch(model, loader, optimizer, device, n_future=100):
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()
        b = _to_device(batch, device)
        out = model(b)
        pred = out[0] if isinstance(out, (tuple, list)) else out  # (B, n_future)
        target = b['soh_traj'][:, :n_future]                      # (B, T) T<=n_future
        t = min(pred.shape[1], target.shape[1])
        loss = criterion(pred[:, :t], target[:, :t])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def validate(model, loader, device, n_future=100):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            b = _to_device(batch, device)
            out = model(b)
            pred = out[0] if isinstance(out, (tuple, list)) else out
            target = b['soh_traj'][:, :n_future]
            t = min(pred.shape[1], target.shape[1])
            preds.append(pred[:, :t].cpu().numpy())
            trues.append(target[:, :t].cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return float(np.mean(np.abs(preds - trues)))


def train(model, train_loader, val_loader, config, save_path, device='cuda'):
    """返回加载最优 checkpoint 的模型。"""
    t_cfg    = config.get('train', {})
    d_cfg    = config.get('data', {})
    lr       = t_cfg.get('lr', 1e-3)
    wd       = t_cfg.get('weight_decay', 1e-4)
    epochs   = t_cfg.get('epochs', 300)
    patience = t_cfg.get('patience', 30)
    n_future = d_cfg.get('n_future', 100)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_mae = float('inf')
    no_improve = 0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, n_future)
        val_mae    = validate(model, val_loader, device, n_future)
        scheduler.step()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f'  Epoch {epoch:3d}/{epochs} | loss={train_loss:.6f} '
                  f'| val_mae={val_mae:.6f} | best={best_val_mae:.6f}')

        if no_improve >= patience:
            print(f'  Early stop at epoch {epoch}')
            break

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    print(f'  Best val MAE: {best_val_mae:.6f}')
    return model
