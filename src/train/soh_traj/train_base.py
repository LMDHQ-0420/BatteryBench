"""
train/soh_traj/train_base.py — SOH 退化轨迹预测标准训练流程
适用: mlp, gru, bigru, lstm, bilstm, cnn, dlinear, patchtst,
      autoformer, itransformer, transformer, micn
目标: batch['soh_traj'] shape (B, n_future)，用 soh_traj_len mask 的 MSE 损失。
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def _masked_mse(pred, target, tmask):
    """MSE only over未来段（trajectory_mask=1 的位置）。"""
    m = tmask > 0
    if m.sum() == 0:
        return (pred.sum() * 0.0)
    return F.mse_loss(pred[m], target[m])


def _masked_mae(pred, target, tmask):
    m = tmask > 0
    if m.sum() == 0:
        return 0.0
    return float(torch.mean(torch.abs(pred[m] - target[m])).item())


def train_one_epoch(model, loader, optimizer, device, n_future=5000):
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()
        b = _to_device(batch, device)
        out = model(b)
        pred   = out[0] if isinstance(out, (tuple, list)) else out  # (B, n_future)
        target = b['soh_traj'][:, :n_future]                        # (B, n_future)
        tmask  = b['trajectory_mask'][:, :n_future]                 # (B, n_future) 只未来段
        loss = _masked_mse(pred, target, tmask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def validate(model, loader, device, n_future=5000):
    model.eval()
    preds, trues, masks_all = [], [], []

    with torch.no_grad():
        for batch in loader:
            b = _to_device(batch, device)
            out = model(b)
            pred = out[0] if isinstance(out, (tuple, list)) else out
            preds.append(pred.cpu())
            trues.append(b['soh_traj'][:, :n_future].cpu())
            masks_all.append(b['trajectory_mask'][:, :n_future].cpu())

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    tmask = torch.cat(masks_all, dim=0)
    return _masked_mae(preds, trues, tmask)


def train(model, train_loader, val_loader, config, save_path, device='cuda'):
    """返回加载最优 checkpoint 的模型。"""
    t_cfg    = config.get('train', {})
    d_cfg    = config.get('data', {})
    lr       = t_cfg.get('lr', 1e-3)
    wd       = t_cfg.get('weight_decay', 1e-4)
    epochs   = t_cfg.get('epochs', 300)
    patience = t_cfg.get('patience', 30)
    n_future = d_cfg.get('n_future', 5000)

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
