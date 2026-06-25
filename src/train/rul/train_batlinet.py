"""
train/rul/train_batlinet.py — BatLiNet RUL 专用训练流程

差异点:
  1. 损失函数: MSE(intra) + λ * MSE(inter-cell 差值对比)，由 model.compute_loss() 实现
  2. 验证时需用 train_loader 作为 reference pool（model.set_reference / clear_reference）
  3. batch_size 建议 ≤ 8（inter-cell pairs 二次方增长）
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader


def _to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def train_one_epoch(model, loader, optimizer, device, use_log_rul=False):
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()
        loss = model.compute_loss(batch, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def validate(model, loader, device, use_log_rul=False, ref_loader=None):
    model.eval()

    if ref_loader is not None:
        ref_Qs, ref_ys = [], []
        with torch.no_grad():
            for batch in ref_loader:
                ref_Qs.append(batch['Q'])
                ref_ys.append(batch['rul'])
        model.set_reference(
            torch.cat(ref_Qs, dim=0).to(device),
            torch.cat(ref_ys, dim=0).to(device),
        )

    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            b = _to_device(batch, device)
            out = model(b)
            pred = out[0] if isinstance(out, (tuple, list)) else out
            p = pred.cpu().numpy().flatten()
            t = b['rul'].cpu().numpy().flatten()
            if use_log_rul:
                p, t = np.expm1(np.clip(p, -10, 20)), np.expm1(t)
            preds.extend(p.tolist())
            trues.extend(t.tolist())

    model.clear_reference()
    return float(np.mean(np.abs(np.array(preds) - np.array(trues))))


def train(model, train_loader, val_loader, config, save_path, device='cuda'):
    """返回加载最优 checkpoint 的模型。"""
    t_cfg = config.get('train', {})
    lr          = t_cfg.get('lr', 1e-3)
    wd          = t_cfg.get('weight_decay', 1e-4)
    epochs      = t_cfg.get('epochs', 300)
    patience    = t_cfg.get('patience', 30)
    use_log_rul = t_cfg.get('use_log_rul', False)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_mae = float('inf')
    no_improve = 0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, use_log_rul)
        val_mae    = validate(model, val_loader, device, use_log_rul, ref_loader=train_loader)
        scheduler.step()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f'  Epoch {epoch:3d}/{epochs} | loss={train_loss:.4f} '
                  f'| val_mae={val_mae:.2f} | best={best_val_mae:.2f}')

        if no_improve >= patience:
            print(f'  Early stop at epoch {epoch}')
            break

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    print(f'  Best val MAE: {best_val_mae:.2f}')
    return model
