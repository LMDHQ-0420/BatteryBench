"""
train/rul/train_base.py — RUL 预测标准训练流程
适用: mlp, gru, bigru, lstm, bilstm, cnn, dlinear, patchtst,
      autoformer, itransformer, transformer, micn, ic2ml

标签: EOL（= RUL + n_cycles），StandardScaler 归一化后训练，inverse_transform 后评估。
Scaler 保存至 save_path.replace('.pt', '_scaler.pkl')，evaluate.py 加载使用。
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


def _to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def _collect_eols(loader) -> np.ndarray:
    eols = []
    for batch in loader:
        key = 'eol' if 'eol' in batch else 'rul'
        eols.extend(batch[key].cpu().numpy().flatten().tolist())
    return np.array(eols, dtype=np.float32).reshape(-1, 1)


def train_one_epoch(model, loader, optimizer, device, scaler):
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()
        b = _to_device(batch, device)
        out = model(b)
        pred = out[0] if isinstance(out, (tuple, list)) else out

        eol_raw = b.get('eol', b['rul'])                            # (B, 1)
        eol_norm = torch.FloatTensor(
            scaler.transform(eol_raw.cpu().numpy().reshape(-1, 1))
        ).to(device)

        loss = criterion(pred.squeeze(-1), eol_norm.squeeze(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def validate(model, loader, device, scaler):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            b = _to_device(batch, device)
            out = model(b)
            pred = out[0] if isinstance(out, (tuple, list)) else out
            p_norm = pred.cpu().numpy().reshape(-1, 1)
            p = scaler.inverse_transform(p_norm).flatten()
            t = b.get('eol', b['rul']).cpu().numpy().flatten()
            preds.extend(p.tolist())
            trues.extend(t.tolist())

    return float(np.mean(np.abs(np.array(preds) - np.array(trues))))


def train(model, train_loader, val_loader, config, save_path, device='cuda'):
    """返回加载最优 checkpoint 的模型。Scaler 保存至 save_path 旁。"""
    t_cfg    = config.get('train', {})
    lr       = t_cfg.get('lr', 1e-3)
    wd       = t_cfg.get('weight_decay', 1e-4)
    epochs   = t_cfg.get('epochs', 300)
    patience = t_cfg.get('patience', 30)

    # fit EOL scaler on training set
    eol_train = _collect_eols(train_loader)
    scaler = StandardScaler()
    scaler.fit(eol_train)
    scaler_path = save_path.replace('.pt', '_scaler.pkl')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_mae = float('inf')
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_mae    = validate(model, val_loader, device, scaler)
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
    print(f'  Best val MAE (EOL): {best_val_mae:.2f}')
    return model
