"""
train/soh_point/train_batlinet.py — BatLiNet SOH point 专用训练流程

差异点:
  1. 损失函数: MSE(intra) + λ * MSE(inter-cell 差值对比)，由 model.compute_loss() 实现
  2. 验证时需用 train_loader 作为 reference pool
  3. target key: batch['soh_point']
"""

import os
import numpy as np
import torch


def _to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def train_one_epoch(model, loader, optimizer, device):
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


def _build_reference(loader, device, n_ref=64, s_cap=2048):
    """遍历一次 train_loader，累积一个不超过 n_ref 的参考池（reservoir sampling）。
    full_cycle_mode 下样本数巨大（每圈一个样本），不能把整个训练集拼进内存，
    只保留至多 n_ref 条参考样本；每条样本的 S 也截断到 s_cap，避免个别超长
    序列（如 22000 圈）导致 forward 时 dQ=(B*R,S,N) 显存爆炸。"""
    ref_Qs, ref_ys = [], []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            Q = batch['Q'].cpu()               # (b, S_i, N)
            y = batch['soh_point'].cpu()
            if Q.shape[1] > s_cap:
                Q = Q[:, :s_cap]
            for i in range(Q.shape[0]):
                seen += 1
                if len(ref_Qs) < n_ref:
                    ref_Qs.append(Q[i]); ref_ys.append(y[i])
                else:
                    j = np.random.randint(0, seen)
                    if j < n_ref:
                        ref_Qs[j] = Q[i]; ref_ys[j] = y[i]
    S_max = max(q.shape[0] for q in ref_Qs)
    padded = []
    for q in ref_Qs:
        pad = S_max - q.shape[0]
        if pad > 0:
            q = torch.cat([q, torch.zeros(pad, q.shape[1])], dim=0)
        padded.append(q)
    return torch.stack(padded, dim=0).to(device), torch.stack(ref_ys, dim=0).to(device)


def validate(model, loader, device, ref_Q=None, ref_y=None):
    model.eval()

    if ref_Q is not None:
        model.set_reference(ref_Q, ref_y)

    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            b = _to_device(batch, device)
            out = model(b)
            pred = out[0] if isinstance(out, (tuple, list)) else out
            preds.extend(pred.cpu().numpy().flatten().tolist())
            trues.extend(b['soh_point'].cpu().numpy().flatten().tolist())

    model.clear_reference()
    return float(np.mean(np.abs(np.array(preds) - np.array(trues))))


def train(model, train_loader, val_loader, config, save_path, device='cuda'):
    t_cfg = config.get('train', {})
    lr      = t_cfg.get('lr', 1e-3)
    wd      = t_cfg.get('weight_decay', 1e-4)
    epochs  = t_cfg.get('epochs', 300)
    patience = t_cfg.get('patience', 30)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_mae = float('inf')
    no_improve = 0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ref_Q, ref_y = _build_reference(train_loader, device, n_ref=model.n_ref)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_mae    = validate(model, val_loader, device, ref_Q=ref_Q, ref_y=ref_y)
        scheduler.step()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f'  Epoch {epoch:3d}/{epochs} | loss={train_loss:.4f} '
                  f'| val_mae={val_mae:.4f} | best={best_val_mae:.4f}')

        if no_improve >= patience:
            print(f'  Early stop at epoch {epoch}')
            break

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    print(f'  Best val MAE: {best_val_mae:.4f}')
    return model
