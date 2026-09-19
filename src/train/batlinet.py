"""Shared training loop for the charge-curve BatLiNet adaptation."""

import os

import numpy as np
import torch


def _to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def _reference(loader, device, task, limit):
    target_key = {'rul': 'eol', 'soh_point': 'soh_point', 'soh_traj': 'soh_traj'}[task]
    curves, targets, masks = [], [], []
    for batch in loader:
        for index in range(len(batch['cycle_curve_data'])):
            curves.append(batch['cycle_curve_data'][index])
            targets.append(batch[target_key][index])
            masks.append(batch['curve_attn_mask'][index])
            if len(curves) == limit:
                return (torch.stack(curves).to(device), torch.stack(targets).to(device),
                        torch.stack(masks).to(device))
    return (torch.stack(curves).to(device), torch.stack(targets).to(device),
            torch.stack(masks).to(device))


def _validation_mae(model, loader, device, task, reference):
    model.eval()
    model.set_reference(*reference)
    errors = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            prediction = model(batch)[0]
            if task == 'rul':
                target = batch['eol']
                errors.append(torch.abs(prediction - target).flatten())
            elif task == 'soh_point':
                errors.append(torch.abs(prediction - batch['soh_point']).flatten())
            else:
                mask = batch['trajectory_mask'][:, :prediction.shape[1]] > 0
                errors.append(torch.abs(prediction - batch['soh_traj'][:, :prediction.shape[1]])[mask])
    model.clear_reference()
    values = torch.cat([value for value in errors if value.numel()])
    return float(values.mean().cpu())


def train_model(model, train_loader, val_loader, config, save_path, device, task):
    train_cfg = config.get('train', {})
    epochs = train_cfg.get('epochs', 300)
    patience = train_cfg.get('patience', 30)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.get('lr', 1e-3),
        weight_decay=train_cfg.get('weight_decay', 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    reference = _reference(train_loader, device, task, model.n_ref)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best, stale = float('inf'), 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model.compute_loss(batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        score = _validation_mae(model, val_loader, device, task, reference)
        scheduler.step()
        if score < best:
            best, stale = score, 0
            torch.save(model.state_dict(), save_path)
        else:
            stale += 1
        if epoch == 1 or epoch % 20 == 0:
            print(f'  Epoch {epoch:3d}/{epochs} | loss={np.mean(losses):.4f} | val_mae={score:.4f} | best={best:.4f}')
        if stale >= patience:
            print(f'  Early stop at epoch {epoch}')
            break

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    print(f'  Best val MAE: {best:.4f}')
    return model
