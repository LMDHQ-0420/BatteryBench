"""
train/soh_traj/train_severson.py — Severson ElasticNet baseline for SOH trajectory.

Severson 是 3 特征线性模型，无法直接产出 5000 维轨迹。此处对齐其能力：
目标 = 未来段（trajectory_mask=1）真实 SOH 的均值（反归一化后），
预测同样为该标量，指标在标量层面计算。metrics 与其它 soh_traj 模型不完全同口径，
但保持 Severson 作为经典 ML 下界基线的定位。
"""

import os
import pickle
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler


def _delta_q_feature(Q: np.ndarray, useable: int) -> list:
    late = Q[useable - 1]
    early = Q[min(9, useable - 1)]
    dq = late - early
    return [float(np.var(dq)), float(np.min(dq)), float(np.mean(dq))]


def _extract_features(dataset) -> np.ndarray:
    feats = []
    for i in range(len(dataset)):
        s = dataset[i]
        feats.append(_delta_q_feature(s['Q'].numpy(), int(s['useable_cycle'])))
    return np.array(feats, dtype=float)


def _get_targets(dataset, eol_threshold: float = 0.80) -> np.ndarray:
    """未来段真实 SOH 均值（反归一化回 [thr,1] 区间）。"""
    ys = []
    scale = 1.0 - eol_threshold
    for i in range(len(dataset)):
        s = dataset[i]
        traj = s['soh_traj'].numpy()
        tmask = s['trajectory_mask'].numpy() > 0
        if tmask.sum() > 0:
            norm_mean = float(traj[tmask].mean())
        else:
            norm_mean = 0.0
        ys.append(norm_mean * scale + eol_threshold)
    return np.array(ys, dtype=float)


def _metrics(preds, y_test) -> dict:
    mae  = float(np.mean(np.abs(preds - y_test)))
    mse  = float(np.mean((preds - y_test) ** 2))
    rmse = float(np.sqrt(mse))
    mask = np.abs(y_test) > 1e-6
    rel_err = np.abs(preds[mask] - y_test[mask]) / np.abs(y_test[mask])
    mape  = float(np.mean(rel_err)) if mask.any() else float('nan')
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape}


def train(train_ds, test_ds, save_path: str = None, eol_threshold: float = 0.80) -> dict:
    X_train, y_train = _extract_features(train_ds), _get_targets(train_ds, eol_threshold)
    X_test,  y_test  = _extract_features(test_ds),  _get_targets(test_ds, eol_threshold)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = ElasticNetCV(cv=5, max_iter=10000)
    model.fit(X_train, y_train)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({'scaler': scaler, 'model': model, 'eol_threshold': eol_threshold}, f)

    return _metrics(model.predict(X_test), y_test)


def evaluate(test_ds, save_path: str) -> dict:
    with open(save_path, 'rb') as f:
        obj = pickle.load(f)
    scaler, model = obj['scaler'], obj['model']
    thr = obj.get('eol_threshold', 0.80)
    X_test = scaler.transform(_extract_features(test_ds))
    y_test = _get_targets(test_ds, thr)
    return _metrics(model.predict(X_test), y_test)
