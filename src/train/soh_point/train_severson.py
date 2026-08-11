"""
train/soh_point/train_severson.py — Severson ElasticNet for SOH point estimation.
特征: 当前观测圈 Q(V) 曲线本身的 variance/min/mean（单圈输入，无法再取跨圈 ΔQ）。
标签: 当前观测圈的 SOH。
"""

import os
import pickle
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler


def _q_feature(Q: np.ndarray) -> list:
    q = Q[0]  # (1, n_grid) -> (n_grid,)，当前这一圈的 Q(V) 曲线
    return [float(np.var(q)), float(np.min(q)), float(np.mean(q))]


def _extract_features(dataset) -> np.ndarray:
    feats = []
    for i in range(len(dataset)):
        s = dataset[i]
        feats.append(_q_feature(s['Q'].numpy()))
    return np.array(feats, dtype=float)


def _get_targets(dataset) -> np.ndarray:
    return np.array([float(dataset[i]['soh_point'].item()) for i in range(len(dataset))])


def _metrics(preds, y_test) -> dict:
    mae  = float(np.mean(np.abs(preds - y_test)))
    mse  = float(np.mean((preds - y_test) ** 2))
    rmse = float(np.sqrt(mse))
    mask = y_test > 1e-6
    rel_err = np.abs(preds[mask] - y_test[mask]) / y_test[mask]
    mape  = float(np.mean(rel_err)) if mask.any() else float('nan')
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape}


def train(train_ds, test_ds, save_path: str = None) -> dict:
    X_train, y_train = _extract_features(train_ds), _get_targets(train_ds)
    X_test,  y_test  = _extract_features(test_ds),  _get_targets(test_ds)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = ElasticNetCV(cv=5, max_iter=10000)
    model.fit(X_train, y_train)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({'scaler': scaler, 'model': model}, f)

    return _metrics(model.predict(X_test), y_test)


def evaluate(test_ds, save_path: str) -> dict:
    with open(save_path, 'rb') as f:
        obj = pickle.load(f)
    scaler, model = obj['scaler'], obj['model']
    X_test = scaler.transform(_extract_features(test_ds))
    y_test = _get_targets(test_ds)
    return _metrics(model.predict(X_test), y_test)
