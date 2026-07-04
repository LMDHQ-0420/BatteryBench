"""
train/soh_point/train_severson.py — Severson ElasticNet for SOH point estimation.
特征: ΔQ(V) = Q[最后观测圈] - Q[第10圈] 的 variance/min/mean。
标签: 最后观测圈的 SOH。
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
