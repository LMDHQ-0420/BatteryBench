"""
train_severson.py — Severson ElasticNet 训练流程

差异点:
  1. 无 GPU，无反向传播，使用 sklearn
  2. 输入为 BatteryDataset，提取 [Var(ΔQ), Min(ΔQ), Mean(ΔQ)] 三个特征
  3. 直接返回 metrics，无需 save_path / checkpoint
"""

import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

from src.data.dataset import BatteryDataset


def _extract_features(dataset: BatteryDataset) -> np.ndarray:
    feats = []
    for i in range(len(dataset)):
        dq = dataset[i]['delta_q'].numpy()
        feats.append([float(np.var(dq)), float(np.min(dq)), float(np.mean(dq))])
    return np.array(feats, dtype=float)


def _get_ruls(dataset: BatteryDataset) -> np.ndarray:
    return np.array([float(dataset[i]['rul'].item()) for i in range(len(dataset))])


def train(train_ds: BatteryDataset, test_ds: BatteryDataset) -> dict:
    """
    训练并评估 Severson baseline。
    返回 {'mae', 'rmse', 'mape', 'acc15'}。
    """
    X_train, y_train = _extract_features(train_ds), _get_ruls(train_ds)
    X_test,  y_test  = _extract_features(test_ds),  _get_ruls(test_ds)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = ElasticNetCV(cv=5, max_iter=10000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae  = float(np.mean(np.abs(preds - y_test)))
    rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
    mask = y_test > 1e-6
    rel_err = np.abs(preds[mask] - y_test[mask]) / y_test[mask]
    mape  = float(np.mean(rel_err) * 100)
    acc15 = float(np.mean(rel_err <= 0.15) * 100)

    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'acc15': acc15}
