"""
train/rul/train_severson.py — Severson ElasticNet baseline for RUL/BLP.
Reference: Severson et al., Nature Energy 2019.

特征：从公开的历史充电 V/I/Q 曲线提取统计量。
标签: EOL（绝对总寿命）。
"""

import os
import pickle
import numpy as np
from src.evaluate.output import PredictionWriter
from src.models.adapted import curve_features
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler


def _extract_features(dataset) -> np.ndarray:
    return np.stack([
        curve_features(dataset[index]['cycle_curve_data'].numpy(),
                       int(dataset[index]['useable_cycle']))
        for index in range(len(dataset))
    ])

def _get_labels(dataset) -> np.ndarray:
    # EOL 绝对值
    return np.array([float(dataset[i]['eol'].item()) for i in range(len(dataset))])


def _metrics(preds, y_test) -> dict:
    mae  = float(np.mean(np.abs(preds - y_test)))
    mse  = float(np.mean((preds - y_test) ** 2))
    rmse = float(np.sqrt(mse))
    mask = y_test > 1e-6
    rel_err = np.abs(preds[mask] - y_test[mask]) / y_test[mask]
    mape  = float(np.mean(rel_err))        if mask.any() else float('nan')
    acc15 = float(np.mean(rel_err <= 0.15)) if mask.any() else float('nan')
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'acc15': acc15}


def train(train_ds, test_ds, save_path: str = None) -> dict:
    X_train, y_train = _extract_features(train_ds), _get_labels(train_ds)
    X_test,  y_test  = _extract_features(test_ds),  _get_labels(test_ds)

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


def evaluate(test_ds, save_path: str, output_dir=None) -> dict:
    with open(save_path, 'rb') as f:
        obj = pickle.load(f)
    scaler, model = obj['scaler'], obj['model']
    X_test = scaler.transform(_extract_features(test_ds))
    y_test = _get_labels(test_ds)
    prediction = model.predict(X_test)
    if output_dir:
        writer = PredictionWriter(output_dir, test_ds, 'rul')
        writer.write(y_test, prediction)
        writer.close()
    return _metrics(prediction, y_test)
