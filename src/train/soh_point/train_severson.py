import os
import pickle
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


def _get_targets(dataset: BatteryDataset) -> np.ndarray:
    return np.array([float(dataset[i]['soh_point'].item()) for i in range(len(dataset))])


def _metrics(preds, y_test) -> dict:
    mae  = float(np.mean(np.abs(preds - y_test)))
    mse  = float(np.mean((preds - y_test) ** 2))
    rmse = float(np.sqrt(mse))
    mask = y_test > 1e-6
    rel_err = np.abs(preds[mask] - y_test[mask]) / y_test[mask]
    mape  = float(np.mean(rel_err))        if mask.any() else float('nan')
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape}


def train(train_ds: BatteryDataset, test_ds: BatteryDataset,
          save_path: str = None) -> dict:
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


def evaluate(test_ds: BatteryDataset, save_path: str) -> dict:
    with open(save_path, 'rb') as f:
        obj = pickle.load(f)
    scaler, model = obj['scaler'], obj['model']
    X_test = scaler.transform(_extract_features(test_ds))
    y_test = _get_targets(test_ds)
    return _metrics(model.predict(X_test), y_test)
