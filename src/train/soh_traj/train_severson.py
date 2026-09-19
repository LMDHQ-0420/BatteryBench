"""Severson linear baseline adapted to historical charge V/I/Q curves."""

import os
import pickle

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.evaluate.output import PredictionWriter
from src.models.adapted import curve_features


def _features(dataset):
    return np.stack([
        curve_features(dataset[index]['cycle_curve_data'].numpy(),
                       int(dataset[index]['useable_cycle']))
        for index in range(len(dataset))
    ])


def _targets(dataset):
    truth = np.stack([dataset[index]['soh_traj'].numpy() for index in range(len(dataset))])
    mask = np.stack([dataset[index]['trajectory_mask'].numpy() for index in range(len(dataset))])
    return truth, mask


def _denormalize(values, threshold):
    return values * (1.0 - threshold) + threshold


def _metrics(prediction, truth, mask, threshold):
    prediction = _denormalize(prediction, threshold)
    truth = _denormalize(truth, threshold)
    valid = mask > 0
    error = prediction[valid] - truth[valid]
    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    denominator = np.abs(truth[valid])
    safe = denominator > 1e-6
    mape = float(np.mean(np.abs(error[safe]) / denominator[safe]))
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape}


def train(train_ds, test_ds, save_path=None):
    scaler = StandardScaler().fit(_features(train_ds))
    truth_train, _ = _targets(train_ds)
    model = Ridge(alpha=1.0).fit(scaler.transform(_features(train_ds)), truth_train)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as file:
            pickle.dump({'scaler': scaler, 'model': model}, file)
    truth, mask = _targets(test_ds)
    prediction = model.predict(scaler.transform(_features(test_ds)))
    return _metrics(prediction, truth, mask, test_ds.soh_threshold)


def evaluate(test_ds, save_path, output_dir=None):
    with open(save_path, 'rb') as file:
        saved = pickle.load(file)
    truth, mask = _targets(test_ds)
    prediction = saved['model'].predict(saved['scaler'].transform(_features(test_ds)))
    if output_dir:
        writer = PredictionWriter(output_dir, test_ds, 'soh_traj', n_future=prediction.shape[1])
        writer.write(_denormalize(truth, test_ds.soh_threshold),
                     _denormalize(prediction, test_ds.soh_threshold), mask > 0)
        writer.close()
    return _metrics(prediction, truth, mask, test_ds.soh_threshold)
