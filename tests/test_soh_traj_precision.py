import importlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

evaluate = importlib.import_module('src.evaluate.soh_traj.evaluate').evaluate


class NearZeroSOH(Dataset):
    _samples = [(0, 1)]
    _batteries = [{'dataset_name': 'synthetic', 'cell_id': 'near_zero'}]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {'soh_traj': torch.tensor([-3.9999, -3.99, 0.5]),
                'prediction': torch.tensor([-3.9998, -3.985, 0.25]),
                'trajectory_mask': torch.tensor([1.0, 1.0, 0.0])}


class FixedPrediction(torch.nn.Module):
    def forward(self, batch):
        return batch['prediction'], None


class TrajectoryPrecisionTest(unittest.TestCase):
    def test_saved_near_zero_soh_uses_metric_precision_before_float32_storage(self):
        dataset = NearZeroSOH()
        with tempfile.TemporaryDirectory() as temp:
            metrics = evaluate(FixedPrediction(), DataLoader(dataset, batch_size=1), 'cpu',
                               n_future=3, eol_threshold=0.8, output_dir=temp)
            with np.load(Path(temp) / 'trajectories.npz') as archive:
                truth = archive['truth'].copy()
                prediction = archive['prediction'].copy()
                mask = archive['mask'].copy()
            sample = dataset[0]
            for name, stored in [('soh_traj', truth), ('prediction', prediction)]:
                expected = (sample[name].numpy().astype(np.float64) * (1.0 - 0.8) + 0.8).astype(np.float32)
                np.testing.assert_array_equal(stored[0], expected)
                self.assertEqual(stored.dtype, np.float32)
            self.assertEqual(mask.dtype, np.bool_)
            selected_truth = truth[mask].astype(np.float64)
            selected_prediction = prediction[mask].astype(np.float64)
            recomputed_mape = np.mean(np.abs(selected_prediction - selected_truth) / np.abs(selected_truth))
            np.testing.assert_allclose(recomputed_mape, metrics['mape'], rtol=1e-6, atol=1e-8)
            # 该样本能暴露旧实现的 float32 反归一化误差。
            old_truth = (sample['soh_traj'] * (1.0 - 0.8) + 0.8).numpy()
            self.assertNotEqual(float(old_truth[0]), float(truth[0, 0]))


if __name__ == '__main__':
    unittest.main()
