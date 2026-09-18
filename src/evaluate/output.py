"""逐样本结果保存；轨迹先写磁盘数组，避免积累整套预测到内存。"""

import csv
from pathlib import Path

import numpy as np


class PredictionWriter:
    def __init__(self, output_dir, dataset, task, n_future=None):
        self.directory = Path(output_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.task = task
        self.index = 0
        self.file = (self.directory / 'predictions.csv').open('w', newline='')
        self.csv = csv.writer(self.file)
        columns = ['sample_index', 'dataset', 'cell_id', 'observation_cycle']
        if task == 'rul':
            columns += ['true_eol', 'predicted_eol']
        elif task == 'soh_point':
            columns += ['true_soh', 'predicted_soh']
        elif n_future is None:
            columns += ['true_future_mean_soh', 'predicted_future_mean_soh']
        self.csv.writerow(columns)
        self.arrays = {}
        if n_future is not None:
            for name, dtype in [('prediction', 'float32'), ('truth', 'float32'), ('mask', 'bool')]:
                self.arrays[name] = np.lib.format.open_memmap(
                    self.directory / f'.{name}.npy', mode='w+', dtype=dtype,
                    shape=(len(dataset), n_future))

    def write(self, truth, prediction, mask=None):
        start = self.index
        for offset in range(len(prediction)):
            index = start + offset
            bidx, cycle = self.dataset._samples[index]
            battery = self.dataset._batteries[bidx]
            row = [index, battery['dataset_name'], battery['cell_id'], cycle]
            if not self.arrays:
                row += [float(truth[offset]), float(prediction[offset])]
            self.csv.writerow(row)
        self.index += len(prediction)
        if self.arrays:
            self.arrays['truth'][start:self.index] = truth
            self.arrays['prediction'][start:self.index] = prediction
            self.arrays['mask'][start:self.index] = mask

    def close(self):
        self.file.close()
        if self.arrays:
            for array in self.arrays.values():
                array.flush()
            np.savez_compressed(self.directory / 'trajectories.npz', **self.arrays)
            for array in self.arrays.values():
                array._mmap.close()
            self.arrays.clear()
            for name in ['prediction', 'truth', 'mask']:
                (self.directory / f'.{name}.npy').unlink()
