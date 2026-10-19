"""Input and padding checks for charge-curve adaptations."""

import unittest

import torch

from src.models.registry import get_spec
from src.models.registry import ALL_MODELS
from scripts.run_pipeline import build_jobs, make_phases


class ChargeModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_point_and_history_outputs(self):
        for task in ('soh_point', 'soh_traj', 'rul'):
            for name in ('cnn_astlstm', 'hcnn'):
                with self.subTest(task=task, model=name):
                    cfg = {'data': {'task': task, 'n_cycles': 4, 'n_future': 17},
                           'model': {'dropout': 0.0}}
                    model = get_spec(name, task).build_fn(cfg)
                    steps = 1 if task == 'soh_point' else 4
                    channels = 2 if task == 'soh_point' else 3
                    curves = torch.randn(2, steps, channels, 400)
                    mask = torch.ones(2, steps)
                    output, _ = model({'cycle_curve_data': curves, 'curve_attn_mask': mask})
                    self.assertEqual(tuple(output.shape), (2, 17 if task == 'soh_traj' else 1))
                    output.square().mean().backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all()
                                        for p in model.parameters() if p.grad is not None))

    def test_registered_in_all_tasks_and_full_schedule(self):
        names = {'cnn_astlstm', 'hcnn'}
        self.assertTrue(all(names <= ALL_MODELS[task]
                            for task in ('soh_point', 'soh_traj', 'rul')))
        jobs = build_jobs({task: names for task in ('soh_point', 'soh_traj', 'rul')})
        phases = make_phases(jobs, [0, 1, 2, 3])
        self.assertEqual(sum(len(queue) for _, queue, _ in phases), 150)
        self.assertEqual(len(phases[-1][1]), 0)

    def test_padding_and_hcnn_capacity_channel(self):
        cfg = {'data': {'task': 'rul', 'n_cycles': 4}, 'model': {'dropout': 0.0}}
        observed = torch.randn(2, 2, 3, 400)
        padded = torch.cat((observed, torch.randn(2, 2, 3, 400)), dim=1)
        for name in ('cnn_astlstm', 'hcnn'):
            with self.subTest(model=name):
                model = get_spec(name, 'rul').build_fn(cfg).eval()
                with torch.no_grad():
                    short, _ = model({'cycle_curve_data': observed,
                                      'curve_attn_mask': torch.ones(2, 2)})
                    long, _ = model({'cycle_curve_data': padded,
                                     'curve_attn_mask': torch.tensor([[1, 1, 0, 0]] * 2)})
                torch.testing.assert_close(short, long)
        model = get_spec('hcnn', 'rul').build_fn(cfg).eval()
        changed = observed.clone()
        changed[:, :, 2] += 1000
        with torch.no_grad():
            base, _ = model({'cycle_curve_data': observed,
                             'curve_attn_mask': torch.ones(2, 2)})
            other, _ = model({'cycle_curve_data': changed,
                              'curve_attn_mask': torch.ones(2, 2)})
        torch.testing.assert_close(base, other)


if __name__ == '__main__':
    unittest.main()
