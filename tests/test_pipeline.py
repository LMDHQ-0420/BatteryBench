import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('pipeline', Path(__file__).resolve().parents[1] / 'scripts/run_pipeline.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class PipelineTests(unittest.TestCase):
    def test_gpu_slot_limits_and_wait_for_all_jobs(self):
        for slots in (1, 4, 6):
            with self.subTest(slots=slots):
                active = {1: 0, 2: 0, 3: 0}
                maximum = dict(active)
                completed = []
                lock = threading.Lock()
                def worker(job, args, gpu, phase, sets, future):
                    with lock:
                        active[gpu] += 1
                        maximum[gpu] = max(maximum[gpu], active[gpu])
                    time.sleep(0.01)
                    with lock:
                        active[gpu] -= 1
                        completed.append(job)
                    return True
                args = SimpleNamespace(gpus=[1, 2, 3])
                with patch.object(pipeline, 'run_job', worker):
                    failures = pipeline.run_phase(list(range(35)), args, 'test', slots, set(), 5)
                self.assertEqual(failures, [])
                self.assertEqual(sorted(completed), list(range(35)))
                self.assertTrue(all(v <= slots for v in maximum.values()))
                self.assertEqual(active, {1: 0, 2: 0, 3: 0})

    def test_checkpoint_or_old_json_does_not_count_as_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(results_dir=Path(temp))
            job = dict(domain='calb', task='soh_point', model='gru', seed=42)
            directory = pipeline.seed_dir(job, args)
            directory.mkdir(parents=True)
            for path in pipeline.checkpoints(job, args):
                path.write_bytes(b'checkpoint')
            metrics = dict(mae=1, mse=1, rmse=1, mape=1)
            result = dict(domain='calb', task='soh_point', model='gru', splits=[metrics] * 3,
                          mean=metrics, std=metrics)
            (directory / 'results.json').write_text(json.dumps(result))
            self.assertFalse(pipeline.result_complete(job, args, set(), 5))
            result['splits'] = [dict(split_idx=i, n_samples=2, n_batteries=1, **metrics)
                                for i in range(1, 4)]
            (directory / 'results.json').write_text(json.dumps(result))
            for i in range(1, 4):
                output = directory / 'test' / f'split{i}'
                output.mkdir(parents=True)
                with (output / 'predictions.csv').open('w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['sample_index', 'dataset', 'cell_id', 'observation_cycle',
                                     'true_soh', 'predicted_soh'])
                    writer.writerows([[0, 'CALB', 'A', 1, 1, 1], [1, 'CALB', 'A', 2, 1, 1]])
            (directory.parent / 'summary.json').write_text(json.dumps({'seeds': {'42': metrics}}))
            self.assertTrue(pipeline.result_complete(job, args, set(), 5))
            (directory / 'test/split2/predictions.csv').write_text('truncated')
            self.assertFalse(pipeline.result_complete(job, args, set(), 5))

    def test_resume_evaluation_failure_does_not_retrain(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(results_dir=Path(temp) / 'results',
                                   state_dir=Path(temp) / 'state', log_dir=Path(temp) / 'log')
            args.log_dir.mkdir()
            job = dict(domain='calb', task='soh_point', model='gru', seed=42)
            calls = []
            def operation(job, args, gpu, op, log):
                calls.append(op)
                if op == 'train':
                    for path in pipeline.checkpoints(job, args):
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b'completed_training')
                else:
                    raise RuntimeError('temporary evaluation failure')
            with patch.object(pipeline, 'command', operation):
                self.assertFalse(pipeline.run_job(job, args, 1, 'train_small', set(), 5))
                self.assertFalse(pipeline.run_job(job, args, 1, 'train_small', set(), 5))
            self.assertEqual(calls, ['train', 'evaluate', 'evaluate'])



    def test_child_sees_only_assigned_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'scripts').mkdir()
            (root / 'scripts/evaluate.py').write_text(
                "import os, sys\nassert os.environ['CUDA_VISIBLE_DEVICES'] == '2'\n"
                "assert sys.argv[sys.argv.index('--gpu') + 1] == '0'\n")
            args = SimpleNamespace(config=root / 'config.yaml', results_dir=root / 'results',
                                   job_threads=3)
            job = dict(domain='calb', task='rul', model='gru', seed=42)
            with patch.object(pipeline, 'ROOT', root):
                pipeline.command(job, args, 2, 'evaluate', root / 'test.log')
            self.assertEqual(len(pipeline.CHILDREN), 0)

    def test_partial_training_failure_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(results_dir=Path(temp) / 'results',
                                   state_dir=Path(temp) / 'state', log_dir=Path(temp) / 'log')
            args.log_dir.mkdir()
            job = dict(domain='calb', task='soh_point', model='gru', seed=42)
            calls = []
            def operation(job, args, gpu, op, log):
                calls.append(op)
                path = pipeline.checkpoints(job, args)[0]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'half_trained')
                raise RuntimeError('interrupted training')
            with patch.object(pipeline, 'command', operation):
                self.assertFalse(pipeline.run_job(job, args, 1, 'train_small', set(), 5))
                self.assertFalse(pipeline.run_job(job, args, 1, 'train_small', set(), 5))
            self.assertEqual(calls, ['train', 'train'])



    def test_all_batlinet_jobs_run_last_and_training_is_exclusive(self):
        large = [dict(model='batlinet'), dict(model='mlp')]
        small = [dict(model='lstm'), dict(model='batlinet')]
        phases = pipeline.make_phases(large, small)
        flattened = [(name, job, slots) for name, jobs, slots in phases for job in jobs]
        seen_batlinet = False
        for name, job, slots in flattened:
            if job['model'] == 'batlinet':
                seen_batlinet = True
                if name.startswith('train'):
                    self.assertEqual(slots, 1)
            else:
                self.assertFalse(seen_batlinet)
        self.assertEqual(len(flattened), len(large + small))
        self.assertEqual(len({id(job) for _, job, _ in flattened}), len(flattened))


if __name__ == '__main__':
    unittest.main()
