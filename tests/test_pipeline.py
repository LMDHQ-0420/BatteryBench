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
        for gpu_slots in ({0: 2, 1: 3, 2: 3, 3: 3}, {1: 1, 2: 1, 3: 1}):
            with self.subTest(gpu_slots=gpu_slots):
                active = {gpu: 0 for gpu in gpu_slots}
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

                with patch.object(pipeline, 'run_job', worker):
                    failures = pipeline.run_phase(
                        list(range(35)), SimpleNamespace(), 'test', gpu_slots, set(), 5,
                    )
                self.assertEqual(failures, [])
                self.assertEqual(sorted(completed), list(range(35)))
                self.assertTrue(all(maximum[gpu] <= limit
                                    for gpu, limit in gpu_slots.items()))
                self.assertTrue(all(value == 0 for value in active.values()))

    def test_checkpoint_or_old_json_does_not_count_as_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(results_dir=Path(temp))
            job = dict(domain='calb', task='soh_point', model='gru', seed=1)
            directory = pipeline.seed_dir(job, args)
            directory.mkdir(parents=True)
            for path in pipeline.checkpoints(job, args):
                path.write_bytes(b'checkpoint')
            metrics = dict(mae=1, mse=1, rmse=1, mape=1)
            result = dict(domain='calb', task='soh_point', model='gru', **metrics)
            (directory / 'results.json').write_text(json.dumps(result))
            self.assertFalse(pipeline.result_complete(job, args, set(), 5))
            result.update(n_samples=2, n_batteries=1)
            (directory / 'results.json').write_text(json.dumps(result))
            output = directory / 'test'
            output.mkdir(parents=True)
            with (output / 'predictions.csv').open('w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['sample_index', 'dataset', 'cell_id', 'observation_cycle',
                                 'true_soh', 'predicted_soh'])
                writer.writerows([[0, 'CALB', 'A', 1, 1, 1], [1, 'CALB', 'A', 2, 1, 1]])
            (directory.parent / 'summary.json').write_text(json.dumps({'seeds': {'1': metrics}}))
            self.assertTrue(pipeline.result_complete(job, args, set(), 5))
            (directory / 'test/predictions.csv').write_text('truncated')
            self.assertFalse(pipeline.result_complete(job, args, set(), 5))

    def test_resume_evaluation_failure_does_not_retrain(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(results_dir=Path(temp) / 'results',
                                   state_dir=Path(temp) / 'state', log_dir=Path(temp) / 'log')
            args.log_dir.mkdir()
            job = dict(domain='calb', task='soh_point', model='gru', seed=1)
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
                self.assertFalse(pipeline.run_job(job, args, 1, 'standard_domains', set(), 5))
                self.assertFalse(pipeline.run_job(job, args, 1, 'standard_domains', set(), 5))
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
            job = dict(domain='calb', task='rul', model='gru', seed=1)
            with patch.object(pipeline, 'ROOT', root):
                pipeline.command(job, args, 2, 'evaluate', root / 'test.log')
            self.assertEqual(len(pipeline.CHILDREN), 0)

    def test_partial_training_failure_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(results_dir=Path(temp) / 'results',
                                   state_dir=Path(temp) / 'state', log_dir=Path(temp) / 'log')
            args.log_dir.mkdir()
            job = dict(domain='calb', task='soh_point', model='gru', seed=1)
            calls = []
            def operation(job, args, gpu, op, log):
                calls.append(op)
                path = pipeline.checkpoints(job, args)[0]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'half_trained')
                raise RuntimeError('interrupted training')
            with patch.object(pipeline, 'command', operation):
                self.assertFalse(pipeline.run_job(job, args, 1, 'standard_domains', set(), 5))
                self.assertFalse(pipeline.run_job(job, args, 1, 'standard_domains', set(), 5))
            self.assertEqual(calls, ['train', 'train'])



    def test_phase_order_counts_and_gpu_limits(self):
        models = {task: {'gru', 'batlinet'} for task in pipeline.TASKS}
        jobs = pipeline.build_jobs(models)
        phases = pipeline.make_phases(jobs, [0, 1, 2, 3])

        self.assertEqual(
            [name for name, _, _ in phases],
            ['four_level_soh_traj_seed1', 'four_level_soh_point_seed1',
             'four_level_rul_seed1', 'four_level_seeds2_to5',
             'standard_domains', 'batlinet_all'],
        )
        self.assertEqual([len(queue) for _, queue, _ in phases], [1, 1, 1, 12, 60, 75])
        self.assertTrue(all(slots == {0: 2, 1: 3, 2: 3, 3: 3}
                            for _, _, slots in phases[:-1]))
        self.assertEqual(phases[-1][2], {1: 1, 2: 1, 3: 1})
        self.assertTrue(all(job['model'] != 'batlinet'
                            for _, queue, _ in phases[:-1] for job in queue))
        self.assertTrue(all(job['model'] == 'batlinet' for job in phases[-1][1]))

        scheduled = [pipeline.job_id(job) for _, queue, _ in phases for job in queue]
        self.assertEqual(len(scheduled), len(jobs))
        self.assertEqual(len(set(scheduled)), len(jobs))


if __name__ == '__main__':
    unittest.main()
