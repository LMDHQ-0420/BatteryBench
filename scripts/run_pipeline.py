"""先评估迁移的可信权重，再训练并评估其他实验；支持中断后重跑。"""

import argparse
import ast
import concurrent.futures
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STOP = threading.Event()
PRINT_LOCK = threading.Lock()
CHILD_LOCK = threading.Lock()
CHILDREN = set()


def say(message):
    with PRINT_LOCK:
        print(time.strftime('[%Y-%m-%d %H:%M:%S]'), message, flush=True)


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False))
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def job_id(job):
    return f"{job['domain']}_{job['task']}_{job['model']}_seed{job['seed']}"


def seed_dir(job, args):
    return args.results_dir / job['domain'] / job['task'] / job['model'] / f"seed{job['seed']}"


def checkpoints(job, args):
    labels = ['best'] if job['domain'] == 'four_level' else ['split1', 'split2', 'split3']
    extension = '.pkl' if job['model'] == 'severson' else '.pt'
    names = [label + extension for label in labels]
    if job['task'] == 'rul' and job['model'] not in ('batlinet', 'severson'):
        names += [label + '_scaler.pkl' for label in labels]
    return [seed_dir(job, args) / name for name in names]


def fingerprint(job, args):
    result = {}
    for path in checkpoints(job, args):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'缺少权重或 scaler：{path}')
        result[path.name] = [path.stat().st_size, path.stat().st_mtime_ns]
    return result


def npy_shape(stream):
    if stream.read(6) != b'\x93NUMPY':
        raise ValueError('不是 NPY 数组')
    version = tuple(stream.read(2))
    width = 2 if version == (1, 0) else 4
    size = struct.unpack('<H' if width == 2 else '<I', stream.read(width))[0]
    return ast.literal_eval(stream.read(size).decode('latin1'))['shape']


def prediction_complete(directory, job, n_samples, n_future):
    path = directory / 'predictions.csv'
    if not path.is_file():
        return False
    with path.open(newline='') as file:
        reader = csv.reader(file)
        header = next(reader, [])
        required = ['sample_index', 'dataset', 'cell_id', 'observation_cycle']
        if job['task'] == 'rul':
            required += ['true_eol', 'predicted_eol']
        elif job['task'] == 'soh_point':
            required += ['true_soh', 'predicted_soh']
        if header != required:
            return False
        count = 0
        for row in reader:
            if len(row) != len(header) or int(row[0]) != count:
                return False
            count += 1
        if count != n_samples:
            return False
    if job['task'] == 'soh_traj':
        with zipfile.ZipFile(directory / 'trajectories.npz') as archive:
            for name in ('truth', 'prediction', 'mask'):
                with archive.open(name + '.npy') as stream:
                    if npy_shape(stream) != (n_samples, n_future):
                        return False
    return True


def result_complete(job, args, expected_sets, n_future):
    """不以权重或 results.json 单独存在判定完成；检查新格式的全套产物。"""
    import math
    directory = seed_dir(job, args)
    try:
        fingerprint(job, args)
        result = read_json(directory / 'results.json')
        if (result['domain'], result['task'], result['model']) != (
                job['domain'], job['task'], job['model']):
            return False
        keys = ['mae', 'mse', 'rmse', 'mape'] + (['acc15'] if job['task'] == 'rul' else [])
        def metrics(record):
            return all(isinstance(record.get(key), (int, float)) and math.isfinite(record[key])
                       for key in keys)
        if job['domain'] == 'four_level':
            records = result['test_sets']
            pairs = [(record['level'], record['dataset']) for record in records]
            if len(pairs) != len(expected_sets) or set(pairs) != expected_sets:
                return False
            summaries = result['level_summary']
            active_levels = set()
            for record in records:
                n = record['n_samples']
                if record.get('status') == 'empty':
                    if n != 0 or record['n_batteries'] != 0:
                        return False
                    continue
                active_levels.add(record['level'])
                if n <= 0 or record['n_batteries'] <= 0 or not metrics(record):
                    return False
                output = directory / 'test' / f"{record['level']}_{record['dataset']}"
                if not prediction_complete(output, job, n, n_future):
                    return False
            if set(summaries) != active_levels or not all(metrics(x) for x in summaries.values()):
                return False
        else:
            records = result['splits']
            if len(records) != 3 or [x['split_idx'] for x in records] != [1, 2, 3]:
                return False
            if not metrics(result['mean']) or not metrics(result['std']):
                return False
            for record in records:
                n = record['n_samples']
                if n <= 0 or record['n_batteries'] <= 0 or not metrics(record):
                    return False
                if not prediction_complete(directory / 'test' / f"split{record['split_idx']}",
                                           job, n, n_future):
                    return False
        summary = read_json(directory.parent / 'summary.json')
        return str(job['seed']) in summary['seeds']
    except (OSError, ValueError, KeyError, TypeError, IndexError, EOFError, zipfile.BadZipFile):
        return False


def stop_children(signum=None, frame=None):
    STOP.set()
    with CHILD_LOCK:
        for process in list(CHILDREN):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def command(job, args, gpu, operation, logfile):
    if STOP.is_set():
        raise InterruptedError('调度已停止')
    argv = [sys.executable, '-u', str(ROOT / 'scripts' / f'{operation}.py'),
            '--domain', job['domain'], '--task', job['task'], '--model', job['model'],
            '--seed', str(job['seed']), '--gpu', '0', '--config', str(args.config),
            '--save_dir', str(args.results_dir / job['domain'] / job['task'])]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       OMP_NUM_THREADS=str(args.job_threads), MKL_NUM_THREADS=str(args.job_threads),
                       OPENBLAS_NUM_THREADS=str(args.job_threads), MPLBACKEND='Agg')
    with logfile.open('a') as output:
        output.write(f'\nGPU={gpu} operation={operation} argv={argv!r}\n')
        output.flush()
        with CHILD_LOCK:
            if STOP.is_set():
                raise InterruptedError('调度已停止')
            process = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            CHILDREN.add(process)
        try:
            while process.poll() is None:
                if STOP.wait(1):
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
            returncode = process.wait()
        finally:
            with CHILD_LOCK:
                CHILDREN.discard(process)
    if STOP.is_set():
        raise InterruptedError('子进程已停止')
    if returncode:
        raise RuntimeError(f'{operation} 退出码 {returncode}；日志：{logfile}')


def run_job(job, args, gpu, phase, expected_sets, n_future):
    name = job_id(job)
    statefile = args.state_dir / 'jobs' / (name + '.json')
    try:
        state = read_json(statefile) if statefile.exists() else {}
    except (OSError, ValueError):
        state = {}
    logfile = args.log_dir / (name + '.log')
    try:
        # 只有完整训练成功且权重未变时，才能在评估失败后免于重复训练。
        trained = False
        if state.get('training_complete'):
            try:
                trained = state.get('checkpoint_fingerprint') == fingerprint(job, args)
            except ValueError:
                pass
        if not trained:
            state = {'training_complete': False}
            say(f'START train GPU={gpu} {name}')
            write_json(statefile, {**state, 'job': job, 'status': 'training'})
            # 移除失效指标；半途权重始终不会被当成已完成。
            (seed_dir(job, args) / 'results.json').unlink(missing_ok=True)
            command(job, args, gpu, 'train', logfile)
            state = {'training_complete': True, 'checkpoint_fingerprint': fingerprint(job, args)}
        write_json(statefile, {**state, 'job': job, 'status': 'evaluating'})
        say(f'START evaluate GPU={gpu} {name}')
        command(job, args, gpu, 'evaluate', logfile)
        if not result_complete(job, args, expected_sets, n_future):
            raise RuntimeError(f'评估退出成功但新格式结果不完整；日志：{logfile}')
        write_json(statefile, {**state, 'job': job, 'status': 'complete'})
        say(f'DONE {name}')
        return True
    except Exception as error:
        write_json(statefile, {**state, 'job': job, 'status': 'failed', 'error': str(error)})
        say(f'FAILED {name}: {error}')
        return False


def run_phase(jobs, args, phase, slots_per_gpu, expected_sets, n_future):
    say(f'PHASE {phase}: {len(jobs)} jobs, GPUs={args.gpus}, 每卡 {slots_per_gpu} 槽')
    slots = [gpu for gpu in args.gpus for _ in range(slots_per_gpu)]
    pending = iter(jobs)
    running = {}
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as executor:
        def submit(gpu):
            if STOP.is_set():
                return
            job = next(pending, None)
            if job is not None:
                future = executor.submit(run_job, job, args, gpu, phase, expected_sets, n_future)
                running[future] = (gpu, job)
        for gpu in slots:
            submit(gpu)
        while running:
            done, _ = concurrent.futures.wait(running, timeout=1,
                        return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                gpu, job = running.pop(future)
                if not future.result():
                    failures.append(job_id(job))
                submit(gpu)
    return failures



def make_phases(large, small):
    # BatLiNet 训练显存占用大，排在全部普通模型之后并独占单卡。
    large_other = [job for job in large if job['model'] != 'batlinet']
    small_other = [job for job in small if job['model'] != 'batlinet']
    batlinet = [job for job in large + small if job['model'] == 'batlinet']
    return [('train_small', small_other, 4),
            ('train_large', large_other, 1),
            ('train_batlinet', batlinet, 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--large-models', nargs='+', default=['batlinet'],
                        help='独占 GPU 的模型名，或 task/model，例如 soh_traj/mlp')
    parser.add_argument('--job-threads', type=int, default=3)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/default.yaml')
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--state-dir', type=Path, default=ROOT / 'pipeline_state')
    parser.add_argument('--log-dir', type=Path,
                        default=ROOT / 'log' / time.strftime('%Y-%m-%d_%H-%M-%S_pipeline'))
    parser.add_argument('--dry-run', action='store_true', help='只输出调度计划，不训练或评估')
    args = parser.parse_args()
    if 0 in args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 1:
        parser.error('GPU 0 必须保留空闲；GPU 编号不能重复')
    if args.job_threads < 1:
        parser.error('--job-threads 必须为正整数')
    for name in ('config', 'results_dir', 'state_dir', 'log_dir'):
        setattr(args, name, getattr(args, name).resolve())
    import yaml
    config = yaml.safe_load(args.config.read_text())
    n_future = config['data'].get('n_future', 5000)
    four = yaml.safe_load((ROOT / 'configs/domains/four_level.yaml').read_text())
    expected_sets = {(x['level'], Path(x['dir']).name) for x in four['data']['test_sets']}
    # 从注册表生成全部实验。输入定义改变后，旧权重不再进入可信恢复路径。
    registry = ast.parse((ROOT / 'src/models/registry.py').read_text())
    node = next(x for x in registry.body if isinstance(x, ast.AnnAssign)
                and isinstance(x.target, ast.Name) and x.target.id == '_REGISTRY')
    models = {ast.literal_eval(k): {ast.literal_eval(m) for m in v.keys}
              for k, v in zip(node.value.keys, node.value.values)}
    jobs = [
        {'domain': domain, 'task': task, 'model': model, 'seed': seed}
        for domain in ('li_ion', 'calb', 'na_ion', 'zn_ion', 'four_level')
        for task in sorted(models)
        for model in sorted(models[task])
        for seed in (1, 7, 42, 123, 2024)
    ]
    valid_large = {m for names in models.values() for m in names} | {
        f'{t}/{m}' for t, names in models.items() for m in names}
    if not set(args.large_models) <= valid_large:
        parser.error('存在未注册的 --large-models 名称')
    large, small, finished = [], [], []
    for job in jobs:
        if result_complete(job, args, expected_sets, n_future):
            finished.append(job)
        elif (job['model'] == 'batlinet' or job['model'] in args.large_models
              or f"{job['task']}/{job['model']}" in args.large_models):
            large.append(job)
        else:
            small.append(job)
    say(f'总计 {len(jobs)}；新格式已完成 {len(finished)}；'
        f'待训大模型 {len(large)}；待训其余模型 {len(small)}')
    say(f'GPU 0 保留；测评 {len(args.gpus)*6} 槽；'
        f'大模型训练 {len(args.gpus)} 槽；其余训练 {len(args.gpus)*4} 槽；'
        f'大模型={args.large_models}')
    phases = make_phases(large, small)
    for phase, queue, slots in phases:
        if queue:
            say(f'计划 {phase}: {len(queue)} 个实验，每卡 {slots} 槽')
    if args.dry_run:
        return 0
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / 'pipeline.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('已有同一状态目录的调度脚本在运行')
        import torch
        if not torch.cuda.is_available() or max(args.gpus) >= torch.cuda.device_count():
            parser.error('所需 GPU 不可用；请在 zw@BatteryBench 环境启动')
        args.log_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.log_dir / 'plan.json', {
                   'train_large': large, 'train_small': small, 'already_complete': finished,
                   'phases': [{'name': name, 'jobs': queue, 'slots_per_gpu': slots}
                              for name, queue, slots in phases]})
        signal.signal(signal.SIGINT, stop_children)
        signal.signal(signal.SIGTERM, stop_children)
        failures = []
        # 阶段间等待全部进程结束，避免训练占用测评槽或大模型与其他任务争显存。
        for phase, queue, slots in phases:
            if STOP.is_set():
                break
            if not queue:
                continue
            failures += run_phase(queue, args, phase, slots, expected_sets, n_future)
        write_json(args.log_dir / 'run_summary.json', {'failed_jobs': failures,
                   'interrupted': STOP.is_set(), 'state_dir': str(args.state_dir)})
        say(f'运行结束；失败 {len(failures)}；日志 {args.log_dir}')
        return 130 if STOP.is_set() else (1 if failures else 0)


if __name__ == '__main__':
    os.chdir(ROOT)
    raise SystemExit(main())
