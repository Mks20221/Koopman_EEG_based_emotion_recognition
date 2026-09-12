"""M1_plain k=5 head replay with its OWN validation-grid selection.
Reuses saved trial-disjoint plans and frozen source checkpoints; no source training.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from snn_core import SNN
from snn_protocol import TrainScaler, digest, atomic_json
from snn_baseline import Progress, evaluate, fit_head, freeze, prediction_file
from snn_replay_shots import SampleIndex, load_data, parent_records, read_json, sha
from snn_k5_stats import collect, summarize, write_outputs, read_prediction, align

VERSION = 'snn-plain-head-verified-1.0'
CHOICE_SOURCE = 'validation_grid_on_M1_plain'


def load_plan(path, index, subject):
    record = read_json(path)
    support = index.indices(record['support']['5'])
    query = index.indices(record['query'])
    d = index.data
    if not np.all(d.subject[np.r_[support, query]] == subject):
        raise ValueError(f'{path}: support/query subject differs')
    if len(support) != 15 or not np.array_equal(np.bincount(d.y[support], minlength=3), [5, 5, 5]):
        raise ValueError(f'{path}: expected five support segments per each of three classes')
    if set(d.y[query].tolist()) != {0, 1, 2}:
        raise ValueError(f'{path}: query lacks a class')
    support_trials = {index.trial_key(int(i)) for i in support}
    if len(support_trials) != 3 or support_trials & {index.trial_key(int(i)) for i in query}:
        raise ValueError(f'{path}: expected one support trial/class, disjoint from query trials')
    return {'support': support, 'query': query, 'saved': record}


def prepare(args, cfg, folds, data):
    index = SampleIndex(data)
    tests, vals, files = {}, {}, {}
    for fold in folds:
        seed, subject = fold['seed'], fold['test_subject']
        relative = Path(f'seed{seed}') / f's{subject:02d}'
        source, replay = args.parent / relative, args.replay / relative
        for name in ('M1_plain_source.pt', 'scaler.npz', 'complete.json'):
            files[str((source / name).resolve())] = sha(source / name)
        if 'M1_plain' not in fold['source_training']:
            raise ValueError(f'M1_plain source training absent: {relative}')
        for repeat in range(args.repeats):
            plans = []
            for code in ('M2', 'M3'):
                path = replay / f'{code}_support_r{repeat}.json'
                plan = load_plan(path, index, subject)
                files[str(path.resolve())] = sha(path)
                plans.append(plan)
                expected = {'sample_id': data.sample_id[plan['query']], 'y_true': data.y[plan['query']]}
                for method in ('none', 'head'):
                    prediction = replay / f'{code}_r{repeat}_k5_{method}.npz'
                    align(expected, read_prediction(prediction, subject))
                    files[str(prediction.resolve())] = sha(prediction)
            if not all(np.array_equal(plans[0][k], plans[1][k]) for k in ('support', 'query')):
                raise ValueError('M0/M1 saved support/query plans differ')
            tests[(seed, subject, repeat)] = plans[0]
        for person in fold['val_subjects']:
            path = replay / f'validation_s{person:02d}_samples.json'
            vals[(seed, subject, person)] = load_plan(path, index, person)
            files[str(path.resolve())] = sha(path)
    return tests, vals, files


def choose_head(model, X, data, fold, vals, cfg, device, log):
    best, chosen, table = float('inf'), None, []
    for steps in cfg.adapt_steps:
        for lr in cfg.adapt_lrs:
            scores = {}
            for person in fold['val_subjects']:
                plan = vals[(fold['seed'], fold['test_subject'], person)]
                candidate = fit_head(model, X, data.y, plan['support'], device, steps, lr)
                result, _ = evaluate(candidate, X, data.y, plan['query'], device, cfg.batch_size)
                scores[str(person)] = result['ce']
            mean = float(np.mean(list(scores.values())))
            if not np.isfinite(mean):
                raise FloatingPointError('Nonfinite validation CE')
            table.append({'steps': steps, 'lr': lr, 'validation_ce_by_subject': scores, 'validation_ce': mean})
            if mean < best:
                best, chosen = mean, {'steps': steps, 'lr': lr}
    if chosen is None:
        raise ValueError('Empty validation hyperparameter grid')
    log(f'M1_plain own validation choice={chosen}; CE={best:.6f}')
    return {'source': CHOICE_SOURCE, 'choices': chosen, 'validation_grid': table,
            'validation_subjects': fold['val_subjects']}


def run_fold(args, cfg, data, fold, tests, vals, device, output, log, fingerprint):
    seed, subject = fold['seed'], fold['test_subject']
    source = args.parent / f'seed{seed}' / f's{subject:02d}'
    with np.load(source / 'scaler.npz', allow_pickle=False) as saved:
        scaler = TrainScaler()
        scaler.mean, scaler.scale = saved['mean'], saved['scale']
    X = scaler.transform(data.X)
    model = SNN(X.shape[-1], 3, cfg.hidden, cfg.latent, True, cfg.coding_steps).to(device)
    model.load_state_dict(torch.load(source / 'M1_plain_source.pt', map_location=device, weights_only=True))
    freeze(model)
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    log.stage = f's{subject:02d}: select head hyperparameters on M1_plain validation subjects'
    selection = choose_head(model, X, data, fold, vals, cfg, device, log)
    result = {'seed': seed, 'test_subject': subject, 'run_fingerprint': fingerprint,
              'choices': {'M1_plain': selection}, 'rows': []}
    for person in fold['val_subjects']:
        atomic_json(output / f'validation_s{person:02d}_samples.json', vals[(seed, subject, person)]['saved'])
    for repeat in range(args.repeats):
        log.stage = f's{subject:02d}: M1_plain repeat={repeat} k=5'
        plan = tests[(seed, subject, repeat)]
        atomic_json(output / f'M1_plain_support_r{repeat}.json', plan['saved'])
        none, predicted_none = evaluate(model, X, data.y, plan['query'], device, cfg.batch_size)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        candidate = fit_head(model, X, data.y, plan['support'], device, **selection['choices'])
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        head, predicted_head = evaluate(candidate, X, data.y, plan['query'], device, cfg.batch_size)
        changed_head = max(float((p - original[name]).abs().max().item())
                           for name, p in candidate.state_dict().items() if name.startswith('head.'))
        for name, value in candidate.state_dict().items():
            if not name.startswith('head.') and not torch.equal(value, original[name]):
                raise AssertionError(f'Frozen backbone changed: {name}')
        for name, value in model.state_dict().items():
            if not torch.equal(value, original[name]):
                raise AssertionError('Source model mutated while fitting a candidate head')
        for method, score, predicted in (('none', none, predicted_none), ('head', head, predicted_head)):
            prediction_file(output / f'M1_plain_r{repeat}_k5_{method}.npz', data, plan['query'], predicted)
            result['rows'].append({'model': 'M1_plain', 'method': method, 'repeat': repeat,
                                   'phase': 'calibrated_query', 'segments_per_class': 5,
                                   'n_support': 15, 'n_query': len(plan['query']),
                                   'fit_seconds': seconds if method == 'head' else 0.,
                                   'head_max_abs_change': changed_head if method == 'head' else 0.,
                                   'backbone_unchanged': True, **score})
        log(f's{subject:02d} r={repeat}: none={none["accuracy"]:.4f} head={head["accuracy"]:.4f}; '
            f'head parameter change={changed_head:.6g}; prediction changes={int((predicted_none != predicted_head).sum())}')
        atomic_json(output / 'partial.json', result)
    atomic_json(output / 'complete.json', result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, default=Path('results/snn_repaired_seed_session1'))
    p.add_argument('--replay', type=Path, default=Path('results/snn_replay_fixed_v2'))
    p.add_argument('--out', type=Path, default=Path('results/snn_plain_head_verified_k5'))
    p.add_argument('--feature-cache', type=Path, default=Path('data_cache/snn_ordered_de'))
    p.add_argument('--features-npz', type=Path)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--test-subjects', type=int, nargs='+')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    if min(args.repeats, args.threads) < 1:
        p.error('Counts must be positive')
    for name in ('parent', 'replay', 'out'):
        setattr(args, name, getattr(args, name).resolve())
    if any(args.out == source or source in args.out.parents for source in (args.parent, args.replay)):
        p.error('Use a separate output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out / 'RUNNING.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    log, admitted = Progress(args.out / 'run.log'), False
    try:
        log(f'{VERSION}: checking existing inputs; torch={torch.__version__}')
        torch.set_num_threads(args.threads)
        parent, reference = read_json(args.parent / 'manifest.json'), read_json(args.replay / 'manifest.json')
        cfg = SimpleNamespace(**parent['protocol']['config'])
        if reference['protocol']['parent_fingerprint'] != parent['fingerprint']:
            raise ValueError('Replay and source runs have different parent fingerprints')
        if args.repeats > reference['protocol']['repeats']:
            raise ValueError('Requested repeats exceed saved replay plans')
        for name, expected in parent['protocol']['code'].items():
            if sha(HERE / name) != expected:
                raise ValueError(f'Parent dependency differs: {name}; restore the matching source file')
        folds = [f for f in parent_records(args, cfg) if f['seed'] == args.seed]
        if not folds:
            raise ValueError('No completed parent folds for requested seed')
        scope = sorted(f['test_subject'] for f in folds)
        selected = sorted(set(args.test_subjects or scope))
        if not set(selected) <= set(scope):
            raise ValueError('Requested subjects are absent from the completed parent folds')
        data = load_data(args, parent, log)
        tests, vals, files = prepare(args, cfg, folds, data)
        log('Saved support/query plans checked; initializing device')
        device = torch.device('cpu' if args.device == 'cpu' else
                              'cuda' if torch.cuda.is_available() else 'cpu')
        if args.device == 'cuda' and device.type != 'cuda':
            raise RuntimeError('CUDA requested but unavailable')
        torch.zeros(1, device=device).sum().item()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        protocol = {'version': VERSION, 'parent_fingerprint': parent['fingerprint'],
                    'reference_fingerprint': reference['fingerprint'], 'input_files': files,
                    'data_hash': data.fingerprint(), 'source': CHOICE_SOURCE,
                    'seed': args.seed, 'repeats': args.repeats, 'scope': scope,
                    'grid': {'steps': cfg.adapt_steps, 'lr': cfg.adapt_lrs},
                    'device': str(device), 'torch': str(torch.__version__),
                    'code': {name: sha(HERE / name) for name in
                             ('snn_plain_head_verified.py', 'snn_k5_stats.py', 'snn_replay_shots.py')}}
        fingerprint = digest(protocol)
        manifest = args.out / 'manifest.json'
        if manifest.exists():
            if not args.resume or read_json(manifest)['fingerprint'] != fingerprint:
                raise ValueError('Output exists or resume protocol changed; use --resume with the same inputs or a new --out')
        else:
            atomic_json(manifest, {'fingerprint': fingerprint, 'protocol': protocol})
        admitted = True
        completed = {}
        for path in args.out.glob(f'seed{args.seed}/s*/complete.json'):
            f = read_json(path)
            if f['run_fingerprint'] != fingerprint:
                raise ValueError('Completed fold fingerprint differs')
            completed[f['test_subject']] = f
        for fold in folds:
            subject = fold['test_subject']
            if subject not in selected:
                continue
            if subject in completed:
                log(f'resume: skip complete s{subject:02d}')
            else:
                output = args.out / f'seed{args.seed}' / f's{subject:02d}'
                output.mkdir(parents=True, exist_ok=True)
                completed[subject] = run_fold(args, cfg, data, fold, tests, vals, device, output, log, fingerprint)
            people = sorted(completed)
            rows, provenance = collect(args.replay, args.out, people, args.repeats, args.seed)
            summary = summarize(rows, people, args.repeats, args.seed)
            write_outputs(args.out, rows, summary, provenance)
            atomic_json(args.out / 'rows.json', [completed[s] for s in people])
            atomic_json(args.out / 'status.json', {'state': 'running', 'completed_subjects': people})
        atomic_json(args.out / 'status.json', {'state': 'completed', 'completed_subjects': sorted(completed),
                    'complete_loso': set(completed) == set(cfg.subjects), 'fingerprint': fingerprint})
        log(f'COMPLETE: {len(completed)} folds; own-validation head replay and prediction statistics in {args.out}')
    except BaseException as exc:
        if admitted or not (args.out / 'manifest.json').exists():
            atomic_json(args.out / 'status.json', {'state': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        'message': str(exc)})
        log(f'{type(exc).__name__}: {exc}')
        raise
    finally:
        log.close()
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
