"""Frozen SNN calibration replay, compatible with the snn-repair-1.0 parent format.
The budget counts segments within ONE support trial per class. sample_id is opaque.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from snn_core import SNN, IntrinsicAdapter
from snn_protocol import (Dataset, TrainScaler, atomic_json, digest, load_npz,
                          local_seed, synthetic_data, trial_plan)
from snn_baseline import Progress, evaluate, fit_head, fit_q, freeze, prediction_file

VERSION = 'snn-replay-2.0'
METHODS = ('none', 'head', 'lowdim')
MODEL_BASE = {'M2': 'M0', 'M3': 'M1'}
REPRO_ATOL = 1e-6


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SampleIndex:
    """O(1) exact ID lookup. Never parse ID suffixes or assume lexical sorting."""
    def __init__(self, data):
        self.data = data
        self.by_id = {str(s): i for i, s in enumerate(data.sample_id)}
        if len(self.by_id) != len(data.X):
            raise ValueError('Duplicate sample_id values')

    def indices(self, ids):
        if not isinstance(ids, list) or not ids:
            raise ValueError('Expected a nonempty list of sample IDs')
        try:
            out = np.array([self.by_id[str(s)] for s in ids], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f'Saved sample ID absent from parent features: {exc.args[0]!r}') from exc
        if len(np.unique(out)) != len(out):
            raise ValueError('Repeated sample IDs inside a support/query list')
        return out

    def trial_key(self, i):
        d = self.data
        return int(d.subject[i]), int(d.session[i]), str(d.trial[i])


def make_plan(index, subject, anchors, query, seed):
    """Retain each original anchor, then add a deterministic permutation of its trial.
    Nonoverlap is inherited from the parent preprocessor's overlap=0, not ID gaps.
    """
    d = index.data
    classes = list(range(len(np.unique(d.y))))
    anchors = np.asarray(anchors, dtype=np.int64)
    query = np.asarray(query, dtype=np.int64)
    if len(anchors) != len(classes) or sorted(d.y[anchors].tolist()) != classes:
        raise ValueError('support["1"] must contain ONE anchor for EACH class, not a class-indexed mapping')
    if not len(query) or len(np.unique(query)) != len(query):
        raise ValueError('Query list must be nonempty and contain unique samples')
    if not np.all(d.subject[np.r_[anchors, query]] == subject):
        raise ValueError('Support/query references a different subject')
    if set(d.y[query].tolist()) != set(classes):
        raise ValueError('Parent query does not contain all classes')
    qtrials = {index.trial_key(int(i)) for i in query}
    if qtrials & {index.trial_key(int(i)) for i in anchors}:
        raise ValueError('Support and query share an original trial')
    orders, capacities = {}, {}
    for c in classes:
        anchor = int(anchors[d.y[anchors] == c][0])
        subj, sess, trial = index.trial_key(anchor)
        pool = np.flatnonzero((d.subject == subj) & (d.session == sess) & (d.trial.astype(str) == trial))
        if not np.all(d.y[pool] == c):
            raise ValueError('Mixed labels within a support trial')
        rest = pool[pool != anchor]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(subject), int(c)]))
        orders[c] = np.r_[anchor, rng.permutation(rest)].astype(np.int64)
        capacities[c] = int(len(pool))
    # Preserve the original class order even if it was not numeric order.
    return {'subject': int(subject), 'anchors': anchors, 'query': query,
            'orders': orders, 'class_order': d.y[anchors].tolist(), 'capacities': capacities}


def support_indices(plan, k):
    if k < 1 or any(n < k for n in plan['capacities'].values()):
        raise ValueError(f'Budget k={k} exceeds this support trial capacity')
    return np.concatenate([plan['orders'][int(c)][:k] for c in plan['class_order']])


def load_saved_plan(path, index, subject, seed):
    record = read_json(path)
    if not isinstance(record.get('support'), dict) or '1' not in record['support']:
        raise ValueError(f'{path}: expected support={{"1": [one sample per class]}}')
    return make_plan(index, subject, index.indices(record['support']['1']),
                     index.indices(record.get('query')), seed)


def common_budgets(plans, requested):
    minimum = min(n for p in plans for n in p['capacities'].values())
    requested = sorted(set([1] + list(requested)))
    feasible = [k for k in requested if k <= minimum]
    return {'requested': requested, 'feasible': feasible, 'minimum_capacity': int(minimum),
            'dropped': [k for k in requested if k not in feasible],
            'primary_k': 5 if 5 in feasible else max(feasible)}


def load_data(args, parent, log):
    cfg = SimpleNamespace(**parent['protocol']['config'])
    if args.features_npz:
        data = load_npz(args.features_npz)
    elif cfg.synthetic:
        data = synthetic_data()
    else:
        cfg.feature_cache = args.feature_cache
        cfg.refresh_features = False
        # Reuse the exact loader/cache implementation of the parent experiment.
        data = local_seed(cfg, log)
    mask = np.isin(data.subject, cfg.subjects) & (data.session == cfg.session)
    data = Dataset(**{k: getattr(data, k)[mask] for k in Dataset.__dataclass_fields__}).validate()
    actual = data.fingerprint()
    if actual != parent['protocol']['data_hash']:
        raise ValueError('Feature fingerprint differs from parent; restore the parent data/cache before replay')
    if list(data.X.shape) != parent['data_shape']:
        raise ValueError('Feature dimensions differ from parent manifest')
    return data


def parent_records(args, cfg):
    records = read_json(args.parent / 'rows.json')
    if not isinstance(records, list) or not records:
        raise ValueError('Parent rows.json must contain completed folds')
    seen = set()
    for record in records:
        key = (int(record['seed']), int(record['test_subject']))
        if key in seen:
            raise ValueError(f'Duplicate parent fold: {key}')
        seen.add(key)
        directory = args.parent / f'seed{key[0]}' / f's{key[1]:02d}'
        if read_json(directory / 'complete.json') != record:
            raise ValueError(f'Parent complete.json and rows.json disagree for {key}')
        if set(record['train_subjects']) & set(record['val_subjects']):
            raise ValueError('Parent training/validation subjects overlap')
        if key[1] in record['train_subjects'] + record['val_subjects']:
            raise ValueError('Parent test subject occurs in training/validation')
        if set(record['train_subjects'] + record['val_subjects'] + [key[1]]) != set(cfg.subjects):
            raise ValueError('Parent fold does not partition its declared population')
        for model, base in MODEL_BASE.items():
            if model not in record['adaptation']:
                raise ValueError(f'{model} adaptation checkpoint was not trained in the parent')
            for name in (f'{base}_source.pt', f'{model}_adapter.pt', 'scaler.npz'):
                if not (directory / name).is_file():
                    raise FileNotFoundError(directory / name)
    return sorted(records, key=lambda f: (f['seed'], f['test_subject']))


def prepare_plans(args, cfg, records, data):
    """Global feasibility includes ALL parent folds, repeats, models, and validation plans.
    A one-fold smoke run therefore has the same budget as the later full run.
    """
    index = SampleIndex(data)
    tests, vals, capacity_records, parent_files = {}, {}, [], {}
    for fold in records:
        seed, test = int(fold['seed']), int(fold['test_subject'])
        directory = args.parent / f'seed{seed}' / f's{test:02d}'
        for name in ('complete.json', 'scaler.npz', 'M0_source.pt', 'M1_source.pt',
                     'M2_adapter.pt', 'M3_adapter.pt'):
            path = directory / name
            parent_files[str(path.relative_to(args.parent))] = sha(path)
        for model in MODEL_BASE:
            for repeat in range(args.repeats):
                path = directory / f'{model}_support_r{repeat}.json'
                plan = load_saved_plan(path, index, test, 200000 + seed * 100 + repeat)
                tests[(seed, test, model, repeat)] = plan
                parent_files[str(path.relative_to(args.parent))] = sha(path)
                capacity_records.append({'role': 'test_support', 'seed': seed, 'test': test,
                                         'model': model, 'repeat': repeat, 'capacities': plan['capacities']})
        for repeat in range(args.repeats):
            a, b = (tests[(seed, test, m, repeat)] for m in MODEL_BASE)
            if not np.array_equal(a['anchors'], b['anchors']) or not np.array_equal(a['query'], b['query']):
                raise ValueError(f'M2/M3 parent support/query differs in s{test}, repeat {repeat}')
        for person in fold['val_subjects']:
            # Identical to snn_baseline.select_adaptation for the original k=1 grid.
            supports, query, _ = trial_plan(data, person, (1,), seed + 9000 + person, cfg.query_trials)
            plan = make_plan(index, person, supports[1], query, 300000 + seed * 100 + person)
            vals[(seed, test, person)] = plan
            capacity_records.append({'role': 'validation_support', 'seed': seed, 'test': test,
                                     'subject': person, 'capacities': plan['capacities']})
    budgets = common_budgets(list(tests.values()) + list(vals.values()), args.segments_per_class)
    budgets['checked_plans'] = capacity_records
    return tests, vals, budgets, parent_files


def load_models(directory, model_name, cfg, features, classes, device):
    base = MODEL_BASE[model_name]
    model = SNN(features, classes, cfg.hidden, cfg.latent, base == 'M1', cfg.coding_steps).to(device)
    model.load_state_dict(torch.load(directory / f'{base}_source.pt', map_location=device, weights_only=True))
    freeze(model)
    saved = torch.load(directory / f'{model_name}_adapter.pt', map_location=device, weights_only=True)
    adapter = IntrinsicAdapter(cfg.hidden, cfg.latent, cfg.q_dim).to(device)
    adapter.load_state_dict(saved['adapter'])
    for p in adapter.parameters():
        p.requires_grad_(False)
    prototypes = saved['prototypes'].to(device)
    if tuple(prototypes.shape) != (classes, cfg.latent):
        raise ValueError('Parent prototype dimensions disagree with model')
    return model, adapter, prototypes


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def fit_and_evaluate(method, model, adapter, prototypes, X, data, support, query, device, cfg, choice):
    sync(device)
    start = time.perf_counter()
    extra = {}
    if method == 'head':
        candidate = fit_head(model, X, data.y, support, device, **choice)
        parameters = sum(p.numel() for p in model.head.parameters())
        q = None
    elif method == 'lowdim':
        q = fit_q(model, adapter, X, data.y, support, device, prototypes=prototypes,
                  proto_weight=cfg.proto_weight, q_reg=cfg.q_reg, **choice)
        parameters = cfg.q_dim
        candidate = model
        extra = {'q': q.detach().cpu().tolist(), 'q_norm': float(q.norm().item())}
    else:
        candidate, q, parameters = model, None, 0
    sync(device)
    elapsed = time.perf_counter() - start if method != 'none' else 0.
    result, predicted = evaluate(candidate, X, data.y, query, device, cfg.batch_size,
                                 adapter if method == 'lowdim' else None, q)
    return {**result, 'target_trainable_parameters': parameters, 'fit_seconds': elapsed, **extra}, predicted


def choose_parameters(k, model_name, model, adapter, prototypes, X, data, fold, vals, device, cfg, log):
    if k == 1:
        # These choices were already selected using the original validation grid.
        # Reusing them makes k=1 an exact replay of the parent benchmark.
        original = fold['adaptation'][model_name]['choices']
        return {m: dict(original[m]) for m in ('head', 'lowdim')}, {'source': 'parent_validation_grid'}
    choices, records = {}, []
    for method in ('head', 'lowdim'):
        best = float('inf')
        for steps in cfg.adapt_steps:
            for lr in cfg.adapt_lrs:
                choice = {'steps': steps, 'lr': lr}
                scores = []
                for subject in fold['val_subjects']:
                    plan = vals[(fold['seed'], fold['test_subject'], subject)]
                    result, _ = fit_and_evaluate(method, model, adapter, prototypes, X, data,
                                                support_indices(plan, k), plan['query'], device, cfg, choice)
                    scores.append(result['ce'])
                value = float(np.mean(scores))
                records.append({'method': method, **choice, 'validation_ce': value})
                if value < best:
                    best, choices[method] = value, choice
        log(f'{model_name} k={k} validation choice {method}: {choices[method]}, CE={best:.5f}')
    return choices, {'source': 'parent_grid_reselected_on_validation_subjects', 'grid': records}


def verify_k1(fold, model, repeat, method, actual):
    matched = [r for r in fold['rows'] if r['phase'] == 'calibrated_query' and r['model'] == model
               and r['method'] == method and r['shots'] == 1 and r['repeat'] == repeat]
    if len(matched) != 1:
        raise ValueError(f'Missing/duplicate parent k=1 result: {model}/{method}/r{repeat}')
    reference = matched[0]
    if (actual['n_support'], actual['n_query']) != (reference['n_support'], reference['n_query']):
        raise ValueError('k=1 sample counts disagree with parent')
    diffs = {m: float(actual[m] - reference[m]) for m in ('accuracy', 'macro_f1', 'event_fraction', 'ce')}
    if any(abs(v) > REPRO_ATOL for v in diffs.values()):
        raise RuntimeError(f'k=1 reproduction mismatch for s{fold["test_subject"]} {model}/{method}/r{repeat}: '
                           f'{diffs}. Check parent device/numeric environment and source files; do not mix results.')
    return diffs


def plan_record(plan, data, budgets):
    return {'support': {str(k): data.sample_id[support_indices(plan, k)].tolist() for k in budgets},
            'query': data.sample_id[plan['query']].tolist(), 'capacities': plan['capacities'],
            'support_trials_per_class': 1, 'budget_unit': 'labeled nonoverlapping segments per class'}


def run_fold(args, cfg, data, parent_fold, tests, vals, budgets, device, out, log, fingerprint):
    seed, subject = int(parent_fold['seed']), int(parent_fold['test_subject'])
    directory = args.parent / f'seed{seed}' / f's{subject:02d}'
    with np.load(directory / 'scaler.npz', allow_pickle=False) as saved:
        scaler = TrainScaler()
        scaler.mean, scaler.scale = saved['mean'], saved['scale']
    X = scaler.transform(data.X)
    result = {'seed': seed, 'test_subject': subject, 'run_fingerprint': fingerprint,
              'rows': [], 'choices': {}, 'k1_reproduction': [], 'segments_tested': budgets['feasible']}
    for person in parent_fold['val_subjects']:
        atomic_json(out / f'validation_s{person:02d}_samples.json',
                    plan_record(vals[(seed, subject, person)], data, budgets['feasible']))
    for name in MODEL_BASE:
        model, adapter, prototypes = load_models(directory, name, cfg, X.shape[-1], len(np.unique(data.y)), device)
        result['choices'][name] = {}
        for repeat in range(args.repeats):
            plan = tests[(seed, subject, name, repeat)]
            atomic_json(out / f'{name}_support_r{repeat}.json', plan_record(plan, data, budgets['feasible']))
        for k in budgets['feasible']:
            log.stage = f's{subject:02d} {name} k={k} validation/calibration'
            choices, selection = choose_parameters(k, name, model, adapter, prototypes, X, data,
                                                    parent_fold, vals, device, cfg, log)
            result['choices'][name][str(k)] = {'choices': choices, **selection}
            for repeat in range(args.repeats):
                plan = tests[(seed, subject, name, repeat)]
                support, query = support_indices(plan, k), plan['query']
                for method in METHODS:
                    score, predicted = fit_and_evaluate(method, model, adapter, prototypes, X, data,
                                                        support, query, device, cfg, choices.get(method))
                    row = {'model': name, 'base': MODEL_BASE[name], 'method': method,
                           'phase': 'calibrated_query', 'repeat': repeat, 'segments_per_class': k,
                           'support_trials_per_class': 1, 'n_support': len(support), 'n_query': len(query),
                           'support_segment_seconds': None if cfg.synthetic or args.features_npz else len(support) * cfg.segment_s,
                           **score}
                    if k == 1:
                        diff = verify_k1(parent_fold, name, repeat, method, row)
                        result['k1_reproduction'].append({'model': name, 'repeat': repeat, 'method': method,
                                                         'differences': diff, 'passed': True})
                    result['rows'].append(row)
                    prediction_file(out / f'{name}_r{repeat}_k{k}_{method}.npz', data, query, predicted)
                recent = result['rows'][-3:]
                log(f's{subject:02d} {name} r={repeat} k={k}: ' +
                    ', '.join(f'{r["method"]}={r["accuracy"]:.4f}' for r in recent))
                atomic_json(out / 'partial.json', result)
    atomic_json(out / 'complete.json', result)
    return result


def paired_summary(folds):
    """Append every repetition/seed, then average WITHIN a subject before inference."""
    comparisons, conditions = {}, {}
    for fold in folds:
        subject, groups = fold['test_subject'], {}
        for row in fold['rows']:
            key = (row['model'], row['segments_per_class'], row['repeat'])
            g = groups.setdefault(key, {})
            if row['method'] in g:
                raise ValueError('Duplicate method row within a repetition')
            g[row['method']] = row
            for metric in ('accuracy', 'macro_f1', 'event_fraction'):
                name = f'{row["model"]}:{row["method"]}:k{row["segments_per_class"]}:{metric}'
                conditions.setdefault(name, {}).setdefault(subject, []).append(row[metric])
        for (model, k, repeat), group in groups.items():
            if set(group) != set(METHODS):
                raise ValueError('Incomplete comparison group')
            for metric in ('accuracy', 'macro_f1', 'event_fraction'):
                for a, b in (('lowdim', 'head'), ('lowdim', 'none'), ('head', 'none')):
                    name = f'{model}:{a}-{b}:k{k}:{metric}'
                    comparisons.setdefault(name, {}).setdefault(subject, []).append(group[a][metric] - group[b][metric])
    output = {}
    for name, people in comparisons.items():
        ids = sorted(people)
        d = np.array([np.mean(people[s]) for s in ids])
        n, sd = len(d), float(d.std(ddof=1)) if len(d) > 1 else None
        half = float(student_t.ppf(.975, n-1) * sd / np.sqrt(n)) if n > 1 else None
        output[name] = {'n_subjects': n, 'mean_delta': float(d.mean()), 'subject_sd': sd,
                        'ci95_t': [float(d.mean()-half), float(d.mean()+half)] if n > 1 else None,
                        'positive': int((d > 0).sum()), 'zero': int((d == 0).sum()),
                        'subject_deltas': dict(zip(map(str, ids), map(float, d)))}
    absolute = {name: {'n_subjects': len(people),
                       'mean': float(np.mean([np.mean(v) for v in people.values()])),
                       'subject_means': {str(s): float(np.mean(v)) for s, v in people.items()}}
                for name, people in conditions.items()}
    return {'unit': 'held-out subject; average repeats and seeds within each subject',
            'intervals': 'exploratory, unadjusted 95% t intervals', 'comparisons': output,
            'condition_means': absolute}


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--test-subjects', type=int, nargs='+', help='Omit to run all completed parent subjects')
    p.add_argument('--repeats', type=int, help='Defaults to parent repeat count')
    p.add_argument('--segments-per-class', type=int, nargs='+', default=[1, 5, 10])
    p.add_argument('--feature-cache', type=Path, default=Path('data_cache/snn_ordered_de'))
    p.add_argument('--features-npz', type=Path, help='Optional exact parent dataset; fingerprint must match')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--out', type=Path, default=Path('results/snn_replay_fixed_v2'))
    p.add_argument('--resume', action='store_true')
    p.add_argument('--check-inputs-only', action='store_true', help='Validate data/plans/budgets without CUDA initialization')
    args = p.parse_args(argv)
    if min(args.segments_per_class) < 1 or args.threads < 1 or (args.repeats is not None and args.repeats < 1):
        p.error('Counts must be positive')
    args.segments_per_class = sorted(set([1] + args.segments_per_class))
    return args


def main(argv=None):
    args = arguments(argv)
    args.parent = args.parent.resolve()
    args.out = args.out.resolve()
    if args.parent == args.out or args.parent in args.out.parents:
        raise ValueError('Replay output must be separate from the parent run directory')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out / 'RUNNING.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f'{lock} exists. Check its PID; only remove it after that process has ended.') from exc
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    log, admitted = Progress(args.out / 'run.log'), False
    try:
        log(f'{VERSION} pid={os.getpid()} torch={torch.__version__}; validating parent files before device initialization')
        torch.set_num_threads(args.threads)
        parent = read_json(args.parent / 'manifest.json')
        cfg = SimpleNamespace(**parent['protocol']['config'])
        if cfg.meta_shots != 1:
            raise ValueError('This replay expects parent meta_shots=1')
        args.repeats = args.repeats if args.repeats is not None else cfg.repeats
        if args.repeats > cfg.repeats:
            raise ValueError('Requested repetitions exceed the saved parent repetitions')
        for name, expected in parent['protocol']['code'].items():
            if sha(HERE / name) != expected:
                raise ValueError(f'Parent dependency changed: {name}. Restore the parent-compatible file; do not rewrite parent hashes.')
        records = parent_records(args, cfg)
        scope = sorted({int(f['test_subject']) for f in records})
        args.test_subjects = sorted(set(args.test_subjects or scope))
        if not set(args.test_subjects).issubset(scope):
            raise ValueError(f'Test subjects must be completed parent subjects: {scope}')
        log.stage = 'load exact parent features'
        data = load_data(args, parent, log)
        log(f'Data fingerprint matches parent; X={data.X.shape}')
        log.stage = 'check all saved support/query lists and validation capacities'
        tests, vals, budgets, file_hashes = prepare_plans(args, cfg, records, data)
        log(f'Global budgets={budgets["feasible"]}; minimum trial capacity={budgets["minimum_capacity"]}; '
            f'dropped={budgets["dropped"]}; primary k={budgets["primary_k"]}')
        if args.check_inputs_only:
            if not (args.out / 'manifest.json').exists():
                atomic_json(args.out / 'feasibility.json', budgets)
            atomic_json(args.out / 'input_check.json', {'passed': True, 'parent_fingerprint': parent['fingerprint'],
                        'checked_parent_folds': len(records), 'global_budgets': budgets['feasible'],
                        'cuda_initialized': False})
            log('INPUT CHECK PASS: dataset, every repetition, trial isolation and common budgets verified')
            return
        log.stage = 'device initialization'
        log(f'Initializing requested device: {args.device}')
        if args.device == 'cpu':
            device = torch.device('cpu')
        else:
            available = torch.cuda.is_available()
            if args.device == 'cuda' and not available:
                raise RuntimeError('CUDA requested but unavailable; use --device cpu for a CPU diagnostic')
            device = torch.device('cuda' if available else 'cpu')
        torch.zeros(1, device=device).sum().item()
        log(f'Device ready: {device}')
        protocol = {'version': VERSION, 'parent': str(args.parent), 'parent_fingerprint': parent['fingerprint'],
                    'parent_files': file_hashes, 'replay_code_sha256': sha(__file__), 'data_hash': data.fingerprint(),
                    'segments_per_class_requested': args.segments_per_class, 'feasible_budgets': budgets['feasible'],
                    'primary_k': budgets['primary_k'], 'repeats': args.repeats, 'scope_subjects': scope,
                    'actual_device': str(device), 'torch_version': str(torch.__version__),
                    'budget_unit': 'labeled nonoverlapping segments per class from one support trial',
                    'validation_selection': 'reuse parent at k=1; same grid on original validation people at k>1'}
        fingerprint = digest(protocol)
        manifest_path = args.out / 'manifest.json'
        if manifest_path.exists():
            old = read_json(manifest_path)
            if not args.resume:
                raise RuntimeError('Replay run exists. Use --resume or a new --out directory')
            if old['fingerprint'] != fingerprint:
                raise RuntimeError('Resume refused: parent, data, code or replay protocol differs')
        else:
            atomic_json(manifest_path, {'fingerprint': fingerprint, 'protocol': protocol, 'synthetic': cfg.synthetic})
        admitted = True
        atomic_json(args.out / 'feasibility.json', budgets)
        completed = {}
        for path in args.out.glob('seed*/s*/complete.json'):
            record = read_json(path)
            if record['run_fingerprint'] != fingerprint:
                raise ValueError('Completed replay fold has a different fingerprint')
            completed[(record['seed'], record['test_subject'])] = record
        selected = [f for f in records if f['test_subject'] in args.test_subjects]
        atomic_json(args.out / 'status.json', {'state': 'running', 'requested_folds': len(selected)})
        for fold in selected:
            key = (fold['seed'], fold['test_subject'])
            out = args.out / f'seed{key[0]}' / f's{key[1]:02d}'
            out.mkdir(parents=True, exist_ok=True)
            if args.resume and key in completed:
                log(f'resume: skip complete seed={key[0]} s{key[1]:02d}')
            else:
                completed[key] = run_fold(args, cfg, data, fold, tests, vals, budgets, device, out, log, fingerprint)
            all_folds = [completed[k] for k in sorted(completed)]
            summary = paired_summary(all_folds)
            summary.update({'completed_subjects': sorted({f['test_subject'] for f in all_folds}),
                            'feasible_budgets': budgets['feasible'], 'primary_k': budgets['primary_k'],
                            'complete_parent_scope': len(completed) == len(records),
                            'complete_loso': len(completed) == len(cfg.subjects) * len(cfg.seeds)})
            atomic_json(args.out / 'rows.json', all_folds)
            atomic_json(args.out / 'summary.json', summary)
            atomic_json(args.out / 'status.json', {'state': 'running', 'completed_folds': len(completed)})
        atomic_json(args.out / 'status.json', {'state': 'completed', 'completed_folds': len(completed),
                    'complete_parent_scope': len(completed) == len(records),
                    'complete_loso': len(completed) == len(cfg.subjects) * len(cfg.seeds),
                    'synthetic': cfg.synthetic, 'fingerprint': fingerprint})
        log(f'COMPLETE: {len(completed)} replay folds. Parent checkpoints unchanged. Output: {args.out}')
    except KeyboardInterrupt:
        if admitted or not (args.out / 'manifest.json').exists():
            atomic_json(args.out / 'status.json', {'state': 'interrupted', 'stage': log.stage})
        log(f'Interrupted during: {log.stage}')
        raise
    except Exception as exc:
        if admitted or not (args.out / 'manifest.json').exists():
            atomic_json(args.out / 'status.json', {'state': 'failed', 'type': type(exc).__name__, 'message': str(exc)})
        log(f'FAILED {type(exc).__name__}: {exc}')
        raise
    finally:
        log.close()
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
