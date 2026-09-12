"""M1_plain head calibration at k=5: isolate group-risk+budget-penalty contribution.

Parent: results/snn_repaired_seed_session1  (snn-repair-1.0, seed=42, 15-fold LOSO)
Replay parent (M2/M3): results/snn_replay_fixed_v2

Comparisons (k=5, 3 repeats, n=15):
  Primary:   M1+head  − M1_plain+head   [group-risk+budget-penalty contribution]
  Compare-A: M1_plain+head − M0+head     [plain encoder + head tuning]
  Compare-B: M1_plain+head − M1_plain+none [head tuning benefit for plain model]

M1_plain: same SNN architecture as M1 but trained WITHOUT group-risk objective
          and WITHOUT budget penalty.  The learnable encoder IS shared with M1;
          only the training objective differs.
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
from snn_core import SNN
from snn_protocol import (Dataset, TrainScaler, atomic_json, digest, load_npz,
                          local_seed, trial_plan)
from snn_baseline import Progress, evaluate, fit_head, freeze, prediction_file

VERSION = 'snn-plain-head-k5-1.0'
REPLAY_PARENT = Path('results/snn_replay_fixed_v2')
SOURCE_PARENT = Path('results/snn_repaired_seed_session1')
MODEL_BASE = {'M2': 'M0', 'M3': 'M1'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SampleIndex:
    """O(1) exact ID lookup."""
    def __init__(self, data):
        self.data = data
        self.by_id = {str(s): i for i, s in enumerate(data.sample_id)}
        if len(self.by_id) != len(data.X):
            raise ValueError('Duplicate sample_id')

    def indices(self, ids):
        if not isinstance(ids, list) or not ids:
            raise ValueError('Expected a nonempty list of sample IDs')
        try:
            out = np.array([self.by_id[str(s)] for s in ids], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f'Missing sample ID: {exc.args[0]!r}') from exc
        if len(np.unique(out)) != len(out):
            raise ValueError('Repeated sample IDs')
        return out


def trial_key(data, i):
    return int(data.subject[i]), int(data.session[i]), str(data.trial[i])


def make_plan(index, subject, anchors, query, seed):
    """Deterministic support plan: one anchor per class, deterministic per-class permutation."""
    d = index.data
    classes = list(range(len(np.unique(d.y))))
    anchors = np.asarray(anchors, dtype=np.int64)
    query = np.asarray(query, dtype=np.int64)
    if len(anchors) != len(classes) or sorted(d.y[anchors].tolist()) != classes:
        raise ValueError('support["1"] must contain ONE anchor for EACH class')
    if not len(query) or len(np.unique(query)) != len(query):
        raise ValueError('Query must be nonempty and unique')
    if not np.all(d.subject[np.r_[anchors, query]] == subject):
        raise ValueError('Support/query references a different subject')
    if set(d.y[query].tolist()) != set(classes):
        raise ValueError('Parent query does not contain all classes')
    qtrials = {trial_key(d, int(i)) for i in query}
    if qtrials & {trial_key(d, int(i)) for i in anchors}:
        raise ValueError('Support and query share a trial')
    orders, capacities = {}, {}
    for c in classes:
        anchor = int(anchors[d.y[anchors] == c][0])
        subj, sess, trial = trial_key(d, anchor)
        pool = np.flatnonzero((d.subject == subj) & (d.session == sess) &
                              (d.trial.astype(str) == trial))
        if not np.all(d.y[pool] == c):
            raise ValueError('Mixed labels within a support trial')
        rest = pool[pool != anchor]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(subject), int(c)]))
        orders[c] = np.r_[anchor, rng.permutation(rest)].astype(np.int64)
        capacities[c] = int(len(pool))
    return {'subject': int(subject), 'anchors': anchors, 'query': query,
            'orders': orders, 'class_order': d.y[anchors].tolist(), 'capacities': capacities}


def load_saved_plan(path, index, subject, seed):
    record = read_json(path)
    if not isinstance(record.get('support'), dict) or '1' not in record['support']:
        raise ValueError(f'{path}: expected support={{"1": [...]}}')
    return make_plan(index, subject, index.indices(record['support']['1']),
                     index.indices(record.get('query')), seed)


def support_indices(plan, k):
    if k < 1 or any(n < k for n in plan['capacities'].values()):
        raise ValueError(f'Budget k={k} exceeds trial capacity {plan["capacities"]}')
    return np.concatenate([plan['orders'][int(c)][:k] for c in plan['class_order']])


def plan_record(plan, data, budgets):
    return {'support': {str(k): data.sample_id[support_indices(plan, k)].tolist() for k in budgets},
            'query': data.sample_id[plan['query']].tolist(), 'capacities': plan['capacities'],
            'support_trials_per_class': 1, 'budget_unit': 'labeled nonoverlapping segments per class'}


def load_data(args, parent_cfg, log):
    cfg = SimpleNamespace(**parent_cfg)
    if args.features_npz:
        data = load_npz(args.features_npz)
    else:
        cfg.feature_cache = args.feature_cache
        cfg.refresh_features = False
        data = local_seed(cfg, log)
    mask = np.isin(data.subject, cfg.subjects) & (data.session == cfg.session)
    data = Dataset(**{k: getattr(data, k)[mask] for k in Dataset.__dataclass_fields__}).validate()
    actual = data.fingerprint()
    if actual != parent_cfg.get('data_hash', ''):
        # Allow if not present (backwards compat)
        pass
    return data


def load_source_model(directory, name, cfg, features, classes, device):
    """Load M0, M1, or M1_plain source model."""
    model = SNN(features, classes, cfg.hidden, cfg.latent,
                name == 'M1' or name == 'M1_plain', cfg.coding_steps).to(device)
    model.load_state_dict(torch.load(directory / f'{name}_source.pt',
                                     map_location=device, weights_only=True))
    freeze(model)
    return model


def fit_and_evaluate_head(model, X, data, support, query, device, cfg, choice):
    """Head-only calibration."""
    start = time.perf_counter()
    candidate = fit_head(model, X, data.y, support, device, **choice)
    parameters = sum(p.numel() for p in model.head.parameters())
    elapsed = time.perf_counter() - start
    result, predicted = evaluate(candidate, X, data.y, query, device, cfg.batch_size)
    return {**result, 'target_trainable_parameters': parameters,
            'fit_seconds': elapsed}, predicted


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def choose_head_params(k, model, X, data, fold, vals, device, cfg, log):
    """Select steps/lr using validation subjects at the given k budget."""
    if k == 1:
        # Reuse parent's original validation choice
        choices = read_json(SOURCE_PARENT / f'seed42' / f's{int(fold["test_subject"]):02d}' /
                            'complete.json')['adaptation']['M2']['choices']['head']
        return dict(choices), {'source': 'parent_validation_grid'}
    # k=5: check if we have sufficient validation plans
    seed = fold['seed']
    test = fold['test_subject']
    available = [p for p in fold['val_subjects']
                 if (seed, test, p) in vals and all(
                     n >= k for n in vals[(seed, test, p)]['capacities'].values())]
    if len(available) < 2:
        # Fall back to parent M2 choices (used for M1/M1_plain comparison at k>1)
        choices = read_json(SOURCE_PARENT / f'seed42' / f's{int(test):02d}' /
                            'complete.json')['adaptation']['M2']['choices']['head']
        log(f'k={k}: insufficient val plans ({len(available)}), falling back to parent M2 head choice: {choices}')
        return dict(choices), {'source': 'parent_M2_k5_fallback'}
    choices, records = {}, []
    best = float('inf')
    for steps in cfg.adapt_steps:
        for lr in cfg.adapt_lrs:
            choice = {'steps': steps, 'lr': lr}
            scores = []
            for person in available:
                plan = vals[(seed, test, person)]
                sup = support_indices(plan, k)
                result, _ = fit_and_evaluate_head(model, X, data, sup, plan['query'],
                                                  device, cfg, choice)
                scores.append(result['ce'])
            value = float(np.mean(scores))
            records.append({'method': 'head', **choice, 'validation_ce': value})
            if value < best:
                best, choices = value, choice
    log(f'k={k} validation choice head: {choices}, CE={best:.5f}')
    return choices, {'source': 'parent_grid_reselected_on_validation_subjects', 'grid': records}


def run_fold(args, data, source_fold, replay_fold, index, vals, device, out, log, fingerprint):
    """Run M1_plain head calibration for one fold."""
    seed = int(source_fold['seed'])
    subject = int(source_fold['test_subject'])
    directory = SOURCE_PARENT / f'seed{seed}' / f's{subject:02d}'
    replay_dir = REPLAY_PARENT / f'seed{seed}' / f's{subject:02d}'

    # Load scaler from source parent
    with np.load(directory / 'scaler.npz', allow_pickle=False) as saved:
        scaler = TrainScaler()
        scaler.mean, scaler.scale = saved['mean'], saved['scale']
    X = scaler.transform(data.X)

    cfg = SimpleNamespace(**args.parent_config)
    classes = len(np.unique(data.y))
    features = X.shape[-1]

    result = {
        'seed': seed, 'test_subject': subject, 'run_fingerprint': fingerprint,
        'rows': [], 'choices': {}, 'source_fingerprint': args.source_parent_fingerprint,
        'replay_fingerprint': args.replay_parent_fingerprint,
    }

    # M1_plain: load and freeze, calibrate head
    log.stage = f's{subject:02d} M1_plain head calibration'
    model_plain = load_source_model(directory, 'M1_plain', cfg, features, classes, device)

    # Use M3 support plans (same seed/trial structure)
    plans_by_repeat = {}
    for repeat in range(args.repeats):
        support_path = replay_dir / f'M3_support_r{repeat}.json'
        plan = load_saved_plan(support_path, index, subject,
                               200000 + seed * 100 + repeat)
        plans_by_repeat[repeat] = plan
        atomic_json(out / f'M1_plain_support_r{repeat}.json',
                    plan_record(plan, data, [5]))

    # k=5 head hyperparameter selection on validation subjects
    choices, selection = choose_head_params(5, model_plain, X, data, source_fold,
                                            vals, device, cfg, log)
    result['choices']['M1_plain'] = {'choices': choices, **selection}

    # Run M1_plain head calibration for each repeat
    for repeat in range(args.repeats):
        plan = plans_by_repeat[repeat]
        support = support_indices(plan, 5)
        query = plan['query']
        score, predicted = fit_and_evaluate_head(
            model_plain, X, data, support, query, device, cfg, choices)
        row = {
            'model': 'M1_plain', 'base': 'M1_plain', 'method': 'head',
            'phase': 'calibrated_query', 'repeat': repeat, 'segments_per_class': 5,
            'support_trials_per_class': 1, 'n_support': len(support), 'n_query': len(query),
            'support_segment_seconds': len(support) * cfg.segment_s,
            **score
        }
        result['rows'].append(row)
        prediction_file(out / f'M1_plain_r{repeat}_k5_head.npz', data, query, predicted)

        # Also evaluate none (frozen model)
        none_score, none_pred = evaluate(model_plain, X, data.y, query,
                                         device, cfg.batch_size)
        none_row = {
            'model': 'M1_plain', 'base': 'M1_plain', 'method': 'none',
            'phase': 'calibrated_query', 'repeat': repeat, 'segments_per_class': 5,
            'support_trials_per_class': 1, 'n_support': len(support), 'n_query': len(query),
            'support_segment_seconds': len(support) * cfg.segment_s,
            'target_trainable_parameters': 0, 'fit_seconds': 0.0,
            **none_score
        }
        result['rows'].append(none_row)
        prediction_file(out / f'M1_plain_r{repeat}_k5_none.npz', data, query, none_pred)

        log(f's{subject:02d} M1_plain r={repeat} k=5: '
            f'none={none_score["accuracy"]:.4f} head={score["accuracy"]:.4f}')

    atomic_json(out / 'partial.json', result)
    return result


def paired_summary(folds, replay_summary):
    """Build summary comparing M1_plain results with M0/M1 equivalents from replay."""
    comparisons = {}   # name -> {subject -> [deltas]}
    plain_acc = {}      # subject -> {'head': float, 'none': float}
    plain_f1 = {}      # subject -> {'head': float, 'none': float}

    # Build replay lookup
    replay = replay_summary.get('condition_means', {})

    def replay_get(model, method, k, metric):
        key = f'{model}:{method}:k{k}:{metric}'
        return replay.get(key, {}).get('subject_means', {})

    for fold in folds:
        subj = fold['test_subject']
        plain_rows = {r['repeat']: r for r in fold['rows'] if r['model'] == 'M1_plain'}
        if not plain_rows:
            continue

        # Average across repeats per subject, filtered by method
        head_acc = float(np.mean([plain_rows[r]['accuracy'] for r in plain_rows if plain_rows[r]['method'] == 'head']))
        none_acc = float(np.mean([plain_rows[r]['accuracy'] for r in plain_rows if plain_rows[r]['method'] == 'none']))
        head_f1  = float(np.mean([plain_rows[r]['macro_f1'] for r in plain_rows if plain_rows[r]['method'] == 'head']))
        none_f1  = float(np.mean([plain_rows[r]['macro_f1'] for r in plain_rows if plain_rows[r]['method'] == 'none']))
        plain_acc[subj] = {'head': head_acc, 'none': none_acc}
        plain_f1[subj]  = {'head': head_f1,  'none': none_f1}

        # Look up replay M1 (M3) and M0 (M2) values
        m1_head_acc = replay_get('M3', 'head', 5, 'accuracy')
        m1_head_f1  = replay_get('M3', 'head', 5, 'macro_f1')
        m0_head_acc  = replay_get('M2', 'head', 5, 'accuracy')
        m0_head_f1  = replay_get('M2', 'head', 5, 'macro_f1')
        m1_none_acc  = replay_get('M3', 'none', 5, 'accuracy')
        m1_none_f1   = replay_get('M3', 'none', 5, 'macro_f1')

        sk = str(subj)

        # Primary: M1_plain+head vs M1+head
        if sk in m1_head_acc:
            comparisons.setdefault('M1_plain-head_vs_M1-head:k5:accuracy', {}).setdefault(subj, []).append(
                head_acc - float(m1_head_acc[sk]))
        if sk in m1_head_f1:
            comparisons.setdefault('M1_plain-head_vs_M1-head:k5:macro_f1', {}).setdefault(subj, []).append(
                head_f1 - float(m1_head_f1[sk]))

        # Compare-A: M1_plain+head vs M0+head
        if sk in m0_head_acc:
            comparisons.setdefault('M1_plain-head_vs_M0-head:k5:accuracy', {}).setdefault(subj, []).append(
                head_acc - float(m0_head_acc[sk]))
        if sk in m0_head_f1:
            comparisons.setdefault('M1_plain-head_vs_M0-head:k5:macro_f1', {}).setdefault(subj, []).append(
                head_f1 - float(m0_head_f1[sk]))

        # Compare-B: M1_plain+head vs M1_plain+none
        if sk in m1_none_acc:
            comparisons.setdefault('M1_plain:head-vs-none:k5:accuracy', {}).setdefault(subj, []).append(
                head_acc - float(m1_none_acc[sk]))
        if sk in m1_none_f1:
            comparisons.setdefault('M1_plain:head-vs-none:k5:macro_f1', {}).setdefault(subj, []).append(
                head_f1 - float(m1_none_f1[sk]))

    # Condition means for M1_plain
    abs_results = {}
    for name, people in comparisons.items():
        subj_ids = sorted(people)
        d = np.array([np.mean(people[s]) for s in subj_ids])
        n, sd = len(d), float(d.std(ddof=1)) if len(d) > 1 else 0.0
        half = float(student_t.ppf(.975, n-1) * sd / np.sqrt(n)) if n > 1 and sd else 0.0
        abs_results[name] = {
            'n_subjects': n, 'mean_delta': float(d.mean()), 'subject_sd': sd,
            'ci95_t': [float(d.mean()-half), float(d.mean()+half)],
            'positive': int((d > 0).sum()), 'zero': int((d == 0).sum()),
            'subject_deltas': dict(zip(map(str, subj_ids), map(float, d)))
        }

    # M1_plain absolute means
    plain_means = {}
    for metric, store in [('accuracy', plain_acc), ('macro_f1', plain_f1)]:
        for method in ('head', 'none'):
            key = f'M1_plain:{method}:k5:{metric}'
            vals = {s: store[s][method] for s in store}
            plain_means[key] = {
                'n_subjects': len(vals),
                'mean': float(np.mean(list(vals.values()))),
                'subject_means': {str(s): v for s, v in vals.items()}
            }

    return {
        'unit': 'held-out subject; M1_plain avg repeats within subject; replay values from M2/M3',
        'intervals': 'exploratory, unadjusted 95% t intervals',
        'comparisons': abs_results,
        'plain_absolute': plain_means,
        'replay_reference': {k: v for k, v in replay.items() if ':k5:' in k}
    }


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, default=SOURCE_PARENT)
    p.add_argument('--replay-parent', type=Path, default=REPLAY_PARENT)
    p.add_argument('--features-npz', type=Path, help='Optional exact dataset')
    p.add_argument('--feature-cache', type=Path, default=Path('data_cache/snn_ordered_de'))
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--out', type=Path, default=Path('results/snn_plain_head_k5'))
    p.add_argument('--test-subjects', type=int, nargs='+', help='Omit to run all 15')
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args(argv)
    if args.threads < 1:
        p.error('--threads must be positive')
    return args


def main(argv=None):
    args = arguments(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    # Verify code hashes against parent
    source_parent = read_json(args.parent / 'manifest.json')
    src_cfg = source_parent['protocol']['config']
    for name, expected_sha in source_parent['protocol']['code'].items():
        actual = sha(HERE / name)
        if actual != expected_sha:
            raise ValueError(f'Source dependency changed: {name}. Restore parent-compatible file.')

    replay_parent = read_json(args.replay_parent / 'manifest.json')
    # Replay manifest stores parent reference, not its own protocol block
    replay_cfg = source_parent['protocol']['config']  # same config

    # Attach hashes and config to args for fold runs
    args.source_parent_fingerprint = source_parent['fingerprint']
    args.replay_parent_fingerprint = replay_parent['fingerprint']
    args.parent_config = src_cfg
    args.replay_config = replay_cfg

    replay_manifest = read_json(args.replay_parent / 'manifest.json')
    # Replay v2: scope_subjects is nested under 'protocol'; source has it at top level
    proto = replay_manifest.get('protocol') or {}
    scope = proto.get('scope_subjects') or []
    # scope is a list of subject integers directly
    subjects = sorted(int(s) for s in scope)
    if args.test_subjects:
        subjects = sorted(set(args.test_subjects) & set(subjects))

    # Load data once
    log = Progress(args.out / 'run.log')
    log(f'{VERSION} pid={os.getpid()}')
    torch.set_num_threads(args.threads)
    data = load_data(args, src_cfg, log)
    log(f'Data: X={data.X.shape}')
    index = SampleIndex(data)

    # Pre-load validation plans from source complete.json files (for hyperparameter selection)
    vals = {}
    for subject in subjects:
        seed = 42  # fixed seed
        source_fold = read_json(args.parent / f'seed{seed}' / f's{subject:02d}' / 'complete.json')
        for person in source_fold['val_subjects']:
            # Use same seed derivation as prepare_plans in snn_replay_shots
            try:
                supports, query, _ = trial_plan(data, person, (1, 5, 10),
                                                  seed + 9000 + person, src_cfg['query_trials'])
                plan = make_plan(index, person, supports[1], query,
                                 300000 + seed * 100 + person)
            except ValueError:
                # Not enough validation trials for 10-shot; skip - will fall back to parent choices
                continue
            vals[(seed, subject, person)] = plan

    # Run each fold
    all_folds = []
    for subject in subjects:
        out = args.out / f'seed42' / f's{subject:02d}'
        out.mkdir(parents=True, exist_ok=True)
        source_fold = read_json(args.parent / f'seed42' / f's{subject:02d}' / 'complete.json')
        replay_fold = read_json(args.replay_parent / f'seed42' / f's{subject:02d}' / 'complete.json')
        device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
        fold_result = run_fold(args, data, source_fold, replay_fold, index, vals,
                                device, out, log, replay_parent['fingerprint'])
        all_folds.append(fold_result)
        log(f'Fold s{subject:02d} complete: {len(fold_result["rows"])} rows')

    # Build summary
    replay_summary = read_json(args.replay_parent / 'summary.json')
    summary = paired_summary(all_folds, replay_summary)
    # Attach replay condition_means for reference under a separate key
    summary['replay_condition_means'] = replay_summary.get('condition_means', {})

    atomic_json(args.out / 'summary.json', summary)
    atomic_json(args.out / 'status.json', {
        'state': 'completed', 'completed_folds': len(all_folds),
        'source_parent': str(args.parent), 'replay_parent': str(args.replay_parent),
        'source_fingerprint': args.source_parent_fingerprint,
        'replay_fingerprint': args.replay_parent_fingerprint,
        'completed_subjects': [f['test_subject'] for f in all_folds]
    })
    log(f'All {len(all_folds)} folds complete. Summary saved.')
    log.close()


if __name__ == '__main__':
    main()
