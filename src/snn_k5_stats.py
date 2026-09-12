"""Recompute k=5 statistics from saved predictions; no PyTorch or training.
Every comparison is FIRST minus SECOND, using the same named metric.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t

VERSION = 'snn-k5-stats-1.0'
MODELS = ('M0', 'M1_plain', 'M1')
METHODS = ('none', 'head')
METRICS = ('accuracy', 'macro_f1')
COMPARISONS = (
    ('M1:head', 'M1_plain:head'),
    ('M1_plain:head', 'M0:head'),
    ('M1_plain:head', 'M1_plain:none'),
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def label_array(value, name):
    a = np.asarray(value)
    if a.ndim != 1 or not len(a) or a.dtype.kind not in 'iuf':
        raise ValueError(f'{name}: expected a nonempty numeric vector')
    if not np.isfinite(a).all() or not np.isin(a, [0, 1, 2]).all():
        raise ValueError(f'{name}: expected integer class labels 0, 1, 2')
    return a.astype(np.int64)


def metrics(y_true, y_pred):
    yt, yp = label_array(y_true, 'y_true'), label_array(y_pred, 'y_pred')
    if yt.shape != yp.shape:
        raise ValueError('Prediction and label lengths differ')
    cm = np.bincount(3 * yt + yp, minlength=9).reshape(3, 3)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2. * tp, denom, out=np.zeros(3), where=denom > 0)
    return {'accuracy': float(tp.sum() / len(yt)), 'macro_f1': float(f1.mean())}


def read_prediction(path, subject=None):
    with np.load(path, allow_pickle=False) as p:
        ids = np.asarray(p['sample_id']).astype(str)
        yt, yp = label_array(p['y_true'], 'y_true'), label_array(p['y_pred'], 'y_pred')
        if ids.shape != yt.shape or yt.shape != yp.shape:
            raise ValueError(f'{path}: inconsistent array dimensions')
        if len(set(ids.tolist())) != len(ids):
            raise ValueError(f'{path}: duplicate query sample_id')
        if subject is not None and 'subject' in p and not np.all(p['subject'] == subject):
            raise ValueError(f'{path}: unexpected subject metadata')
    return {'sample_id': ids, 'y_true': yt, 'y_pred': yp}


def align(reference, actual):
    lookup = {sid: i for i, sid in enumerate(actual['sample_id'].tolist())}
    if set(lookup) != set(reference['sample_id'].tolist()):
        raise ValueError('Query sample_id sets differ between conditions')
    ix = np.array([lookup[sid] for sid in reference['sample_id']], dtype=np.int64)
    if not np.array_equal(reference['y_true'], actual['y_true'][ix]):
        raise ValueError('Query labels disagree after sample_id alignment')
    return actual['y_pred'][ix]


def collect(replay, plain, subjects, repeats=3, seed=42):
    rows, files, selections = [], {}, []
    for subject in subjects:
        relative = Path(f'seed{seed}') / f's{subject:02d}'
        for repeat in range(repeats):
            reference = None
            for model in MODELS:
                code = {'M0': 'M2', 'M1': 'M3', 'M1_plain': 'M1_plain'}[model]
                base = plain if model == 'M1_plain' else replay
                for method in METHODS:
                    path = Path(base) / relative / f'{code}_r{repeat}_k5_{method}.npz'
                    prediction = read_prediction(path, subject)
                    if reference is None:
                        reference = prediction
                    predicted = align(reference, prediction)
                    rows.append({'seed': seed, 'test_subject': subject, 'repeat': repeat,
                                 'model': model, 'method': method, 'segments_per_class': 5,
                                 'n_query': len(predicted), **metrics(reference['y_true'], predicted),
                                 'prediction_file': str(path.resolve())})
                    files[str(path.resolve())] = sha(path)
        directory = Path(plain) / relative
        saved = next((p for p in (directory / 'complete.json', directory / 'partial.json') if p.exists()), None)
        choice = read_json(saved).get('choices', {}).get('M1_plain', {}) if saved else {}
        source = str(choice.get('source', 'not_recorded'))
        selections.append({'test_subject': subject, 'source': source,
                           'fallback_recorded': 'fallback' in source.lower(),
                           'choice': choice.get('choices')})
    return rows, {'input_file_sha256': files, 'query_alignment_passed': True,
                  'plain_selection_records': selections}


def summarize(rows, subjects, repeats=3, seed=42):
    indexed = {}
    for row in rows:
        key = (row['test_subject'], row['model'], row['method'], row['repeat'])
        if key in indexed:
            raise ValueError(f'Duplicate result row: {key}')
        if row['seed'] != seed or row['segments_per_class'] != 5:
            raise ValueError('Mixed seeds or budgets in this summary')
        indexed[key] = row
    expected = {(s, model, method, r) for s in subjects for model in MODELS
                for method in METHODS for r in range(repeats)}
    if set(indexed) != expected:
        raise ValueError(f'Incomplete/unexpected conditions: missing={expected-set(indexed)}, extra={set(indexed)-expected}')
    conditions = {}
    for model in MODELS:
        for method in METHODS:
            conditions[f'{model}:{method}'] = {}
            for metric in METRICS:
                people = {str(s): float(np.mean([indexed[(s, model, method, r)][metric]
                                                for r in range(repeats)])) for s in subjects}
                conditions[f'{model}:{method}'][metric] = {
                    'mean': float(np.mean(list(people.values()))), 'subject_means': people}
    comparisons = []
    for first, second in COMPARISONS:
        for metric in METRICS:
            a, b = conditions[first][metric], conditions[second][metric]
            delta = np.array([a['subject_means'][str(s)] - b['subject_means'][str(s)] for s in subjects])
            mean = float(delta.mean())
            if not np.isclose(mean, a['mean'] - b['mean'], atol=1e-12, rtol=0):
                raise AssertionError('Paired mean differs from difference of absolute means')
            n = len(delta)
            sd = float(delta.std(ddof=1)) if n > 1 else None
            half = float(student_t.ppf(.975, n-1) * sd / np.sqrt(n)) if n > 1 else None
            comparisons.append({'first': first, 'second': second, 'metric': metric,
                                'direction': 'first_minus_second', 'n_subjects': n,
                                'mean_delta': mean, 'subject_sd': sd,
                                'ci95_t': [mean-half, mean+half] if n > 1 else None,
                                'positive': int((delta > 0).sum()), 'zero': int((delta == 0).sum()),
                                'negative': int((delta < 0).sum()),
                                'subject_deltas': dict(zip(map(str, subjects), map(float, delta)))})
    return {'version': VERSION, 'seed': seed, 'segments_per_class': 5, 'repeats': repeats,
            'subjects': subjects, 'n_prediction_files': len(rows),
            'unit': 'held-out subject; average repeats before paired comparisons',
            'intervals': 'exploratory unadjusted 95% t intervals',
            'conditions': conditions, 'comparisons': comparisons}


def report_text(summary, provenance):
    lines = ['# k=5 预测文件重算结果', '',
             '全部差值按“前项 − 后项”计算；先平均同一被试的重复，再进行被试间统计。', '',
             '| 模型 | 方法 | Accuracy | Macro-F1 |', '| --- | --- | ---: | ---: |']
    for model in MODELS:
        for method in METHODS:
            item = summary['conditions'][f'{model}:{method}']
            lines.append(f'| {model} | {method} | {item["accuracy"]["mean"]:.4f} | {item["macro_f1"]["mean"]:.4f} |')
    lines += ['', '差值和区间单位为百分点；区间为探索性、未校正的 95% t 区间。', '',
              '| 前项 − 后项 | 指标 | 均值差 | 95% CI | 正/零/负 |', '| --- | --- | ---: | --- | --- |']
    for c in summary['comparisons']:
        ci = c['ci95_t']
        interval = f'[{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]' if ci else '单被试，不计算区间'
        lines.append(f'| {c["first"]} − {c["second"]} | {c["metric"]} | {c["mean_delta"]*100:+.2f} | '
                     f'{interval} | {c["positive"]}/{c["zero"]}/{c["negative"]} |')
    fallback = [s['test_subject'] for s in provenance['plain_selection_records'] if s['fallback_recorded']]
    lines += ['', 'M1_plain 超参数来源（读取折记录，供核对）：', '', '| 被试 | 来源 |', '| --- | --- |']
    lines += [f'| {s["test_subject"]} | {s["source"]} |' for s in provenance['plain_selection_records']]
    if fallback:
        lines += ['', f'发现回退选参记录：{fallback}。这些旧结果不满足为 M1_plain 独立在验证被试上选参的约定。']
    lines += ['', '统计重算核对了各条件 query 的样本标识与标签。它本身不检查源训练或支持集选取，也不自动判定创新有效。', '']
    return '\n'.join(lines)


def write_outputs(out, rows, summary, provenance):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'rows_from_predictions.json', rows)
    atomic_json(out / 'summary_from_predictions.json', summary)
    atomic_json(out / 'prediction_provenance.json', provenance)
    (out / 'report_from_predictions.md').write_text(report_text(summary, provenance), encoding='utf-8')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', type=Path, default=Path('results/snn_replay_fixed_v2'))
    p.add_argument('--plain', type=Path, default=Path('results/snn_plain_head_k5'))
    p.add_argument('--out', type=Path, default=Path('results/snn_k5_summary_from_predictions'))
    p.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 16)))
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)
    args.subjects = sorted(set(args.subjects))
    if args.repeats < 1 or min(args.subjects) < 1:
        p.error('Counts must be positive')
    for source in (args.replay, args.plain):
        if args.out.resolve() == source.resolve() or source.resolve() in args.out.resolve().parents:
            p.error('--out must be separate from input experiment directories')
    rows, provenance = collect(args.replay, args.plain, args.subjects, args.repeats, args.seed)
    summary = summarize(rows, args.subjects, args.repeats, args.seed)
    write_outputs(args.out, rows, summary, provenance)
    print(report_text(summary, provenance), flush=True)
    print(f'Saved: {args.out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
