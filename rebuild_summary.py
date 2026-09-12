"""Rebuild summary.json from raw npz files."""
import numpy as np, json, os
from scipy.stats import t as t_dist
from sklearn.metrics import f1_score

subjects = list(range(1, 16))
plain_dir = 'E:/Python/Study/Koopman_EEG/results/snn_plain_head_k5/seed42'
replay_dir = 'E:/Python/Study/Koopman_EEG/results/snn_replay_fixed_v2/seed42'

# ── 1. M1_plain raw: partial.json + npz ──────────────────────────────────────
plain_raw = {}   # s -> r -> method -> {acc, f1}
for s in subjects:
    ppath = os.path.join(plain_dir, 's{:02d}/partial.json'.format(s))
    with open(ppath) as f:
        partial = json.load(f)
    plain_raw[s] = {}
    for row in partial['rows']:
        r, m = row['repeat'], row['method']
        if m not in ('head', 'none'):
            continue
        npz = np.load(os.path.join(plain_dir, 's{:02d}/M1_plain_r{}_k5_{}.npz'.format(s, r, m)),
                      allow_pickle=True)
        yt, yp = npz['y_true'], npz['y_pred']
        acc = float((yt == yp).mean())
        f1v = float(f1_score(yt, yp, average='macro', zero_division=0))
        plain_raw[s].setdefault(r, {})[m] = {'accuracy': acc, 'macro_f1': f1v}

# Average 3 repeats per subject
plain_avg = {}   # s -> method -> {acc, f1}
for s in subjects:
    plain_avg[s] = {}
    for m in ('head', 'none'):
        reps = plain_raw[s]
        accs = [reps[r][m]['accuracy'] for r in reps if m in reps.get(r, {})]
        f1s  = [reps[r][m]['macro_f1']  for r in reps if m in reps.get(r, {})]
        plain_avg[s][m] = {'accuracy': np.mean(accs), 'macro_f1': np.mean(f1s)}

# ── 2. Replay M2/M3 raw: complete.json ────────────────────────────────────────
replay_by_repeat = {}   # (model, method, k) -> s -> [accs]
for s in subjects:
    fpath = os.path.join(replay_dir, 's{:02d}/complete.json'.format(s))
    with open(fpath) as f:
        complete = json.load(f)
    for row in complete['rows']:
        if row['segments_per_class'] != 5:
            continue
        model, method = row['model'], row['method']
        if method not in ('head', 'none'):
            continue
        key = (model, method)
        replay_by_repeat.setdefault(key, {}).setdefault(s, []).append(row['accuracy'])

replay_avg = {key: {s: np.mean(v) for s, v in d.items()}
              for key, d in replay_by_repeat.items()}

# ── 3. Paired stats ────────────────────────────────────────────────────────────
def paired(delta_subj, subjects):
    deltas = np.array([delta_subj[s] for s in subjects])
    n = len(deltas)
    mean_d = np.mean(deltas)
    sd = np.std(deltas, ddof=1)
    t = t_dist.ppf(0.975, df=n-1)
    ci = (mean_d - t*sd/np.sqrt(n), mean_d + t*sd/np.sqrt(n))
    return {
        'mean_delta': float(mean_d), 'sd': float(sd),
        'ci95_t': [float(c) for c in ci],
        'positive': int(np.sum(deltas > 0)), 'zero': int(np.sum(deltas == 0)),
        'subject_deltas': {str(s): float(deltas[i]) for i, s in enumerate(subjects)}
    }

# ── 4. Comparisons ────────────────────────────────────────────────────────────
comparisons = {}

# A: M1_plain head vs M1(M3) head
da = {s: plain_avg[s]['head']['accuracy'] - replay_avg[('M3','head')][s] for s in subjects}
df = {s: plain_avg[s]['head']['macro_f1']  - replay_avg[('M3','head')][s] for s in subjects}
comparisons['M1_plain-head_vs_M1-head:k5:accuracy'] = paired(da, subjects)
comparisons['M1_plain-head_vs_M1-head:k5:macro_f1'] = paired(df, subjects)

# B: M1_plain head vs M0(M2) head
da = {s: plain_avg[s]['head']['accuracy'] - replay_avg[('M2','head')][s] for s in subjects}
df = {s: plain_avg[s]['head']['macro_f1']  - replay_avg[('M2','head')][s] for s in subjects}
comparisons['M1_plain-head_vs_M0-head:k5:accuracy'] = paired(da, subjects)
comparisons['M1_plain-head_vs_M0-head:k5:macro_f1'] = paired(df, subjects)

# C: M1_plain head vs none  [positive = head better]
da = {s: plain_avg[s]['head']['accuracy'] - plain_avg[s]['none']['accuracy'] for s in subjects}
df = {s: plain_avg[s]['head']['macro_f1']  - plain_avg[s]['none']['macro_f1']  for s in subjects}
comparisons['M1_plain:head-vs-none:k5:accuracy'] = paired(da, subjects)
comparisons['M1_plain:head-vs-none:k5:macro_f1'] = paired(df, subjects)

# ── 5. Absolute means ──────────────────────────────────────────────────────────
plain_absolute = {}
for metric, mkey in [('accuracy','accuracy'), ('macro_f1','macro_f1')]:
    for method in ('head', 'none'):
        key = 'M1_plain:{}:k5:{}'.format(method, metric)
        vals = {s: plain_avg[s][method][mkey] for s in subjects}
        plain_absolute[key] = {
            'n_subjects': len(vals),
            'mean': float(np.mean(list(vals.values()))),
            'subject_means': {str(s): float(v) for s, v in vals.items()}
        }

replay_reference = {}
for model, method, met in [
    ('M2','none','accuracy'), ('M2','head','accuracy'),
    ('M2','none','macro_f1'), ('M2','head','macro_f1'),
    ('M3','none','accuracy'), ('M3','head','accuracy'),
    ('M3','none','macro_f1'), ('M3','head','macro_f1')]:
    key = '{}:{}:k5:{}'.format(model, method, met)
    vals = replay_avg[(model, method)]
    replay_reference[key] = {
        'n_subjects': len(subjects),
        'mean': float(np.mean([vals[s] for s in subjects])),
        'subject_means': {str(s): float(vals[s]) for s in subjects}
    }

# ── 6. Save ───────────────────────────────────────────────────────────────────
summary = {
    'unit': 'held-out subject; M1_plain avg across 3 repeats; replay from raw complete.json',
    'intervals': 'exploratory, unadjusted 95% t intervals',
    'comparisons': comparisons,
    'plain_absolute': plain_absolute,
    'replay_reference': replay_reference
}
out = 'E:/Python/Study/Koopman_EEG/results/snn_plain_head_k5/summary.json'
with open(out, 'w') as f:
    json.dump(summary, f, indent=2)
print('Saved:', out)

# ── 7. Verified print ─────────────────────────────────────────────────────────
print()
print('=' * 70)
print('{:>18} {:>6} {:>8} {:>8}'.format('Model', 'Method', 'Acc', 'F1'))
print('=' * 70)

def print_row(label, acc, f1v):
    print('{:>18} {:>6} {:8.4f} {:8.4f}'.format(label, '', acc, f1v))

def print_model(label, method, src):
    if src == 'plain':
        acc = plain_absolute['M1_plain:{}:k5:accuracy'.format(method)]['mean']
        f1v = plain_absolute['M1_plain:{}:k5:macro_f1'.format(method)]['mean']
    else:
        src_m = 'M2' if 'M2' in label else 'M3'
        acc = replay_reference['{}:{}:k5:accuracy'.format(src_m, method)]['mean']
        f1v = replay_reference['{}:{}:k5:macro_f1'.format(src_m, method)]['mean']
    print('{:>18} {:>6} {:8.4f} {:8.4f}'.format(label, method, acc, f1v))

print_model('M0 (M2 source)', 'none', 'replay')
print_model('M0 (M2 source)', 'head', 'replay')
print_model('M1 (M3 source)', 'none', 'replay')
print_model('M1 (M3 source)', 'head', 'replay')
print_model('M1_plain', 'none', 'plain')
print_model('M1_plain', 'head', 'plain')

print()
print('=== PAIRED COMPARISONS ===')
for name, c in comparisons.items():
    print('{}'.format(name))
    print('  Delta={:+.4f}  95% CI=[{:+.4f}, {:+.4f}]  pos={}/15  zero={}/15'.format(
        c['mean_delta'], c['ci95_t'][0], c['ci95_t'][1], c['positive'], c['zero']))
    sd = [c['subject_deltas'][str(s)] for s in subjects]
    print('  per-subject: [{}]'.format(', '.join('{:.4f}'.format(v) for v in sd)))
