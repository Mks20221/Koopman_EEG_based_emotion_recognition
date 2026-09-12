"""Train-only preprocessing, trial-disjoint few-shot plans, and local SEED adapter."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt


FEATURE_VERSION = 'ordered-de-v1'
BANDS = ((1,4), (4,8), (8,13), (13,30), (30,45))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    trial: np.ndarray
    sample_id: np.ndarray

    def validate(self):
        n = len(self.X)
        if self.X.ndim != 3 or min(self.X.shape) < 1:
            raise ValueError('X must be nonempty [segments, ordered frames, features]')
        if any(np.asarray(v).shape != (n,) for v in
               (self.y, self.subject, self.session, self.trial, self.sample_id)):
            raise ValueError('Each metadata array must have one entry per segment')
        if not np.isfinite(self.X).all():
            raise ValueError('Nonfinite input features')
        if len(np.unique(self.sample_id)) != n:
            raise ValueError('sample_id must be globally unique')
        if not np.array_equal(np.unique(self.y), np.arange(len(np.unique(self.y)))):
            raise ValueError('Labels must be consecutive integers starting at 0')
        # Each (person, session, trial) has exactly one emotion label.
        for key in set(zip(self.subject.tolist(), self.session.tolist(), self.trial.tolist())):
            mask = (self.subject == key[0]) & (self.session == key[1]) & (self.trial == key[2])
            if len(np.unique(self.y[mask])) != 1:
                raise ValueError(f'Inconsistent labels inside trial {key}')
        return self

    def fingerprint(self):
        h = hashlib.sha256()
        for v in (self.X, self.y, self.subject, self.session, self.trial, self.sample_id.astype('U')):
            a = np.ascontiguousarray(v)
            h.update(str((a.shape, a.dtype.str)).encode())
            h.update(a.tobytes())
        return h.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as p:
        return Dataset(**{k: p[k] for k in Dataset.__dataclass_fields__}).validate()


def save_npz(path, data):
    np.savez_compressed(path, **{k: getattr(data, k) for k in Dataset.__dataclass_fields__})


class TrainScaler:
    def fit(self, X, indices):
        self.mean = X[indices].mean(axis=(0,1), keepdims=True)
        self.scale = np.maximum(X[indices].std(axis=(0,1), keepdims=True), 1e-5)
        return self

    def transform(self, X):
        return np.clip((X-self.mean)/self.scale, -5., 5.).astype(np.float32)


def fold_subjects(population, test, val_count=2, specified=None):
    population = sorted(map(int, population))
    if test not in population:
        raise ValueError('Test subject outside population')
    if specified is not None:
        item = specified[str(test)]
        train, val = list(map(int,item['train'])), list(map(int,item['val']))
    else:
        # Fixed cyclic validation subjects, independent of model/seed/results.
        at = population.index(test)
        val = [population[(at + k + 1) % len(population)] for k in range(val_count)]
        train = [s for s in population if s not in val and s != test]
    if (not train or not val or len(set(train)) != len(train) or len(set(val)) != len(val)
        or set(train)&set(val) or test in train+val
        or set(train+val+[test]) != set(population)):
        raise ValueError('Fold must partition ALL selected people into disjoint train/val/test')
    return train, val


def trial_plan(data, subject, shots=(1,2,3), seed=0, query_trials_per_class=2):
    """Reserve fixed query trials, then nested support pools across shot budgets.
    A shot = ONE labeled segment from ONE distinct trial, per class.
    Unused segments of support trials are excluded from query evaluation.
    """
    support = {int(k): [] for k in shots}
    query, reservations = [], []
    for c in np.unique(data.y):
        # Independent class streams keep query trials fixed even when the
        # requested maximum support budget changes between runs.
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(subject), int(c)]))
        idx = np.flatnonzero((data.subject == subject) & (data.y == c))
        keys = sorted(set(zip(data.session[idx].tolist(), data.trial[idx].tolist())))
        need = max(shots) + query_trials_per_class
        if len(keys) < need:
            raise ValueError(f's{subject}, class {c}: {len(keys)} trials, need {need}; '
                             'reduce --shots or --query-trials explicitly')
        order = rng.permutation(len(keys))
        qkeys = [keys[i] for i in order[:query_trials_per_class]]
        skeys = [keys[i] for i in order[query_trials_per_class:query_trials_per_class+max(shots)]]
        chosen = []
        for sess, trial in skeys:
            candidates = idx[(data.session[idx] == sess) & (data.trial[idx] == trial)]
            chosen.append(int(rng.choice(candidates)))
        for k in support:
            support[k].extend(chosen[:k])
        for sess, trial in qkeys:
            query.extend(idx[(data.session[idx] == sess) & (data.trial[idx] == trial)].tolist())
        reservations.append({'class':int(c), 'support_trials':skeys, 'query_trials':qkeys})
    return {k:np.array(v,dtype=np.int64) for k,v in support.items()}, np.array(query,dtype=np.int64), reservations


def balanced_batches(subjects, rng, per_subject=8, max_steps=0):
    pools = [np.flatnonzero(subjects == s) for s in np.unique(subjects)]
    steps = max_steps or max(int(np.ceil(len(p)/per_subject)) for p in pools)
    for _ in range(steps):
        # All source subjects occur in every batch; no padded fake samples.
        batch = np.concatenate([rng.choice(p, per_subject, replace=len(p)<per_subject) for p in pools])
        yield rng.permutation(batch)


def de_sequence(segs, fs, frame_s=1., hop_s=.5):
    if segs.ndim != 3:
        raise ValueError('Expected raw segments [N, channels, samples]')
    size, hop = int(round(frame_s*fs)), int(round(hop_s*fs))
    if size < 2 or hop < 1 or segs.shape[-1] < size or fs <= 90:
        raise ValueError('Invalid frame, hop, segment length, or sampling rate')
    starts = list(range(0, segs.shape[-1]-size+1, hop))
    out = np.empty((len(segs),len(starts),segs.shape[1],len(BANDS)),np.float32)
    for bi, (lo,hi) in enumerate(BANDS):
        sos = butter(4, (lo,hi), fs=fs, btype='bandpass', output='sos')
        filtered = sosfiltfilt(sos, segs, axis=-1)
        for ti, start in enumerate(starts):
            var = filtered[...,start:start+size].var(-1)
            out[:,ti,:,bi] = .5*np.log(2*np.pi*np.e*np.maximum(var,1e-12))
    return out.reshape(len(segs),len(starts),-1)


def local_seed(args, log):
    # These three modules are supplied by the user's existing project.
    import config
    from preprocess import PreprocConfig, build_segments
    import preprocess, data as source_data
    cfg = PreprocConfig(win_s=args.segment_s, overlap=0., drop_bad_channels=True,
                        reject_abs_thr=args.reject_abs_thr)
    cfg_dict = asdict(cfg) if hasattr(cfg,'__dataclass_fields__') else vars(cfg)
    sources = {m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
               for m in (config,preprocess,source_data)}
    fs = float(config.SEED_FS)
    cache_root = Path(args.feature_cache)
    cache_root.mkdir(parents=True,exist_ok=True)
    blocks = []
    for subject in args.subjects:
        key = digest({'version':FEATURE_VERSION,'sources':sources,'cfg':cfg_dict,'fs':fs,
                      'subject':subject,'session':args.session,'frame':args.frame_s,
                      'hop':args.hop_s,'data_version':args.data_version})
        cache = cache_root / (key + '.npz')
        log(f'data s{subject:02d} session={args.session}: ' + ('cache load' if cache.exists() and not args.refresh_features else 'preprocess start'))
        if cache.exists() and not args.refresh_features:
            block = load_npz(cache)
        else:
            raw = build_segments('seed',subject,args.session,cfg=cfg,use_cache=not args.refresh_features)
            tkey = next((k for k in ('trial','trial_id','trial_ids') if k in raw),None)
            if tkey is None:
                raise KeyError('build_segments must return original trial IDs as trial/trial_id/trial_ids; do not fabricate them from segments')
            segs, labels = np.asarray(raw['segs']), np.asarray(raw['y']).reshape(-1)
            # SEED convention supports {-1,0,1} or already mapped {0,1,2}.
            if set(np.unique(labels)).issubset({-1,0,1}) and -1 in labels:
                labels = labels + 1
            if set(np.unique(labels)) != {0,1,2}:
                raise ValueError(f's{subject}: expected all three SEED classes; observed {np.unique(labels)}')
            log(f'data s{subject:02d}: {len(labels)} segments; ordered DE extraction')
            feats = []
            for start in range(0,len(segs),64):
                feats.append(de_sequence(segs[start:start+64],fs,args.frame_s,args.hop_s))
                log(f'data s{subject:02d}: DE {min(start+64,len(segs))}/{len(segs)}')
            trials = np.asarray(raw[tkey]).reshape(-1).astype(str)
            block = Dataset(np.concatenate(feats),labels.astype(np.int64),
                            np.full(len(labels),subject,np.int64),np.full(len(labels),args.session,np.int64),
                            trials,np.array([f's{subject}:session{args.session}:trial{t}:segment{i}' for i,t in enumerate(trials)])).validate()
            tmp = cache.with_name(cache.stem+'.tmp.npz')
            save_npz(tmp,block)
            tmp.replace(cache)
        blocks.append(block)
    shapes = {b.X.shape[1:] for b in blocks}
    if len(shapes) != 1:
        raise ValueError(f'Feature layouts differ across subjects: {shapes}')
    return Dataset(**{k:np.concatenate([getattr(b,k) for b in blocks]) for k in Dataset.__dataclass_fields__}).validate()


def synthetic_data(seed=314):
    """Small independent trial groups; ONLY for engineering verification."""
    rng = np.random.default_rng(seed)
    arrays = [[] for _ in range(6)]
    for subject in range(1,6):
        offset = rng.normal(0,.25,12)
        for c in range(3):
            for trial in range(5):
                for segment in range(2):
                    x = rng.normal(0,.5,(5,12)) + offset
                    x[:,4*c:4*c+4] += 1.
                    x[:,c] += np.linspace(-.5,.5,5)
                    values = (x,c,subject,1,str(5*c+trial),f's{subject}:t{5*c+trial}:i{segment}')
                    for a,v in zip(arrays,values): a.append(v)
    return Dataset(np.array(arrays[0],np.float32),*(np.asarray(a) for a in arrays[1:])).validate()
