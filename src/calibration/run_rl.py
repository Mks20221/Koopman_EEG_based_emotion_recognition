"""Fixed-budget PPO and four-way LOSO experiment. python -m src.calibration.run_rl"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DEPS = ROOT / '.runtime' / 'rl_deps'
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import torch
import stable_baselines3 as sb3
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from .adapter import train_source_model, SimpleMLP
from .run import load_all_windows, WindowScaler, build_folds
from .data_loader import SEED_EXT_ROOT, _subject_files
from .control import ACTIONS, STATE_NAMES, make_task, run_controller, select_fixed, sync
from .rl_env import AdaptationEnv


CONFIG = dict(
    session=1, hidden=128, source_epochs=50, source_lr=1e-3, source_weight_decay=1e-4,
    seed=42, episodes=512, episode_steps=9, feature='de_movingAve', features=310,
    feedback_trials=list(range(1, 10)), evaluation_trials=list(range(10, 16)),
    budgets=[0, 3, 6, 9], actions=ACTIONS, state=STATE_NAMES,
    dropout='train mode p=0.3 at input and hidden for ALL head updates; eval for prediction/reward',
    source_selection='minimum validation subject trial-equal CE over 50 trained epochs, last tie',
    inner_split='pseudo-subject and next cyclic inner validation excluded; remaining 11 fit model/scaler',
    feedback_order='training permutation per episode; validation/test 1..9',
    source_trial_unit='(subject, trial); each recording trial equal weight',
    algorithm='stable_baselines3.PPO', sb3_version='2.4.1', gymnasium_version='1.0.0',
    ppo=dict(learning_rate=3e-4, n_steps=144, batch_size=72, n_epochs=10,
             gamma=1.0, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
             vf_coef=0.5, max_grad_norm=0.5, normalize_advantage=True,
             target_kl=None),
    policy_architecture=dict(pi=[64, 64], vf=[64, 64]), policy_activation='Tanh',
    checkpoint_interval_episodes=16, checkpoint_rule='max mean validation trial accuracy at 3/6/9; first tie',
    normalization='VecNormalize observations only, training tasks only; frozen checkpoint stats at validation/test',
    reward='trial-equal independent evaluation CE before minus after; unscaled, no reward normalization',
    random_action='uniform five actions; numpy.default_rng(42) freshly initialized each outer fold',
    primary='subject-paired RL_policy minus fixed_head window accuracy at feedback 9',
    interpretation='single-seed exploratory offline SEED feedback simulation',
)


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def parameter_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def snapshot(out):
    files = list((ROOT / 'src' / 'calibration').glob('*.py'))
    files += [ROOT / 'src' / 'config.py', ROOT / 'tests' / 'test_calibration_rl.py',
              ROOT / 'docs' / 'calibration_protocol.md', ROOT / 'docs' / 'rl_calibration_protocol.md',
              ROOT / 'src' / 'calibration' / 'requirements-rl.txt']
    manifest = {}
    for file in files:
        relative = file.relative_to(ROOT)
        target = out / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
        manifest[str(relative)] = sha(file)
    write_json(out / 'source_manifest.json', manifest)
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    diff = subprocess.run(['git', 'diff', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    (out / 'working_tree.diff').write_text(diff.stdout, encoding='utf-8')
    write_json(out / 'environment.json', dict(python=sys.version, torch=torch.__version__,
               sb3=sb3.__version__, gymnasium=gymnasium.__version__, numpy=np.__version__,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               platform=platform.platform(), git_head=git.stdout.strip(), executable=sys.executable))


def get_source(data, train, val, forbidden, path, device, log):
    assert not (set(train) & (set(forbidden) | {val}))
    assert val not in forbidden
    provenance = dict(train_subjects=list(train), scaler_fit_subjects=list(train),
                      early_stop_subject=val, forbidden_subjects=list(forbidden), checkpoint=str(path.name))
    if path.exists():
        saved = torch.load(path, map_location=device)
        assert saved['provenance'] == provenance
        model = SimpleMLP(hidden=128).to(device)
        model.load_state_dict(saved['model_state'])
        scaler = WindowScaler()
        scaler.mean, scaler.scale = saved['scaler_mean'], saved['scaler_scale']
        return model.eval(), scaler, provenance
    torch.manual_seed(42)
    x = np.concatenate([data[s]['X_flat'] for s in train])
    y = np.concatenate([data[s]['y_flat'] for s in train])
    ids = np.concatenate([s * 100 + data[s]['trial_ids_flat'] for s in train])
    assert len(np.unique(ids)) == len(train) * 15
    scaler = WindowScaler().fit(x)
    sync(device)
    start = time.perf_counter()
    model, info = train_source_model(x, y, ids, data[val]['X_flat'], data[val]['y_flat'],
        data[val]['trial_ids_flat'], scaler.mean, scaler.scale, device=device, hidden=128,
        epochs=50, lr=1e-3, weight_decay=1e-4)
    sync(device)
    info['training_seconds'] = time.perf_counter() - start
    info['parameter_hash'] = parameter_hash(model)
    torch.save(dict(model_state=model.state_dict(), scaler_mean=scaler.mean, scaler_scale=scaler.scale,
                    provenance=provenance, train_info=info), path)
    write_json(path.with_suffix('.json'), dict(provenance=provenance, train_info=info, sha256=sha(path)))
    log(f'  source {path.name}: {info["training_seconds"]:.1f}s, selected epoch {info["selected_epoch_one_based"]}')
    return model.eval(), scaler, provenance


def normalized_action(policy, mean, var, state):
    obs = np.clip((state - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32)
    return int(policy.predict(obs, deterministic=True)[0])


def train_policy(tasks, validation, out, device, log):
    env = AdaptationEnv(tasks, device, out / 'training_steps.jsonl')
    vector = VecNormalize(DummyVecEnv([lambda: env]), norm_obs=True, norm_reward=False, clip_obs=10.)
    policy = PPO('MlpPolicy', vector, **CONFIG['ppo'], seed=42, device='cpu',
                 policy_kwargs=dict(net_arch=CONFIG['policy_architecture']), verbose=0)
    initial = {k: v.detach().clone() for k, v in policy.policy.state_dict().items()}
    policy.save(out / 'policy_initial')
    start = time.perf_counter()
    history, best_score, best_episode = [], -np.inf, None
    for block in range(32):
        policy.learn(total_timesteps=144, reset_num_timesteps=False)
        expected = (block + 1) * 16
        assert env.completed == expected and policy.num_timesteps == expected * 9
        mean, var = vector.obs_rms.mean.copy(), vector.obs_rms.var.copy()
        score = run_controller(validation,
            lambda state, fb: normalized_action(policy, mean, var, state), device)['score']
        record = dict(episodes=expected, timesteps=policy.num_timesteps, validation_score=score,
                      recent_reward=float(np.mean([e['reward'] for e in env.episode_returns[-16:]])),
                      elapsed_seconds=time.perf_counter()-start,
                      ppo_updates=int(policy._n_updates),
                      train_metrics={k:float(v) for k,v in policy.logger.name_to_value.items()
                                     if k.startswith('train/') and np.isscalar(v)})
        history.append(record)
        if score > best_score:
            best_score, best_episode = score, expected
            policy.save(out / 'policy_best')
            vector.save(out / 'normalizer_best.pkl')
            np.savez(out / 'normalizer_best.npz', mean=mean, var=var, count=vector.obs_rms.count)
        write_json(out / 'policy_selection.json', history)
        log(f'  PPO {expected}/512 episodes; validation={score:.4f}; best={best_score:.4f} @ {best_episode}; {record["elapsed_seconds"]:.0f}s')
    policy.save(out / 'policy_last')
    vector.save(out / 'normalizer_last.pkl')
    write_json(out / 'training_rewards.json', env.episode_returns)
    delta = sum(float((v.detach()-initial[k]).square().sum()) for k,v in policy.policy.state_dict().items()) ** .5
    assert delta > 0 and env.completed == 512 and policy.num_timesteps == 4608
    training = dict(episodes=env.completed, timesteps=policy.num_timesteps,
                    parameter_l2_change=delta, best_episode=best_episode, best_validation_score=best_score,
                    seconds=time.perf_counter()-start, ppo_updates=int(policy._n_updates),
                    classifier_update_steps=sum(e['update_steps'] for e in env.episode_returns),
                    policy_sha256=sha(out/'policy_best.zip'),
                    normalizer_sha256=sha(out/'normalizer_best.npz'))
    write_json(out / 'policy_training.json', training)
    vector.close()
    return training


def run_fold(fold, data, out, device, log):
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    test, val, train = fold['test'], fold['val'], fold['train']
    write_json(out / 'split.json', fold)
    source, scaler, provenance = get_source(data, train, val, [test], out/'source_outer.pt', device, log)
    validation = make_task(val, data[val], source, scaler, device, provenance)
    fixed_start = time.perf_counter()
    fixed, candidates = select_fixed(validation, device)
    fixed_selection_seconds = time.perf_counter()-fixed_start
    write_json(out/'fixed_selection.json', dict(selected=fixed, candidates=candidates,
                                             seconds=fixed_selection_seconds))
    tasks = []
    for i, pseudo in enumerate(train):
        inner_val = train[(i + 1) % len(train)]
        inner_train = [s for s in train if s not in (pseudo, inner_val)]
        model, inner_scaler, inner_prov = get_source(data, inner_train, inner_val, [test, val, pseudo],
            out/f'source_pseudo_{pseudo:02d}.pt', device, log)
        tasks.append(make_task(pseudo, data[pseudo], model, inner_scaler, device, inner_prov))
    write_json(out/'source_mapping.json', dict(outer=provenance,
               pseudo_tasks=[dict(pseudo_subject=t.subject, **t.provenance) for t in tasks]))
    training = train_policy(tasks, validation, out, device, log)
    selected = PPO.load(out/'policy_best.zip', device='cpu')
    selected.policy.set_training_mode(False)
    for param in selected.policy.parameters():
        param.requires_grad_(False)
    with np.load(out/'normalizer_best.npz') as norm:
        mean, var = norm['mean'].copy(), norm['var'].copy()
    policy_hash = parameter_hash(selected.policy)
    # Only now build the final test task. No test signal is used in training/selection.
    test_task = make_task(test, data[test], source, scaler, device, provenance)
    random = np.random.default_rng(42)
    choosers = {'none': lambda state, fb: 0,
                'fixed_head': lambda state, fb: fixed['action'],
                'random_action': lambda state, fb: int(random.integers(5)),
                'RL_policy': lambda state, fb: normalized_action(selected, mean, var, state)}
    records, predictions, costs = [], [], []
    for method, chooser in choosers.items():
        result = run_controller(test_task, chooser, device, method=method, save_predictions=True)
        for record in result['records']:
            if method == 'RL_policy':
                record['normalized_state'] = np.clip((np.asarray(record['state'])-mean)/np.sqrt(var+1e-8),-10,10).tolist()
        records.extend(result['records'])
        predictions.extend(result['predictions'])
        costs.append(dict(subject=test, method=method, update_steps=result['update_steps'],
                          update_seconds=result['update_seconds'], decision_seconds=result['decision_seconds']))
        log(f'  test s{test} {method}: {result["update_steps"]} updates (metrics reserved for final report)')
    assert parameter_hash(selected.policy) == policy_hash
    assert all(t.subject in train for t in tasks)
    write_json(out/'feedback_records.json', records)
    write_json(out/'costs.json', costs)
    with open(out/'predictions.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    # First-fold acceptance gates are also enforced on every subsequent fold.
    assert len(predictions) == 4 * 4 * sum(len(t.y) for t in test_task.evaluation)
    assert all(len(r['visible_trials']) == r['feedback'] for r in records)
    finished = dict(subject=test, seconds=time.perf_counter()-start, complete=True,
                    policy_training=training, frozen_test_policy_hash=policy_hash,
                    predictions_sha256=sha(out/'predictions.csv'))
    write_json(out/'complete.json', finished)
    return finished


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--subjects', type=int, nargs='+', default=list(range(1,16)))
    args = parser.parse_args()
    assert sb3.__version__ == CONFIG['sb3_version'] and gymnasium.__version__ == CONFIG['gymnasium_version']
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    out = args.out or ROOT/'results'/'calibration'/('rl_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    config = dict(CONFIG, device=args.device, subjects=args.subjects, torch_threads=1)
    if args.resume:
        assert json.loads((out/'config.json').read_text(encoding='utf-8')) == json.loads(json.dumps(config))
        manifest = json.loads((out/'source_manifest.json').read_text(encoding='utf-8'))
        assert all(sha(ROOT/p) == digest for p,digest in manifest.items()), 'Source changed: resume refused'
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/'config.json', config)
        snapshot(out)
    log_file = open(out/'run.log','a',encoding='utf-8', buffering=1)
    def log(message):
        line = f'[{datetime.now().isoformat(timespec="seconds")}] {message}'
        print(line, flush=True)
        log_file.write(line+'\n')
    log(f'Start {out}; exact 512 complete episodes per fold, seed 42')
    completed = []
    try:
        data = load_all_windows(1)
        files = [Path(SEED_EXT_ROOT)/_subject_files(s)[0] for s in range(1,16)]
        files.append(Path(SEED_EXT_ROOT)/'label.mat')
        write_json(out/'data_manifest.json', {str(p):dict(size=p.stat().st_size, sha256=sha(p)) for p in files})
        folds = build_folds(list(range(1,16)))
        for fold in folds:
            if fold['test'] not in args.subjects:
                continue
            fold_out = out/f'fold_{fold["test"]:02d}'
            if args.resume and (fold_out/'complete.json').exists():
                result = json.loads((fold_out/'complete.json').read_text())
            else:
                log(f'Fold {fold["test"]}/15, train={fold["train"]}, val={fold["val"]}, test={fold["test"]}')
                result = run_fold(fold, data, fold_out, args.device, log)
            completed.append(result)
            estimate = np.mean([r['seconds'] for r in completed])*(len(args.subjects)-len(completed))
            write_json(out/'status.json', dict(complete=False, completed_subjects=[r['subject'] for r in completed],
                        remaining_seconds_estimate=estimate))
            log(f'Completed fold {fold["test"]}: {result["seconds"]:.1f}s; estimated remaining {estimate/60:.1f} min')
        from .report_rl import build_report
        build_report(out)
        write_json(out/'status.json', dict(complete=len(completed)==15,
                    completed_subjects=[r['subject'] for r in completed], requested_subjects=args.subjects))
        log(f'Finished {len(completed)}/15 folds; prediction-recomputed report saved.')
    except BaseException as exc:
        write_json(out/'failure.json', dict(error=repr(exc), completed_subjects=[r['subject'] for r in completed]))
        raise
    finally:
        log_file.close()


if __name__ == '__main__':
    main()
