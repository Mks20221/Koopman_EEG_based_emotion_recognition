"""Recompute all research metrics from actual saved window probabilities."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .control import METHODS, BUDGETS


def metrics(frame):
    p = frame[['p0','p1','p2']].to_numpy()
    assert np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-6)
    pred = p.argmax(1)
    assert np.array_equal(pred, frame.y_pred)
    grouped = frame.groupby('trial')
    assert (grouped.y_true.nunique() == 1).all()
    trial_p = grouped[['p0','p1','p2']].mean()
    trial_y = grouped.y_true.first()
    return dict(window_accuracy=float(accuracy_score(frame.y_true, pred)),
                window_macro_f1=float(f1_score(frame.y_true, pred, labels=[0,1,2], average='macro', zero_division=0)),
                trial_accuracy=float(accuracy_score(trial_y, trial_p.to_numpy().argmax(1))),
                trial_macro_f1=float(f1_score(trial_y, trial_p.to_numpy().argmax(1), labels=[0,1,2], average='macro', zero_division=0)))


def build_report(out):
    out = Path(out)
    rows, costs, trainings, all_predictions = [], [], [], []
    completed = sorted(p.parent for p in out.glob('fold_*/complete.json'))
    for fold in completed:
        frame = pd.read_csv(fold/'predictions.csv')
        subject = int(frame.subject.iloc[0])
        assert frame.subject.nunique() == 1
        assert not frame.duplicated(['subject','method','feedback','trial','window']).any()
        reference = frame[(frame.method=='none') & (frame.feedback==0)].sort_values(['trial','window'])
        assert set(reference.trial) == set(range(10,16))
        assert set(frame.method) == set(METHODS) and set(frame.feedback) == set(BUDGETS)
        for method in METHODS:
            for budget in BUDGETS:
                part = frame[(frame.method==method) & (frame.feedback==budget)].sort_values(['trial','window'])
                assert np.array_equal(part[['trial','window','y_true']].to_numpy(),
                                      reference[['trial','window','y_true']].to_numpy())
                if budget == 0 or method == 'none':
                    assert np.allclose(part[['p0','p1','p2']], reference[['p0','p1','p2']], rtol=0, atol=0)
                rows.append(dict(subject=subject,method=method,feedback=budget,**metrics(part)))
        all_predictions.append(frame)
        costs.extend(json.loads((fold/'costs.json').read_text()))
        training = json.loads((fold/'policy_training.json').read_text())
        assert training['episodes']==512 and training['timesteps']==4608 and training['parameter_l2_change']>0
        training['subject'] = subject
        training['pseudo_source_training_seconds'] = sum(json.loads(p.read_text())['train_info']['training_seconds']
                                                         for p in fold.glob('source_pseudo_*.json'))
        training['outer_source_training_seconds'] = json.loads((fold/'source_outer.json').read_text())['train_info']['training_seconds']
        training['fixed_selection_seconds'] = json.loads((fold/'fixed_selection.json').read_text())['seconds']
        trainings.append(training)
    df = pd.DataFrame(rows)
    cost_df = pd.DataFrame(costs)
    df.to_csv(out/'metrics_from_predictions.csv', index=False)
    pd.concat(all_predictions,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    cost_df.to_csv(out/'costs.csv',index=False)
    pd.DataFrame(trainings).to_csv(out/'training_costs.csv',index=False)
    measures = ['window_accuracy','window_macro_f1','trial_accuracy','trial_macro_f1']
    table = df.groupby(['feedback','method'])[measures].mean().reset_index()
    table.to_csv(out/'comparison_from_predictions.csv',index=False)
    primary = df[df.feedback==9].pivot(index='subject', columns='method', values='window_accuracy')
    primary['difference'] = primary.RL_policy-primary.fixed_head
    cost_steps = cost_df.pivot(index='subject',columns='method',values='update_steps')
    primary['fixed_steps'], primary['rl_steps'] = cost_steps.fixed_head,cost_steps.RL_policy
    primary['extra_steps'] = primary.rl_steps-primary.fixed_steps
    primary.to_csv(out/'paired_primary.csv')
    diff = primary.difference.to_numpy()
    n = len(diff)
    ci = stats.t.interval(.95, n-1, loc=diff.mean(), scale=stats.sem(diff)) if n>1 and np.std(diff)>0 else (float(diff.mean()),float(diff.mean()))
    summary = dict(n_subjects=n, complete_loso=n==15, seed=42,
                   primary=dict(mean_difference=float(diff.mean()), ci95=list(map(float,ci)),
                                positive_subjects=primary.index[diff>0].tolist(),
                                tied_subjects=primary.index[diff==0].tolist(),
                                negative_subjects=primary.index[diff<0].tolist(),
                                paired_t_p=float(stats.ttest_1samp(diff,0).pvalue) if n>1 else None),
                   mean_metrics=table.to_dict('records'),
                   mean_costs=cost_df.groupby('method')[['update_steps','update_seconds','decision_seconds']].mean().reset_index().to_dict('records'),
                   rl_training_seconds=sum(t['seconds']+t['pseudo_source_training_seconds'] for t in trainings),
                   rl_training_head_steps=sum(t['classifier_update_steps'] for t in trainings),
                   provenance='metrics recomputed from predictions.csv; paired unit = subject; exploratory unadjusted 95% t CI')
    (out/'summary_from_predictions.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines = ['# RL-controlled calibration: prediction-recomputed results', '',
             f'Completed {n}/15 subjects. SEED session 1, seed 42; offline simulated label feedback.',
             'Single seed exploratory study. All metrics below are subject means. Historical baseline scores are not reused.', '',
             '| Feedback | Method | Window accuracy | Window Macro-F1 | Trial accuracy | Trial Macro-F1 |',
             '|---:|---|---:|---:|---:|---:|']
    for b in BUDGETS:
        for m in METHODS:
            r=table[(table.feedback==b)&(table.method==m)].iloc[0]
            lines.append(f'| {b} | {m} | '+ ' | '.join(f'{100*r[k]:.2f}%' for k in measures)+' |')
    lines += ['',f'Primary RL_policy - fixed_head window accuracy at feedback 9: **{100*diff.mean():+.2f} percentage points**.',
              f'Exploratory paired 95% t CI: [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] pp (n={n}).',
              'A confidence interval crossing zero does not establish equivalence. Training reward is not a success criterion.', '',
              '| Subject | Fixed window acc | RL window acc | Difference (pp) | Fixed steps | RL steps |',
              '|---:|---:|---:|---:|---:|---:|']
    for s,r in primary.iterrows():
        lines.append(f'| {s} | {100*r.fixed_head:.2f}% | {100*r.RL_policy:.2f}% | {100*r.difference:+.2f} | {int(r.fixed_steps)} | {int(r.rl_steps)} |')
    lines += ['', '| Method | Mean target update steps | Mean update seconds | Mean prediction/state/action seconds |',
              '|---|---:|---:|---:|']
    for r in summary['mean_costs']:
        lines.append(f'| {r["method"]} | {r["update_steps"]:.2f} | {r["update_seconds"]:.3f} | {r["decision_seconds"]:.3f} |')
    lines += ['',f'Additional RL training: {summary["rl_training_seconds"]/60:.2f} minutes, including pseudo-source training and policy validation; '
              f'{summary["rl_training_head_steps"]} classifier head gradient steps in reward episodes (validation steps additional).',
              'Shared outer source training and fixed hyperparameter selection costs are in training_costs.csv.',
              'Target update/decision timers exclude evaluation-set scoring and source loading; CUDA is synchronized at timer boundaries.',
              'All adaptation methods retain Adam state and use the same cumulative trial-weighted update with train-mode dropout p=0.3.',
              'The policy is frozen on the final test subject. Test evaluation labels are used only to compute reports, never rewards or actions.',
              '', 'Artifacts: source_snapshot/, config.json, fold_*/source_mapping.json, source_*.pt (model + scaler),',
              'policy_best.zip + normalizer_best.npz, policy_selection.json, training_steps.jsonl, feedback_records.json,',
              'predictions.csv, metrics_from_predictions.csv, paired_primary.csv and costs.csv.',
              '', 'PPO implementation: https://stable-baselines3.readthedocs.io/en/v2.4.1/modules/ppo.html']
    (out/'report_from_predictions.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(3,5,figsize=(16,8),sharex=True)
    for ax,fold in zip(axes.flat,completed):
        rewards=json.loads((fold/'training_rewards.json').read_text())
        values=np.array([r['reward'] for r in rewards])
        ax.plot(np.arange(1,len(values)+1),values,alpha=.18,lw=.5)
        ax.plot(np.arange(16,len(values)+1),np.convolve(values,np.ones(16)/16,mode='valid'))
        ax.axhline(0,color='black',lw=.5)
        ax.set_title(fold.name)
        ax.set_xlabel('Training episode')
        ax.set_ylabel('CE reduction')
    fig.suptitle('Training reward (16-episode moving mean); not a test metric')
    fig.tight_layout()
    fig.savefig(out/'training_reward_curves.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric in zip(axes,['window_accuracy','trial_accuracy']):
        for method in METHODS:
            sub=table[table.method==method].sort_values('feedback')
            ax.plot(sub.feedback,sub[metric]*100,marker='o',label=method)
        ax.set(xlabel='Labeled feedback trials',ylabel=metric+' (%)',xticks=BUDGETS)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out/'comparison_curves.png',dpi=160)
    plt.close(fig)
    return summary


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('out',type=Path)
    build_report(parser.parse_args().out)
