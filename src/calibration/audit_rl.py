"""Post-run audit of saved checkpoints, raw-data scalers, logs and PPO step counts.

This module does not train or change experiment configuration.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from .run_rl import ROOT, PPO, sha, parameter_hash, write_json
from .run import load_all_windows, WindowScaler
from .adapter import SimpleMLP
from .control import make_task, evaluation_loss


def audit(out, device='cpu'):
    out=Path(out)
    torch.set_num_threads(1)
    data=load_all_windows(1)
    manifest=json.loads((out/'source_manifest.json').read_text(encoding='utf-8'))
    source_match={name:sha(ROOT/name)==digest for name,digest in manifest.items()}
    assert all(source_match.values()),'Training source differs from frozen snapshot'
    data_manifest=json.loads((out/'data_manifest.json').read_text())
    assert all(sha(path)==v['sha256'] for path,v in data_manifest.items())
    fold_audits=[]
    for folder in sorted(p.parent for p in out.glob('fold_*/complete.json')):
        split=json.loads((folder/'split.json').read_text())
        source_audits=[]
        for file in sorted(folder.glob('source_*.pt')):
            saved=torch.load(file,map_location=device)
            prov=saved['provenance']
            train,val,forbidden=prov['train_subjects'],prov['early_stop_subject'],prov['forbidden_subjects']
            assert set(train).isdisjoint(set(forbidden)|{val}) and val not in forbidden
            assert prov['scaler_fit_subjects']==train
            assert split['test'] in forbidden
            if 'pseudo' in file.name:
                pseudo=int(file.stem.split('_')[-1])
                assert pseudo in split['train'] and pseudo in forbidden
                assert split['val'] in forbidden and val in split['train']
                assert len(train)==11
            else:
                assert train==split['train'] and val==split['val'] and len(train)==13
            x=np.concatenate([data[s]['X_flat'] for s in train])
            np.testing.assert_array_equal(saved['scaler_mean'],x.mean(0))
            np.testing.assert_array_equal(saved['scaler_scale'],np.maximum(x.std(0),1e-8))
            model=SimpleMLP().to(device)
            model.load_state_dict(saved['model_state'])
            model.eval()
            assert parameter_hash(model)==saved['train_info']['parameter_hash']
            scaler=WindowScaler()
            scaler.mean,scaler.scale=saved['scaler_mean'],saved['scaler_scale']
            task=make_task(val,data[val],model,scaler,device,prov)
            ce=evaluation_loss(model,task.feedback+task.evaluation)
            assert abs(ce-saved['train_info']['best_val_loss'])<2e-5
            source_audits.append(dict(checkpoint=file.name,scaler_exact_match=True,
                                     saved_validation_ce_error=abs(ce-saved['train_info']['best_val_loss'])))
        assert len(source_audits)==14
        rows=[json.loads(line) for line in (folder/'training_steps.jsonl').read_text().splitlines()]
        assert len(rows)==4608
        for episode in range(1,513):
            part=rows[(episode-1)*9:episode*9]
            assert [r['episode'] for r in part]==[episode]*9
            assert [r['feedback'] for r in part]==list(range(1,10))
            assert sorted(r['trial'] for r in part)==list(range(1,10))
            assert all(r['pseudo_subject'] in split['train'] for r in part)
            assert abs(sum(r['reward'] for r in part)-(part[0]['eval_ce_before']-part[-1]['eval_ce_after']))<1e-6
            for r in part:
                if r['action']==0:
                    assert r['reward']==0 and r['update_steps']==0
        selection=json.loads((folder/'policy_selection.json').read_text())
        assert len(selection)==32 and selection[-1]['timesteps']==4608
        train_info=json.loads((folder/'policy_training.json').read_text())
        best=max(selection,key=lambda r:r['validation_score'])
        assert best['episodes']==train_info['best_episode']
        initial=PPO.load(folder/'policy_initial.zip',device='cpu')
        selected=PPO.load(folder/'policy_best.zip',device='cpu')
        last=PPO.load(folder/'policy_last.zip',device='cpu')
        def actor_delta(model):
            a=initial.policy.state_dict(); b=model.policy.state_dict()
            keys=[k for k in a if 'policy_net' in k or 'action_net' in k]
            return sum(float((a[k]-b[k]).square().sum()) for k in keys)**.5
        assert actor_delta(selected)>0 and actor_delta(last)>0
        counts=sorted(set(int(v['step']) for v in last.policy.optimizer.state.values()))
        assert counts==[640] and last.num_timesteps==4608 and last._n_updates==320
        test_records=json.loads((folder/'feedback_records.json').read_text())
        for r in test_records:
            assert len(r['state'])==15 and len(r['visible_trials'])==r['feedback']
            assert r['visible_trials']==list(range(1,r['feedback']+1))
        fold_audits.append(dict(subject=split['test'],sources=source_audits,environment_steps=len(rows),
                               complete_episodes=512,ppo_training_rounds=32,optimization_epochs=320,
                               policy_adam_steps=640,selected_actor_l2_change=actor_delta(selected),
                               final_actor_l2_change=actor_delta(last)))
        print(f'AUDIT PASS {folder.name}: 14 source/scaler matches; 512x9 steps; actor trained; 640 Adam steps',flush=True)
    result=dict(passed=True,completed_folds=len(fold_audits),complete_loso=len(fold_audits)==15,
                source_snapshot_matches=source_match,data_hashes_match=True,folds=fold_audits)
    write_json(out/'artifact_audit.json',result)
    target=out/'audit_source'
    target.mkdir(exist_ok=True)
    shutil.copy2(__file__,target/'audit_rl.py')
    write_json(target/'manifest.json',dict(file='audit_rl.py',sha256=sha(__file__),
               role='post-run artifact verification, not used to train or choose models'))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('out',type=Path)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    audit(args.out,args.device)
