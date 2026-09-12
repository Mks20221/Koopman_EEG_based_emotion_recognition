"""Repaired executable SNN experiment. Run from project root: python -u -m src.snn_baseline --help.
M0=fixed encoder; M1=learned encoder+group risk+cost penalty;
M2=source-learned intrinsic adaptation of M0; M3=the same adaptation of M1.
Only engineering smoke tests use --synthetic. No real-data claims are embedded.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import platform
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats as scipy_stats

# Compatible with both python -m src.snn_baseline and python src/snn_baseline.py.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from snn_core import SNN, IntrinsicAdapter, group_objective, prototype_loss
from snn_protocol import (Dataset, TrainScaler, atomic_json, digest, fold_subjects,
                          trial_plan, balanced_batches, local_seed, load_npz, synthetic_data)

VERSION = 'snn-repair-1.0'


class Progress:
    def __init__(self, path):
        self.path, self.stage = Path(path), 'startup'
        self.lock, self.done = threading.Lock(), threading.Event()
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()

    def __call__(self, message):
        line = f'[{time.strftime("%H:%M:%S")}] {message}'
        with self.lock:
            print(line, flush=True)
            with self.path.open('a', encoding='utf-8') as f: f.write(line+'\n')

    def heartbeat(self):
        while not self.done.wait(30):
            self(f'heartbeat: {self.stage}; process elapsed={time.monotonic()-self.started:.0f}s')

    def close(self):
        self.done.set()
        self.thread.join(timeout=1)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def metrics(y, prediction, classes):
    return {'accuracy':float(accuracy_score(y,prediction)),
            'macro_f1':float(f1_score(y,prediction,labels=list(range(classes)),average='macro',zero_division=0))}


def tensor_batch(X, y, indices, device):
    return (torch.as_tensor(X[indices],dtype=torch.float32,device=device),
            torch.as_tensor(y[indices],dtype=torch.long,device=device))


def evaluate(model, X, y, indices, device, batch_size=128, adapter=None, q=None):
    model.eval()
    predicted, total, ce_total = [], {}, 0.
    with torch.no_grad():
        for start in range(0,len(indices),batch_size):
            ix = indices[start:start+batch_size]
            xb, yb = tensor_batch(X,y,ix,device)
            logits, _, counters = model(xb,adapter,q)
            predicted.extend(logits.argmax(-1).cpu().tolist())
            ce_total += F.cross_entropy(logits,yb,reduction='sum').item()
            for k,v in counters.items(): total[k] = total.get(k,0.) + v.sum().item()
    if not len(indices): raise ValueError('Cannot evaluate empty data')
    result = metrics(y[indices],predicted,model.classes)
    result.update({k:v/len(indices) for k,v in total.items()})
    result['ce'] = ce_total/len(indices)
    return result, np.asarray(predicted,dtype=np.int64)


def subject_validation(model, X, data, people, device, batch_size, adapter=None, plans=None, fit_cfg=None):
    results = []
    for s in people:
        if adapter is None:
            idx = np.flatnonzero(data.subject == s)
            result, _ = evaluate(model,X,data.y,idx,device,batch_size)
        else:
            support, query, _ = plans[s]
            ix = support[next(iter(support))]
            q = fit_q(model,adapter,X,data.y,ix,device,**fit_cfg)
            result, _ = evaluate(model,X,data.y,query,device,batch_size,adapter,q)
        results.append(result)
    return {k:float(np.mean([r[k] for r in results])) for k in results[0]}


def train_source(args, data, X, train_people, val_people, name, seed, device, log):
    # Reseed BEFORE construction; M0/M1 start with equal shared weights.
    seed_all(seed)
    model = SNN(X.shape[-1],len(np.unique(data.y)),args.hidden,args.latent,
                name != 'M0',args.coding_steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    tr = np.flatnonzero(np.isin(data.subject,train_people))
    rng = np.random.default_rng(seed)
    best_score, best_state, history = float('inf'), None, []
    for epoch in range(1,args.epochs+1):
        log.stage = f'{name} seed={seed} epoch={epoch}/{args.epochs}'
        model.train()
        losses, start = [], time.monotonic()
        for relative in balanced_batches(data.subject[tr],rng,args.per_subject,args.steps_per_epoch):
            ix = tr[relative]
            xb,yb = tensor_batch(X,data.y,ix,device)
            sb = torch.as_tensor(data.subject[ix],device=device)
            logits,_,counts = model(xb)
            loss = group_objective(logits,yb,sb,counts['event_fraction'],name=='M1',
                                   args.budget,args.budget_weight,args.group_temperature)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            losses.append(loss.item())
        val = subject_validation(model,X,data,val_people,device,args.batch_size)
        # Equal-subject validation CE for every source model; no test data selection.
        if val['ce'] < best_score:
            best_score, best_state = val['ce'],copy.deepcopy(model.state_dict())
            best_epoch = epoch
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'val':val})
        log(f'{name} epoch {epoch}/{args.epochs} loss={np.mean(losses):.4f} '
            f'valCE={val["ce"]:.4f} valAcc={val["accuracy"]:.4f} elapsed={time.monotonic()-start:.1f}s')
    model.load_state_dict(best_state)
    model.eval()
    return model, {'best_epoch':best_epoch,'selection':'mean validation-subject CE','history':history}


def freeze(model):
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)


def source_prototypes(model,X,data,train_people,device,batch_size):
    ix = np.flatnonzero(np.isin(data.subject,train_people))
    means = []
    model.eval()
    with torch.no_grad():
        # Equal-subject class prototypes, not dominated by people with more segments.
        for s in train_people:
            rows = []
            for c in range(model.classes):
                ci = ix[(data.subject[ix]==s)&(data.y[ix]==c)]
                if not len(ci): raise ValueError(f'Train subject {s} has no class {c}')
                features = []
                for at in range(0,len(ci),batch_size):
                    xb,_ = tensor_batch(X,data.y,ci[at:at+batch_size],device)
                    features.append(model(xb)[1])
                rows.append(torch.cat(features).mean(0))
            means.append(torch.stack(rows))
    return torch.stack(means).mean(0).detach()


def fit_q(model,adapter,X,y,support,device,steps,lr,prototypes,proto_weight,q_reg,
          create_graph=False):
    # No query argument: query labels cannot enter target fitting.
    xb,yb = tensor_batch(X,y,support,device)
    q = torch.zeros(adapter.q_dim,device=device,requires_grad=True)
    for _ in range(steps):
        logits,representation,_ = model(xb,adapter,q)
        objective = F.cross_entropy(logits,yb) + q_reg*q.square().mean()
        if proto_weight:
            objective = objective + proto_weight*prototype_loss(representation,yb,prototypes)
        grad, = torch.autograd.grad(objective,q,create_graph=create_graph)
        q = (q-lr*grad).clamp(-2.,2.)
        if not create_graph: q = q.detach().requires_grad_(True)
    return q if create_graph else q.detach()


def train_adapter(args,model,X,data,train_people,val_people,seed,device,log):
    freeze(model)
    seed_all(seed+1701)
    adapter = IntrinsicAdapter(model.hidden,model.latent,args.q_dim).to(device)
    prototypes = source_prototypes(model,X,data,train_people,device,args.batch_size)
    optimizer = torch.optim.Adam(adapter.parameters(),lr=args.meta_lr)
    rng = np.random.default_rng(seed+1701)
    plans = {s:trial_plan(data,s,(args.meta_shots,),seed+7000+s,args.query_trials) for s in val_people}
    fit_cfg = dict(steps=args.meta_inner_steps,lr=args.meta_inner_lr,prototypes=prototypes,
                   proto_weight=args.proto_weight,q_reg=args.q_reg)
    best_state, best_ce, history = None, float('inf'), []
    for epoch in range(1,args.meta_epochs+1):
        log.stage = f'meta epoch={epoch}/{args.meta_epochs}'
        losses=[]
        for episode in range(args.meta_episodes):
            person = int(rng.choice(train_people))
            supports,query,_ = trial_plan(data,person,(args.meta_shots,),int(rng.integers(2**31)),args.query_trials)
            # Trial-disjoint query sampled by class to keep meta graphs small.
            query = np.concatenate([rng.choice(query[data.y[query]==c],
                    min(args.meta_query_per_class,np.sum(data.y[query]==c)),replace=False)
                    for c in range(model.classes)])
            q = fit_q(model,adapter,X,data.y,supports[args.meta_shots],device,create_graph=True,**fit_cfg)
            xb,yb = tensor_batch(X,data.y,query,device)
            logits,rep,_ = model(xb,adapter,q)
            loss = F.cross_entropy(logits,yb) + args.proto_weight*prototype_loss(rep,yb,prototypes)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(),5.)
            optimizer.step()
            losses.append(loss.item())
        val = subject_validation(model,X,data,val_people,device,args.batch_size,adapter,plans,fit_cfg)
        if val['ce'] < best_ce:
            best_ce,best_state,best_epoch=val['ce'],copy.deepcopy(adapter.state_dict()),epoch
        history.append({'epoch':epoch,'meta_query_loss':float(np.mean(losses)),'val':val})
        log(f'meta epoch {epoch}/{args.meta_epochs} queryLoss={np.mean(losses):.4f} valCE={val["ce"]:.4f}')
    adapter.load_state_dict(best_state)
    # Only q is trainable during target calibration.
    for p in adapter.parameters(): p.requires_grad_(False)
    return adapter,prototypes,{'best_epoch':best_epoch,'history':history}


def fit_head(model,X,y,support,device,steps,lr):
    """Standard target classifier-only tuning control, same labeled support."""
    candidate = copy.deepcopy(model)
    freeze(candidate)
    for p in candidate.head.parameters(): p.requires_grad_(True)
    xb,yb = tensor_batch(X,y,support,device)
    with torch.no_grad(): rep = candidate(xb)[1]
    optimizer = torch.optim.SGD(candidate.head.parameters(),lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(candidate.head(rep),yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    freeze(candidate)
    return candidate


def select_adaptation(args,model,adapter,prototypes,X,data,val_people,seed,device,log):
    # Select steps/lr on validation subjects and their trial-disjoint queries.
    # Never select using the final test subject.
    plans = {s:trial_plan(data,s,(args.meta_shots,),seed+9000+s,args.query_trials) for s in val_people}
    choices, table = {}, []
    for method in ('lowdim','lowdim_no_proto','head'):
        best, choice = float('inf'), None
        for steps in args.adapt_steps:
            for lr in args.adapt_lrs:
                scores=[]
                for s,(supports,query,_) in plans.items():
                    sup=supports[args.meta_shots]
                    if method=='head':
                        tuned=fit_head(model,X,data.y,sup,device,steps,lr)
                        score,_=evaluate(tuned,X,data.y,query,device,args.batch_size)
                    else:
                        q=fit_q(model,adapter,X,data.y,sup,device,steps,lr,prototypes,
                                args.proto_weight if method=='lowdim' else 0.,args.q_reg)
                        score,_=evaluate(model,X,data.y,query,device,args.batch_size,adapter,q)
                    scores.append(score['ce'])
                value=float(np.mean(scores))
                table.append({'method':method,'steps':steps,'lr':lr,'val_ce':value})
                if value<best: best,choice=value,{'steps':steps,'lr':lr}
        choices[method]=choice
        log(f'validation choice {method}: {choice}, CE={best:.4f}')
    return choices,table


def prediction_file(path,data,indices,predicted):
    np.savez_compressed(path,sample_id=data.sample_id[indices],y_true=data.y[indices],
                        y_pred=predicted,subject=data.subject[indices],session=data.session[indices],
                        trial=data.trial[indices])


def run_fold(args,data,test,seed,out,device,log,specified):
    train_people,val_people=fold_subjects(np.unique(data.subject),test,2,specified)
    tr=np.flatnonzero(np.isin(data.subject,train_people))
    scaler=TrainScaler().fit(data.X,tr)
    X=scaler.transform(data.X)
    log(f'fold s{test:02d} seed={seed}: train_people={train_people}, val_people={val_people}, '
        f'test={test}; train_segments={len(tr)}')
    if set(data.y[tr]) != set(np.unique(data.y)): raise ValueError('Source training lacks a class')
    fold={'test_subject':int(test),'seed':seed,'train_subjects':train_people,'val_subjects':val_people,
          'rows':[],'source_training':{},'adaptation':{}}
    np.savez_compressed(out/'scaler.npz',mean=scaler.mean,scale=scaler.scale)
    sources = ['M0','M1'] if args.models==['all'] else sorted(set(
        {'M2':'M0','M3':'M1'}.get(m,m) for m in args.models))
    adapt_names = {'M0':'M2','M1':'M3'}
    for base in sources:
        model,trace=train_source(args,data,X,train_people,val_people,base,seed,device,log)
        torch.save(model.state_dict(),out/f'{base}_source.pt')
        atomic_json(out/f'{base}_training.json',trace)
        fold['source_training'][base]={'best_epoch':trace['best_epoch']}
        all_test=np.flatnonzero(data.subject==test)
        score,pred=evaluate(model,X,data.y,all_test,device,args.batch_size)
        prediction_file(out/f'{base}_all_test.npz',data,all_test,pred)
        fold['rows'].append({'model':base,'phase':'zero_shot_all','method':'none','shots':0,'repeat':0,
                            'n_query':len(all_test),**score})
        log(f'{base} s{test:02d} zero-shot acc={score["accuracy"]:.4f} '
            f'event_fraction={score["event_fraction"]:.4f}')
        atomic_json(out/'partial.json',fold)
        aname=adapt_names.get(base)
        if aname is None or not (args.models==['all'] or aname in args.models): continue
        log.stage=f'{aname}: source meta training'
        adapter,prototypes,meta_trace=train_adapter(args,model,X,data,train_people,val_people,seed,device,log)
        torch.save({'adapter':adapter.state_dict(),'prototypes':prototypes.cpu()},out/f'{aname}_adapter.pt')
        choices,table=select_adaptation(args,model,adapter,prototypes,X,data,val_people,seed,device,log)
        fold['adaptation'][aname]={'best_meta_epoch':meta_trace['best_epoch'],'choices':choices,'validation_grid':table}
        atomic_json(out/f'{aname}_meta_training.json',meta_trace)
        for repeat in range(args.repeats):
            # Same support/query plan for M2 and M3, every seed, and all controls.
            supports,query,reservations=trial_plan(data,test,args.shots,100000+test*100+repeat,args.query_trials)
            atomic_json(out/f'{aname}_support_r{repeat}.json',{
                'support':{str(k):data.sample_id[ix].tolist() for k,ix in supports.items()},
                'query':data.sample_id[query].tolist(),'reserved_trials':reservations})
            zero,pred=evaluate(model,X,data.y,query,device,args.batch_size)
            prediction_file(out/f'{aname}_r{repeat}_query_none.npz',data,query,pred)
            for shots,support in supports.items():
                common={'model':aname,'base':base,'phase':'calibrated_query','shots':shots,'repeat':repeat,
                        'n_support':len(support),'n_query':len(query),
                        'support_segment_seconds':None if args.synthetic or args.npz else len(support)*args.segment_s}
                fold['rows'].append({**common,'method':'none','target_trainable_parameters':0,**zero})
                for method in ('lowdim','lowdim_no_proto','head'):
                    cfg=choices[method]
                    started=time.monotonic()
                    if method=='head':
                        candidate=fit_head(model,X,data.y,support,device,**cfg)
                        fitted=time.monotonic()-started
                        score,pred=evaluate(candidate,X,data.y,query,device,args.batch_size)
                        params=sum(p.numel() for p in candidate.head.parameters())
                        extra={}
                    else:
                        q=fit_q(model,adapter,X,data.y,support,device,prototypes=prototypes,
                                proto_weight=args.proto_weight if method=='lowdim' else 0.,q_reg=args.q_reg,**cfg)
                        fitted=time.monotonic()-started
                        score,pred=evaluate(model,X,data.y,query,device,args.batch_size,adapter,q)
                        params=args.q_dim
                        extra={'q':q.cpu().tolist(),'q_norm':float(q.norm())}
                    prediction_file(out/f'{aname}_r{repeat}_k{shots}_{method}.npz',data,query,pred)
                    fold['rows'].append({**common,'method':method,'target_trainable_parameters':params,
                                        'fit_seconds':fitted,**extra,**score})
                log(f'{aname} s{test:02d} repeat={repeat} shots/class={shots}: '
                    f'none={zero["accuracy"]:.4f}, lowdim={fold["rows"][-3]["accuracy"]:.4f}, '
                    f'head={fold["rows"][-1]["accuracy"]:.4f}')
                atomic_json(out/'partial.json',fold)
    atomic_json(out/'complete.json',fold)
    return fold


def paired_summary(folds):
    """Average seeds/repeats within each held-out subject before paired statistics."""
    comparisons={}
    for fold in folds:
        subject=fold['test_subject']
        rows=fold['rows']
        source={r['model']:r for r in rows if r['phase']=='zero_shot_all'}
        for first,second in (('M1','M0'),('M1_plain','M0'),('M1','M1_plain')):
            if first in source and second in source:
                for metric in ('accuracy','macro_f1','event_fraction'):
                    key=f'{first}-{second}:all:{metric}'
                    comparisons.setdefault(key,{}).setdefault(subject,[]).append(source[first][metric]-source[second][metric])
        groups={}
        for r in rows:
            if r['phase']=='calibrated_query':
                groups.setdefault((r['model'],r['shots'],r['repeat']),{})[r['method']]=r
        for (name,shots,repeat),g in groups.items():
            for control in ('none','head','lowdim_no_proto'):
                if 'lowdim' not in g or control not in g: continue
                for metric in ('accuracy','macro_f1','event_fraction'):
                    key=f'{name}:lowdim-{control}:k{shots}:{metric}'
                    comparisons.setdefault(key,{}).setdefault(subject,[]).append(g['lowdim'][metric]-g[control][metric])
    result={}
    for name,people in comparisons.items():
        ids=sorted(people)
        d=np.array([np.mean(people[s]) for s in ids])
        n=len(d)
        std=float(d.std(ddof=1)) if n>1 else None
        ci=None
        if n>1:
            half=float(scipy_stats.t.ppf(.975,n-1)*std/np.sqrt(n))
            ci=[float(d.mean()-half),float(d.mean()+half)]
        result[name]={'n_subjects':n,'mean_delta':float(d.mean()),'subject_sd':std,'ci95_t':ci,
                      'subject_deltas':dict(zip(map(str,ids),map(float,d))),
                      'positive_subjects':int((d>0).sum()),'zero_subjects':int((d==0).sum())}
    return {'unit':'held-out subject; average seeds/repeats first',
            'status':'descriptive exploratory intervals; no automatic novelty/success verdict',
            'comparisons':result}


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic',action='store_true',help='Engineering data, not SEED results')
    p.add_argument('--npz',type=Path,help='Prepared X[N,T,F], y, subject, session, trial, sample_id')
    p.add_argument('--subjects',type=int,nargs='+')
    p.add_argument('--test-subjects',type=int,nargs='+')
    p.add_argument('--session',type=int,default=1)
    p.add_argument('--folds-json',type=Path,help='Explicit {test_id:{train:[...],val:[...]}}')
    p.add_argument('--models',nargs='+',choices=['M0','M1','M1_plain','M2','M3','all'],default=['M0','M1'])
    p.add_argument('--seeds',type=int,nargs='+',default=[42])
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--latent',type=int,default=32)
    p.add_argument('--coding-steps',type=int,default=2)
    p.add_argument('--per-subject',type=int,default=8)
    p.add_argument('--steps-per-epoch',type=int,default=0,help='0=from largest source subject; positive=explicit cap')
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--budget',type=float,default=.15,help='Soft event AC fraction target, not guaranteed')
    p.add_argument('--budget-weight',type=float,default=1.)
    p.add_argument('--group-temperature',type=float,default=.25)
    p.add_argument('--q-dim',type=int,default=8)
    p.add_argument('--meta-epochs',type=int,default=5)
    p.add_argument('--meta-episodes',type=int,default=12)
    p.add_argument('--meta-lr',type=float,default=.003)
    p.add_argument('--meta-inner-lr',type=float,default=.5)
    p.add_argument('--meta-inner-steps',type=int,default=3)
    p.add_argument('--meta-shots',type=int,default=1)
    p.add_argument('--meta-query-per-class',type=int,default=8)
    p.add_argument('--proto-weight',type=float,default=.1)
    p.add_argument('--q-reg',type=float,default=.001)
    p.add_argument('--shots',type=int,nargs='+',default=[1,2,3])
    p.add_argument('--query-trials',type=int,default=2)
    p.add_argument('--adapt-steps',type=int,nargs='+',default=[5,10])
    p.add_argument('--adapt-lrs',type=float,nargs='+',default=[.01,.1,.5])
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--segment-s',type=float,default=4.)
    p.add_argument('--frame-s',type=float,default=1.)
    p.add_argument('--hop-s',type=float,default=.5)
    p.add_argument('--reject-abs-thr',type=float,default=15.)
    p.add_argument('--feature-cache',type=Path,default=Path('data_cache/snn_ordered_de'))
    p.add_argument('--data-version',default='local-seed-v1',help='Change if underlying raw data change')
    p.add_argument('--refresh-features',action='store_true')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--out',type=Path,default=Path('results/snn_repaired'))
    p.add_argument('--resume',action='store_true',help='Skip complete folds after exact manifest match')
    args=p.parse_args(argv)
    if args.synthetic and args.npz: p.error('Choose only one of --synthetic and --npz')
    if 'all' in args.models and args.models!=['all']: p.error('all must be the only model token')
    positive=('epochs','hidden','latent','coding_steps','per_subject','batch_size','meta_epochs',
              'meta_episodes','meta_inner_steps','meta_shots','meta_query_per_class','q_dim',
              'query_trials','repeats','threads','session')
    if any(getattr(args,k)<1 for k in positive): p.error('Counts must be positive')
    if args.steps_per_epoch<0: p.error('steps-per-epoch must be >=0')
    if not 0<=args.budget<=1 or args.group_temperature<=0: p.error('Invalid budget or group temperature')
    if min(args.shots+args.adapt_steps)<1 or min(args.adapt_lrs+[args.lr,args.meta_lr,args.meta_inner_lr])<=0:
        p.error('Shots, update steps, and learning rates must be positive')
    args.shots=sorted(set(args.shots))
    args.seeds=sorted(set(args.seeds))
    args.subjects=sorted(set(args.subjects or (range(1,6) if args.synthetic else range(1,16))))
    args.test_subjects=sorted(set(args.test_subjects or args.subjects))
    return args


def main(argv=None):
    args=arguments(argv)
    args.out.mkdir(parents=True,exist_ok=True)
    # Atomic exclusive creation prevents simultaneous writes to a run directory.
    lock=args.out/'RUNNING.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'{lock} exists. Check its PID; after a killed process, remove only this stale lock before --resume.')
    os.write(fd,str(os.getpid()).encode()); os.close(fd)
    log=Progress(args.out/'run.log')
    admitted=False
    try:
        torch.set_num_threads(args.threads)
        device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else
                            'cpu' if args.device=='auto' else args.device)
        if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
        log(f'{VERSION} pid={os.getpid()} python={platform.python_version()} torch={torch.__version__} device={device}')
        torch.ones(1,device=device).sum().item()  # fail visibly during device initialization
        log.stage='load data'
        data=synthetic_data() if args.synthetic else load_npz(args.npz) if args.npz else local_seed(args,log)
        mask=np.isin(data.subject,args.subjects)&(data.session==args.session)
        data=Dataset(**{k:getattr(data,k)[mask] for k in Dataset.__dataclass_fields__}).validate()
        if set(np.unique(data.subject))!=set(args.subjects): raise ValueError('Selected subjects missing from data')
        if not set(args.test_subjects).issubset(args.subjects): raise ValueError('Test subjects must be in the population')
        specified=json.loads(args.folds_json.read_text(encoding='utf-8-sig')) if args.folds_json else None
        # Validate all folds and few-shot group availability BEFORE expensive training.
        do_adapt=args.models==['all'] or bool({'M2','M3'}&set(args.models))
        for test in args.test_subjects: fold_subjects(args.subjects,test,2,specified)
        if do_adapt:
            for person in args.subjects:
                trial_plan(data,person,tuple(sorted(set(args.shots+[args.meta_shots]))),0,args.query_trials)
        code_hash={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                   for name in ('snn_baseline.py','snn_core.py','snn_protocol.py')}
        # Runtime scheduling/cache flags do not change experimental content.
        excluded={'out','resume','refresh_features','feature_cache','test_subjects','device','npz','folds_json'}
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in excluded}
        protocol={'config':config,'code':code_hash,'data_hash':data.fingerprint(),'explicit_folds':specified,
                  'fold_rule':'explicit' if specified else 'cyclic-next-two validation subjects',
                  'actual_device':str(device),'torch_version':torch.__version__}
        manifest={'version':VERSION,'fingerprint':digest(protocol),'protocol':protocol,
                  'data_shape':list(data.X.shape),'synthetic':args.synthetic,
                  'note':'New ordered-DE protocol; old static-DE scores are not a paired baseline.'}
        manifest_path=args.out/'manifest.json'
        if manifest_path.exists():
            old=json.loads(manifest_path.read_text())
            if not args.resume: raise RuntimeError('Output already contains a run. Use a new --out or --resume.')
            if old['fingerprint']!=manifest['fingerprint']: raise RuntimeError('Resume refused: data/code/config/protocol differs; use a new output directory')
        else: atomic_json(manifest_path,manifest)
        admitted=True
        atomic_json(args.out/'status.json',{'state':'running','pid':os.getpid(),'test_subjects':args.test_subjects})
        log(f'dataset X={data.X.shape}; run fingerprint={manifest["fingerprint"][:12]}; '
            'statistics unit=held-out subject')
        # Include previously completed subjects when a resume request covers
        # only a subset; never discard their rows from the aggregate files.
        completed_folds={}
        for path in args.out.glob('seed*/s*/complete.json'):
            item=json.loads(path.read_text())
            completed_folds[(item['seed'],item['test_subject'])]=item
        for seed in args.seeds:
            for test in args.test_subjects:
                out=args.out/f'seed{seed}'/f's{test:02d}'
                out.mkdir(parents=True,exist_ok=True)
                completed=out/'complete.json'
                if args.resume and completed.exists():
                    log(f'resume: skip complete seed={seed} s{test:02d}')
                    fold=json.loads(completed.read_text())
                else:
                    fold=run_fold(args,data,test,seed,out,device,log,specified)
                completed_folds[(seed,test)]=fold
                folds=[completed_folds[k] for k in sorted(completed_folds)]
                summary=paired_summary(folds)
                summary['population_subjects']=args.subjects
                summary['completed_subjects']=sorted({f['test_subject'] for f in folds})
                summary['complete_loso']=len(folds)==len(args.seeds)*len(args.subjects)
                atomic_json(args.out/'summary.json',summary)
                atomic_json(args.out/'status.json',{'state':'running','completed_folds':len(folds),
                                                   'population_folds':len(args.seeds)*len(args.subjects)})
        atomic_json(args.out/'rows.json',folds)
        atomic_json(args.out/'status.json',{'state':'completed','completed_folds':len(folds),
                                           'complete_loso':len(folds)==len(args.seeds)*len(args.subjects),
                                           'synthetic':args.synthetic,'fingerprint':manifest['fingerprint']})
        log(f'COMPLETE: {len(folds)} folds; outputs in {args.out.resolve()}')
    except Exception as exc:
        if admitted or not (args.out/'manifest.json').exists():
            atomic_json(args.out/'status.json',{'state':'failed','type':type(exc).__name__,'message':str(exc)})
        log(f'FAILED {type(exc).__name__}: {exc}')
        raise
    finally:
        log.close()
        lock.unlink(missing_ok=True)


if __name__=='__main__': main()
