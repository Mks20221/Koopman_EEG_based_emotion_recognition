"""Tests for leakage, gradients, event accounting, and actual adaptation."""
import copy
import math
import sys
import unittest
import tempfile
import types
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from snn_core import SNN, IntrinsicAdapter, spike, group_objective
from snn_protocol import synthetic_data, TrainScaler, trial_plan, fold_subjects, balanced_batches, de_sequence, local_seed
from snn_baseline import fit_q, freeze, fit_head, paired_summary, evaluate

class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data=synthetic_data()

    def setUp(self):
        torch.manual_seed(123)
        self.model=SNN(12,3,12,8,True)
        self.X=torch.tensor(self.data.X[:6])
        self.y=torch.tensor([0,1,2,0,1,2])

    def test_discrete_forward_and_atan_backward(self):
        x=torch.tensor([-1.,0.,1.],requires_grad=True)
        out=spike(x)
        self.assertEqual(out.tolist(),[0.,1.,1.])
        out.sum().backward()
        torch.testing.assert_close(x.grad,1/(1+(math.pi*x).square()))

    def test_initial_backbones_and_encoders_are_equal(self):
        torch.manual_seed(42); a=SNN(12,3,12,8,False)
        torch.manual_seed(42); b=SNN(12,3,12,8,True)
        for av,bv in zip(a.state_dict().values(),b.state_dict().values()):
            torch.testing.assert_close(av,bv,rtol=0,atol=0)
        torch.testing.assert_close(a(self.X)[0],b(self.X)[0],rtol=0,atol=0)

    def test_encoder_and_both_lif_layers_receive_gradients(self):
        logits,_,c=self.model(self.X)
        loss=group_objective(logits,self.y,torch.tensor([1,1,2,2,3,3]),c['event_fraction'],True,budget=0.)
        loss.backward()
        for p in (self.model.encoder.raw_threshold,self.model.lif1.raw_tau,self.model.lif2.raw_threshold):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(),0)

    def test_budget_penalty_is_connected_to_encoder(self):
        _,_,c=self.model(self.X)
        grad,=torch.autograd.grad(c['event_fraction'].mean(),self.model.encoder.raw_threshold)
        self.assertGreater(grad.abs().sum().item(),0)

    def test_states_reset_between_batches(self):
        optim=torch.optim.SGD(self.model.parameters(),.01)
        for _ in range(2):
            out=self.model(self.X)[0]
            repeated=self.model(self.X)[0]
            torch.testing.assert_close(out,repeated,rtol=0,atol=0)
            optim.zero_grad(); F.cross_entropy(out,self.y).backward(); optim.step()

    def test_zero_q_exactly_matches_base_and_parameters_stay_bounded(self):
        adapter=IntrinsicAdapter(12,8,8)
        q=torch.zeros(8,requires_grad=True)
        torch.testing.assert_close(self.model(self.X)[0],self.model(self.X,adapter,q)[0],rtol=0,atol=0)
        for lif,delta in zip((self.model.lif1,self.model.lif2),adapter(torch.full((8,),1e8))):
            tau,threshold=lif.parameters_with_delta(delta)
            self.assertTrue(((tau>1.1)&(tau<20)).all())
            self.assertTrue(((threshold>.1)&(threshold<2)).all())

    def test_nonzero_q_changes_actual_forward_dynamics(self):
        adapter=IntrinsicAdapter(12,8,8)
        x=torch.tensor(self.data.X[:30])
        base=self.model(x)[0]
        adapted=self.model(x,adapter,torch.full((8,),2.))[0]
        self.assertTrue((base!=adapted).any())

    def test_q_is_fitted_and_backbone_remains_frozen(self):
        adapter=IntrinsicAdapter(12,8,8)
        freeze(self.model)
        for p in adapter.parameters(): p.requires_grad_(False)
        before=copy.deepcopy(self.model.state_dict())
        prior=adapter.projection.detach().clone()
        q=fit_q(self.model,adapter,self.data.X,self.data.y,np.arange(6),'cpu',3,.5,
                torch.zeros(3,8),.1,.001)
        self.assertGreater(q.norm().item(),0)
        for k,v in self.model.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        torch.testing.assert_close(adapter.projection,prior,rtol=0,atol=0)

    def test_meta_query_gradient_updates_adaptation_space(self):
        adapter=IntrinsicAdapter(12,8,8)
        freeze(self.model)
        q=fit_q(self.model,adapter,self.data.X,self.data.y,np.arange(6),'cpu',2,.5,
                torch.zeros(3,8),.1,.001,create_graph=True)
        query=torch.tensor(self.data.X[6:12])
        F.cross_entropy(self.model(query,adapter,q)[0],torch.tensor(self.data.y[6:12],dtype=torch.long)).backward()
        self.assertIsNotNone(adapter.projection.grad)
        self.assertTrue(torch.isfinite(adapter.projection.grad).all())
        self.assertGreater(adapter.projection.grad.abs().sum().item(),0)

    def test_head_control_only_changes_head(self):
        freeze(self.model)
        before=copy.deepcopy(self.model.state_dict())
        tuned=fit_head(self.model,self.data.X,self.data.y,np.arange(6),'cpu',2,.1)
        for k,v in tuned.state_dict().items():
            if not k.startswith('head.'): torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        self.assertFalse(torch.equal(tuned.head.weight,before['head.weight']))

    def test_discrete_spikes_and_fanout_counts_all_samples(self):
        _,_,c=self.model(self.X,return_spikes=True)
        e,h1,h2=c['spike_tensors']
        for s in (e,h1,h2): self.assertTrue(((s==0)|(s==1)).all())
        expected=e.sum((1,2))*12+h1.sum((1,2))*8
        torch.testing.assert_close(c['event_ac'],expected)
        full,_=evaluate(self.model,self.data.X,self.data.y,np.arange(30),'cpu',batch_size=30)
        chunks,_=evaluate(self.model,self.data.X,self.data.y,np.arange(30),'cpu',batch_size=7)
        for key in ('event_ac','input_spikes','hidden1_spikes','accuracy'):
            self.assertAlmostEqual(full[key],chunks[key],places=5)

    def test_scaler_uses_source_training_only(self):
        train=np.flatnonzero(self.data.subject<3)
        a=TrainScaler().fit(self.data.X,train)
        altered=self.data.X.copy(); altered[self.data.subject>=3]+=10000
        b=TrainScaler().fit(altered,train)
        np.testing.assert_array_equal(a.mean,b.mean)
        np.testing.assert_array_equal(a.scale,b.scale)

    def test_nested_support_and_fixed_trial_disjoint_queries(self):
        supports,query,_=trial_plan(self.data,1,(1,2,3),9)
        def keys(ix): return set(zip(self.data.subject[ix],self.data.session[ix],self.data.trial[ix]))
        for k,support in supports.items():
            self.assertEqual(len(support),3*k)
            self.assertEqual(len(keys(support)),3*k)
            self.assertFalse(keys(support)&keys(query))
        self.assertTrue(set(supports[1])<=set(supports[2])<=set(supports[3]))
        support2,query2,_=trial_plan(self.data,1,(1,),9)
        np.testing.assert_array_equal(query,query2)
        np.testing.assert_array_equal(supports[1],support2[1])

    def test_person_partition_and_equal_subject_batches(self):
        train,val=fold_subjects([1,2,3,4,5],1)
        self.assertEqual((train,val),([4,5],[2,3]))
        with self.assertRaises(ValueError): fold_subjects([1,2,3,4,5],1,specified={'1':{'train':[1,4,5],'val':[2,3]}})
        subjects=np.array([1,1,1,1,1,2,3,3])
        for ix in balanced_batches(subjects,np.random.default_rng(9),4,3):
            self.assertEqual(len(ix),12)
            np.testing.assert_array_equal(np.unique(subjects[ix],return_counts=True)[1],[4,4,4])

    def test_aggregate_statistical_unit_is_subject(self):
        folds=[]
        for person in [1,2]:
            for seed in [42,43,44]:
                folds.append({'test_subject':person,'seed':seed,'rows':[
                    {'model':name,'phase':'zero_shot_all','accuracy':acc,'macro_f1':acc,'event_fraction':.1}
                    for name,acc in [('M0',.5),('M1',.5+.01*person)]]})
        result=paired_summary(folds)['comparisons']['M1-M0:all:accuracy']
        self.assertEqual(result['n_subjects'],2)
        self.assertAlmostEqual(result['mean_delta'],.015)

    def test_de_layout_has_real_ordered_frames(self):
        t=np.arange(800)/200
        raw=np.stack([np.sin(2*np.pi*10*t)*(1+t),np.sin(2*np.pi*20*t)])
        features=de_sequence(raw[None],200)
        self.assertEqual(features.shape,(1,7,10))
        self.assertGreater(features[0,-1,2],features[0,0,2])
        self.assertGreater(features[0,0,8],features[0,0,7])

    def test_local_preprocess_adapter_preserves_trials_and_reuses_cache(self):
        from argparse import Namespace
        from dataclasses import make_dataclass
        with tempfile.TemporaryDirectory() as tmp:
            modules={name:types.ModuleType(name) for name in ('config','preprocess','data')}
            for name,module in modules.items():
                path=Path(tmp)/(name+'.py'); path.write_text('# adapter fixture')
                module.__file__=str(path)
            modules['config'].SEED_FS=200
            modules['preprocess'].PreprocConfig=make_dataclass('Cfg',[
                ('win_s',float),('overlap',float),('drop_bad_channels',bool),('reject_abs_thr',float)])
            calls=[]
            def build(dataset,subject,session,cfg,use_cache):
                calls.append((dataset,subject,session,use_cache))
                return {'segs':np.random.default_rng(subject).normal(size=(30,2,800)),
                        'y':np.repeat(np.arange(3),10),'trial':np.repeat(np.arange(15),2)}
            modules['preprocess'].build_segments=build
            args=Namespace(segment_s=4.,reject_abs_thr=15.,feature_cache=Path(tmp)/'features',
                           subjects=[1,2],session=1,frame_s=1.,hop_s=.5,data_version='test',refresh_features=False)
            with patch.dict(sys.modules,modules):
                first=local_seed(args,lambda message:None)
                second=local_seed(args,lambda message:None)
            self.assertEqual(len(calls),2)
            self.assertEqual(first.X.shape,(60,7,10))
            self.assertEqual(len(np.unique(first.trial)),15)
            self.assertEqual(first.fingerprint(),second.fingerprint())

if __name__=='__main__': unittest.main(verbosity=2)
