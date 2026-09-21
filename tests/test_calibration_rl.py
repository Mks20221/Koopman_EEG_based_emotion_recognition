"""Causal isolation, identical fixed path, optimizer continuity and actual PPO learning."""
from pathlib import Path
import copy
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'.runtime'/'rl_deps'))
from src.calibration.adapter import SimpleMLP, CalibConfig, CalibrationAdapter, TrialEqualLossCalculator, train_source_model
from src.calibration.control import Trial, Task, FeedbackSession, apply_action, run_controller, select_fixed, ACTIONS, STATE_NAMES
from src.calibration.rl_env import AdaptationEnv
from src.calibration.run import run_hp_selection, WindowScaler, build_folds
from src.calibration.run_rl import get_source, parameter_hash
from stable_baselines3 import PPO


def task():
    torch.manual_seed(42)
    model=SimpleMLP(4,8,3)
    trials=tuple(Trial(i+1,torch.randn(3+i%3,4),torch.full((3+i%3,),i%3,dtype=torch.long)) for i in range(15))
    return Task(1,model,trials[:9],trials[9:],{})


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_trial_weighting_unequal_lengths(self):
        ids=np.array([101,101,102,201,201,201])
        w=TrialEqualLossCalculator.weights(ids,'cpu')
        for t in np.unique(ids):
            self.assertAlmostEqual(float(w[ids==t].sum()),1/3,places=6)

    def test_fixed_path_identical_for_every_action(self):
        t=task()
        for action in range(5):
            a=CalibrationAdapter(t.source,CalibConfig(),'cpu')
            b=CalibrationAdapter(t.source,CalibConfig(),'cpu')
            for fb in range(1,4):
                visible=t.feedback[:fb]
                torch.manual_seed(100+fb)
                apply_action(a,action,visible)
                torch.manual_seed(100+fb)
                b.apply_fixed_head_continuous(torch.cat([v.x for v in visible]),torch.cat([v.y for v in visible]),
                    np.concatenate([np.full(len(v.y),v.number) for v in visible]),n_steps=ACTIONS[action][1],lr=ACTIONS[action][0])
                for x,y in zip(a.model.parameters(),b.model.parameters()):
                    torch.testing.assert_close(x,y,atol=0,rtol=0)

    def test_continuous_adam_lr_skip_and_frozen_features(self):
        t=task(); a=CalibrationAdapter(t.source,CalibConfig(),'cpu')
        optimizer=id(a._optimizer)
        apply_action(a,1,t.feedback[:1])
        step=float(a._optimizer.state[a.model.fc2.weight]['step'])
        before=parameter_hash(a.model)
        apply_action(a,0,t.feedback[:2])
        self.assertEqual(before,parameter_hash(a.model))
        apply_action(a,4,t.feedback[:3])
        self.assertEqual(optimizer,id(a._optimizer))
        self.assertEqual(float(a._optimizer.state[a.model.fc2.weight]['step']),step+20)
        self.assertEqual(a._optimizer.param_groups[0]['lr'],1e-3)
        torch.testing.assert_close(t.source.fc1.weight,a.model.fc1.weight,atol=0,rtol=0)
        self.assertFalse(torch.equal(t.source.fc2.weight,a.model.fc2.weight))

    def test_evaluation_labels_cannot_change_state_or_actions(self):
        a=task(); b=copy.deepcopy(a)
        for t in b.evaluation:
            t.y[:]=(t.y+1)%3
        choose=lambda state,fb: int(np.argmax(state[:3]))+1
        r1=run_controller(a,choose,'cpu'); r2=run_controller(b,choose,'cpu')
        self.assertEqual([r['action'] for r in r1['records']],[r['action'] for r in r2['records']])
        self.assertEqual([r['state'] for r in r1['records']],[r['state'] for r in r2['records']])

    def test_future_feedback_is_invisible_and_current_prediction_precedes_update(self):
        t=task(); s=FeedbackSession(t.source,t.feedback,range(9),'cpu')
        t.source.eval()
        with torch.no_grad(): expected=t.source(t.feedback[0].x).softmax(-1).mean(0).numpy()
        state=s.prepare()
        np.testing.assert_allclose(state[:3],expected)
        self.assertEqual(len(state),len(STATE_NAMES))
        self.assertEqual(len(s.visible),1)
        for future in t.feedback[1:]: future.y[:]=(future.y+1)%3
        other=FeedbackSession(t.source,t.feedback,range(9),'cpu')
        np.testing.assert_array_equal(state,other.prepare())

    def test_reward_independent_and_telescopes(self):
        env=AdaptationEnv([task()],'cpu')
        env.reset(seed=42); total=0
        for i in range(9):
            _,r,done,_,_=env.step(1)
            total+=r
            self.assertEqual(done,i==8)
        self.assertAlmostEqual(total,env.initial_loss-env.before)
        self.assertEqual(env.completed,1)

    def test_real_ppo_changes_policy_parameters(self):
        env=AdaptationEnv([task()],'cpu')
        p=PPO('MlpPolicy',env,n_steps=18,batch_size=9,n_epochs=2,seed=42,device='cpu')
        before=parameter_hash(p.policy)
        p.learn(36)
        self.assertNotEqual(before,parameter_hash(p.policy))
        self.assertEqual(env.completed,4)

    def test_source_selection_does_not_keep_random_init_when_losses_above_one(self):
        model=SimpleMLP(4,8,3)
        with torch.no_grad(): model.fc2.bias.copy_(torch.tensor([10.,-10.,0.]))
        before=parameter_hash(model)
        x=np.ones((9,4),np.float32); y=np.ones(9,dtype=np.int64); ids=np.repeat([1,2,3],3)
        with patch('src.calibration.adapter.SimpleMLP',return_value=model):
            selected,info=train_source_model(x,y,ids,x,y,ids,np.zeros(4),np.ones(4),epochs=2)
        self.assertGreater(info['best_val_loss'],1)
        self.assertTrue(info['selected_from_epoch_after_init'])
        self.assertNotEqual(before,parameter_hash(selected))

    def test_source_and_scaler_exclude_pseudo_and_outer_holdouts(self):
        data={s:dict(X_flat=np.full((6,310),s,dtype=np.float32),y_flat=np.zeros(6,dtype=np.int64),
                     trial_ids_flat=np.arange(6)) for s in range(1,16)}
        # Fifteen unique trial IDs for production invariant.
        for s in data:
            data[s]=dict(X_flat=np.full((15,310),s,dtype=np.float32),y_flat=np.zeros(15,dtype=np.int64),trial_ids_flat=np.arange(1,16))
        def fake_train(x,y,ids,xv,yv,idv,mean,scale,**kwargs):
            self.assertEqual(set(x[:,0]),set(range(5,16)))
            self.assertEqual(set(xv[:,0]),{4})
            self.assertAlmostEqual(float(mean[0]),10.)
            return SimpleMLP(),dict(selected_epoch_one_based=1)
        with tempfile.TemporaryDirectory() as tmp,patch('src.calibration.run_rl.train_source_model',side_effect=fake_train):
            _,scaler,prov=get_source(data,list(range(5,16)),4,[1,2,3],Path(tmp)/'source.pt','cpu',lambda x:None)
            self.assertEqual(prov['scaler_fit_subjects'],list(range(5,16)))


if __name__=='__main__':
    unittest.main(verbosity=2)
