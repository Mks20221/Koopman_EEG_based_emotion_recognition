"""Shared causal interaction and head updates for fixed, random and RL control."""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
import torch
from torch.nn import functional as F

from .adapter import CalibrationAdapter, CalibConfig, TrialEqualLossCalculator

ACTIONS = ((None, 0), (1e-4, 5), (1e-4, 20), (1e-3, 5), (1e-3, 20))
STATE_NAMES = ([f'current_mean_p{i}' for i in range(3)] +
               ['current_entropy', 'current_feedback_ce'] +
               [f'feedback_class_fraction_{i}' for i in range(3)] +
               ['feedback_fraction'] + [f'previous_action_{i}' for i in range(6)])
BUDGETS = (0, 3, 6, 9)
METHODS = ('none', 'fixed_head', 'random_action', 'RL_policy')


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


@dataclass
class Trial:
    number: int
    x: torch.Tensor
    y: torch.Tensor


@dataclass
class Task:
    subject: int
    source: object
    feedback: tuple
    evaluation: tuple
    provenance: dict


def make_task(subject, data, source, scaler, device, provenance):
    trials = tuple(Trial(i + 1,
                        torch.as_tensor(scaler.transform(x), dtype=torch.float32, device=device),
                        torch.as_tensor(y, dtype=torch.long, device=device))
                   for i, (x, y) in enumerate(zip(data['X_list'], data['y_list'])))
    return Task(subject, source, trials[:9], trials[9:], provenance)


def apply_action(adapter, action, visible):
    """Single update implementation, including skip; Adam is never reconstructed."""
    lr, steps = ACTIONS[int(action)]
    x = torch.cat([t.x for t in visible])
    y = torch.cat([t.y for t in visible])
    ids = np.concatenate([np.full(len(t.y), t.number) for t in visible])
    adapter.apply_fixed_head_continuous(x, y, ids, n_steps=steps, lr=lr)
    return steps


class FeedbackSession:
    """Does not hold evaluation data. predict -> reveal -> state -> action -> update."""
    def __init__(self, source, feedback, order, device):
        self.adapter = CalibrationAdapter(source, CalibConfig(), device)
        self.feedback = feedback
        self.order = list(order)
        self.device = device
        self.visible = []
        self.previous = 5  # dedicated START token
        self.total_steps = 0
        self.seconds = 0.0
        self.prepared = False

    def prepare(self):
        assert not self.prepared and len(self.visible) < 9
        trial = self.feedback[self.order[len(self.visible)]]
        self.adapter.model.eval()
        with torch.no_grad():
            logits = self.adapter.model(trial.x)  # before accessing current label
            mean_p = logits.softmax(-1).mean(0)
            entropy = -(mean_p * mean_p.clamp_min(1e-12).log()).sum()
            loss = F.cross_entropy(logits, trial.y)  # label revealed now
        self.visible.append(trial)
        counts = np.bincount([int(t.y[0]) for t in self.visible], minlength=3)
        previous = np.eye(6, dtype=np.float32)[self.previous]
        self.state = np.concatenate((mean_p.cpu().numpy(), [entropy.item(), loss.item()],
                                     counts / len(self.visible), [len(self.visible)/9], previous)).astype(np.float32)
        assert len(self.state) == len(STATE_NAMES) and np.isfinite(self.state).all()
        self.prepared = True
        return self.state.copy()

    def act(self, action):
        assert self.prepared and 0 <= int(action) < len(ACTIONS)
        sync(self.device)
        start = time.perf_counter()
        steps = apply_action(self.adapter, action, self.visible)
        sync(self.device)
        elapsed = time.perf_counter() - start
        self.total_steps += steps
        self.seconds += elapsed
        record = dict(feedback=len(self.visible), trial=self.visible[-1].number,
                      state=self.state.tolist(), action=int(action), lr=ACTIONS[int(action)][0],
                      update_steps=steps, cumulative_update_steps=self.total_steps,
                      update_seconds=elapsed, cumulative_update_seconds=self.seconds,
                      visible_trials=[t.number for t in self.visible])
        self.previous = int(action)
        self.prepared = False
        return record


@torch.no_grad()
def evaluation_loss(model, trials):
    model.eval()
    x, y = torch.cat([t.x for t in trials]), torch.cat([t.y for t in trials])
    ids = np.concatenate([np.full(len(t.y), t.number) for t in trials])
    return float(TrialEqualLossCalculator.compute(model, x, y, ids, x.device))


@torch.no_grad()
def trial_accuracy(model, trials):
    model.eval()
    correct = [int(model(t.x).softmax(-1).mean(0).argmax()) == int(t.y[0]) for t in trials]
    return float(np.mean(correct))


@torch.no_grad()
def predict_rows(model, trials, subject, method, budget):
    model.eval()
    rows = []
    for trial in trials:
        probs = model(trial.x).softmax(-1).cpu().numpy()
        labels = trial.y.cpu().numpy()
        for i, p in enumerate(probs):
            rows.append(dict(subject=subject, method=method, feedback=budget,
                             trial=trial.number, window=i, y_true=int(labels[i]),
                             p0=float(p[0]), p1=float(p[1]), p2=float(p[2]),
                             y_pred=int(p.argmax())))
    return rows


def run_controller(task, choose_action, device, seed=42, method='validation', save_predictions=False):
    """Evaluation labels are consulted only AFTER actions, solely for scoring."""
    devices = [torch.device(device).index or 0] if str(device).startswith('cuda') else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        session = FeedbackSession(task.source, task.feedback, range(9), device)
        records, predictions, scores = [], [], {}
        inference_seconds = 0.0
        for fb in range(10):
            if fb in BUDGETS:
                scores[fb] = trial_accuracy(session.adapter.model, task.evaluation)
                if save_predictions:
                    predictions.extend(predict_rows(session.adapter.model, task.evaluation,
                                                    task.subject, method, fb))
            if fb == 9:
                break
            sync(device)
            start = time.perf_counter()
            state = session.prepare()
            action = int(choose_action(state, fb))
            sync(device)
            decision_seconds = time.perf_counter() - start
            inference_seconds += decision_seconds
            record = session.act(action)
            record.update(method=method, subject=task.subject, decision_seconds=decision_seconds,
                          cumulative_decision_seconds=inference_seconds)
            records.append(record)
        assert all(torch.equal(a, b) for a, b in zip(task.source.fc1.parameters(),
                                                    session.adapter.model.fc1.parameters()))
        return dict(score=float(np.mean([scores[b] for b in (3, 6, 9)])), scores=scores,
                    records=records, predictions=predictions,
                    update_steps=session.total_steps, update_seconds=session.seconds,
                    decision_seconds=inference_seconds)


def select_fixed(task, device, seed=42):
    """Existing rule: mean trial accuracy at feedback 3/6/9, first tie wins."""
    candidates = []
    for action in range(1, 5):
        result = run_controller(task, lambda state, fb, a=action: a, device, seed)
        candidates.append(dict(action=action, lr=ACTIONS[action][0], steps=ACTIONS[action][1],
                               score=result['score'], scores=result['scores']))
    best = max(candidates, key=lambda row: row['score'])
    return best, candidates
