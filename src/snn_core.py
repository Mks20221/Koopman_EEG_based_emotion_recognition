"""Discrete SNN, differentiable event counts, and zero-preserving intrinsic adaptation.
No snntorch dependency. States are local to each forward call.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class _ATan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        a = ctx.alpha
        return grad * (a / 2) / (1 + (math.pi * a * x / 2).square()), None


def spike(x):
    return _ATan.apply(x, 2.0)


def bounded(raw, lo, hi):
    return lo + (hi - lo) * raw.sigmoid()


def inverse_bound(value, lo, hi):
    p = (value - lo) / (hi - lo)
    return math.log(p / (1 - p))


class SpikeEncoder(nn.Module):
    """Signed ON/OFF cumulative code. Input [batch, real EEG frames, features].
    One positive threshold per feature, shared by ON/OFF rails; no redundant gain.
    Coding microsteps repeat a frame's current, not a claim of extra EEG samples.
    """
    def __init__(self, features, learnable=False, coding_steps=2):
        super().__init__()
        self.raw_threshold = nn.Parameter(
            torch.full((features,), inverse_bound(1.0, .25, 4.0)),
            requires_grad=learnable)
        self.coding_steps = coding_steps

    def forward(self, x):
        threshold = bounded(self.raw_threshold, .25, 4.0).repeat(2)
        current = torch.cat((x.relu(), (-x).relu()), -1)
        state = x.new_zeros(x.shape[0], 2 * x.shape[-1])
        emitted = []
        for frame in current.unbind(1):
            for _ in range(self.coding_steps):
                state = state + frame / self.coding_steps
                s = spike(state - threshold)
                # Soft reset retains overshoot; detach only the reset event.
                state = state - s.detach() * threshold
                emitted.append(s)
        return torch.stack(emitted, 1)


class LIF(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.raw_tau = nn.Parameter(torch.full((width,), inverse_bound(3., 1.1, 20.)))
        self.raw_threshold = nn.Parameter(torch.full((width,), inverse_bound(.5, .1, 2.)))

    def parameters_with_delta(self, delta=None):
        dt = dh = 0.
        if delta is not None:
            dt, dh = delta
        return (bounded(self.raw_tau + dt, 1.1, 20.),
                bounded(self.raw_threshold + dh, .1, 2.))

    def step(self, current, mem, delta=None):
        tau, threshold = self.parameters_with_delta(delta)
        decay = torch.exp(-1. / tau)
        mem = decay * mem + current
        s = spike(mem - threshold)
        return s, mem - s.detach() * threshold


class SNN(nn.Module):
    def __init__(self, features, classes=3, hidden=64, latent=32,
                 learnable_encoder=False, coding_steps=2):
        super().__init__()
        self.encoder = SpikeEncoder(features, learnable_encoder, coding_steps)
        self.fc1 = nn.Linear(2 * features, hidden)
        self.lif1 = LIF(hidden)
        self.fc2 = nn.Linear(hidden, latent)
        self.lif2 = LIF(latent)
        self.head = nn.Linear(latent, classes)
        self.features, self.hidden, self.latent, self.classes = features, hidden, latent, classes

    def forward(self, x, adapter=None, q=None, return_spikes=False):
        encoded = self.encoder(x)
        batch, steps, _ = encoded.shape
        m1, m2 = x.new_zeros(batch, self.hidden), x.new_zeros(batch, self.latent)
        deltas = adapter(q) if adapter is not None else (None, None)
        h1, h2 = [], []
        for inp in encoded.unbind(1):
            s1, m1 = self.lif1.step(self.fc1(inp), m1, deltas[0])
            s2, m2 = self.lif2.step(self.fc2(s1), m2, deltas[1])
            h1.append(s1)
            h2.append(s2)
        h1, h2 = torch.stack(h1, 1), torch.stack(h2, 1)
        representation = h2.mean(1)
        logits = self.head(representation)
        counts = [v.sum((1, 2)) for v in (encoded, h1, h2)]
        # Event-driven theoretical AC estimate for the two spike-driven layers.
        # Readout runs ONCE on a real-valued mean; report its dense MACs separately.
        event_ac = counts[0] * self.hidden + counts[1] * self.latent
        max_event_ac = steps * (2 * self.features * self.hidden + self.hidden * self.latent)
        stats = {
            'input_spikes': counts[0], 'hidden1_spikes': counts[1], 'hidden2_spikes': counts[2],
            'event_ac': event_ac, 'event_fraction': event_ac / max_event_ac,
            'total_spike_rate': sum(counts) / (steps * (2*self.features+self.hidden+self.latent)),
            'readout_mac': x.new_full((batch,), self.latent * self.classes),
            'neuron_updates': x.new_full((batch,), steps * (self.hidden + self.latent)),
            'encoder_updates': x.new_full((batch,), steps * 2 * self.features),
        }
        if return_spikes:
            stats['spike_tensors'] = (encoded, h1, h2)
        return logits, representation, stats


class IntrinsicAdapter(nn.Module):
    """A source-learned low-rank map q -> bounded changes of LIF raw parameters.
    q=0 gives EXACTLY the base network. Only q is fitted on a target subject.
    """
    def __init__(self, hidden, latent, q_dim=8, max_delta=.75):
        super().__init__()
        self.widths = (hidden, latent)
        self.q_dim, self.max_delta = q_dim, max_delta
        self.projection = nn.Parameter(torch.randn(2 * (hidden + latent), q_dim) * .05)

    def forward(self, q):
        if q is None or q.shape != (self.q_dim,):
            raise ValueError(f'q must have shape ({self.q_dim},)')
        d = self.max_delta * torch.tanh(self.projection @ q)
        a, b, c, e = d.split((self.widths[0], self.widths[0], self.widths[1], self.widths[1]))
        return ((a, b), (c, e))


def group_objective(logits, y, subjects, event_fraction, robust=False,
                    budget=.15, budget_weight=1., temperature=.25):
    """Smooth worst-subject CE and squared per-subject budget violation.
    The penalty is a soft constraint, NOT a guarantee that a budget is met.
    """
    ce = F.cross_entropy(logits, y, reduction='none')
    groups = torch.unique(subjects)
    losses = torch.stack([ce[subjects == s].mean() for s in groups])
    rates = torch.stack([event_fraction[subjects == s].mean() for s in groups])
    if robust:
        risk = temperature * (torch.logsumexp(losses / temperature, 0) - math.log(len(groups)))
        penalty = (rates - budget).relu().square().mean()
        return risk + budget_weight * penalty
    return losses.mean()


def prototype_loss(representation, y, prototypes):
    # Source prototypes are frozen and obtained ONLY from source training people.
    return (representation - prototypes[y]).square().mean()
