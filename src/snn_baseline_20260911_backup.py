"""
SNN Baseline for EEG Emotion Recognition (SEED)
================================================
Architecture: DE features → CNN encoder → LIF SNN → mean-pool readout
Supports M0/M1 (encoding variants) and M2/M3 (adaptation variants).

Four models:
  M0: vanilla encoding, no adaptation
  M1: trainable threshold encoding, no adaptation
  M2: vanilla encoding, low-dim dynamics adaptation
  M3: trainable threshold encoding, low-dim dynamics adaptation
"""

from __future__ import annotations

import json, os, time, copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import butter, filtfilt

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import RESULTS_DIR, SEED_FS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from data import load_trials
from preprocess import build_segments, PreprocConfig


# ─────────────────────────────────────────────
# DE feature extraction (matches exp_separability.py)
# ─────────────────────────────────────────────
DE_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def de_features(segs: np.ndarray, fs: float, bands=DE_BANDS) -> np.ndarray:
    """
    segs: (N, C, L)  raw EEG segments
    returns: (N, C * len(bands)) DE features
    """
    n, C, L = segs.shape
    nyq = fs / 2.0
    feats = np.zeros((n, C, len(bands)))
    for bi, (lo, hi) in enumerate(bands):
        b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
        filtered = filtfilt(b, a, segs, axis=-1)
        var = filtered.var(axis=-1)
        feats[:, :, bi] = 0.5 * np.log(2 * np.pi * np.e * np.maximum(var, 1e-12))
    return feats.reshape(n, -1)  # (N, C*bands)


def build_de_dataset(dataset="seed", subjects=None, sessions=(1, 2, 3),
                     preproc_cfg=None):
    """
    Load all specified subjects/sessions, extract DE features.
    Returns X_all (N, C*bands), y_all (N,), subj_all (N,)
    """
    if preproc_cfg is None:
        preproc_cfg = PreprocConfig()

    X_all, y_all, subj_all, sess_all = [], [], [], []
    for subj in (subjects or list_subjects(dataset)):
        for ses in sessions:
            try:
                data = build_segments(dataset, subj, ses,
                                      cfg=preproc_cfg, use_cache=False)
                segs = data['segs']
                y = data['y']
            except FileNotFoundError:
                continue
            feats = de_features(segs, SEED_FS)
            X_all.append(feats)
            y_all.append(y)
            # Use (subject, session) pair as subject identifier for LOSO
            subj_all.append(np.full(len(y), subj))
            sess_all.append(np.full(len(y), ses))
    X_all = np.concatenate(X_all, 0)
    y_all = np.concatenate(y_all, 0)
    subj_all = np.concatenate(subj_all, 0)
    return X_all, y_all, subj_all


def list_subjects(dataset="seed"):
    if dataset == "seed":
        return list(range(1, 16))
    raise ValueError(f"Unknown dataset: {dataset}")


# ─────────────────────────────────────────────
# Model components
# ─────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    Light CNN encoder for DE feature vector.
    Input: (B, C*bands)  — reshaped to (B, bands, C) for 1D conv
    Output: (B, z_dim) continuous values normalized for LIF firing
    """
    def __init__(self, n_channels=51, n_bands=5, hidden=64, z_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_bands, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, z_dim),
            # BN normalizes to ~N(0,1) so LIF threshold=1.0 is appropriate
            nn.BatchNorm1d(z_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C*bands) -> (B, bands, C)
        B = x.shape[0]
        n_channels, n_bands = 51, 5
        x = x.view(B, n_bands, n_channels)
        return self.net(x)  # (B, z_dim)


class LIFNeurons(nn.Module):
    """
    Trainable LIF with straight-through estimator (STE) surrogate gradient.
    No snntorch dependency - avoids graph-retention bugs in some torch versions.
    Gradient flows through the spike decision via STE.
    """
    def __init__(self, dim: int, threshold=1.0, tau=5.0, learn_thr=True, learn_tau=True):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=learn_thr)
        self.log_tau = nn.Parameter(torch.tensor(tau).log(), requires_grad=learn_tau)

    @property
    def beta(self):
        return torch.exp(-1.0 / torch.clamp(torch.exp(self.log_tau), min=0.1))

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)
        mem = self.beta * mem + (1 - self.beta) * x
        # STE surrogate: spike is a normalized soft quantity, gradient = 1/thr
        spike = (mem - self.threshold).clamp(min=0) / (self.threshold + 1e-6)
        spike = spike.clamp(0, 1)
        # Hard reset using detached spike value
        mem = mem - spike.detach() * self.threshold
        return spike, mem


class SNNDecoder(nn.Module):
    """
    Temporal readout: pool spikes over time, then classify.
    """
    def __init__(self, z_dim=32, n_classes=3, time_steps=8):
        super().__init__()
        self.time_steps = time_steps
        self.readout = nn.Sequential(
            nn.Flatten(),
            nn.Linear(z_dim * time_steps, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        # spikes: (B, T, z_dim)
        return self.readout(spikes)


class MeanReadout(nn.Module):
    """Simple mean-pool over time steps."""
    def __init__(self, z_dim=32, n_classes=3):
        super().__init__()
        self.classifier = nn.Linear(z_dim, n_classes)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        # spikes: (B, T, z_dim)
        return self.classifier(spikes.mean(dim=1))


# ─────────────────────────────────────────────
# Full SNN model
# ─────────────────────────────────────────────
class EEGSNN(nn.Module):
    """
    Base EEG SNN: encoder + LIF SNN layers + readout.
    Encodes DE features into spike trains, processes with LIF, classifies.
    """
    def __init__(self, n_channels=51, n_bands=5, z_dim=32, n_classes=3,
                 n_neurons=64, time_steps=8, readout_type="mean"):
        super().__init__()
        self.time_steps = time_steps
        self.encoder = CNNEncoder(n_channels, n_bands, hidden=n_neurons, z_dim=z_dim)

        # Two LIF layers
        # lif2 threshold=0.3: driven by lif1 spikes which are (mem-thr)/thr ≈ 0.1-0.3
        self.lif1 = LIFNeurons(z_dim, threshold=1.0, tau=2.0)
        self.lif2 = LIFNeurons(z_dim, threshold=0.3, tau=2.0)

        self.readout = MeanReadout(z_dim, n_classes) if readout_type == "mean" \
                       else SNNDecoder(z_dim, n_classes, time_steps)

    def forward(self, x: torch.Tensor, adaptation_vec: Optional[torch.Tensor] = None):
        """
        x: (B, C*bands) DE features
        adaptation_vec: optional (B, adapt_dim) — for M2/M3 adaptation
        Returns: logits (B, n_classes)
        """
        B = x.shape[0]
        T = self.time_steps

        # Encode continuous features to spike trains
        h = self.encoder(x)  # (B, z_dim)
        # Broadcast to time dimension
        h = h.unsqueeze(1).expand(B, T, -1)  # (B, T, z_dim)

        # Process through LIF layers (accumulate outputs, no in-place writes)
        mem1, mem2 = None, None
        spikes_out = []
        for t in range(T):
            x_t = h[:, t, :]  # (B, z_dim)
            sp1, mem1 = self.lif1(x_t, mem1)
            sp2, mem2 = self.lif2(sp1, mem2)
            spikes_out.append(sp2)
        spikes = torch.stack(spikes_out, dim=1)  # (B, T, z_dim)

        return self.readout(spikes)


# ─────────────────────────────────────────────
# Trainable threshold encoding (Innovation 1)
# ─────────────────────────────────────────────
class TrainableThresholdEncoder(nn.Module):
    """
    Maps continuous DE features to spike trains using learnable channel-wise thresholds.
    Uses soft sigmoid gating (surrogate gradient) for gradient flow.
    """
    def __init__(self, n_channels=51, n_bands=5, groups_per_band=1, threshold_init=0.0):
        super().__init__()
        total_units = n_channels * n_bands * groups_per_band
        # Learnable threshold and scale
        self.threshold = nn.Parameter(torch.zeros(1) + threshold_init)
        self.scale = nn.Parameter(torch.ones(1) * 10.0)  # steep sigmoid → hard spike
        # Per-feature gain
        self.log_gain = nn.Parameter(torch.zeros(total_units))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C*bands) continuous DE features
        Returns: soft spike tensor (B, C*bands) in [0,1] via sigmoid gating
        """
        gain = F.softplus(self.log_gain) + 1e-6
        x_norm = x / gain
        # Soft encoding: sigmoid surrogate for spike
        return torch.sigmoid(self.scale * (x_norm - self.threshold))


class EEGSNNWithEncoding(nn.Module):
    """
    M1/M3: SNN with trainable threshold encoding.
    Continuous DE features → soft threshold encoder → LIF SNN → readout.
    """
    def __init__(self, n_channels=51, n_bands=5, z_dim=32, n_classes=3,
                 n_neurons=64, time_steps=8, readout_type="mean",
                 trainable_encoding=True):
        super().__init__()
        self.time_steps = time_steps
        self.trainable_encoding = trainable_encoding

        if trainable_encoding:
            self.threshold_encoder = TrainableThresholdEncoder(
                n_channels, n_bands, threshold_init=0.0)
            # Project soft-spike features to z_dim
            self.spike_proj = nn.Sequential(
                nn.Linear(n_channels * n_bands, z_dim),
                nn.BatchNorm1d(z_dim),
                nn.ReLU(),
            )
            # Learnable gain to match encoder output scale (~N(0,1) after BN)
            self.spike_gain = nn.Parameter(torch.ones(z_dim) * 5.0)
        else:
            self.threshold_encoder = None
            self.encoder = CNNEncoder(n_channels, n_bands, hidden=n_neurons, z_dim=z_dim)

        # Two LIF layers — same config as EEGSNN
        self.lif1 = LIFNeurons(z_dim, threshold=1.0, tau=2.0)
        self.lif2 = LIFNeurons(z_dim, threshold=0.3, tau=2.0)

        self.readout = MeanReadout(z_dim, n_classes)

    def forward(self, x: torch.Tensor, adaptation_vec: Optional[torch.Tensor] = None):
        B = x.shape[0]
        T = self.time_steps

        if self.trainable_encoding:
            # Encode with learnable thresholds → project to z_dim
            x_enc = self.threshold_encoder(x)  # (B, C*bands)
            h = self.spike_proj(x_enc) * self.spike_gain  # (B, z_dim)
        else:
            h = self.encoder(x)  # (B, z_dim)

        h = h.unsqueeze(1).expand(B, T, -1)  # (B, T, z_dim)

        mem1, mem2 = None, None
        spikes_out = []
        for t in range(T):
            x_t = h[:, t, :]
            sp1, mem1 = self.lif1(x_t, mem1)
            sp2, mem2 = self.lif2(sp1, mem2)
            spikes_out.append(sp2)
        spikes = torch.stack(spikes_out, dim=1)  # (B, T, z_dim)

        return self.readout(spikes)


# ─────────────────────────────────────────────
# Low-dimensional dynamics adaptation (Innovation 2)
# ─────────────────────────────────────────────
class LowDimAdaptationMLP(nn.Module):
    """
    Maps a low-dimensional adaptation vector q_s to multiplicative modulation
    on the LIF output spikes. This avoids in-place parameter changes that
    break autograd graphs.
    """
    def __init__(self, in_dim=8, n_neurons=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, n_neurons),
        )

    def forward(self, q: torch.Tensor):
        """
        q: (B, in_dim) adaptation vector
        Returns: modulation tensor (B, n_neurons) ∈ [0,1]
        """
        mod = torch.sigmoid(self.fc(q))  # (B, n_neurons)
        return mod


class EEGSNNWithAdaptation(nn.Module):
    """
    M2/M3: SNN with low-dimensional dynamics adaptation.
    Uses output modulation (not in-place parameter changes) for stability.
    """
    def __init__(self, n_channels=51, n_bands=5, z_dim=32, n_classes=3,
                 n_neurons=64, time_steps=8, readout_type="mean",
                 adapt_dim=8, trainable_encoding=False):
        super().__init__()
        self.time_steps = time_steps
        self.adapt_dim = adapt_dim

        if trainable_encoding:
            self.threshold_encoder = TrainableThresholdEncoder(
                n_channels, n_bands, threshold_init=0.0)
            self.spike_proj = nn.Sequential(
                nn.Linear(n_channels * n_bands, z_dim),
                nn.BatchNorm1d(z_dim),
                nn.ReLU(),
            )
            self.spike_gain = nn.Parameter(torch.ones(z_dim) * 5.0)
        else:
            self.threshold_encoder = None
            self.encoder = CNNEncoder(n_channels, n_bands, hidden=n_neurons, z_dim=z_dim)

        # LIF layers — same config as EEGSNN (tau=2.0)
        self.lif1 = LIFNeurons(z_dim, threshold=1.0, tau=2.0)
        self.lif2 = LIFNeurons(z_dim, threshold=0.3, tau=2.0)

        # Adaptation: modulates lif2 output spikes
        self.adapt_mlp = LowDimAdaptationMLP(in_dim=adapt_dim, n_neurons=z_dim)

        # Learnable per-subject adaptation vector
        self.q_s = nn.Parameter(torch.zeros(adapt_dim))

        self.readout = MeanReadout(z_dim, n_classes)

    def encode(self, x):
        B = x.shape[0]
        T = self.time_steps
        if self.threshold_encoder is not None:
            x_enc = self.threshold_encoder(x)
            h = self.spike_proj(x_enc) * self.spike_gain
        else:
            h = self.encoder(x)
        h = h.unsqueeze(1).expand(B, T, -1)
        return h

    def forward(self, x: torch.Tensor, q: Optional[torch.Tensor] = None):
        """
        x: (B, C*bands) DE features
        q: optional (B, adapt_dim) adaptation vector; uses self.q_s if None
        Returns: logits (B, n_classes)
        """
        B = x.shape[0]
        T = self.time_steps

        # Encode
        h = self.encode(x)

        # Get adaptation modulation
        if q is None:
            q = self.q_s.unsqueeze(0).expand(B, -1)
        mod = self.adapt_mlp(q)  # (B, z_dim)

        mem1, mem2 = None, None
        spikes_out = []
        for t in range(T):
            x_t = h[:, t, :]
            sp1, mem1 = self.lif1(x_t, mem1)
            sp2, mem2 = self.lif2(sp1, mem2)
            # Apply adaptation as multiplicative modulation on sp2
            sp2 = sp2 * mod
            spikes_out.append(sp2)
        spikes = torch.stack(spikes_out, dim=1)

        return self.readout(spikes)


# ─────────────────────────────────────────────
# Lightweight ANN baseline (same architecture, no spikes)
# ─────────────────────────────────────────────
class EEGCNN(nn.Module):
    """ANN with same encoder architecture as SNN for fair comparison."""
    def __init__(self, n_channels=51, n_bands=5, hidden=64, z_dim=32, n_classes=3):
        super().__init__()
        self.encoder = CNNEncoder(n_channels, n_bands, hidden=hidden, z_dim=z_dim)
        self.classifier = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.classifier(h)


# ─────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────
def loso_split(X, y, subj, test_subj):
    """LOSO split: test on test_subj, train on all others."""
    train_mask = subj != test_subj
    test_mask = subj == test_subj
    return X[train_mask], y[train_mask], X[test_mask], y[test_mask]


def per_subject_split(X, y, subj, val_ratio=0.2, seed=42):
    """Split within each subject for validation monitoring."""
    np.random.seed(seed)
    X_tr, y_tr, X_va, y_va = [], [], [], []
    for s in np.unique(subj):
        mask = subj == s
        xs, ys = X[mask], y[mask]
        n = len(xs)
        perm = np.random.permutation(n)
        k = max(1, int(n * val_ratio))
        va_idx = perm[:k]
        tr_idx = perm[k:]
        X_tr.append(xs[tr_idx]); y_tr.append(ys[tr_idx])
        X_va.append(xs[va_idx]); y_va.append(ys[va_idx])
    return (np.concatenate(X_tr), np.concatenate(y_tr),
            np.concatenate(X_va), np.concatenate(y_va))


def subject_aware_collate(batch_data, batch_labels, batch_subjects, max_subjects_per_batch=8):
    """
    Group by subject and sample equally to avoid one subject dominating a batch.
    """
    unique_subjs = np.unique(batch_subjects)
    np.random.shuffle(unique_subjs)
    outputs = []
    for s in unique_subjs[:max_subjects_per_batch]:
        mask = batch_subjects == s
        outputs.append((batch_data[mask], batch_labels[mask]))
    # Interleave subjects
    data_chunks = [o[0] for o in outputs]
    label_chunks = [o[1] for o in outputs]
    max_len = max(len(d) for d in data_chunks)
    # Pad and stack
    X_batched = []
    y_batched = []
    for d, l in zip(data_chunks, label_chunks):
        pad_len = max_len - len(d)
        if pad_len > 0:
            d = np.concatenate([d, np.zeros((pad_len, d.shape[1]))], axis=0)
            l = np.concatenate([l, np.zeros(pad_len, dtype=int)], axis=0)
        X_batched.append(d)
        y_batched.append(l)
    X_b = torch.tensor(np.stack(X_batched, 1).reshape(-1, X_batched[0].shape[1]), dtype=torch.float32)
    y_b = torch.tensor(np.concatenate(y_batched), dtype=torch.long)
    return X_b, y_b


def train_model(model, X_tr, y_tr, X_va, y_va,
                epochs=100, lr=1e-3, batch_size=32, weight_decay=1e-4,
                seed=0, verbose=False):
    """Standard training loop with validation monitoring."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = DEVICE
    model = model.to(device)

    # Per-subject normalization
    subj_ids = np.unique(np.tile(np.arange(len(X_tr) // 15 + 1), 15))[:len(X_tr)]

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.LongTensor(y_tr).to(device)
    X_va_t = torch.FloatTensor(X_va).to(device)
    y_va_t = torch.LongTensor(y_va).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()

    best_va_acc = -1
    best_state = copy.deepcopy(model.state_dict())
    patience = 20
    no_improve = 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=device)
        n_batches = 0
        epoch_loss = 0.0

        for i in range(0, len(perm), batch_size):
            idx = perm[i:i+batch_size]
            if len(idx) < 2:
                continue  # skip: BatchNorm needs ≥2 samples
            xb, yb = X_tr_t[idx], y_tr_t[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        sched.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(X_va_t)
            va_acc = (va_logits.argmax(1) == y_va_t).float().mean().item()
            tr_logits = model(X_tr_t[:512])
            tr_acc = (tr_logits.argmax(1) == y_tr_t[:512]).float().mean().item()

        if va_acc > best_va_acc:
            best_va_acc = va_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (ep + 1) % 20 == 0:
            print(f"  ep {ep+1:3d} | loss {epoch_loss/n_batches:.4f} "
                  f"| tr {tr_acc:.3f} va {va_acc:.3f} | best {best_va_acc:.3f}")

        if no_improve >= patience:
            if verbose:
                print(f"  Early stop at ep {ep+1}")
            break

    # Load best state for caller to use directly; don't call model.load_state_dict here
    # (snntorch Leaky has internal buffers that create graph references)
    model.load_state_dict(best_state)
    return best_va_acc, best_state


def evaluate_model(model, X_te, y_te, device=None):
    """Evaluate on test set."""
    if device is None:
        device = DEVICE
    model.eval()
    X_te_t = torch.FloatTensor(X_te).to(device)
    y_te_t = torch.LongTensor(y_te).to(device)
    with torch.no_grad():
        logits = model(X_te_t)
        preds = logits.argmax(1)
        acc = (preds == y_te_t).float().mean().item()
    return acc, preds.cpu().numpy()


def count_operations(model, X, device=None):
    """
    Estimate synaptic operations for SNN.
    Returns: (n_syn_ops, spike_rate) per component.
    For LIF: count = FLOPs for membrane update + reset.
    """
    if device is None:
        device = DEVICE
    model.eval()
    X_t = torch.FloatTensor(X[:32]).to(device)

    # Hook to count spikes
    spike_counts = {}
    def hook_spike(name):
        def _hook(module, input, output):
            if isinstance(output, tuple):
                spike_counts[name] = output[0].sum().item()
            else:
                spike_counts[name] = output.sum().item()
        return _hook

    handles = []
    for name, module in model.named_modules():
        if 'lif' in name.lower():
            handles.append(module.register_forward_hook(hook_spike(name)))

    with torch.no_grad():
        _ = model(X_t)

    for h in handles:
        h.remove()

    total_spikes = sum(spike_counts.values())
    n_steps = model.time_steps if hasattr(model, 'time_steps') else 8
    n_samples = X.shape[0]
    avg_spike_rate = total_spikes / (n_samples * n_steps)
    return total_spikes, avg_spike_rate, spike_counts


# ─────────────────────────────────────────────
# Main experiment runner
# ─────────────────────────────────────────────
@dataclass
class SNNConfig:
    model_type: str = "M0"         # M0, M1, M2, M3
    z_dim: int = 32
    n_neurons: int = 64
    time_steps: int = 8
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 32
    val_ratio: float = 0.2
    weight_decay: float = 1e-4
    adapt_dim: int = 8             # for M2/M3
    n_calibration: int = 1         # shots per class for M2/M3 adaptation
    seed: int = 42
    n_subjects: int = 15
    sessions: tuple = (1, 2, 3)
    preproc_code_version: int = 5

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if not k.startswith('_')}


def build_model(config: SNNConfig, n_channels=51, n_bands=5, n_classes=3):
    """Factory to build model by type."""
    kwargs = dict(
        n_channels=n_channels, n_bands=n_bands,
        z_dim=config.z_dim, n_classes=n_classes,
        n_neurons=config.n_neurons, time_steps=config.time_steps,
    )
    if config.model_type == "M0":
        return EEGSNN(**kwargs)
    elif config.model_type == "M1":
        return EEGSNNWithEncoding(**kwargs, trainable_encoding=True)
    elif config.model_type == "M2":
        return EEGSNNWithAdaptation(**kwargs, trainable_encoding=False, adapt_dim=config.adapt_dim)
    elif config.model_type == "M3":
        return EEGSNNWithAdaptation(**kwargs, trainable_encoding=True, adapt_dim=config.adapt_dim)
    elif config.model_type == "ANN":
        return EEGCNN(n_channels=n_channels, n_bands=n_bands,
                      hidden=config.n_neurons, z_dim=config.z_dim, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def run_loso(config: SNNConfig, subjects=None, sessions=(1, 2, 3),
             preproc_cfg=None, results_dir=None):
    """
    Run full LOSO experiment for a given model type.
    Returns dict with per-subject results and summary.
    """
    if preproc_cfg is None:
        preproc_cfg = PreprocConfig()
    if results_dir is None:
        results_dir = Path(RESULTS_DIR) / "snn_baseline"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"[{config.model_type}] Loading data...")
    X_all, y_all, subj_all = build_de_dataset(
        subjects=subjects, sessions=sessions, preproc_cfg=preproc_cfg)
    print(f"  Data: X={X_all.shape}, y={y_all.shape}, n_subjects={len(np.unique(subj_all))}")

    subjects = subjects or list(range(1, 16))
    rows = []
    total_start = time.time()

    for test_subj in subjects:
        t0 = time.time()
        X_tr, y_tr, X_te, y_te = loso_split(X_all, y_all, subj_all, test_subj)

        # Per-subject normalization
        mu_tr = X_tr.mean(0, keepdims=True)
        std_tr = X_tr.std(0, keepdims=True) + 1e-8
        X_tr = (X_tr - mu_tr) / std_tr
        X_te = (X_te - mu_tr) / std_tr

        # Within-subject val split for model selection
        X_tr_s, y_tr_s, X_va_s, y_va_s = per_subject_split(
            X_tr, y_tr, np.full(len(y_tr), 0), val_ratio=config.val_ratio, seed=config.seed)

        # Build and train model (with early stopping on val split)
        model = build_model(config)
        best_va_acc, best_state = train_model(
            model, X_tr_s, y_tr_s, X_va_s, y_va_s,
            epochs=config.epochs, lr=config.lr,
            batch_size=config.batch_size, weight_decay=config.weight_decay,
            seed=config.seed, verbose=False)
        va_acc = best_va_acc

        # Re-train on full training set (same early stopping mechanism)
        model_full = build_model(config)
        best_full_va_acc, best_full_state = train_model(
            model_full, X_tr, y_tr, X_va_s, y_va_s,
            epochs=config.epochs, lr=config.lr,
            batch_size=config.batch_size, weight_decay=config.weight_decay,
            seed=config.seed, verbose=False)

        # Use the early-stopped checkpoint from step 2 (which trained on full data)
        model_full.load_state_dict(best_full_state)
        te_acc, preds = evaluate_model(model_full, X_te, y_te)

        # Count operations
        try:
            total_ops, spike_rate, _ = count_operations(model_full, X_te)
        except Exception:
            total_ops, spike_rate = -1, -1.0

        row = dict(
            subject=test_subj,
            val_acc=va_acc,
            test_acc=te_acc,
            spike_ops=int(total_ops) if total_ops >= 0 else -1,
            spike_rate=float(spike_rate) if spike_rate >= 0 else -1.0,
            n_train=len(y_tr),
            n_test=len(y_te),
            time_s=time.time() - t0,
        )
        rows.append(row)
        print(f"  subj {test_subj:2d}: va={va_acc:.3f} te={te_acc:.3f} "
              f"spike_ops={total_ops:.0f} rate={spike_rate:.2f} [{row['time_s']:.1f}s]")

    total_time = time.time() - total_start

    # Summary
    test_accs = [r['test_acc'] for r in rows]
    val_accs = [r['val_acc'] for r in rows]
    summary = dict(
        config=config.to_dict(),
        model_type=config.model_type,
        mean_test_acc=np.mean(test_accs),
        std_test_acc=np.std(test_accs),
        mean_val_acc=np.mean(val_accs),
        per_subject=test_accs,
        total_time_s=total_time,
    )

    # Save
    out_file = results_dir / f"{config.model_type}_rows.json"
    with open(out_file, 'w') as f:
        json.dump(dict(rows=rows, summary=summary), f, indent=2)

    print(f"\n[{config.model_type}] Summary: test_acc={summary['mean_test_acc']:.3f}±{summary['std_test_acc']:.3f} "
          f"({len(test_accs)} subjects, {total_time:.0f}s total)")
    return rows, summary


def run_all_models(model_types=("M0", "M1", "M2", "M3"), subjects=None, sessions=(1, 2, 3),
                   preproc_cfg=None, results_dir=None, **kwargs):
    """Run all four model types and compare."""
    all_summaries = {}
    for mt in model_types:
        config = SNNConfig(model_type=mt, **kwargs)
        rows, summary = run_loso(config, subjects=subjects, sessions=sessions,
                                  preproc_cfg=preproc_cfg, results_dir=results_dir)
        all_summaries[mt] = summary
        # Save comparison
        comp_file = Path(results_dir if results_dir else RESULTS_DIR) / "snn_baseline" / "comparison.json"
        with open(comp_file, 'w') as f:
            json.dump(all_summaries, f, indent=2)
    return all_summaries


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="M0", choices=["M0", "M1", "M2", "M3", "ANN", "all"])
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 4)))  # smoke test: 3 subjects
    parser.add_argument("--epochs", type=int, default=30)  # quick smoke
    parser.add_argument("--z-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--time-steps", type=int, default=8)
    args = parser.parse_args()

    print(f"Config: model={args.model}, subjects={args.subjects}, epochs={args.epochs}")
    print(f"Device: {DEVICE}")

    preproc_cfg = PreprocConfig(win_s=4.0, overlap=0.0, drop_bad_channels=True,
                                  reject_abs_thr=15.0)

    if args.model == "all":
        run_all_models(
            model_types=("M0", "M1", "M2", "M3"),
            subjects=args.subjects,
            sessions=(1,),  # smoke: session 1 only
            preproc_cfg=preproc_cfg,
            epochs=args.epochs,
            z_dim=args.z_dim,
            lr=args.lr,
            time_steps=args.time_steps,
        )
    else:
        config = SNNConfig(
            model_type=args.model,
            n_subjects=len(args.subjects),
            sessions=(1,),
            epochs=args.epochs,
            z_dim=args.z_dim,
            lr=args.lr,
            time_steps=args.time_steps,
        )
        run_loso(config, subjects=args.subjects, sessions=(1,),
                 preproc_cfg=preproc_cfg)
