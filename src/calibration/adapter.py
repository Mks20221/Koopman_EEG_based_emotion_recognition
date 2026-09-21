# -*- coding: utf-8 -*-
"""Calibration adapter: fine-tune a pre-trained MLP on new subject data.

Provides an explicit interface for updating the model given visible labeled data,
so it can be driven by a fixed strategy (this round) or by a future RL agent.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class CalibConfig:
    lr: float = 1e-4
    steps: int = 5
    batch_size: int = 32
    weight_decay: float = 1e-4


class SimpleMLP(nn.Module):
    """Shallow MLP for window-level DE classification.

    Architecture:
        input(310) -> dropout -> linear(128) -> relu -> dropout -> linear(3)
    """

    def __init__(self, in_features: int = 310, hidden: int = 128, classes: int = 3):
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, classes)
        self.in_features = in_features
        self.hidden = hidden
        self.classes = classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def feature_parameters(self):
        """Parameters of the feature extractor (fc1)."""
        return [self.fc1.weight, self.fc1.bias]

    def classifier_parameters(self):
        """Parameters of the classification head (fc2)."""
        return [self.fc2.weight, self.fc2.bias]


class TrialEqualLossCalculator:
    """Helper to compute per-trial-equal-weighted CE loss.

    Given X, y, and trial_ids (all same length), computes:
        L = (1/T) * sum_t mean(CE(windows_in_trial_t))
    where T is the number of distinct trials, not classes.
    """

    @staticmethod
    def weights(trial_ids, device):
        """One unit of total weight per distinct subject/trial, then normalize."""
        if torch.is_tensor(trial_ids):
            trial_ids = trial_ids.detach().cpu().numpy()
        _, inverse, counts = np.unique(trial_ids, return_inverse=True, return_counts=True)
        return torch.as_tensor(1.0 / (len(counts) * counts[inverse]),
                               dtype=torch.float32, device=device)

    @staticmethod
    def compute(model, X_t, y_t, trial_ids, device, weights=None):
        """Compute trial-equal-weighted CE loss (requires grad enabled)."""
        logits = model(X_t.to(device))  # (N, 3)
        ce_per_window = F.cross_entropy(logits, y_t.to(device), reduction='none')  # (N,)

        if weights is None:
            weights = TrialEqualLossCalculator.weights(trial_ids, device)
        return (ce_per_window * weights).sum()


class CalibrationAdapter:
    """Encapsulates continuous calibration update from a source model.

    One adapter = one model copy + one optimizer (initialized once).
    The optimizer state is preserved across apply_fixed_head_continuous calls.
    """

    def __init__(self, model: SimpleMLP, config: CalibConfig, device: str = "cpu"):
        self.source_model = copy.deepcopy(model)
        self.model = copy.deepcopy(model)
        self.config = config
        self.device = device
        self.model.to(device)

        # Initialize optimizer ONCE; state is kept across calls
        for p in self.model.feature_parameters():
            p.requires_grad = False
        self._optimizer = torch.optim.Adam(
            self.model.classifier_parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    def reset_from_source(self):
        """Reset model to source, reinitialize optimizer."""
        self.model = copy.deepcopy(self.source_model)
        self.model.to(self.device)
        for p in self.model.feature_parameters():
            p.requires_grad = False
        self._optimizer = torch.optim.Adam(
            self.model.classifier_parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def _prepare_data(self, X, y, trial_ids=None):
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)
        trial_ids_t = torch.as_tensor(trial_ids, dtype=torch.long, device=self.device) \
            if trial_ids is not None else None
        return X_t, y_t, trial_ids_t

    def apply_fixed_head_continuous(
        self, X_visible, y_visible, trial_ids_visible, n_steps=None, lr=None
    ):
        """Fine-tune classifier head continuously from current model state.

        The feature extractor (fc1) is frozen throughout.
        The optimizer is initialized once at __init__ and retains momentum.
        One call = one feedback round (one trial).
        """
        steps = n_steps if n_steps is not None else self.config.steps
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        if lr is not None:
            for group in self._optimizer.param_groups:
                group['lr'] = lr
        if steps == 0:
            self.model.eval()
            return self.model

        X_t, y_t, trial_ids_t = self._prepare_data(X_visible, y_visible, trial_ids_visible)
        trial_ids_np = trial_ids_t.cpu().numpy()
        weights = TrialEqualLossCalculator.weights(trial_ids_np, self.device)

        self.model.train()   # enable dropout in training mode
        for step in range(steps):
            self._optimizer.zero_grad()
            loss = TrialEqualLossCalculator.compute(
                self.model, X_t, y_t, trial_ids_np, self.device, weights=weights
            )
            loss.backward()
            self._optimizer.step()

        self.model.eval()   # back to eval mode for evaluation
        return self.model

    def apply_none(self):
        """Return source model unchanged (deepcopy)."""
        return copy.deepcopy(self.source_model)


def batch_evaluate_window(
    model, X, y, device="cpu", batch_size=128
):
    """Window-level evaluation.

    Returns window-level accuracy and Macro-F1.
    """
    model.eval()
    model.to(device)
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[start:start+batch_size], dtype=torch.float32, device=device)
            logits = model(xb)
            all_preds.append(logits.argmax(-1).cpu().numpy())
    preds = np.concatenate(all_preds)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, labels=[0, 1, 2], average="macro", zero_division=0)
    return {"window_accuracy": float(acc), "window_macro_f1": float(f1)}, preds


def batch_evaluate_trial(
    model, X_by_trial, y_by_trial, device="cpu"
):
    """Trial-level evaluation: average logits per trial then argmax.

    X_by_trial : list of np.ndarray, each (n_win_i, 310)
    y_by_trial : list of np.ndarray, each (n_win_i,) — all same label
    """
    model.eval()
    model.to(device)
    trial_preds, trial_true = [], []
    with torch.no_grad():
        for X_trial, y_trial in zip(X_by_trial, y_by_trial):
            xb = torch.as_tensor(X_trial, dtype=torch.float32, device=device)
            logits = model(xb)  # (n_win, 3)
            # Average probabilities across windows in this trial
            probs = logits.softmax(-1).mean(0)  # (3,)
            trial_preds.append(probs.argmax(-1).item())
            trial_true.append(int(y_trial[0]))
    trial_preds = np.array(trial_preds)
    trial_true = np.array(trial_true)
    acc = accuracy_score(trial_true, trial_preds)
    f1 = f1_score(trial_true, trial_preds, labels=[0, 1, 2], average="macro", zero_division=0)
    return {"trial_accuracy": float(acc), "trial_macro_f1": float(f1)}, trial_preds


def train_source_model(
    X_train, y_train, trial_ids_train,
    X_val, y_val, trial_ids_val,
    scaler_mean, scaler_scale,
    device="cpu", hidden=128, epochs=50, lr=1e-3, weight_decay=1e-4,
    logger=None,
):
    """Train source MLP on training-subject window data with fitted scaler.

    Returns (trained_model, train_info dict).
    train_info: all metrics from the SAME best_checkpoint model, plus init model metrics.
    """
    # Apply scaler
    X_tr = (X_train - scaler_mean) / scaler_scale
    X_va = (X_val - scaler_mean) / scaler_scale

    model = SimpleMLP(in_features=X_train.shape[1], hidden=hidden, classes=3)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_tr_t = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    trial_tr_t = torch.as_tensor(trial_ids_train, dtype=torch.long, device=device)
    X_va_t = torch.as_tensor(X_va, dtype=torch.float32, device=device)
    y_va_t = torch.as_tensor(y_val, dtype=torch.long, device=device)
    trial_va_t = torch.as_tensor(trial_ids_val, dtype=torch.long, device=device)

    # ---- Metrics for init (untrained) model ----
    model.eval()
    with torch.no_grad():
        init_train_loss = TrialEqualLossCalculator.compute(
            model, X_tr_t, y_tr_t, trial_tr_t.cpu().numpy(), device
        ).item()
        logits_init = model(X_tr_t)
        preds_init = logits_init.argmax(-1).cpu().numpy()
        train_acc_init = float((preds_init == y_tr_t.cpu().numpy()).mean())
        init_val_loss = TrialEqualLossCalculator.compute(
            model, X_va_t, y_va_t, trial_va_t.cpu().numpy(), device
        ).item()

    # ---- Train ----
    best_model = copy.deepcopy(model)
    best_val_score = float("inf")   # lower val loss = better; init to +inf so epoch-0 is captured
    best_epoch = 0
    best_val_loss_at_best = init_val_loss
    best_train_loss_at_best = init_train_loss
    best_train_acc_at_best = train_acc_init
    final_model = copy.deepcopy(model)
    final_train_loss = init_train_loss

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(y_tr_t), device=device)
        X_tr_t = torch.index_select(X_tr_t, 0, perm)
        y_tr_t = torch.index_select(y_tr_t, 0, perm)
        trial_tr_t = torch.index_select(trial_tr_t, 0, perm)

        optimizer.zero_grad()
        loss = TrialEqualLossCalculator.compute(
            model, X_tr_t, y_tr_t, trial_tr_t.cpu().numpy(), device
        )
        loss.backward()
        optimizer.step()
        final_train_loss = loss.item()
        final_model = copy.deepcopy(model)

        # Evaluate on validation
        model.eval()
        with torch.no_grad():
            val_loss = TrialEqualLossCalculator.compute(
                model, X_va_t, y_va_t, trial_va_t.cpu().numpy(), device
            ).item()
            val_score = val_loss  # lower = better

        if val_score <= best_val_score:
            best_val_score = val_score
            best_model = copy.deepcopy(model)
            best_epoch = ep
            best_val_loss_at_best = val_loss
            # compute train metrics on best model
            best_model.eval()
            with torch.no_grad():
                logits_best = best_model(X_tr_t)
                preds_best = logits_best.argmax(-1).cpu().numpy()
                best_train_loss_at_best = TrialEqualLossCalculator.compute(
                    best_model, X_tr_t, y_tr_t, trial_tr_t.cpu().numpy(), device
                ).item()
                best_train_acc_at_best = float((preds_best == y_tr_t.cpu().numpy()).mean())

    # ---- Regression check: all val losses > 1 ----
    if best_val_loss_at_best > 1.0:
        if logger:
            logger(
                f"  [REGRESSION WARNING] best_val_loss={best_val_loss_at_best:.4f} > 1.0 "
                f"(suggests severe distribution mismatch)"
            )

    train_info = {
        # init model
        "init_train_loss": float(init_train_loss),
        "init_val_loss": float(init_val_loss),
        "train_acc_init": float(train_acc_init),
        # best checkpoint model (selected by val loss)
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss_at_best),
        "best_train_loss": float(best_train_loss_at_best),
        "train_acc_best": float(best_train_acc_at_best),
        # last epoch model
        "final_train_loss": float(final_train_loss),
        # checkpoint selection tracking
        "selected_from_epoch_after_init": True,
        "selected_epoch_one_based": int(best_epoch) + 1,
    }

    if logger:
        logger(
            f"  Source: init_loss={init_train_loss:.4f} val={init_val_loss:.4f} acc={train_acc_init:.4f} | "
            f"best_ep={best_epoch} val_loss={best_val_loss_at_best:.4f} | "
            f"final_loss={final_train_loss:.4f} | "
            f"selected_trained_epoch={best_epoch + 1}"
        )

    return best_model, train_info
