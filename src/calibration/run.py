# -*- coding: utf-8 -*-
"""Per-trial calibration experiment on SEED ExtractedFeatures DE data.

Run from project root:
    python -m src.calibration.run

Experiment design (per subject):
  - 15-fold LOSO: each fold has 1 test subject, 1 val subject, 13 train subjects
  - Source model: MLP trained on 13 source subjects (all windows, with scaler)
  - Calibration:
      * trials 1-9: calibration set (labels revealed sequentially)
      * trials 10-15: evaluation set (labels never revealed)
  - Methods: none (no update), fixed_head (freeze features, update classifier only)
  - Hyperparameter selection: on validation subject, continuous 9-step adaptation
  - Feedback budgets: {0, 3, 6, 9} with continuous model updates in between
  - Metrics: window-level and trial-level accuracy / Macro-F1
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROJECT_ROOT as CONFIG_PROJECT_ROOT
PROJECT_ROOT = Path(CONFIG_PROJECT_ROOT)
from src.calibration.data_loader import (
    load_de_session, flatten_session,
    SEED_EXT_ROOT, SEED_N_SUBJECTS, SEED_N_TRIALS,
)
from src.calibration.adapter import (
    SimpleMLP, CalibrationAdapter, CalibConfig,
    batch_evaluate_window, batch_evaluate_trial,
    train_source_model,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="SEED per-trial calibration experiment")
    parser.add_argument("--session", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--source_epochs", type=int, default=50)
    parser.add_argument("--source_lr", type=float, default=1e-3)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading (all windows, no aggregation)
# ---------------------------------------------------------------------------
def load_all_windows(session: int):
    """Load all windows for all 15 subjects.

    Returns dict: subject -> {
        X_list, y_list, trial_ids_list, window_ids_list, labels_raw,
        X_flat (all windows), y_flat, trial_ids_flat
    }
    """
    data = {}
    for s in range(1, SEED_N_SUBJECTS + 1):
        X_list, y_list, trial_ids_list, window_ids_list, raw = load_de_session(s, session)
        X_flat, y_flat, trial_ids_flat = flatten_session(X_list, y_list, trial_ids_list)
        data[s] = dict(
            X_list=X_list, y_list=y_list,
            trial_ids_list=trial_ids_list, window_ids_list=window_ids_list,
            labels_raw=raw,
            X_flat=X_flat, y_flat=y_flat, trial_ids_flat=trial_ids_flat,
        )
    return data


def build_folds(subjects: list[int]):
    """Build 15 LOSO folds. Each fold: test=one subject, val=next, train=rest."""
    folds = []
    for test_subj in subjects:
        test_idx = subjects.index(test_subj)
        val_subj = subjects[(test_idx + 1) % len(subjects)]
        train_subjs = [s for s in subjects if s not in (test_subj, val_subj)]
        folds.append(dict(test=test_subj, val=val_subj, train=train_subjs))
    return folds


# ---------------------------------------------------------------------------
# Scaler fitting
# ---------------------------------------------------------------------------
class WindowScaler:
    """StandardScaler fitted on training windows only."""

    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.scale = np.maximum(X.std(axis=0), 1e-8)
        return self

    def transform(self, X):
        return (X - self.mean) / self.scale

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ---------------------------------------------------------------------------
# HP selection on validation subject (continuous, same as test)
# ---------------------------------------------------------------------------
def run_hp_selection(source_model, scaler, val_data, calib_trials, eval_trials,
                     lr_choices, steps_choices, device):
    """Run full 9-step continuous adaptation for each HP candidate.

    Scaler is applied ONCE before any evaluation. Each HP candidate gets its
    own adapter (model copy + optimizer). Data is pre-normalized here.

    Returns best HP and per-candidate score history.
    """
    best_hp, best_score = None, -1.0
    all_scores = []

    # Normalize val data ONCE with the fold's scaler
    val_calib_X_norm = [
        scaler.transform(val_data['X_list'][t - 1]) for t in calib_trials
    ]
    val_calib_y = [val_data['y_list'][t - 1] for t in calib_trials]
    val_calib_tid = [val_data['trial_ids_list'][t - 1] for t in calib_trials]
    val_eval_X_norm = [
        scaler.transform(val_data['X_list'][t - 1]) for t in eval_trials
    ]
    val_eval_y = [val_data['y_list'][t - 1] for t in eval_trials]

    for lr in lr_choices:
        for steps in steps_choices:
            cfg = CalibConfig(lr=lr, steps=steps)
            adapter = CalibrationAdapter(source_model, cfg, device=device)
            # Evaluate at fb=0
            _, preds_0 = batch_evaluate_trial(adapter.model, val_eval_X_norm, val_eval_y, device)
            acc_0 = accuracy_score([y[0] for y in val_eval_y], preds_0)

            acc_at_fb = {}
            for fb_i in range(len(calib_trials)):
                adapter.apply_fixed_head_continuous(
                    np.concatenate(val_calib_X_norm[:fb_i + 1]),
                    np.concatenate(val_calib_y[:fb_i + 1]),
                    np.concatenate(val_calib_tid[:fb_i + 1]),
                    n_steps=steps,
                )
                fb = fb_i + 1
                if fb in (3, 6, 9):
                    _, preds = batch_evaluate_trial(
                        adapter.model, val_eval_X_norm, val_eval_y, device
                    )
                    acc_at_fb[fb] = accuracy_score([y[0] for y in val_eval_y], preds)
            acc_9 = acc_at_fb[9]

            score = np.mean([acc_at_fb.get(3, 0), acc_at_fb.get(6, 0), acc_at_fb.get(9, 0)])
            all_scores.append(dict(
                lr=lr, steps=steps, score=score,
                acc_0=acc_0, acc_3=acc_at_fb.get(3, 0),
                acc_6=acc_at_fb.get(6, 0), acc_9=acc_9,
            ))
            if score > best_score:
                best_score = score
                best_hp = dict(lr=lr, steps=steps)

    return best_hp, best_score, all_scores


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_calibration(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    results_dir = Path(args.results_dir or PROJECT_ROOT / "results" / "calibration" / run_id)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] run_id={run_id}")
    print(f"[{time.strftime('%H:%M:%S')}] device={device}, session={args.session}")

    config = dict(
        run_id=run_id,
        session=args.session,
        hidden=args.hidden,
        source_epochs=args.source_epochs,
        source_lr=args.source_lr,
        seed=args.seed,
        device=device,
        n_subjects=SEED_N_SUBJECTS,
        n_trials=SEED_N_TRIALS,
        feature="de_movingAve (62ch x 5bands = 310d)",
        architecture="MLP 310->dropout(0.3)->linear(128)->relu->dropout(0.3)->linear(3)",
        calib_lr_choices=[1e-4, 1e-3],
        calib_steps_choices=[5, 20],
        feedback_budgets=[0, 3, 6, 9],
        eval_trials=list(range(10, 16)),
        calib_trials=list(range(1, 10)),
    )
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"[{time.strftime('%H:%M:%S')}] Loading SEED session {args.session}...")
    all_data = load_all_windows(args.session)

    # Quick stats
    total_train_windows = sum(
        all_data[s]['X_flat'].shape[0] for s in range(1, 14)
    )
    print(f"  Fold 1 train windows (s3-s15): {total_train_windows}")
    print(f"  eval trials: {config['eval_trials']}, calib trials: {config['calib_trials']}")

    subjects = list(range(1, SEED_N_SUBJECTS + 1))
    folds = build_folds(subjects)

    calib_trials = np.array(config['calib_trials'])   # 1-9
    eval_trials = np.array(config['eval_trials'])     # 10-15

    all_results = []

    for fold_idx, fold in enumerate(folds):
        test_subj = fold["test"]
        val_subj = fold["val"]
        train_subjs = fold["train"]

        print(f"\n[{time.strftime('%H:%M:%S')}] Fold {fold_idx+1}/15 | test=s{test_subj}, val=s{val_subj}, train={train_subjs}")

        # ---- Fit scaler on training subjects only ----
        train_X_list = [all_data[s]['X_flat'] for s in train_subjs]
        train_y_list = [all_data[s]['y_flat'] for s in train_subjs]
        X_train_all = np.concatenate(train_X_list, axis=0)
        y_train_all = np.concatenate(train_y_list, axis=0)
        trial_train_raw = np.concatenate([s * 100 + all_data[s]['trial_ids_flat'] for s in train_subjs], axis=0)
        print(f"  Train windows: {X_train_all.shape[0]}")

        scaler = WindowScaler()
        scaler.fit(X_train_all)
        scaler_params = dict(mean=scaler.mean.tolist(), scale=scaler.scale.tolist())

        # ---- Train source model ----
        X_val_raw = all_data[val_subj]['X_flat']
        y_val_raw = all_data[val_subj]['y_flat']
        trial_val_raw = all_data[val_subj]['trial_ids_flat']

        source_model, train_info = train_source_model(
            X_train=X_train_all, y_train=y_train_all, trial_ids_train=trial_train_raw,
            X_val=X_val_raw, y_val=y_val_raw, trial_ids_val=trial_val_raw,
            scaler_mean=scaler.mean, scaler_scale=scaler.scale,
            device=device, hidden=args.hidden,
            epochs=args.source_epochs, lr=args.source_lr,
        )
        print(
            f"  Source: init_loss={train_info['init_train_loss']:.4f} "
            f"val={train_info['init_val_loss']:.4f} acc={train_info['train_acc_init']:.4f} | "
            f"best_ep={train_info['best_epoch']} val_loss={train_info['best_val_loss']:.4f} | "
            f"final_loss={train_info['final_train_loss']:.4f} | "
            f"after_init={'YES' if train_info['selected_from_epoch_after_init'] else 'NO (epoch-0 selected)'}"
        )

        # ---- HP selection on validation subject ----
        val_data = all_data[val_subj]
        best_hp, best_score, hp_log = run_hp_selection(
            source_model, scaler, val_data, calib_trials, eval_trials,
            config["calib_lr_choices"], config["calib_steps_choices"], device,
        )
        print(f"  HP val: best_hp={best_hp}, score={best_score:.4f}")

        # ---- Prepare test subject data (normalized ONCE with fold's scaler) ----
        test_data = all_data[test_subj]
        test_calib_X_norm = [
            scaler.transform(test_data['X_list'][t - 1]) for t in calib_trials
        ]
        test_calib_y = [test_data['y_list'][t - 1] for t in calib_trials]
        test_calib_tid = [test_data['trial_ids_list'][t - 1] for t in calib_trials]
        test_eval_X_norm = [
            scaler.transform(test_data['X_list'][t - 1]) for t in eval_trials
        ]
        test_eval_y = [test_data['y_list'][t - 1] for t in eval_trials]
        test_eval_trial_ids = [t for t in eval_trials]   # trial numbers for CSV

        cfg_best = CalibConfig(lr=best_hp["lr"], steps=best_hp["steps"])

        fold_record = dict(
            subject=test_subj, val_subject=val_subj,
            best_hp=best_hp, scaler_params=scaler_params,
            train_info=train_info,
            hp_selection_log=hp_log,
            none={}, fixed_head={},
            none_window={}, fixed_head_window={},
            predictions=[],   # per-row dicts: {method, fb, trial_id, y_true, y_pred, window_idx}
        )

        # Helper: collect predictions for a model and set of normalized X
        def collect_trial_predictions(model, X_norm_list, y_list, trial_nums, method, fb):
            rows = []
            for trial_idx, (X_trial_norm, y_trial, trial_num) in enumerate(
                    zip(X_norm_list, y_list, trial_nums)):
                model.eval()
                xb = torch.as_tensor(X_trial_norm, dtype=torch.float32, device=device)
                with torch.no_grad():
                    logits = model(xb)
                    probs = logits.softmax(-1).cpu().numpy()   # (n_win, 3)
                    window_preds = logits.argmax(-1).cpu().numpy()   # (n_win,)
                # Trial prediction: average probabilities
                trial_prob_avg = probs.mean(0)
                trial_pred = int(trial_prob_avg.argmax())
                y_true = int(y_trial[0])
                for win_idx, (w_pred, w_prob) in enumerate(zip(window_preds, probs)):
                    rows.append(dict(
                        method=method,
                        feedback_budget=fb,
                        trial_id=int(trial_num),
                        window_idx=int(win_idx),
                        y_true=int(y_true),
                        y_pred_window=int(w_pred),
                        y_pred_trial=int(trial_pred),
                        trial_prob_0=float(trial_prob_avg[0]),
                        trial_prob_1=float(trial_prob_avg[1]),
                        trial_prob_2=float(trial_prob_avg[2]),
                    ))
            return rows

        # === none (source model, no adaptation) ===
        adapter_none = CalibrationAdapter(source_model, cfg_best, device=device)
        model_none = adapter_none.apply_none()
        model_none.eval()
        metrics_win_none, _ = batch_evaluate_window(
            model_none,
            np.concatenate(test_eval_X_norm, axis=0),
            np.concatenate(test_eval_y, axis=0),
            device=device,
        )
        metrics_trial_none, _ = batch_evaluate_trial(
            model_none, test_eval_X_norm, test_eval_y, device=device,
        )
        fold_record["none"][0] = metrics_trial_none
        fold_record["none_window"][0] = metrics_win_none
        fold_record["predictions"].extend(collect_trial_predictions(
            model_none, test_eval_X_norm, test_eval_y, test_eval_trial_ids, "none", 0))

        # === fixed_head: continuous adaptation ===
        adapter_fh = CalibrationAdapter(source_model, cfg_best, device=device)

        # fb=0 (same as source model before any updates)
        metrics_trial_0, _ = batch_evaluate_trial(
            adapter_fh.model, test_eval_X_norm, test_eval_y, device=device,
        )
        metrics_win_0, _ = batch_evaluate_window(
            adapter_fh.model,
            np.concatenate(test_eval_X_norm, axis=0),
            np.concatenate(test_eval_y, axis=0),
            device=device,
        )
        fold_record["fixed_head"][0] = metrics_trial_0
        fold_record["fixed_head_window"][0] = metrics_win_0
        fold_record["predictions"].extend(collect_trial_predictions(
            adapter_fh.model, test_eval_X_norm, test_eval_y, test_eval_trial_ids, "fixed_head", 0))

        print(
            f"  s{test_subj} fixed_head fb=0: "
            f"trial_acc={metrics_trial_0['trial_accuracy']:.4f} "
            f"trial_f1={metrics_trial_0['trial_macro_f1']:.4f} "
            f"win_acc={metrics_win_0['window_accuracy']:.4f} "
            f"win_f1={metrics_win_0['window_macro_f1']:.4f}"
        )

        # Continuous updates: one per trial, model+optimizer state persists
        for fb_i in range(len(calib_trials)):
            adapter_fh.apply_fixed_head_continuous(
                np.concatenate(test_calib_X_norm[:fb_i + 1]),
                np.concatenate(test_calib_y[:fb_i + 1]),
                np.concatenate(test_calib_tid[:fb_i + 1]),
                n_steps=best_hp["steps"],
            )
            fb = fb_i + 1
            if fb in (3, 6, 9):
                metrics_trial, _ = batch_evaluate_trial(
                    adapter_fh.model, test_eval_X_norm, test_eval_y, device=device,
                )
                metrics_win, _ = batch_evaluate_window(
                    adapter_fh.model,
                    np.concatenate(test_eval_X_norm, axis=0),
                    np.concatenate(test_eval_y, axis=0),
                    device=device,
                )
                fold_record["fixed_head"][fb] = metrics_trial
                fold_record["fixed_head_window"][fb] = metrics_win
                fold_record["predictions"].extend(collect_trial_predictions(
                    adapter_fh.model, test_eval_X_norm, test_eval_y,
                    test_eval_trial_ids, "fixed_head", fb))
                print(
                    f"  s{test_subj} fixed_head fb={fb}: "
                    f"trial_acc={metrics_trial['trial_accuracy']:.4f} "
                    f"trial_f1={metrics_trial['trial_macro_f1']:.4f} "
                    f"win_acc={metrics_win['window_accuracy']:.4f} "
                    f"win_f1={metrics_win['window_macro_f1']:.4f}"
                )

        # Ensure fb=9 is recorded (may already be set by loop above)
        if 9 not in fold_record["fixed_head"]:
            metrics_trial, _ = batch_evaluate_trial(
                adapter_fh.model, test_eval_X_norm, test_eval_y, device=device,
            )
            metrics_win, _ = batch_evaluate_window(
                adapter_fh.model,
                np.concatenate(test_eval_X_norm, axis=0),
                np.concatenate(test_eval_y, axis=0),
                device=device,
            )
            fold_record["fixed_head"][9] = metrics_trial
            fold_record["fixed_head_window"][9] = metrics_win
            fold_record["predictions"].extend(collect_trial_predictions(
                adapter_fh.model, test_eval_X_norm, test_eval_y,
                test_eval_trial_ids, "fixed_head", 9))
            print(
                f"  s{test_subj} fixed_head fb=9: "
                f"trial_acc={metrics_trial['trial_accuracy']:.4f} "
                f"trial_f1={metrics_trial['trial_macro_f1']:.4f} "
                f"win_acc={metrics_win['window_accuracy']:.4f} "
                f"win_f1={metrics_win['window_macro_f1']:.4f}"
            )

        print(
            f"  s{test_subj} none fb=0: "
            f"trial_acc={metrics_trial_none['trial_accuracy']:.4f} "
            f"trial_f1={metrics_trial_none['trial_macro_f1']:.4f} "
            f"win_acc={metrics_win_none['window_accuracy']:.4f} "
            f"win_f1={metrics_win_none['window_macro_f1']:.4f}"
        )

        # Verify: check that fc1 is frozen during adaptation
        with torch.no_grad():
            fc1_diff = (adapter_fh.model.fc1.weight - source_model.fc1.weight).abs().max().item()
        if fc1_diff > 1e-8:
            print(f"  [WARNING] fc1 weight changed during adaptation: max_diff={fc1_diff:.2e}")
        else:
            print(f"  fc1 frozen: max_diff={fc1_diff:.2e}")

        all_results.append(fold_record)

        with open(results_dir / "run.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # ---- Aggregate results ----
    print(f"\n[{time.strftime('%H:%M:%S')}] Aggregating results...")

    summary = {}
    for fb in [0, 3, 6, 9]:
        none_trial_acc = [r["none"].get(fb, r["none"][0])["trial_accuracy"] for r in all_results]
        none_trial_f1 = [r["none"].get(fb, r["none"][0])["trial_macro_f1"] for r in all_results]
        fh_trial_acc = [r["fixed_head"].get(fb, r["fixed_head"][0])["trial_accuracy"] for r in all_results]
        fh_trial_f1 = [r["fixed_head"].get(fb, r["fixed_head"][0])["trial_macro_f1"] for r in all_results]
        none_win_acc = [r["none_window"].get(fb, r["none_window"][0])["window_accuracy"] for r in all_results]
        fh_win_acc = [r["fixed_head_window"].get(fb, r["fixed_head_window"][0])["window_accuracy"] for r in all_results]
        diff_trial_acc = [fh_trial_acc[i] - none_trial_acc[i] for i in range(len(all_results))]

        summary[fb] = {
            "trial": {
                "none": {
                    "accuracy_mean": float(np.mean(none_trial_acc)),
                    "accuracy_std": float(np.std(none_trial_acc)),
                    "macro_f1_mean": float(np.mean(none_trial_f1)),
                    "macro_f1_std": float(np.std(none_trial_f1)),
                },
                "fixed_head": {
                    "accuracy_mean": float(np.mean(fh_trial_acc)),
                    "accuracy_std": float(np.std(fh_trial_acc)),
                    "macro_f1_mean": float(np.mean(fh_trial_f1)),
                    "macro_f1_std": float(np.std(fh_trial_f1)),
                },
                "diff": {
                    "accuracy_mean": float(np.mean(diff_trial_acc)),
                    "accuracy_std": float(np.std(diff_trial_acc)),
                },
            },
            "window": {
                "none_accuracy_mean": float(np.mean(none_win_acc)),
                "fh_accuracy_mean": float(np.mean(fh_win_acc)),
            },
        }

    subject_table = []
    for r in all_results:
        row = dict(subject=r["subject"])
        for fb in [0, 3, 6, 9]:
            row[f"none_{fb}_trial_acc"] = r["none"].get(fb, r["none"][0])["trial_accuracy"]
            row[f"fh_{fb}_trial_acc"] = r["fixed_head"].get(fb, r["fixed_head"][0])["trial_accuracy"]
            row[f"fh_{fb}_trial_f1"] = r["fixed_head"].get(fb, r["fixed_head"][0])["trial_macro_f1"]
            row[f"fh_{fb}_win_acc"] = r["fixed_head_window"].get(fb, r["fixed_head_window"][0])["window_accuracy"]
        subject_table.append(row)

    summary["per_subject"] = subject_table

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Predictions CSV (true per-window rows, recomputed from stored predictions) ----
    rows = []
    for r in all_results:
        subj = r["subject"]
        for p_row in r["predictions"]:
            row = dict(subject=subj)
            row.update(p_row)
            rows.append(row)

    if rows:
        import csv
        with open(results_dir / "predictions.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys(), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    # ---- Verify predictions: recompute metrics from stored rows ----
    from sklearn.metrics import accuracy_score, f1_score

    verification_errors = []
    for r in all_results:
        subj = r["subject"]
        for fb in [0, 3, 6, 9]:
            for method in ("none", "fixed_head"):
                key_trial = method
                key_win = method + "_window"
                stored_trial = r[key_trial].get(fb, r[key_trial].get(0))
                stored_win = r[key_win].get(fb, r[key_win].get(0))

                # Recompute from predictions
                rows_for_m = [p for p in r["predictions"]
                               if p["method"] == method and p["feedback_budget"] == fb]

                trial_true = [p["y_true"] for p in rows_for_m]
                trial_pred = [p["y_pred_trial"] for p in rows_for_m]
                # Deduplicate by trial_id (one row per trial)
                trial_seen = {}
                for p in rows_for_m:
                    if p["trial_id"] not in trial_seen:
                        trial_seen[p["trial_id"]] = (p["y_true"], p["y_pred_trial"])
                trial_true_unique = [v[0] for v in trial_seen.values()]
                trial_pred_unique = [v[1] for v in trial_seen.values()]
                win_true = [p["y_true"] for p in rows_for_m]
                win_pred = [p["y_pred_window"] for p in rows_for_m]

                recomputed_trial_acc = accuracy_score(trial_true_unique, trial_pred_unique)
                recomputed_trial_f1 = f1_score(trial_true_unique, trial_pred_unique,
                                                labels=[0, 1, 2], average="macro", zero_division=0)
                recomputed_win_acc = accuracy_score(win_true, win_pred)
                recomputed_win_f1 = f1_score(win_true, win_pred,
                                              labels=[0, 1, 2], average="macro", zero_division=0)

                # Compare
                if abs(recomputed_trial_acc - stored_trial["trial_accuracy"]) > 1e-9:
                    verification_errors.append(
                        f"s{subj} {method} fb={fb} trial_acc: stored={stored_trial['trial_accuracy']:.4f} "
                        f"recomputed={recomputed_trial_acc:.4f}"
                    )
                if abs(recomputed_win_acc - stored_win["window_accuracy"]) > 1e-9:
                    verification_errors.append(
                        f"s{subj} {method} fb={fb} win_acc: stored={stored_win['window_accuracy']:.4f} "
                        f"recomputed={recomputed_win_acc:.4f}"
                    )

    if verification_errors:
        print("\n[PREDICTION VERIFICATION ERRORS]")
        for e in verification_errors:
            print(f"  {e}")
    else:
        print("\n[Prediction verification PASSED: all stored metrics match recomputed]")

    # ---- Run log ----
    with open(results_dir / "run.log", "w") as f:
        f.write(f"run_id: {run_id}\n")
        f.write(f"session: {args.session}\n")
        f.write(f"device: {device}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"hidden: {args.hidden}\n")
        f.write(f"source_epochs: {args.source_epochs}\n")
        f.write(f"feature: {config['feature']}\n")
        f.write(f"architecture: {config['architecture']}\n")
        f.write(f"completed_at: {datetime.now().isoformat()}\n")
        f.write(f"n_subjects_completed: {len(all_results)}\n")
        f.write("\nPer-feedback summary (trial-level):\n")
        for fb in [0, 3, 6, 9]:
            s = summary[fb]["trial"]
            f.write(
                f"  fb={fb}: "
                f"none={s['none']['accuracy_mean']:.4f}±{s['none']['accuracy_std']:.4f} "
                f"fh={s['fixed_head']['accuracy_mean']:.4f}±{s['fixed_head']['accuracy_std']:.4f} "
                f"diff={s['diff']['accuracy_mean']:.4f}±{s['diff']['accuracy_std']:.4f}\n"
            )
        f.write("\nPer-feedback summary (window-level):\n")
        for fb in [0, 3, 6, 9]:
            s = summary[fb]["window"]
            f.write(
                f"  fb={fb}: none_win={s['none_accuracy_mean']:.4f} "
                f"fh_win={s['fh_accuracy_mean']:.4f}\n"
            )

    print(f"\n[{time.strftime('%H:%M:%S')}] Done! Results in {results_dir}")
    print("Per-feedback (trial-level):")
    for fb in [0, 3, 6, 9]:
        s = summary[fb]["trial"]
        print(
            f"  fb={fb}: none={s['none']['accuracy_mean']:.4f}±{s['none']['accuracy_std']:.4f} "
            f"fh={s['fixed_head']['accuracy_mean']:.4f}±{s['fixed_head']['accuracy_std']:.4f} "
            f"diff={s['diff']['accuracy_mean']:.4f}±{s['diff']['accuracy_std']:.4f}"
        )
    print("Per-feedback (window-level):")
    for fb in [0, 3, 6, 9]:
        s = summary[fb]["window"]
        print(f"  fb={fb}: none_win={s['none_accuracy_mean']:.4f} fh_win={s['fh_accuracy_mean']:.4f}")

    return summary


if __name__ == "__main__":
    args = parse_args()
    run_calibration(args)
