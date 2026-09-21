# -*- coding: utf-8 -*-
"""Calibration module for per-trial few-shot adaptation on SEED.

Implements the calibration experiment described in docs/calibration_protocol.md.

Submodules
----------
data_loader : Load SEED ExtractedFeatures DE data (window-level).
adapter     : Model adaptation strategies and evaluation.
run         : Main experiment entry point.
plot        : Adaptation curve plotting.
"""
from .data_loader import load_de_session, flatten_session, SEED_EXT_ROOT, SEED_N_SUBJECTS, SEED_N_TRIALS
from .adapter import (
    SimpleMLP,
    CalibrationAdapter,
    CalibConfig,
    batch_evaluate_window,
    batch_evaluate_trial,
    train_source_model,
)

__all__ = [
    "load_de_session",
    "flatten_session",
    "SEED_EXT_ROOT",
    "SEED_N_SUBJECTS",
    "SEED_N_TRIALS",
    "SimpleMLP",
    "CalibrationAdapter",
    "CalibConfig",
    "WindowScaler",
    "batch_evaluate_window",
    "batch_evaluate_trial",
    "train_source_model",
]
