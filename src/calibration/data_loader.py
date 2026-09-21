# -*- coding: utf-8 -*-
"""Load SEED ExtractedFeatures DE data for calibration experiments.

Data format per .mat file:
  - de_movingAve{t}: shape (62, n_windows, 5) → (channels, windows, bands)
  - label.mat: shape (1, 15), values {-1, 0, 1}

Each trial returns all windows; no trial-level aggregation here.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat

SEED_EXT_ROOT = os.environ.get(
    "SEED_EXT_ROOT",
    r"E:\Python\DATA\SEED\ExtractedFeatures"
)
SEED_FS = 200
SEED_LABEL_MAP = {-1: 0, 0: 1, 1: 2}
SEED_N_SUBJECTS = 15
SEED_N_TRIALS = 15


def _subject_files(subject: int) -> list[str]:
    """Return sorted list of .mat filenames for a subject (by date)."""
    pre = Path(SEED_EXT_ROOT)
    files = sorted(
        [f.name for f in pre.glob(f"{subject}_*.mat")],
        key=lambda fn: re.search(r"_(\d{8})", fn).group(1)
    )
    return files


def load_de_session(subject: int, session: int = 1):
    """Load DE features for one subject and session.

    Returns
    -------
    X_list : list of np.ndarray, each (n_windows_i, 310)
    y_list : list of np.ndarray, each (n_windows_i,) — all same label per trial
    trial_ids_list : list of np.ndarray, each (n_windows_i,) — all same trial id
    window_ids_list : list of np.ndarray, each (n_windows_i,)
    labels_raw : np.ndarray, shape (15,) — original {-1,0,1} labels
    """
    files = _subject_files(subject)
    if not (1 <= session <= len(files)):
        raise ValueError(f"Subject {subject} has {len(files)} sessions, got {session}")

    label_path = Path(SEED_EXT_ROOT) / "label.mat"
    all_labels = loadmat(label_path)["label"].ravel()  # (15,)

    mat_path = Path(SEED_EXT_ROOT) / files[session - 1]
    mat = loadmat(mat_path)

    X_list, y_list, trial_ids_list, window_ids_list = [], [], [], []

    for trial_idx in range(1, SEED_N_TRIALS + 1):
        key = f"de_movingAve{trial_idx}"
        if key not in mat:
            raise KeyError(f"{key} not found in {mat_path}")
        de = np.asarray(mat[key])  # (62, n_win, 5)

        # Flatten channels × bands → single feature vector per window
        # de shape: (62, n_win, 5) → (n_win, 62*5)
        n_win = de.shape[1]
        de_flat = de.transpose(1, 0, 2).reshape(n_win, -1)  # (n_win, 310)

        label_raw = all_labels[trial_idx - 1]
        label_mapped = SEED_LABEL_MAP[int(label_raw)]

        X_list.append(de_flat.astype(np.float32))
        y_list.append(np.full(n_win, label_mapped, dtype=np.int64))
        trial_ids_list.append(np.full(n_win, trial_idx, dtype=np.int64))
        window_ids_list.append(np.arange(n_win, dtype=np.int64))

    return X_list, y_list, trial_ids_list, window_ids_list, all_labels


def flatten_session(X_list, y_list, trial_ids_list):
    """Concatenate all trials into single flat arrays.

    Returns
    -------
    X : np.ndarray, shape (total_windows, 310)
    y : np.ndarray, shape (total_windows,)
    trial_ids : np.ndarray, shape (total_windows,)
    """
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    trial_ids = np.concatenate(trial_ids_list, axis=0)
    return X, y, trial_ids
