# -*- coding: utf-8 -*-
"""Plot adaptation curves from calibration run results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_adaptation_curve(results_dir: Path, output_path: Path | None = None):
    results_dir = Path(results_dir)
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found in {results_dir}")

    with open(summary_path) as f:
        summary = json.load(f)

    per_subject = summary["per_subject"]
    n_subjects = len(per_subject)
    feedback_budgets = [0, 3, 6, 9]

    # Collect per-subject differences
    none_accs = {fb: [row[f"none_{fb}_acc"] for row in per_subject] for fb in feedback_budgets}
    fh_accs = {fb: [row[f"fh_{fb}_acc"] for row in per_subject] for fb in feedback_budgets}

    none_mean = [np.mean(none_accs[fb]) for fb in feedback_budgets]
    none_std = [np.std(none_accs[fb]) for fb in feedback_budgets]
    fh_mean = [np.mean(fh_accs[fb]) for fb in feedback_budgets]
    fh_std = [np.std(fh_accs[fb]) for fb in feedback_budgets]

    diff_mean = [fh_mean[i] - none_mean[i] for i in range(len(feedback_budgets))]
    diff_std = [np.std([fh_accs[fb][j] - none_accs[fb][j] for j in range(n_subjects)])
                for fb in feedback_budgets]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: absolute accuracy
    ax = axes[0]
    x = np.arange(len(feedback_budgets))
    ax.errorbar(x, none_mean, yerr=none_std, label="none", marker="o", capsize=4)
    ax.errorbar(x, fh_mean, yerr=fh_std, label="fixed_head", marker="s", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"fb={fb}" for fb in feedback_budgets])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Per-trial Calibration: Absolute Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: difference
    ax = axes[1]
    colors = ["green" if d > 0 else "red" for d in diff_mean]
    ax.bar(x, diff_mean, yerr=diff_std, color=colors, alpha=0.6, capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"fb={fb}" for fb in feedback_budgets])
    ax.set_ylabel("Accuracy difference (fixed_head − none)")
    ax.set_title("Per-trial Calibration: Adaptation Gain")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    out_path = output_path or results_dir / "adaptation_curve.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    plot_adaptation_curve(args.results_dir, args.output)
