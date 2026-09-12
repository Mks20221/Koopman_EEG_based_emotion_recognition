# -*- coding: utf-8 -*-
"""READOUT_CHECK_PLAN 完整被试内实验：Mean/GRU × B/C/D1 × 15被试 × 3划分种子

六组设计（与 READOUT_CHECK_PLAN.md 严格一致）：
  G1: Mean Pooling + B    G2: Mean Pooling + C    G3: Mean Pooling + D1
  G4: GRU 末状态 + B      G5: GRU 末状态 + C      G6: GRU 末状态 + D1

270 次训练组成：
  15 被试 × 3 划分种子(split_seeds=[0,1,2]) × 6 组 = 270 runs
  每 run = 1 subject × 1 split_seed × 1 (readout, variant) 组合，训练 100 epoch

主指标（先验定义，不根据测试结果调整）：
  Δ_C-B_GRU  = G5_test_acc − G4_test_acc
  Δ_C-B_mean = G2_test_acc − G1_test_acc
  Δ_diff     = Δ_C-B_GRU − Δ_C-B_mean

标准化：仅用训练折统计量（与 exp_koopman_gate.py 完全一致）
划分：trial-level 3折（train/val/test），split_seeds=[0,1,2]
帧化：non_overlap（frame_len=100, hop=100）
训练种子：seed_train=0（单一种子，不把 seed 当成被试）
GRU：单向，hidden=16，与 encoder 联合训练
Checkpoint：验证集准确率最优
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

# ---- 路径 setup ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import RESULTS_DIR, SEED_FS
from data import list_subjects
from exp_koopman_gate import (
    FRAMINGS, make_frames, split_by_trial, standardize,
    _pairs, linearity_metrics, latent_health,
)
from koopman_ae import KoopmanModel
from preprocess import PreprocConfig, build_segments

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(RESULTS_DIR, "readout_check", "full_run")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 配置（与 READOUT_CHECK_PLAN.md 严格一致）
# ============================================================
SUBJECTS = list(range(1, 16))           # 1-15
SPLIT_SEEDS = (0, 1, 2)                 # 3 个划分种子
TRAIN_SEED = 0                           # 单一种子，不扩展
FRAMING = "non_overlap"                  # non_overlap
Z_DIM = 32
EPOCHS = 100
LR = 1e-3
BATCH = 32
WEIGHT_DECAY = 1e-4
ALPHA_REC = 1.0
BETA_KOOP = 1.0
GRU_HIDDEN = 16
N_CLASSES = 3

VARIANTS = ("B", "C", "D1")
READOUTS = ("mean", "gru")

# ============================================================
# GRU 读出头
# ============================================================
class GRUReadout(nn.Module):
    def __init__(self, z_dim=32, hidden=32, n_classes=3):
        super().__init__()
        self.gru = nn.GRU(z_dim, hidden, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, z):
        _, h = self.gru(z)
        return self.classifier(h.squeeze(0))


# ============================================================
# 训练：Mean Pooling
# ============================================================
def train_mean(variant, data, z_dim=32, epochs=100, lr=1e-3, batch=32,
               alpha_rec=1.0, beta_koop=1.0, seed_train=0):
    torch.manual_seed(seed_train)
    np.random.seed(seed_train)
    Xtr, ytr, Xva, yva, Xte, yte = data
    n_ch, frame_len = Xtr.shape[2], Xtr.shape[3]

    model = KoopmanModel(variant, n_ch, frame_len, z_dim=z_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()

    def to_t(a, d=torch.float32):
        return torch.as_tensor(a, dtype=d, device=DEVICE)

    Xtr_t, ytr_t = to_t(Xtr), to_t(ytr, torch.long)
    Xva_t, yva_t = to_t(Xva), to_t(yva, torch.long)
    Xte_t, yte_t = to_t(Xte), to_t(yte, torch.long)

    best_va, best_state, best_ep = -1.0, None, -1
    n = len(Xtr_t)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            z, logits = model(xb)
            loss = ce(logits, yb)
            if model.decoder is not None:
                B, T = xb.shape[0], xb.shape[1]
                xr = model.decoder(z.reshape(B * T, -1)).view(B, T, *xb.shape[2:])
                loss = loss + alpha_rec * (((xr - xb) ** 2).mean() / (xb.var() + 1e-8))
            if model.transition is not None:
                zp, zn = z[:, :-1, :], z[:, 1:, :]
                pred = model.transition(zp.reshape(-1, z.shape[-1])).view_as(zn)
                loss = loss + beta_koop * (((pred - zn) ** 2).sum()
                                           / ((zn ** 2).sum() + 1e-8))
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va_acc = (model(Xva_t)[1].argmax(1) == yva_t).float().mean().item()
        if va_acc > best_va:
            best_va, best_ep = va_acc, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z_te, logit_te = model(Xte_t)
        te_acc = (logit_te.argmax(1) == yte_t).float().mean().item()
        z_tr = model(Xtr_t)[0]
    z_te_np, z_tr_np = z_te.cpu().numpy(), z_tr.cpu().numpy()

    out = dict(variant=variant, readout="mean", test_acc=te_acc,
               val_acc=best_va, best_epoch=best_ep,
               n_transition_params=model.n_transition_params())
    out.update(linearity_metrics(z_te_np, z_tr_np))
    out.update(latent_health(z_te_np))
    return out


# ============================================================
# 训练：GRU 末状态
# ============================================================
def train_gru(variant, data, z_dim=32, epochs=100, lr=1e-3, batch=32,
              alpha_rec=1.0, beta_koop=1.0, seed_train=0,
              gru_hidden=16, n_classes=3):
    torch.manual_seed(seed_train)
    np.random.seed(seed_train)
    Xtr, ytr, Xva, yva, Xte, yte = data
    n_ch, frame_len = Xtr.shape[2], Xtr.shape[3]

    encoder = KoopmanModel(variant, n_ch, frame_len, z_dim=z_dim).to(DEVICE)
    readout = GRUReadout(z_dim, gru_hidden, n_classes).to(DEVICE)
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(readout.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()

    def to_t(a, d=torch.float32):
        return torch.as_tensor(a, dtype=d, device=DEVICE)

    Xtr_t, ytr_t = to_t(Xtr), to_t(ytr, torch.long)
    Xva_t, yva_t = to_t(Xva), to_t(yva, torch.long)
    Xte_t, yte_t = to_t(Xte), to_t(yte, torch.long)

    best_va, best_state, best_ep = -1.0, None, -1
    n = len(Xtr_t)

    for ep in range(epochs):
        encoder.train()
        readout.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr_t[idx], ytr_t[idx]

            z = encoder.encode_seq(xb)
            logits = readout(z)
            loss = ce(logits, yb)

            if encoder.decoder is not None:
                B, T = xb.shape[0], xb.shape[1]
                xr = encoder.decoder(z.reshape(B * T, -1)).view(B, T, *xb.shape[2:])
                loss = loss + alpha_rec * (((xr - xb) ** 2).mean() / (xb.var() + 1e-8))
            if encoder.transition is not None:
                zp, zn = z[:, :-1, :], z[:, 1:, :]
                pred = encoder.transition(zp.reshape(-1, z.shape[-1])).view_as(zn)
                loss = loss + beta_koop * (((pred - zn) ** 2).sum()
                                           / ((zn ** 2).sum() + 1e-8))
            opt.zero_grad()
            loss.backward()
            opt.step()

        encoder.eval()
        readout.eval()
        with torch.no_grad():
            z_va = encoder.encode_seq(Xva_t)
            va_logits = readout(z_va)
            va_acc = (va_logits.argmax(1) == yva_t).float().mean().item()
        if va_acc > best_va:
            best_va, best_ep = va_acc, ep
            best_state = {
                "encoder": {k: v.detach().clone() for k, v in encoder.state_dict().items()},
                "readout": {k: v.detach().clone() for k, v in readout.state_dict().items()},
            }

    encoder.load_state_dict(best_state["encoder"])
    readout.load_state_dict(best_state["readout"])
    encoder.eval()
    readout.eval()
    with torch.no_grad():
        z_te = encoder.encode_seq(Xte_t)
        te_acc = (readout(z_te).argmax(1) == yte_t).float().mean().item()
        z_tr = encoder.encode_seq(Xtr_t)

    z_te_np, z_tr_np = z_te.cpu().numpy(), z_tr.cpu().numpy()
    out = dict(variant=variant, readout="gru", test_acc=te_acc,
               val_acc=best_va, best_epoch=best_ep,
               n_transition_params=encoder.n_transition_params())
    out.update(linearity_metrics(z_te_np, z_tr_np))
    out.update(latent_health(z_te_np))
    return out


# ============================================================
# 完整实验
# ============================================================
def run_all():
    cfg = PreprocConfig(win_s=4.0, reject_abs_thr=15.0)
    fp = FRAMINGS[FRAMING]

    # 配置快照
    config = dict(
        subjects=SUBJECTS, split_seeds=list(SPLIT_SEEDS), train_seed=TRAIN_SEED,
        framing=FRAMING, z_dim=Z_DIM, epochs=EPOCHS, lr=LR, batch=BATCH,
        weight_decay=WEIGHT_DECAY, alpha_rec=ALPHA_REC, beta_koop=BETA_KOOP,
        gru_hidden=GRU_HIDDEN, n_classes=N_CLASSES,
        variants=VARIANTS, readouts=READOUTS,
        frame_len=fp["frame_len"], hop=fp["hop"],
        total_runs=len(SUBJECTS) * len(SPLIT_SEEDS) * len(VARIANTS) * len(READOUTS),
        timestamp=datetime.now().isoformat(),
    )
    with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    rows = []
    total_runs = len(SUBJECTS) * len(SPLIT_SEEDS) * len(VARIANTS) * len(READOUTS)
    done = 0

    for subj in SUBJECTS:
        t_subj = time.time()
        r = build_segments("seed", subj, session=1, cfg=cfg)
        segs, y, trial = r["segs"], r["y"], r["trial"]

        for sp in SPLIT_SEEDS:
            tr_m, va_m, te_m = split_by_trial(y, trial, seed=sp)
            s_tr, s_va, s_te = standardize(segs[tr_m], segs[va_m], segs[te_m])
            ytr, yva, yte = y[tr_m], y[va_m], y[te_m]
            Xtr, Xva, Xte = (make_frames(s, **fp) for s in (s_tr, s_va, s_te))
            data = (Xtr, ytr, Xva, yva, Xte, yte)

            for ro in READOUTS:
                for var in VARIANTS:
                    t0 = time.time()
                    if ro == "mean":
                        res = train_mean(var, data, z_dim=Z_DIM,
                                         epochs=EPOCHS, lr=LR, batch=BATCH,
                                         alpha_rec=ALPHA_REC, beta_koop=BETA_KOOP,
                                         seed_train=TRAIN_SEED)
                    else:
                        res = train_gru(var, data, z_dim=Z_DIM,
                                        epochs=EPOCHS, lr=LR, batch=BATCH,
                                        alpha_rec=ALPHA_REC, beta_koop=BETA_KOOP,
                                        seed_train=TRAIN_SEED,
                                        gru_hidden=GRU_HIDDEN, n_classes=N_CLASSES)
                    elapsed = time.time() - t0
                    res.update(subject=subj, split_seed=sp, train_seed=TRAIN_SEED,
                               framing=FRAMING, readout=ro, variant=var,
                               n_train=int(tr_m.sum()), n_val=int(va_m.sum()),
                               n_test=int(te_m.sum()), elapsed=elapsed)
                    rows.append(res)
                    done += 1

                    tag = f"{ro}_{var}_s{subj:02d}_sp{sp}"
                    print(f"  [{done}/{total_runs}] {tag}: acc={res['test_acc']:.3f}  "
                          f"val={res['val_acc']:.3f}  ep={res['best_epoch']}  "
                          f"({elapsed:.1f}s)  tvar={res['temporal_var_frac']:.3f}",
                          flush=True)

        # 每被试落盘一次（防中断丢失）
        with open(os.path.join(OUT_DIR, "rows.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"  -- 被试 {subj} 完成，elapsed={time.time()-t_subj:.1f}s")
        print(f"  -- 进度 {done}/{total_runs}  剩余约 {(time.time()-t_subj)*(total_runs-done)/done:.0f}s")

    return rows, config


# ============================================================
# 汇总
# ============================================================
def summarize(rows, config):
    """统计单位是被试：先在 split_seed 上平均，再做配对统计"""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[(r["readout"], r["variant"])][r["subject"]].append(r["test_acc"])

    # 逐被试均值
    means = {}
    for key, sub_dict in by.items():
        means[key] = {s: float(np.mean(v)) for s, v in sub_dict.items()}

    subjects = sorted(set(r["subject"] for r in rows))
    n = len(subjects)

    print("\n" + "=" * 70)
    print("READOUT_CHECK 完整被试内实验汇总  (统计单位=被试)")
    print("=" * 70)
    print(f"{'组':<20}{'test_acc 均值±std':>22}{'min':>8}{'max':>8}")
    print("-" * 70)
    for ro in READOUTS:
        for var in VARIANTS:
            key = (ro, var)
            vals = np.array([means[key][s] for s in subjects])
            print(f"  {ro}/{var:<15}{vals.mean():>17.3f}±{vals.std():<4.3f}"
                  f"{vals.min():>8.3f}{vals.max():>8.3f}")

    print("\n--- 主比较：Koopman 在不同读出下的贡献 ---")
    comparisons = [
        ("Δ_C-B(GRU)",  "gru",  "C", "B"),
        ("Δ_C-B(Mean)", "mean", "C", "B"),
        ("Δ_C-D1(GRU)", "gru",  "C", "D1"),
        ("Δ_C-D1(Mean)","mean", "C", "D1"),
        ("Δ_diff(C-B)", "diff", "C", "B"),
        ("Δ_diff(C-D1)","diff", "C", "D1"),
    ]

    results = {}
    for name, ro_a, var_a, var_b in comparisons:
        if ro_a == "diff":
            # Δ_diff = (G5-G4) - (G2-G1)
            a_key = ("gru", var_a)
            b_key = ("gru", var_b)
            c_key = ("mean", var_a)
            d_key = ("mean", var_b)
            vals = np.array([(means[a_key][s] - means[b_key][s]) -
                             (means[c_key][s] - means[d_key][s])
                             for s in subjects])
        else:
            a_key = (ro_a, var_a)
            b_key = (ro_a, var_b)
            vals = np.array([means[a_key][s] - means[b_key][s] for s in subjects])

        mu = vals.mean()
        sd = vals.std(ddof=1)
        se = sd / np.sqrt(n)
        ci = 1.96 * se
        t, p = stats.ttest_rel(vals, np.zeros(n))
        n_pos = int((vals > 0).sum())
        p_sign = stats.binomtest(n_pos, n, 0.5).pvalue
        results[name] = dict(mean=float(mu), std=float(sd), ci_lower=float(mu-ci),
                             ci_upper=float(mu+ci), t=float(t), p=float(p),
                             n_pos=n_pos, n=n)
        print(f"  {name:<20}{mu:+.4f}  [95%CI {mu-ci:+.4f},{mu+ci:+.4f}]"
              f"  t={t:.2f} p={p:.3f}  +{n_pos}/{n}  dz={mu/(sd+1e-12):.3f}")

    # 胜负统计
    print("\n--- 逐被试胜负统计（C vs B）---")
    for ro in READOUTS:
        wins = sum(1 for s in subjects if means[("gru" if ro=="gru" else "mean"), "C"][s] >
                                            means[("gru" if ro=="gru" else "mean"), "B"][s])
        print(f"  {ro}: C>B: {wins}/{n} 被试")

    # 保存汇总
    summary = dict(
        config=config,
        n_subjects=n,
        means={f"{ro}/{var}": means[(ro, var)] for ro in READOUTS for var in VARIANTS},
        comparisons=results,
        subjects=subjects,
        timestamp=datetime.now().isoformat(),
    )
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 失败检测
    failures = [r for r in rows if np.isnan(r.get("test_acc", float("nan")))]
    if failures:
        print(f"\n⚠️  {len(failures)} 次运行失败：")
        for f_ in failures:
            print(f"  {f_['readout']}/{f_['variant']}  subj={f_['subject']}  sp={f_['split_seed']}")
    else:
        print(f"\n✅ 所有 {len(rows)} 次运行完成，无失败。")

    return summary


def main():
    print("=" * 60)
    print("READOUT_CHECK 完整被试内实验")
    print(f"DEVICE: {DEVICE}")
    print(f"组合数: {len(SUBJECTS)}被试 × {len(SPLIT_SEEDS)}划分 × "
          f"{len(VARIANTS)}模型 × {len(READOUTS)}读出 = "
          f"{len(SUBJECTS)*len(SPLIT_SEEDS)*len(VARIANTS)*len(READOUTS)} runs")
    print("=" * 60)
    rows, config = run_all()
    summary = summarize(rows, config)
    print(f"\n结果已存: {OUT_DIR}")
    print(f"  config.json  : 运行配置")
    print(f"  rows.json    : {len(rows)} 条逐次运行记录")
    print(f"  summary.json : 汇总统计（含配对差/CI/p值）")


if __name__ == "__main__":
    main()
