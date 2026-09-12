# -*- coding: utf-8 -*-
"""READOUT_CHECK_PLAN 短程检查：在跑完整实验之前验证形状、梯度、数据范围。

最小配置：被试 1，split_seed=0，non_overlap，B/C/D1 各 5 epoch，GRU hidden=16。
"""
from __future__ import annotations

import sys
import os
import time
import collections

import numpy as np
import torch
import torch.nn as nn

# 确保项目根目录在 path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import RESULTS_DIR, SEED_FS
from data import list_subjects
from exp_koopman_gate import (
    FRAMINGS, make_frames, split_by_trial, standardize,
    de_features, fit_logreg, _pairs, linearity_metrics, latent_health,
)
from koopman_ae import VARIANTS, KoopmanModel
from preprocess import PreprocConfig, build_segments

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(RESULTS_DIR, "readout_check")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 短程配置
# ============================================================
SUBJECT = 1
SPLIT_SEED = 0
FRAMING = "non_overlap"
Z_DIM = 32
EPOCHS_SHORT = 5
GRU_HIDDEN = 16
VARIANTS_SHORT = ("B", "C", "D1")


# ============================================================
# GRU 读出头（与 READOUT_CHECK_PLAN 一致）
# ============================================================
class GRUReadout(nn.Module):
    def __init__(self, z_dim=32, hidden=32, n_classes=3):
        super().__init__()
        self.gru = nn.GRU(z_dim, hidden, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, z):          # z: (B, T, z_dim)
        _, h = self.gru(z)        # h: (1, B, hidden)
        return self.classifier(h.squeeze(0))


# ============================================================
# 训练函数（Mean Pooling 版）
# ============================================================
def train_one_mean(variant, data, z_dim=32, epochs=5, lr=1e-3, batch=32,
                   alpha_rec=1.0, beta_koop=1.0, seed_train=0):
    torch.manual_seed(seed_train)
    np.random.seed(seed_train)
    Xtr, ytr, Xva, yva, Xte, yte = data
    n_ch, frame_len = Xtr.shape[2], Xtr.shape[3]

    model = KoopmanModel(variant, n_ch, frame_len, z_dim=z_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
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
            # ---- 梯度检查 ----
            if ep == 0 and i == 0:
                enc_grads = {n: p.grad.norm().item()
                             for n, p in model.encoder.named_parameters()
                             if p.grad is not None}
                print(f"    [GRAD CHECK ep0 batch0] encoder grad norms: {enc_grads}")
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

    return dict(variant=variant, test_acc=te_acc, val_acc=best_va,
                best_epoch=best_ep, z_tr=z_tr_np, z_te=z_te_np,
                n_transition_params=model.n_transition_params()), model


# ============================================================
# 训练函数（GRU 读出头版）
# ============================================================
def train_one_gru(variant, data, z_dim=32, epochs=5, lr=1e-3, batch=32,
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
        lr=lr, weight_decay=1e-4)
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

            z = encoder.encode_seq(xb)                 # (B, T, z_dim)
            logits = readout(z)                         # (B, n_classes)
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
            # ---- 梯度检查 ----
            if ep == 0 and i == 0:
                enc_grads = {n: p.grad.norm().item()
                             for n, p in encoder.encoder.named_parameters()
                             if p.grad is not None}
                ro_grads = {n: p.grad.norm().item()
                            for n, p in readout.named_parameters()
                            if p.grad is not None}
                print(f"    [GRAD CHECK ep0 batch0] encoder grad norms: {enc_grads}")
                print(f"    [GRAD CHECK ep0 batch0] readout grad norms: {ro_grads}")
            opt.step()

        # ---- 验证 ----
        encoder.eval()
        readout.eval()
        with torch.no_grad():
            z_va = encoder.encode_seq(Xva_t)
            va_logits = readout(z_va)
            va_acc = (va_logits.argmax(1) == yva_t).float().mean().item()
        if va_acc > best_va:
            best_va, best_ep = va_acc, ep
            best_state = {
                'encoder': {k: v.detach().clone() for k, v in encoder.state_dict().items()},
                'readout': {k: v.detach().clone() for k, v in readout.state_dict().items()},
            }

    encoder.load_state_dict(best_state['encoder'])
    readout.load_state_dict(best_state['readout'])
    encoder.eval()
    readout.eval()
    with torch.no_grad():
        z_te = encoder.encode_seq(Xte_t)
        te_acc = (readout(z_te).argmax(1) == yte_t).float().mean().item()
        z_tr = encoder.encode_seq(Xtr_t)
    z_te_np, z_tr_np = z_te.cpu().numpy(), z_tr.cpu().numpy()

    return dict(variant=variant, test_acc=te_acc, val_acc=best_va,
                best_epoch=best_ep, z_tr=z_tr_np, z_te=z_te_np,
                n_transition_params=encoder.n_transition_params()), encoder, readout


# ============================================================
# 主检查流程
# ============================================================
print("=" * 60)
print("READOUT_CHECK_PLAN 短程检查")
print("=" * 60)
print(f"DEVICE: {DEVICE}")
print(f"被试={SUBJECT} 划分种子={SPLIT_SEED} 帧化={FRAMING}")
print(f"EPOCHS={EPOCHS_SHORT}  GRU_HIDDEN={GRU_HIDDEN}")
print()

# 1. 加载数据
cfg = PreprocConfig(win_s=4.0, reject_abs_thr=15.0)
r = build_segments("seed", SUBJECT, session=1, cfg=cfg)
segs, y, trial = r["segs"], r["y"], r["trial"]
print(f"[DATA] 段数={len(segs)}  形状={segs.shape}  标签分布={np.bincount(y, minlength=3).tolist()}")

# 2. 划分
tr_m, va_m, te_m = split_by_trial(y, trial, seed=SPLIT_SEED)
s_tr, s_va, s_te = standardize(segs[tr_m], segs[va_m], segs[te_m])
ytr, yva, yte = y[tr_m], y[va_m], y[te_m]
print(f"[SPLIT] train={int(tr_m.sum())}  val={int(va_m.sum())}  test={int(te_m.sum())}")

# 3. 帧化
fp = FRAMINGS[FRAMING]
Xtr, Xva, Xte = (make_frames(s, **fp) for s in (s_tr, s_va, s_te))
print(f"[FRAMING] frame_len={fp['frame_len']}  hop={fp['hop']}  T={Xtr.shape[1]}")
print(f"  Xtr={Xtr.shape}  Xva={Xva.shape}  Xte={Xte.shape}")
data = (Xtr, ytr, Xva, yva, Xte, yte)

# 4. Mean Pooling 组（B/C/D1 各 5 epoch）
print()
print("--- Mean Pooling 组（B/C/D1）---")
mean_results = {}
for variant in VARIANTS_SHORT:
    t0 = time.time()
    res, model = train_one_mean(variant, data, z_dim=Z_DIM,
                                epochs=EPOCHS_SHORT, seed_train=0)
    elapsed = time.time() - t0
    z_tr, z_te = res['z_tr'], res['z_te']
    print(f"  {variant}: test_acc={res['test_acc']:.3f}  val_acc={res['val_acc']:.3f}"
          f"  best_ep={res['best_epoch']}  elapsed={elapsed:.1f}s")
    print(f"    z_tr={z_tr.shape}  z_te={z_te.shape}"
          f"  T={z_tr.shape[1]}  z_dim={z_tr.shape[-1]}")
    # 线性度诊断
    lm = linearity_metrics(z_te, z_tr)
    lh = latent_health(z_te)
    print(f"    R_learned={lm.get('R_lin_learned', float('nan')):.4f}"
          f"  R_ols_train={lm['R_lin_ols_train']:.4f}"
          f"  tvar={lh['temporal_var_frac']:.3f}")
    mean_results[variant] = res

# 5. GRU 组（B/C/D1 各 5 epoch）
print()
print("--- GRU 读出头组（B/C/D1）---")
gru_results = {}
for variant in VARIANTS_SHORT:
    t0 = time.time()
    res, enc, ro = train_one_gru(variant, data, z_dim=Z_DIM,
                                 epochs=EPOCHS_SHORT, seed_train=0,
                                 gru_hidden=GRU_HIDDEN)
    elapsed = time.time() - t0
    z_tr, z_te = res['z_tr'], res['z_te']
    print(f"  {variant}: test_acc={res['test_acc']:.3f}  val_acc={res['val_acc']:.3f}"
          f"  best_ep={res['best_epoch']}  elapsed={elapsed:.1f}s")
    print(f"    z_tr={z_tr.shape}  z_te={z_te.shape}"
          f"  T={z_tr.shape[1]}  z_dim={z_tr.shape[-1]}")
    lm = linearity_metrics(z_te, z_tr)
    lh = latent_health(z_te)
    print(f"    R_learned={lm.get('R_lin_learned', float('nan')):.4f}"
          f"  R_ols_train={lm['R_lin_ols_train']:.4f}"
          f"  tvar={lh['temporal_var_frac']:.3f}")
    gru_results[variant] = res

# 6. 汇总短程比较
print()
print("--- 短程汇总 ---")
for variant in VARIANTS_SHORT:
    dm = mean_results[variant]['test_acc']
    dg = gru_results[variant]['test_acc']
    print(f"  {variant}: mean={dm:.3f}  gru={dg:.3f}  diff(gru-mean)={dg-dm:+.3f}")

delta_c_b_mean = mean_results['C']['test_acc'] - mean_results['B']['test_acc']
delta_c_b_gru  = gru_results['C']['test_acc']  - gru_results['B']['test_acc']
delta_diff = delta_c_b_gru - delta_c_b_mean
print(f"\n  主比较: Δ_C-B(mean)={delta_c_b_mean:+.3f}  Δ_C-B(gru)={delta_c_b_gru:+.3f}")
print(f"          Δ_diff={delta_diff:+.3f}  (GRU下Koopman增益 - Mean下Koopman增益)")

# 7. 保存结果
import json
out_path = os.path.join(OUT_DIR, "short_run.json")
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(dict(
        subject=SUBJECT, split_seed=SPLIT_SEED, framing=FRAMING,
        epochs=EPOCHS_SHORT, gru_hidden=GRU_HIDDEN,
        mean_results={k: {kk: vv for kk, vv in v.items()
                          if kk not in ('z_tr', 'z_te')}
                      for k, v in mean_results.items()},
        gru_results={k: {kk: vv for kk, vv in v.items()
                        if kk not in ('z_tr', 'z_te')}
                     for k, v in gru_results.items()},
        summary=dict(
            delta_c_b_mean=float(delta_c_b_mean),
            delta_c_b_gru=float(delta_c_b_gru),
            delta_diff=float(delta_diff),
        )
    ), fh, indent=2, ensure_ascii=False)
print(f"\n短程结果已存: {out_path}")
