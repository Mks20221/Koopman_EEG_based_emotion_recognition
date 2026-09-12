# -*- coding: utf-8 -*-
"""D6：幅度定义审计。检查 top-k 模态选择依据的 b（只由窗口第一个 Hankel 快照
lstsq 得到）是不是一个公平的"整窗重要性"指标，还是偏向窗口开头的瞬态。
不碰任何标签。"""
import json
import os

import numpy as np
import scipy.linalg as sla
from joblib import Parallel, delayed

import koopman as km
from config import RESULTS_DIR
import diag_tier1 as dt

DT = dt.DT
CFG = km.DEFAULT
N_SEGMENTS = 200
OUT_DIR = os.path.join(RESULTS_DIR, "diagnostics")


def _fit_full(H, r, method="fb"):
    X1, X2 = H[:, :-1], H[:, 1:]
    U, s, Vt = np.linalg.svd(X1, full_matrices=False)
    rr = min(r, len(s), X1.shape[0])
    U, s, Vt = U[:, :rr], s[:rr], Vt[:rr]
    Sinv = np.diag(1.0 / s)
    Af = U.conj().T @ X2 @ Vt.conj().T @ Sinv
    if method == "fb":
        Ub, sb, Vtb = np.linalg.svd(X2, full_matrices=False)
        Ub, sb, Vtb = Ub[:, :rr], sb[:rr], Vtb[:rr]
        Ab = Ub.conj().T @ X1 @ Vtb.conj().T @ np.diag(1.0 / sb)
        try:
            A = sla.sqrtm(Af @ np.linalg.inv(Ab))
            A = np.real_if_close(A, tol=1e6)
            if not np.all(np.isfinite(A)):
                A = Af
        except Exception:
            A = Af
    else:
        A = Af
    lam, W = np.linalg.eig(A)
    Phi = X2 @ Vt.conj().T @ Sinv @ W
    b = np.linalg.lstsq(Phi, H[:, 0], rcond=None)[0]
    return Phi, lam, b


def _d6_one(x, cfg=CFG):
    H = km.hankel_embed(x, cfg.delay)
    Phi, lam, b = _fit_full(H, cfg.rank, cfg.method)
    f = np.angle(lam) / (2 * np.pi * DT)

    keep = (f >= cfg.f_lo) & (f <= cfg.f_hi) & (np.abs(lam) <= cfg.mod_max)
    if keep.sum() == 0:
        keep = (f >= cfg.f_lo) & (f <= cfg.f_hi)
    if keep.sum() == 0:
        keep = np.ones_like(f, dtype=bool)
    keep_idx = np.where(keep)[0]

    phi_norms = np.linalg.norm(Phi, axis=0)
    n_steps = H.shape[1]
    t = np.arange(n_steps)

    amp_official = np.abs(b[keep_idx])
    order_official = np.argsort(-amp_official)[:cfg.top_k]
    official_idx = keep_idx[order_official]

    sum_pow = np.array([np.sum(np.abs(lam[i]) ** (2 * t)) for i in keep_idx])
    C = (np.abs(b[keep_idx]) ** 2) * (phi_norms[keep_idx] ** 2) * sum_pow
    order_C = np.argsort(-C)[:cfg.top_k]
    proposed_idx = keep_idx[order_C]

    set_off, set_pro = set(official_idx.tolist()), set(proposed_idx.tolist())
    jaccard = len(set_off & set_pro) / len(set_off | set_pro) if (set_off | set_pro) else 1.0

    def reconstruct(cols):
        V = lam[cols][:, None] ** t[None, :]
        return np.real(Phi[:, cols] @ (b[cols][:, None] * V))

    def rel_err(cols):
        return float(np.linalg.norm(H - reconstruct(cols)) / (np.linalg.norm(H) + 1e-12))

    return dict(
        phi_norm_cv=float(np.std(phi_norms) / (np.mean(phi_norms) + 1e-12)),
        phi_norm_min=float(phi_norms.min()), phi_norm_max=float(phi_norms.max()),
        jaccard_overlap=float(jaccard),
        n_official=int(len(official_idx)), n_proposed=int(len(proposed_idx)),
        err_official=rel_err(official_idx), err_proposed=rel_err(proposed_idx),
    )


def main():
    xs = dt.sample_segments(N_SEGMENTS, seed=dt.SAMPLE_SEED + 2)
    print(f"[D6] {len(xs)} 段")
    rows = Parallel(n_jobs=-1, verbose=5)(delayed(_d6_one)(x) for x in xs)

    phi_cv = np.array([r["phi_norm_cv"] for r in rows])
    jac = np.array([r["jaccard_overlap"] for r in rows])
    err_off = np.array([r["err_official"] for r in rows])
    err_pro = np.array([r["err_proposed"] for r in rows])

    report = dict(
        n_segments=len(xs),
        phi_norm_cv=dict(median=float(np.median(phi_cv)), mean=float(phi_cv.mean())),
        jaccard_overlap=dict(median=float(np.median(jac)), mean=float(jac.mean()),
                             frac_identical=float(np.mean(jac >= 0.999))),
        err_official=dict(median=float(np.median(err_off)), mean=float(err_off.mean())),
        err_proposed=dict(median=float(np.median(err_pro)), mean=float(err_pro.mean())),
        err_improvement_frac=float(np.mean(err_pro < err_off)),
        err_improvement_median_pct=float(np.median((err_off - err_pro) / (err_off + 1e-12)) * 100),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    with open(os.path.join(OUT_DIR, "d6_raw.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("done")


if __name__ == "__main__":
    main()
