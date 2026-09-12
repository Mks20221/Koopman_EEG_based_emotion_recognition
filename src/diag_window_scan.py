# -*- coding: utf-8 -*-
"""窗长敏感性扫描（第三档，label-free）。T in {4,8,12,16}s，d=16/r=16 固定，
只变窗长。测三件事：奇异值谱间隙位置、正交不变性、官方过滤后重构误差。
不碰任何标签，产出用于决定是否要改窗长走修订流程。"""
import json
import os
import time

import numpy as np
from joblib import Parallel, delayed

import koopman as km
from config import RESULTS_DIR, SEED_FS, SEED_GOOD_CHANNELS
from data import list_subjects, load_trials
from preprocess import PreprocConfig, bandpass, segment

DT = 1.0 / SEED_FS
D = 16
R = 16
T_GRID = (4, 8, 12, 16)
N_SEGMENTS = 120
N_ORTH = 20
SAMPLE_SEED = 17
OUT_DIR = os.path.join(RESULTS_DIR, "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)


def sample_segments_T(win_s, n, seed):
    rng = np.random.default_rng(seed)
    cfg = PreprocConfig(win_s=win_s)
    subjects = list_subjects("seed")
    per_subj = max(1, n // len(subjects))
    xs = []
    for subj in subjects:
        X, y, fs = load_trials("seed", subj, 3)
        t_idx = rng.integers(0, len(X))
        x = bandpass(X[t_idx], fs, cfg)[list(SEED_GOOD_CHANNELS), :]
        segs = segment(x, fs, cfg)
        if len(segs) == 0:
            continue
        m = min(per_subj, len(segs))
        idx = rng.choice(len(segs), size=m, replace=False)
        xs += [segs[i] for i in idx]
    return xs[:n]


def gap_position(x, d=D, max_check=80):
    H = km.hankel_embed(x, d)
    X1 = H[:, :-1]
    s = np.linalg.svd(X1, compute_uv=False)
    s = s[:min(max_check, len(s))]
    gaps = -np.diff(np.log(s + 1e-12))
    idx = int(np.argmax(gaps))
    return idx + 1


def official_recon_err(x, cfg=km.DEFAULT):
    sp = km.dmd_spectrum(x, DT, cfg)
    H = km.hankel_embed(x, cfg.delay)
    n = H.shape[1]
    t = np.arange(n)
    cols = sp["mode_idx"]
    V = sp["lam"][:, None] ** t[None, :] if False else None
    lam_full = sp["lam"]
    Vfull = lam_full[:, None] ** t[None, :]
    recon = np.real(sp["Phi"][:, cols] @ ((np.linalg.lstsq(sp["Phi"], H[:, 0], rcond=None)[0])[cols][:, None] * Vfull[:len(cols)])) if False else None
    return None


def official_recon_err2(x, cfg=km.DEFAULT):
    H = km.hankel_embed(x, cfg.delay)
    X1, X2 = H[:, :-1], H[:, 1:]
    import scipy.linalg as sla
    U, s, Vt = np.linalg.svd(X1, full_matrices=False)
    r = min(cfg.rank, len(s), X1.shape[0])
    U, s, Vt = U[:, :r], s[:r], Vt[:r]
    Sinv = np.diag(1.0 / s)
    Af = U.conj().T @ X2 @ Vt.conj().T @ Sinv
    Ub, sb, Vtb = np.linalg.svd(X2, full_matrices=False)
    Ub, sb, Vtb = Ub[:, :r], sb[:r], Vtb[:r]
    Ab = Ub.conj().T @ X1 @ Vtb.conj().T @ np.diag(1.0 / sb)
    try:
        A = sla.sqrtm(Af @ np.linalg.inv(Ab))
        A = np.real_if_close(A, tol=1e6)
        if not np.all(np.isfinite(A)):
            A = Af
    except Exception:
        A = Af
    lam, W = np.linalg.eig(A)
    Phi = X2 @ Vt.conj().T @ Sinv @ W
    b = np.linalg.lstsq(Phi, H[:, 0], rcond=None)[0]
    f = np.angle(lam) / (2 * np.pi * DT)
    keep = (f >= cfg.f_lo) & (f <= cfg.f_hi) & (np.abs(lam) <= cfg.mod_max)
    if keep.sum() == 0:
        keep = (f >= cfg.f_lo) & (f <= cfg.f_hi)
    if keep.sum() == 0:
        keep = np.ones_like(f, dtype=bool)
    keep_idx = np.where(keep)[0]
    amp = np.abs(b[keep_idx])
    order = np.argsort(-amp)[:cfg.top_k]
    mode_idx = keep_idx[order]

    n = H.shape[1]
    t = np.arange(n)
    V = lam[mode_idx][:, None] ** t[None, :]
    recon = np.real(Phi[:, mode_idx] @ (b[mode_idx][:, None] * V))
    return float(np.linalg.norm(H - recon) / (np.linalg.norm(H) + 1e-12)), len(mode_idx)


def random_orthogonal(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Q, Rm = np.linalg.qr(A)
    return Q @ np.diag(np.sign(np.diag(Rm)))


def orth_w2(x, s_f, s_sigma, seed, cfg=km.DEFAULT):
    n_ch = x.shape[0]
    Q = random_orthogonal(n_ch, seed)
    xq = Q @ x
    sp1 = km.dmd_spectrum(x, DT, cfg)
    sp2 = km.dmd_spectrum(xq, DT, cfg)
    return km.eigval_distance(sp1, sp2, s_f, s_sigma)


def run_for_T(T):
    t0 = time.time()
    xs = sample_segments_T(T, N_SEGMENTS, seed=SAMPLE_SEED)
    print(f"[T={T}s] {len(xs)} 段采样完成")

    gaps = Parallel(n_jobs=-1, verbose=0)(delayed(gap_position)(x) for x in xs)
    gaps = np.array(gaps)

    recon = Parallel(n_jobs=-1, verbose=0)(delayed(official_recon_err2)(x) for x in xs)
    errs = np.array([r[0] for r in recon])
    n_modes = np.array([r[1] for r in recon])

    specs = Parallel(n_jobs=-1, verbose=0)(delayed(km.dmd_spectrum)(x, DT, km.DEFAULT) for x in xs)
    from scipy.stats import iqr
    all_f = np.concatenate([s["f"] for s in specs])
    all_sigma = np.concatenate([s["sigma"] for s in specs])
    s_f = float(iqr(all_f)) or 1.0
    s_sigma = float(iqr(all_sigma)) or 1.0

    orth = Parallel(n_jobs=-1, verbose=0)(
        delayed(orth_w2)(x, s_f, s_sigma, 3000 + i) for i, x in enumerate(xs[:N_ORTH]))
    orth = np.array(orth)

    report = dict(
        T=T, n_segments=len(xs), elapsed_sec=time.time() - t0,
        gap_position=dict(median=float(np.median(gaps)), mean=float(gaps.mean()),
                          frac_ge_17=float(np.mean(gaps >= 17)), frac_ge_9=float(np.mean(gaps >= 9))),
        recon_err_official=dict(median=float(np.median(errs)), mean=float(errs.mean())),
        n_official_modes=dict(median=float(np.median(n_modes)), mean=float(n_modes.mean())),
        s_f=s_f, s_sigma=s_sigma,
        orth_w2=dict(median=float(np.median(orth)), mean=float(orth.mean()), max=float(orth.max()),
                    frac_near_zero=float(np.mean(orth < 0.05))),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    all_reports = {}
    for T in T_GRID:
        all_reports[T] = run_for_T(T)
    with open(os.path.join(OUT_DIR, "d_window_scan.json"), "w", encoding="utf-8") as fh:
        json.dump(all_reports, fh, ensure_ascii=False, indent=2)
    print("\n=== 汇总 ===")
    print(f"{'T(s)':>5}{'gap>=17占比':>14}{'重构误差中位':>14}{'正交W2中位':>12}{'正交W2接近0占比':>16}")
    for T in T_GRID:
        r = all_reports[T]
        print(f"{T:>5}{r['gap_position']['frac_ge_17']*100:>13.1f}%"
              f"{r['recon_err_official']['median']:>14.3f}"
              f"{r['orth_w2']['median']:>12.3f}"
              f"{r['orth_w2']['frac_near_zero']*100:>15.1f}%")


if __name__ == "__main__":
    main()
