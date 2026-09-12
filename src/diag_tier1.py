# -*- coding: utf-8 -*-
"""第一档诊断 D1-D4，严格标签无关，见用户 2026-08-11 诊断任务书。

不碰任何情绪/被试标签。目的是在接受 session3 负结果前排除 λ 分支的实现错误。
D1 若触发停止条件（n_after_topk>8 或 n_unique_freq<n_after_topk），
本脚本会打印明确的 STOP 标记并跳过 D2-D4。

写 results/diagnostics/report_tier1.md，不碰 results/separability/ 下的
session3 官方结果。
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import scipy.linalg as sla
from joblib import Parallel, delayed

import koopman as km
from config import RESULTS_DIR, SEED_FS
from data import list_subjects
from preprocess import build_segments

DT = 1.0 / SEED_FS
CFG = km.DEFAULT
N_SEGMENTS_D1 = 500
N_SEGMENTS_D3 = 60   # 15 试次 x 4 窗口，只从少数几个 (被试,session) 里取
N_SEGMENTS_D4 = 100
SAMPLE_SEED = 2
OUT_DIR = os.path.join(RESULTS_DIR, "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)


# ==========================================================================
# 复刻 koopman.dmd_spectrum 的完整内部过程，但把每一步的中间量都吐出来。
# 数值逻辑必须跟 koopman.py 逐行一致，否则这次审计毫无意义。
# ==========================================================================
def dmd_full_trace(X, dt, cfg=CFG):
    H = km.hankel_embed(X, cfg.delay)
    X1, X2 = H[:, :-1], H[:, 1:]
    U, s, Vt = np.linalg.svd(X1, full_matrices=False)
    r = min(cfg.rank, len(s), X1.shape[0])
    U, s, Vt = U[:, :r], s[:r], Vt[:r]
    Sinv = np.diag(1.0 / s)

    Af = U.conj().T @ X2 @ Vt.conj().T @ Sinv
    A_before_cast = None
    if cfg.method == "fb":
        Ub, sb, Vtb = np.linalg.svd(X2, full_matrices=False)
        Ub, sb, Vtb = Ub[:, :r], sb[:r], Vtb[:r]
        Ab = Ub.conj().T @ X1 @ Vtb.conj().T @ np.diag(1.0 / sb)
        try:
            A_raw = sla.sqrtm(Af @ np.linalg.inv(Ab))
            A_before_cast = A_raw.copy()
            A = np.real_if_close(A_raw, tol=1e6)
            if not np.all(np.isfinite(A)):
                A = Af
        except Exception:
            A = Af
    else:
        A = Af

    lam, W = np.linalg.eig(A)
    Phi = X2 @ Vt.conj().T @ Sinv @ W
    b = np.linalg.lstsq(Phi, H[:, 0], rcond=None)[0]

    f = np.angle(lam) / (2 * np.pi * dt)
    sigma = np.log(np.abs(lam) + 1e-300) / dt

    keep = (f >= cfg.f_lo) & (f <= cfg.f_hi) & (np.abs(lam) <= cfg.mod_max)
    if keep.sum() == 0:
        keep = (f >= cfg.f_lo) & (f <= cfg.f_hi)
    if keep.sum() == 0:
        keep = np.ones_like(f, dtype=bool)
    keep_idx = np.where(keep)[0]
    amp = np.abs(b[keep_idx])
    order = np.argsort(-amp)[:cfg.top_k]
    mode_idx = keep_idx[order]

    return dict(A=A, A_before_cast=A_before_cast, r=r, lam=lam, f=f, sigma=sigma,
                b=b, keep_idx=keep_idx, mode_idx=mode_idx)


# ==========================================================================
# D1：模态计数审计
# ==========================================================================
def _d1_one(x):
    tr = dmd_full_trace(x, DT)
    lam = tr["lam"]
    r = tr["r"]

    A = tr["A"]
    is_complex_dtype = np.iscomplexobj(A)
    a_imag_max = float(np.max(np.abs(A.imag))) if is_complex_dtype else 0.0
    a_real_max = float(np.max(np.abs(A.real))) if is_complex_dtype else float(np.max(np.abs(A)))
    # real_if_close 之前的原始 sqrtm 输出的虚部量级，独立于 tol=1e6 这个转换判据
    raw = tr["A_before_cast"]
    raw_imag_ratio = None
    if raw is not None:
        raw_imag_ratio = float(np.max(np.abs(raw.imag)) / (np.max(np.abs(raw.real)) + 1e-30))

    tol = 1e-6 * (np.max(np.abs(lam)) + 1e-30)
    n_real = int(np.sum(np.abs(lam.imag) < tol))

    # 共轭配对：把每个非实特征值跟"距离最近的共轭"配对，配不上的记为 unpaired
    non_real_idx = np.where(np.abs(lam.imag) >= tol)[0]
    paired = set()
    n_pairs = 0
    unpaired = []
    remaining = list(non_real_idx)
    for i in non_real_idx:
        if i in paired:
            continue
        conj_target = np.conj(lam[i])
        cand = [j for j in remaining if j != i and j not in paired]
        if not cand:
            unpaired.append(int(i)); continue
        dists = [abs(lam[j] - conj_target) for j in cand]
        j_best = cand[int(np.argmin(dists))]
        d_best = dists[int(np.argmin(dists))]
        scale = abs(lam[i]) + 1e-30
        if d_best / scale < 1e-4:
            paired.add(i); paired.add(j_best); n_pairs += 1
        else:
            unpaired.append(int(i))

    f_kept = tr["f"][tr["mode_idx"]]
    n_unique = len(np.unique(np.round(f_kept, 2))) if len(f_kept) else 0

    return dict(
        n_eigs_raw=int(len(lam)), r=int(r),
        n_real=n_real, n_conj_pairs=n_pairs, n_unpaired_complex=len(unpaired),
        n_after_freq_modulus=int(len(tr["keep_idx"])),
        n_after_topk=int(len(tr["mode_idx"])),
        n_unique_freq=n_unique,
        a_is_complex_dtype=bool(is_complex_dtype), a_imag_max=a_imag_max, a_real_max=a_real_max,
        raw_sqrtm_imag_ratio=raw_imag_ratio,
    )


def run_d1(xs, n_jobs=-1):
    print(f"[D1] {len(xs)} 段，逐段模态计数审计...")
    t0 = time.time()
    rows = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_d1_one)(x) for x in xs)
    print(f"[D1] 耗时 {time.time()-t0:.1f}s")

    n_after_topk = np.array([r["n_after_topk"] for r in rows])
    n_after_fm = np.array([r["n_after_freq_modulus"] for r in rows])
    n_unique = np.array([r["n_unique_freq"] for r in rows])
    n_unpaired = np.array([r["n_unpaired_complex"] for r in rows])
    a_complex = np.array([r["a_is_complex_dtype"] for r in rows])
    a_imag_max = np.array([r["a_imag_max"] for r in rows])
    raw_ratio = np.array([r["raw_sqrtm_imag_ratio"] for r in rows if r["raw_sqrtm_imag_ratio"] is not None])

    hist_topk = np.bincount(n_after_topk, minlength=17)

    over8 = int(np.sum(n_after_topk > 8))
    unique_lt_topk = int(np.sum(n_unique < n_after_topk))

    report = dict(
        n_segments=len(xs),
        n_after_topk=dict(mean=float(n_after_topk.mean()), median=float(np.median(n_after_topk)),
                          max=int(n_after_topk.max()), min=int(n_after_topk.min()),
                          histogram=hist_topk.tolist()),
        n_after_freq_modulus=dict(mean=float(n_after_fm.mean()), median=float(np.median(n_after_fm)),
                                  max=int(n_after_fm.max())),
        n_segments_over8=over8, frac_over8=float(over8 / len(xs)),
        n_segments_unique_lt_topk=unique_lt_topk,
        n_unpaired_complex=dict(mean=float(n_unpaired.mean()), max=int(n_unpaired.max()),
                                frac_any_unpaired=float(np.mean(n_unpaired > 0))),
        A_dtype_complex_frac=float(a_complex.mean()),
        A_imag_max=dict(mean=float(a_imag_max.mean()), max=float(a_imag_max.max())),
        raw_sqrtm_imag_to_real_ratio=dict(
            mean=float(raw_ratio.mean()) if len(raw_ratio) else None,
            median=float(np.median(raw_ratio)) if len(raw_ratio) else None,
            max=float(raw_ratio.max()) if len(raw_ratio) else None) if len(raw_ratio) else None,
    )
    return report, rows


# ==========================================================================
# D2：(f,sigma) 尺度平衡审计
# ==========================================================================
def run_d2(sample_f, sample_sigma, s_f, s_sigma, n_pairs=1000, seed=SAMPLE_SEED):
    rng = np.random.default_rng(seed)
    n = len(sample_f)
    ratios = []
    for _ in range(n_pairs):
        i, j = rng.integers(0, n, size=2)
        fi, si, wi = sample_f[i], sample_sigma[i], None
        fj, sj = sample_f[j], sample_sigma[j]
        if len(fi) == 0 or len(fj) == 0:
            continue
        df = (fi[:, None] - fj[None, :]) / s_f
        dsig = (si[:, None] - sj[None, :]) / s_sigma
        Cf = (df ** 2)
        Cs = (dsig ** 2)
        # 用等权近似传输代价分解（不重新跑 OT，只看代价矩阵量级分布，
        # 因为这里要看的是尺度是否失衡，不是精确传输方案）
        ratios.append(float(Cf.mean() / (Cf.mean() + Cs.mean() + 1e-30)))
    ratios = np.array(ratios)
    all_f = np.concatenate(sample_f)
    all_sigma = np.concatenate(sample_sigma)
    return dict(
        s_f=s_f, s_sigma=s_sigma,
        f_stats=dict(min=float(all_f.min()), max=float(all_f.max()),
                    median=float(np.median(all_f)), p1=float(np.percentile(all_f, 1)),
                    p99=float(np.percentile(all_f, 99))),
        sigma_stats=dict(min=float(all_sigma.min()), max=float(all_sigma.max()),
                         median=float(np.median(all_sigma)), p1=float(np.percentile(all_sigma, 1)),
                         p99=float(np.percentile(all_sigma, 99))),
        cost_ratio_Cf_share=dict(median=float(np.median(ratios)), mean=float(ratios.mean()),
                                 p5=float(np.percentile(ratios, 5)), p95=float(np.percentile(ratios, 95))),
    )


# ==========================================================================
# D3：估计器自身可靠性（同试次相邻窗口 vs 跨情绪）
# ==========================================================================
def run_d3(n_subjects=5, n_trials=4, seed=SAMPLE_SEED):
    """用未切段的原始试次数据，手动切 4 个连续 4s 窗口（0-4s,4-8s,8-12s,12-16s），
    完全绕开 build_segments 的伪迹剔除/乱序，保证"相邻"关系的物理意义。"""
    from data import load_trials
    from preprocess import bandpass, PreprocConfig
    from config import SEED_GOOD_CHANNELS

    rng = np.random.default_rng(seed)
    subjects = rng.choice(list_subjects("seed"), size=n_subjects, replace=False)
    cfg_pp = PreprocConfig()
    win = int(round(cfg_pp.win_s * SEED_FS))

    within_trial, cross_trial_same_emo, cross_emo = [], [], []
    for subj in subjects:
        X, y, fs = load_trials("seed", int(subj), 3)
        trial_ids = rng.choice(len(X), size=min(n_trials, len(X)), replace=False)
        specs_by_trial = {}
        for t in trial_ids:
            x = bandpass(X[t], fs, cfg_pp)[list(SEED_GOOD_CHANNELS), :]
            wins = [x[:, k * win:(k + 1) * win] for k in range(4) if (k + 1) * win <= x.shape[1]]
            specs_by_trial[int(t)] = ([km.dmd_spectrum(w, 1.0 / fs) for w in wins], int(y[t]))

        for t, (specs, emo) in specs_by_trial.items():
            for i in range(len(specs) - 1):
                within_trial.append((specs[i], specs[i + 1]))
        tids = list(specs_by_trial.keys())
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                s1, e1 = specs_by_trial[tids[i]]
                s2, e2 = specs_by_trial[tids[j]]
                pair = (s1[0], s2[0])
                if e1 == e2:
                    cross_trial_same_emo.append(pair)
                else:
                    cross_emo.append(pair)

    def s_f_sigma(all_specs):
        fs_ = np.concatenate([s["f"] for pair in all_specs for s in pair])
        return fs_

    all_f = np.concatenate([km.dmd_spectrum(np.zeros((1, 1)), 1.0)["f"] if False else s["f"]
                            for pair in within_trial + cross_trial_same_emo + cross_emo for s in pair])
    from scipy.stats import iqr as _iqr
    all_sigma = np.concatenate([s["sigma"] for pair in within_trial + cross_trial_same_emo + cross_emo for s in pair])
    s_f = float(_iqr(all_f)) or 1.0
    s_sigma = float(_iqr(all_sigma)) or 1.0

    def dists(pairs):
        out = []
        for s1, s2 in pairs:
            out.append(km.eigval_distance(s1, s2, s_f, s_sigma))
        return np.array(out)

    d_within = dists(within_trial)
    d_cross_trial_same_emo = dists(cross_trial_same_emo)
    d_cross_emo = dists(cross_emo)

    def stat(a):
        return dict(n=len(a), median=float(np.median(a)) if len(a) else None,
                   mean=float(a.mean()) if len(a) else None)

    return dict(s_f=s_f, s_sigma=s_sigma,
               d_within_trial=stat(d_within),
               d_cross_trial_same_emotion=stat(d_cross_trial_same_emo),
               d_cross_emotion=stat(d_cross_emo))


# ==========================================================================
# D4：秩容量与重构误差
# ==========================================================================
def _d4_one(x):
    H = km.hankel_embed(x, CFG.delay)
    X1 = H[:, :-1]
    s = np.linalg.svd(X1, compute_uv=False)
    energy_total = float(np.sum(s ** 2))
    E = {r: float(np.sum(s[:r] ** 2) / energy_total) for r in (8, 16, 32, 48, 64) if r <= len(s)}
    stable_rank = float(np.sum(s ** 2) / (s[0] ** 2))

    tr = dmd_full_trace(x, DT)
    lam, Phi, b, mode_idx = tr["lam"], None, None, tr["mode_idx"]
    # 用完整 trace 里已有的量重构（复用 D1 里算过的 Phi 需要额外返回，这里重算一次）
    Xt1, Xt2 = H[:, :-1], H[:, 1:]
    U, sv, Vt = np.linalg.svd(Xt1, full_matrices=False)
    r = min(CFG.rank, len(sv), Xt1.shape[0])
    U, sv, Vt = U[:, :r], sv[:r], Vt[:r]
    Sinv = np.diag(1.0 / sv)
    Phi = Xt2 @ Vt.conj().T @ Sinv @ np.linalg.eig(tr["A"])[1]
    b_full = np.linalg.lstsq(Phi, H[:, 0], rcond=None)[0]
    n_steps = H.shape[1]
    t = np.arange(n_steps)
    cols = mode_idx
    if len(cols) == 0:
        cols = np.arange(r)
    V = tr["lam"][cols][:, None] ** t[None, :]
    recon = np.real(Phi[:, cols] @ (b_full[cols][:, None] * V))
    rel_err = float(np.linalg.norm(H - recon) / (np.linalg.norm(H) + 1e-12))

    return dict(E=E, stable_rank=stable_rank, recon_rel_err=rel_err)


def run_d4(xs, n_jobs=-1):
    print(f"[D4] {len(xs)} 段，秩容量与重构误差...")
    rows = Parallel(n_jobs=n_jobs, verbose=0)(delayed(_d4_one)(x) for x in xs)
    E_by_r = {r: np.array([row["E"].get(r, np.nan) for row in rows]) for r in (8, 16, 32, 48, 64)}
    stable_rank = np.array([row["stable_rank"] for row in rows])
    recon_err = np.array([row["recon_rel_err"] for row in rows])
    return dict(
        E_by_r={str(r): dict(median=float(np.nanmedian(v)), mean=float(np.nanmean(v)))
               for r, v in E_by_r.items()},
        stable_rank=dict(median=float(np.median(stable_rank)), mean=float(stable_rank.mean())),
        recon_rel_err=dict(median=float(np.median(recon_err)), mean=float(recon_err.mean())),
    )


def sample_segments(n, seed=SAMPLE_SEED):
    rng = np.random.default_rng(seed)
    subjects = list_subjects("seed")
    per_subj = max(1, n // len(subjects))
    xs = []
    for subj in subjects:
        r = build_segments("seed", subj, 3)
        m = len(r["segs"])
        idx = rng.choice(m, size=min(per_subj, m), replace=False)
        xs += [r["segs"][i] for i in idx]
    return xs[:n]


def main():
    xs_d1 = sample_segments(N_SEGMENTS_D1, seed=SAMPLE_SEED)
    report_d1, rows_d1 = run_d1(xs_d1)

    print(json.dumps(report_d1, ensure_ascii=False, indent=2))

    stop = report_d1["frac_over8"] > 0 or report_d1["n_segments_unique_lt_topk"] > 0
    with open(os.path.join(OUT_DIR, "d1_raw.json"), "w", encoding="utf-8") as fh:
        json.dump(report_d1, fh, ensure_ascii=False, indent=2)

    if stop:
        print("\n*** STOP: D1 触发停止条件（n_after_topk>8 的段存在，或存在 unique_freq<topk 的段）***")
        write_report_tier1(report_d1, None, None, None, stopped=True)
        return

    print("\n[D1] 未触发停止条件，继续 D2-D4")
    # D2 需要 sample 的 f/sigma（复用 D1 的 500 段自带的谱，但 D1 只存了统计量，
    # 这里为了 D2/D4 需要，独立轻量重算一次官方 dmd_spectrum，用相同抽样）
    specs = Parallel(n_jobs=-1, verbose=5)(delayed(km.dmd_spectrum)(x, DT, CFG) for x in xs_d1)
    f_list = [s["f"] for s in specs]
    sigma_list = [s["sigma"] for s in specs]
    from scipy.stats import iqr
    s_f = float(iqr(np.concatenate(f_list))) or 1.0
    s_sigma = float(iqr(np.concatenate(sigma_list))) or 1.0
    report_d2 = run_d2(f_list, sigma_list, s_f, s_sigma)
    with open(os.path.join(OUT_DIR, "d2_raw.json"), "w", encoding="utf-8") as fh:
        json.dump(report_d2, fh, ensure_ascii=False, indent=2)
    print(json.dumps(report_d2, ensure_ascii=False, indent=2))

    report_d3 = run_d3()
    with open(os.path.join(OUT_DIR, "d3_raw.json"), "w", encoding="utf-8") as fh:
        json.dump(report_d3, fh, ensure_ascii=False, indent=2)
    print(json.dumps(report_d3, ensure_ascii=False, indent=2))

    xs_d4 = sample_segments(N_SEGMENTS_D4, seed=SAMPLE_SEED + 1)
    report_d4 = run_d4(xs_d4)
    with open(os.path.join(OUT_DIR, "d4_raw.json"), "w", encoding="utf-8") as fh:
        json.dump(report_d4, fh, ensure_ascii=False, indent=2)
    print(json.dumps(report_d4, ensure_ascii=False, indent=2))

    write_report_tier1(report_d1, report_d2, report_d3, report_d4, stopped=False)


def write_report_tier1(d1, d2, d3, d4, stopped):
    lines = ["# 第一档诊断报告\n"]
    lines.append(f"停止条件触发：{'是' if stopped else '否'}\n")
    lines.append("## D1 模态计数审计\n")
    lines.append(f"```\n{json.dumps(d1, ensure_ascii=False, indent=2)}\n```\n")
    if d2 is not None:
        lines.append("## D2 (f,sigma) 尺度平衡审计\n")
        lines.append(f"```\n{json.dumps(d2, ensure_ascii=False, indent=2)}\n```\n")
    if d3 is not None:
        lines.append("## D3 估计器自身可靠性\n")
        lines.append(f"```\n{json.dumps(d3, ensure_ascii=False, indent=2)}\n```\n")
    if d4 is not None:
        lines.append("## D4 秩容量与重构误差\n")
        lines.append(f"```\n{json.dumps(d4, ensure_ascii=False, indent=2)}\n```\n")
    path = os.path.join(OUT_DIR, "report_tier1.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\n报告已写 {path}")


if __name__ == "__main__":
    main()
