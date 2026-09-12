# -*- coding: utf-8 -*-
"""标签无关诊断：结局 C 出来之后，先排查是不是工程/超参问题，再接受结论。

PREREGISTER §3 允许"用不含标签信息的准则（如谱重构误差）"重调参数——
这里只做诊断，不产生任何用于判据的数值，全程不碰情绪/被试标签。

三件事：
  1. top-12（按幅度选，再叠加 1-45Hz + mod_max<=1.02 过滤）相对全部 r=16 个
     模态，重构误差涨了多少——过滤是不是把有信息量的模态误杀了。
  2. 被过滤掉的模态集中在哪个频段——如果扎堆在接近 0 Hz，说明是低频漂移
     而不是真实节律在被保留/剔除。
  3. d（时延深度）是否选得合理——用段内切分的样本外预测误差（训练用段的
     前 80%，预测剩下 20%），而不是任何跟标签有关的东西。

运行：python diag_spectrum.py
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import scipy.linalg as sla
from joblib import Parallel, delayed

from config import RESULTS_DIR, SEED_FS
from data import list_subjects
from preprocess import build_segments

DT = 1.0 / SEED_FS
N_PER_SUBJECT = 10          # 每被试随机取 10 段，不看标签
SAMPLE_SEED = 1              # 特意跟主分析（seed=0）不同，避免任何"复用同一次抽样"的联想
D_GRID = (8, 16, 24, 32)
RANK = 16
F_RANGE = (1.0, 45.0)
MOD_MAX = 1.02
TOP_K = 12


def hankel_embed(X, d):
    C, T = X.shape
    return np.vstack([X[:, i:T - d + 1 + i] for i in range(d)])


def _fit(H, r, method="fb"):
    """DMD 拟合核心：给定已经嵌入好的 Hankel 矩阵 H，返回 Phi/lam/b。
    跟 koopman.dmd_spectrum 数值逻辑一致，但不做频段/模长过滤、
    不把 lam 投影回单位圆——诊断要看的就是过滤前的真实情况。"""
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
    f = np.angle(lam) / (2 * np.pi * DT)
    sigma = np.log(np.abs(lam) + 1e-300) / DT
    return dict(Phi=Phi, lam=lam, b=b, f=f, sigma=sigma)


def dmd_core(X, d, r, method="fb"):
    H = hankel_embed(X, d)
    core = _fit(H, r, method)
    core["H"] = H
    return core


def rel_error(H, H_hat):
    return float(np.linalg.norm(H - H_hat) / (np.linalg.norm(H) + 1e-12))


def reconstruct(core, cols):
    lam, Phi, b, n_steps = core["lam"], core["Phi"], core["b"], core["H"].shape[1]
    t = np.arange(n_steps)
    V = lam[cols][:, None] ** t[None, :]
    return Phi[:, cols] @ (b[cols][:, None] * V)


# ==========================================================================
# 诊断 1+2：重构误差 + 频率分布（d=16, r=16, 官方参数）
# ==========================================================================
def _diag_reconstruction(x):
    core = dmd_core(x, 16, RANK, method="fb")
    H = core["H"]
    n = RANK

    err_full = rel_error(H, reconstruct(core, np.arange(n)))

    amp_all = np.abs(core["b"])
    top12_amp_idx = np.argsort(-amp_all)[:TOP_K]
    err_top12_amp_only = rel_error(H, reconstruct(core, top12_amp_idx))

    keep = ((core["f"] >= F_RANGE[0]) & (core["f"] <= F_RANGE[1])
            & (np.abs(core["lam"]) <= MOD_MAX))
    keep_idx = np.where(keep)[0]
    if len(keep_idx) == 0:
        keep_idx = np.arange(n)
    amp_kept = amp_all[keep_idx]
    official_idx = keep_idx[np.argsort(-amp_kept)[:TOP_K]]
    err_official = rel_error(H, reconstruct(core, official_idx))

    discarded_idx = np.setdiff1d(np.arange(n), official_idx)
    return dict(
        err_full=err_full, err_top12_amp_only=err_top12_amp_only, err_official=err_official,
        f_kept=core["f"][official_idx].tolist(), f_discarded=core["f"][discarded_idx].tolist(),
        n_official_modes=len(official_idx))


# ==========================================================================
# 诊断 3：d 敏感性——段内前 80% 拟合，预测后 20%，纯样本外误差，不碰标签
# ==========================================================================
def _diag_d_sensitivity(x, d, holdout_frac=0.2):
    """段内前 80% 拟合 DMD，用模态外推公式 Phi@(b*lam^t) 预测后 20%，
    跟 actual 比相对误差。纯样本外时间切分，不碰任何标签。

    注意：不能直接用降维空间的算子 A（r×r）去乘全维状态——A 是在 POD
    约化坐标系里定义的，套用在原始 C*d 维状态向量上维度对不上、意义也不对。
    模态外推公式在全维空间里是自洽的，用它才对。
    """
    H = hankel_embed(x, d)
    n = H.shape[1]
    n_test = max(1, int(n * holdout_frac))
    n_train = n - n_test
    core = _fit(H[:, :n_train], RANK, method="fb")
    t = np.arange(n_train, n_train + n_test)
    pred = np.real(core["Phi"] @ (core["b"][:, None] * (core["lam"][:, None] ** t[None, :])))
    actual = H[:, n_train:n_train + n_test]
    return rel_error(actual, pred)


def _process_one(x, d_grid):
    rec = _diag_reconstruction(x)
    d_err = {d: _diag_d_sensitivity(x, d) for d in d_grid}
    return rec, d_err


def sample_segments(n_per_subject=N_PER_SUBJECT, seed=SAMPLE_SEED):
    """随机取样，完全不看标签（既不按情绪分层，也不按试次分层）。"""
    rng = np.random.default_rng(seed)
    xs = []
    for subj in list_subjects("seed"):
        r = build_segments("seed", subj, 3)  # session 3，跟主分析一致
        n = len(r["segs"])
        idx = rng.choice(n, size=min(n_per_subject, n), replace=False)
        xs += [r["segs"][i] for i in idx]
    return xs


def main():
    out_dir = os.path.join(RESULTS_DIR, "separability", "diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    xs = sample_segments()
    print(f"诊断样本 {len(xs)} 段（每被试 {N_PER_SUBJECT} 段，纯随机不看标签，seed={SAMPLE_SEED}）")

    t0 = time.time()
    out = Parallel(n_jobs=-1, verbose=5)(
        delayed(_process_one)(x, D_GRID) for x in xs)
    print(f"耗时 {time.time()-t0:.1f}s")

    recs = [o[0] for o in out]
    d_errs = [o[1] for o in out]

    err_full = np.array([r["err_full"] for r in recs])
    err_top12 = np.array([r["err_top12_amp_only"] for r in recs])
    err_official = np.array([r["err_official"] for r in recs])
    n_modes = np.array([r["n_official_modes"] for r in recs])

    f_kept = np.concatenate([r["f_kept"] for r in recs]) if len(recs) else np.array([])
    f_discarded = np.concatenate([r["f_discarded"] for r in recs]) if len(recs) else np.array([])

    print("\n[诊断 1] 重构相对误差（r=16 全部模态 vs 幅度选 top12 vs 官方过滤+top12）")
    print(f"  全部16模态:        median={np.median(err_full):.4f}  mean={err_full.mean():.4f}")
    print(f"  仅幅度选top12:      median={np.median(err_top12):.4f}  mean={err_top12.mean():.4f}")
    print(f"  官方(频段+模长+top12): median={np.median(err_official):.4f}  mean={err_official.mean():.4f}")
    print(f"  官方过滤后平均保留模态数: {n_modes.mean():.2f} / {TOP_K}")

    print("\n[诊断 2] 保留 vs 剔除模态的频率分布")
    for name, arr in (("保留", f_kept), ("剔除", f_discarded)):
        if len(arr) == 0:
            print(f"  {name}: 无"); continue
        print(f"  {name}(n={len(arr)}): median={np.median(arr):.2f}Hz "
              f"  [1-4Hz占比]={np.mean((arr>=1)&(arr<4)):.2%}"
              f"  [4-8Hz]={np.mean((arr>=4)&(arr<8)):.2%}"
              f"  [8-13Hz]={np.mean((arr>=8)&(arr<13)):.2%}"
              f"  [13-30Hz]={np.mean((arr>=13)&(arr<30)):.2%}"
              f"  [30-45Hz]={np.mean((arr>=30)&(arr<=45)):.2%}")

    print("\n[诊断 3] d 敏感性——段内后 20% 的样本外预测相对误差")
    for d in D_GRID:
        vals = np.array([de[d] for de in d_errs])
        print(f"  d={d:>3}: median={np.median(vals):.4f}  mean={vals.mean():.4f}  "
              f"（值越低=该 d 对这批数据的样本外预测越准）")

    summary = dict(
        n_segments=len(xs), sample_seed=SAMPLE_SEED,
        err_full=dict(median=float(np.median(err_full)), mean=float(err_full.mean())),
        err_top12_amp_only=dict(median=float(np.median(err_top12)), mean=float(err_top12.mean())),
        err_official=dict(median=float(np.median(err_official)), mean=float(err_official.mean())),
        n_official_modes_mean=float(n_modes.mean()),
        f_kept_median=float(np.median(f_kept)) if len(f_kept) else None,
        f_discarded_median=float(np.median(f_discarded)) if len(f_discarded) else None,
        d_sensitivity={str(d): dict(median=float(np.median([de[d] for de in d_errs])),
                                    mean=float(np.mean([de[d] for de in d_errs])))
                       for d in D_GRID})
    path = os.path.join(out_dir, "diag_spectrum_summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"\n结果已存 {path}")


if __name__ == "__main__":
    main()
