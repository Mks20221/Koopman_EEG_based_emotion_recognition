# -*- coding: utf-8 -*-
"""第 0 阶段第 2 周 · 可分性检验，严格按 docs/PREREGISTER.md 实现。

任何数值定义（距离、Sep、判据阈值）改动都必须先在 PREREGISTER.md 的修订记录
里写明理由，这里只负责实现，不负责决定。

用法：
    python exp_separability.py --smoke          # 每被试每情绪只抽 3 段，验证管线通不通
    python exp_separability.py                  # 主分析：session 3，1-45Hz，2700 段
    python exp_separability.py --control         # 加 1-30Hz 对照（PREREGISTER §11）
    python exp_separability.py --de              # 加 DE 基线（PREREGISTER §9）
    python exp_separability.py --control --de    # 全套
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import iqr
from sklearn.metrics import silhouette_score

import koopman as km
from config import (CACHE_DIR, RESULTS_DIR, SEED_FS, SEED_GOOD_CHANNELS)
from data import list_subjects
from preprocess import DEFAULT as PREPROC_DEFAULT
from preprocess import PreprocConfig, build_segments, stratified_sample

# 影响数值结果的代码版本，改动计算逻辑必须 +1。
CODE_VERSION = 1

# PREREGISTER §1、§4：主分析集 session 3，确认集 session 1、2
MAIN_SESSION = 3
CONFIRM_SESSIONS = (1, 2)
N_PER_CLASS = 60
SAMPLE_SEED = 0

# PREREGISTER §6
K_NN = 5
N_PERM = 1000
PERM_SEED = 0

# PREREGISTER §9
DE_BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))

# PREREGISTER 修订 2（2026-08-14）：窗长 4s→8s，伪迹阈值 15→20，label-free 依据见
# 修订记录。session 3 自本修订起退出候选，正式检验改在 session 1、2 上跑。
PREPROC_T8 = PreprocConfig(win_s=8.0, reject_abs_thr=20.0)
CONFIRM_SESSIONS_NEW = (1, 2)


# ==========================================================================
# 1. 谱估计（每段独立，可并行），分层缓存
# ==========================================================================
def _cache_path(dataset, session, cfg, n_per_class, sample_seed, preproc_cfg):
    d = os.path.join(CACHE_DIR, "spectra")
    os.makedirs(d, exist_ok=True)
    key = (f"{dataset}_ses{session}_{cfg.key()}_{preproc_cfg.key()}"
           f"_n{n_per_class}_sd{sample_seed}_v{CODE_VERSION}")
    return os.path.join(d, key + ".npz")


def _estimate_one(x, fs, cfg, n_channels):
    dt = 1.0 / fs
    sp = km.dmd_spectrum(x.astype(np.float64), dt, cfg)
    Q = km.spatial_modes(sp, n_channels)
    return sp["f"], sp["sigma"], sp["w"], sp["lam"], Q


def _to_object_array(lst):
    """np.array(list, dtype=object) 在元素是形状不一致（但同 ndim、部分维度
    恰好相等）的数组时会先尝试常规多维堆叠再失败——即使显式给了 dtype=object。
    这里的 Q 列表就是这种情况：段落保留的模态数偶尔 < top_k，导致 (51,12) 与
    (51,7) 混在一起。手动逐个赋值绕开 numpy 的形状推断。"""
    arr = np.empty(len(lst), dtype=object)
    for i, x in enumerate(lst):
        arr[i] = x
    return arr


def collect_samples(dataset="seed", session=MAIN_SESSION, cfg=km.DEFAULT,
                    n_per_class=N_PER_CLASS, sample_seed=SAMPLE_SEED,
                    subjects=None, use_cache=True, n_jobs=-1,
                    preproc_cfg=PREPROC_DEFAULT):
    """每 (被试×情绪) 抽 n_per_class 段（PREREGISTER §4），逐段估谱。

    subjects=None 时用全部被试（正式结果必须如此）；传子集仅用于 --smoke 调试，
    调试结果不落正式缓存路径（cache path 不含 subjects，故子集不写缓存，
    只在内存里跑一遍确认代码通）。

    preproc_cfg 必须显式传入并参与缓存 key（PREREGISTER 修订2起窗长可变，
    不传会默默落回 T=4s 的 PREPROC_DEFAULT，缓存也会跟旧结果冲突）。
    """
    subs_full = list_subjects(dataset)
    is_full = subjects is None or list(subjects) == subs_full
    subjects = list(subjects) if subjects is not None else subs_full

    path = _cache_path(dataset, session, cfg, n_per_class, sample_seed, preproc_cfg)
    if is_full and use_cache and os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return dict(subj=z["subj"], y=z["y"], trial=z["trial"],
                    f=list(z["f"]), sigma=list(z["sigma"]), w=list(z["w"]),
                    lam=list(z["lam"]), Q=list(z["Q"]),
                    meta=json.loads(str(z["meta"])))

    xs, subj_all, y_all, trial_all = [], [], [], []
    reject_stats = {}
    for subj in subjects:
        r = build_segments(dataset, subj, session, cfg=preproc_cfg)
        idx = stratified_sample(r["y"], r["trial"], n_per_class, sample_seed)
        for i in idx:
            xs.append(r["segs"][i])
        subj_all += [subj] * len(idx)
        y_all += r["y"][idx].tolist()
        trial_all += r["trial"][idx].tolist()
        reject_stats[subj] = r["stats"]["reject_rate"]

    n_channels = xs[0].shape[0]
    t0 = time.time()
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_estimate_one)(x, SEED_FS, cfg, n_channels) for x in xs)
    elapsed = time.time() - t0

    f_list = [o[0] for o in out]
    sigma_list = [o[1] for o in out]
    w_list = [o[2] for o in out]
    lam_list = [o[3] for o in out]
    Q_list = [o[4] for o in out]

    meta = dict(dataset=dataset, session=session, cfg=asdict(cfg),
                preproc_cfg=asdict(preproc_cfg),
                n_per_class=n_per_class, sample_seed=sample_seed,
                n_segments=len(xs), n_channels=n_channels,
                reject_rate_by_subject=reject_stats, elapsed_sec=elapsed,
                code_version=CODE_VERSION, subjects=subjects)

    result = dict(subj=np.array(subj_all), y=np.array(y_all), trial=np.array(trial_all),
                 f=f_list, sigma=sigma_list, w=w_list, lam=lam_list, Q=Q_list, meta=meta)

    if is_full:
        np.savez_compressed(
            path, subj=result["subj"], y=result["y"], trial=result["trial"],
            f=_to_object_array(f_list), sigma=_to_object_array(sigma_list),
            w=_to_object_array(w_list), lam=_to_object_array(lam_list),
            Q=_to_object_array(Q_list), meta=json.dumps(meta))
    return result


# ==========================================================================
# 2. 距离矩阵（按行并行，把 3.6M 次 emd2/svd 调用摊到 n 个任务里）
# ==========================================================================
def _row_dist(i, f, sigma, w, Q, s_f, s_sigma, n):
    d_lam = np.zeros(n)
    d_phi = np.zeros(n)
    si = dict(f=f[i], sigma=sigma[i], w=w[i])
    for j in range(i + 1, n):
        sj = dict(f=f[j], sigma=sigma[j], w=w[j])
        d_lam[j] = km.eigval_distance(si, sj, s_f, s_sigma)
        d_phi[j] = km.grassmann_distance(Q[i], Q[j])
    return d_lam, d_phi


def build_distance_matrices(sample, n_jobs=-1):
    f, sigma, w, Q = sample["f"], sample["sigma"], sample["w"], sample["Q"]
    n = len(f)
    s_f = float(iqr(np.concatenate(f))) or 1.0
    s_sigma = float(iqr(np.concatenate(sigma))) or 1.0

    t0 = time.time()
    rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_row_dist)(i, f, sigma, w, Q, s_f, s_sigma, n) for i in range(n))
    elapsed = time.time() - t0

    D_lam = np.zeros((n, n))
    D_phi = np.zeros((n, n))
    for i, (dl, dp) in enumerate(rows):
        D_lam[i, :] += dl; D_lam[:, i] += dl
        D_phi[i, :] += dp; D_phi[:, i] += dp
    return D_lam, D_phi, s_f, s_sigma, elapsed


# ==========================================================================
# 3. 指标：Sep / silhouette / kNN / 置换检验（PREREGISTER §6）
# ==========================================================================
def compute_sep(D, primary_labels, secondary_fixed_labels):
    """Sep = mean(d | 次要标签相同, 主要标签不同) / mean(d | 次要标签相同, 主要标签相同)。"""
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    same_sec = secondary_fixed_labels[iu[0]] == secondary_fixed_labels[iu[1]]
    same_pri = primary_labels[iu[0]] == primary_labels[iu[1]]
    d = D[iu]
    num = d[same_sec & ~same_pri]
    den = d[same_sec & same_pri]
    num_m = float(num.mean()) if len(num) else float("nan")
    den_m = float(den.mean()) if len(den) else float("nan")
    return num_m / (den_m + 1e-12), num_m, den_m


def permutation_test_sep(D, primary_labels, secondary_fixed_labels,
                         n_perm=N_PERM, seed=PERM_SEED):
    rng = np.random.default_rng(seed)
    obs, _, _ = compute_sep(D, primary_labels, secondary_fixed_labels)
    lbl = primary_labels.copy()
    null = np.empty(n_perm)
    for k in range(n_perm):
        rng.shuffle(lbl)
        null[k] = compute_sep(D, lbl, secondary_fixed_labels)[0]
    p = float((null >= obs).mean())
    return float(obs), p


def knn_loo_acc(D, labels, k=K_NN):
    n = D.shape[0]
    Dm = D.copy()
    np.fill_diagonal(Dm, np.inf)
    correct = 0
    for i in range(n):
        nn = np.argpartition(Dm[i], k)[:k]
        vals, counts = np.unique(labels[nn], return_counts=True)
        pred = vals[np.argmax(counts)]
        correct += int(pred == labels[i])
    return correct / n


def full_report(D_lam, D_phi, y, subj, tag=""):
    """PREREGISTER §6/§10：2x2 Sep 表 + p + silhouette + kNN。"""
    rows = {}
    for name, D in (("lam", D_lam), ("phi", D_phi)):
        sep_emo, p_emo = permutation_test_sep(D, y, subj)
        sep_subj, p_subj = permutation_test_sep(D, subj, y)
        rows[name] = dict(
            sep_emo=sep_emo, p_emo=p_emo,
            sep_subj=sep_subj, p_subj=p_subj,
            sil_emo=float(silhouette_score(D, y, metric="precomputed")),
            sil_subj=float(silhouette_score(D, subj, metric="precomputed")),
            knn_emo=knn_loo_acc(D, y), knn_subj=knn_loo_acc(D, subj))
        r = rows[name]
        print(f"  [{tag}/{name}] Sep_emo={r['sep_emo']:.3f}(p={r['p_emo']:.4f})  "
              f"Sep_subj={r['sep_subj']:.3f}(p={r['p_subj']:.4f})  "
              f"sil_emo={r['sil_emo']:.3f}  sil_subj={r['sil_subj']:.3f}  "
              f"kNN_emo={r['knn_emo']:.3f}  kNN_subj={r['knn_subj']:.3f}")
    return rows


# ==========================================================================
# 4. DE 特征基线（PREREGISTER §9）
# ==========================================================================
def de_features(segs, fs, bands=DE_BANDS):
    n, C, L = segs.shape
    nyq = fs / 2.0
    feats = np.zeros((n, C, len(bands)))
    for bi, (lo, hi) in enumerate(bands):
        b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
        filtered = filtfilt(b, a, segs, axis=-1)
        var = filtered.var(axis=-1)
        feats[:, :, bi] = 0.5 * np.log(2 * np.pi * np.e * np.maximum(var, 1e-12))
    return feats.reshape(n, -1)


def build_de_distance(dataset, session, subjects, n_per_class, sample_seed,
                      preproc_cfg=PREPROC_DEFAULT):
    """独立于谱估计流程，直接从 build_segments 重新取同一批段算 DE。"""
    segs_all, y_all, subj_all = [], [], []
    for subj in subjects:
        r = build_segments(dataset, subj, session, cfg=preproc_cfg)
        idx = stratified_sample(r["y"], r["trial"], n_per_class, sample_seed)
        segs_all.append(r["segs"][idx])
        y_all += r["y"][idx].tolist()
        subj_all += [subj] * len(idx)
    segs = np.concatenate(segs_all)
    feats = de_features(segs, SEED_FS)
    D = squareform(pdist(feats, metric="euclidean"))
    return D, np.array(y_all), np.array(subj_all)


# ==========================================================================
# 5. 出图
# ==========================================================================
def plot_heatmaps(D_lam, D_phi, y, subj, out_dir, tag=""):
    order_emo = np.argsort(y, kind="stable")
    order_subj = np.argsort(subj, kind="stable")
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    specs = [(D_lam, order_emo, "特征值距离 · 按情绪排序"),
             (D_lam, order_subj, "特征值距离 · 按被试排序"),
             (D_phi, order_emo, "特征向量距离 · 按情绪排序"),
             (D_phi, order_subj, "特征向量距离 · 按被试排序")]
    for ax, (D, order, title) in zip(axes.ravel(), specs):
        im = ax.imshow(D[np.ix_(order, order)], cmap="viridis")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    path = os.path.join(out_dir, f"距离矩阵热图{tag}.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  -> {path}")


def plot_spectral_points(sample, out_dir, tag=""):
    f = np.concatenate(sample["f"]); sigma = np.concatenate(sample["sigma"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(f, sigma, s=3, alpha=0.15, color="#2E75B6")
    ax.set_xlabel("f (Hz)"); ax.set_ylabel("sigma (1/s)")
    ax.set_title(f"保留模态的 (f,sigma) 分布{tag}", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, f"谱点分布{tag}.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  -> {path}")


# ==========================================================================
# 6. 判据（PREREGISTER §7/§8/§9）
# ==========================================================================
def judge(rows_main, rows_1_30, de_rows):
    lam = rows_main["lam"]
    c1 = lam["sep_emo"] >= 1.5 and lam["p_emo"] < 0.01
    c1_130 = rows_1_30["lam"]["sep_emo"] >= 1.5 if rows_1_30 else None
    c2 = lam["sep_emo"] > lam["sep_subj"]
    phi = rows_main["phi"]
    c3 = phi["sep_subj"] > phi["sep_emo"] and phi["sep_subj"] >= 1.5
    c4 = lam["sil_emo"] > lam["sil_subj"]

    print("\n[判据 §7]")
    print(f"  1. Sep_emo(lam)>=1.5 且 p<0.01: {c1}"
          f"{'（1-30Hz 也需成立: ' + str(c1_130) + '）' if c1_130 is not None else ''}")
    print(f"  2. Sep_emo(lam) > Sep_subj(lam): {c2}")
    print(f"  3. Sep_subj(phi) > Sep_emo(phi) 且 >=1.5: {c3}")
    print(f"  4. sil(情绪|lam) > sil(被试|lam): {c4}")

    all4 = c1 and (c1_130 is None or c1_130) and c2 and c3 and c4
    weak = 1.1 < lam["sep_emo"] < 1.5
    fail = lam["sep_emo"] <= 1.1
    d_case = c1 and c2 and c4 and not c3

    if all4:
        outcome = "A 成立"
    elif d_case:
        outcome = "D 半成立（3 不过，需删去“差异吸收进模态”的理论主张）"
    elif weak:
        outcome = "B 弱成立"
    elif fail:
        outcome = "C 不成立"
    else:
        outcome = "介于 B/C 之间，需人工核查（未落入任何预定义区间）"
    print(f"\n[结局 §8] {outcome}")

    if de_rows is not None:
        incr = lam["knn_emo"] >= de_rows["knn_emo"]
        print(f"\n[DE 增量判据 §9] kNN_emo(lambda)={lam['knn_emo']:.3f} "
              f">= kNN_emo(DE)={de_rows['knn_emo']:.3f}: {incr}")
    return outcome


# ==========================================================================
# main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="每被试每情绪只抽 3 段，验证管线")
    ap.add_argument("--control", action="store_true", help="加做 1-30Hz 对照（PREREGISTER §11）")
    ap.add_argument("--de", action="store_true", help="加做 DE 特征基线（PREREGISTER §9）")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--session", type=int, default=MAIN_SESSION)
    ap.add_argument("--win-s", type=float, default=4.0,
                    help="预处理段长(s)。4.0=修订1原参数(session3用过)，8.0=修订2新参数")
    args = ap.parse_args()

    preproc_cfg = PREPROC_T8 if args.win_s == 8.0 else PreprocConfig(win_s=args.win_s)
    tag_suffix = f"_win{args.win_s:g}s" if args.win_s != 4.0 else ""

    out_dir = os.path.join(RESULTS_DIR, "separability")
    os.makedirs(out_dir, exist_ok=True)

    subjects = None
    n_per_class = N_PER_CLASS
    if args.smoke:
        subjects = list_subjects("seed")[:3]
        n_per_class = 3
        print(f"[SMOKE] 被试={subjects}  每类={n_per_class} 段\n")

    print("=" * 70)
    print(f"可分性检验  session={args.session}  win_s={args.win_s}s  1-45Hz 主分析")
    print("=" * 70)

    sample = collect_samples(session=args.session, cfg=km.DEFAULT,
                             n_per_class=n_per_class, subjects=subjects,
                             n_jobs=args.n_jobs, preproc_cfg=preproc_cfg)
    n = len(sample["y"])
    print(f"共 {n} 段，谱估计耗时 {sample['meta']['elapsed_sec']:.1f}s")
    print("逐被试剔除率:",
          {k: f"{v*100:.1f}%" for k, v in sample["meta"]["reject_rate_by_subject"].items()})

    D_lam, D_phi, s_f, s_sigma, dist_elapsed = build_distance_matrices(sample, n_jobs=args.n_jobs)
    print(f"s_f={s_f:.3f}  s_sigma={s_sigma:.3f}  距离矩阵耗时 {dist_elapsed:.1f}s")

    y, subj = sample["y"], sample["subj"]
    rows_main = full_report(D_lam, D_phi, y, subj, tag="1-45Hz")

    if not args.smoke:
        plot_heatmaps(D_lam, D_phi, y, subj, out_dir, tag=tag_suffix)
        plot_spectral_points(sample, out_dir, tag=tag_suffix)

    rows_1_30 = None
    if args.control:
        print("\n" + "=" * 70)
        print("1-30Hz 对照（PREREGISTER §11）")
        print("=" * 70)
        sample_c = collect_samples(session=args.session, cfg=km.CONTROL_1_30HZ,
                                   n_per_class=n_per_class, subjects=subjects,
                                   n_jobs=args.n_jobs, preproc_cfg=preproc_cfg)
        D_lam_c, D_phi_c, *_ = build_distance_matrices(sample_c, n_jobs=args.n_jobs)
        rows_1_30 = full_report(D_lam_c, D_phi_c, sample_c["y"], sample_c["subj"], tag="1-30Hz")

    de_rows = None
    if args.de:
        print("\n" + "=" * 70)
        print("DE 特征基线（PREREGISTER §9）")
        print("=" * 70)
        subs = subjects or list_subjects("seed")
        D_de, y_de, subj_de = build_de_distance("seed", args.session, subs,
                                                 n_per_class, SAMPLE_SEED,
                                                 preproc_cfg=preproc_cfg)
        de_rows = full_report(D_de, D_de, y_de, subj_de, tag="DE")["lam"]

    outcome = judge(rows_main, rows_1_30, de_rows)

    summary = dict(session=args.session, win_s=args.win_s, preproc_cfg=asdict(preproc_cfg),
                   n_segments=n, s_f=s_f, s_sigma=s_sigma,
                   reject_rate_by_subject=sample["meta"]["reject_rate_by_subject"],
                   rows_main=rows_main, rows_1_30=rows_1_30, de_rows=de_rows,
                   outcome=outcome, smoke=args.smoke)
    summary_path = os.path.join(out_dir, f"summary_ses{args.session}{tag_suffix}"
                                f"{'_smoke' if args.smoke else ''}.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"\n结果已存 {summary_path}")


if __name__ == "__main__":
    main()
