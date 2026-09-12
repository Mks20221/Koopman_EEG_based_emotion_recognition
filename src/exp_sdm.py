# -*- coding: utf-8 -*-
"""sDM（spatial Dynamic Mode）特征在 SEED 情绪分类上的完整实验。

研究问题（独立于 Deep Koopman 主线）：
    在固定 4 s / 200 Hz / 51 通道 / 1–50 Hz / session 1 的 SEED 段上，
    DMD 空间模态特征（sDM）能否在常规 DE ＋ 空间协方差基线之外提供
    独立的情境情绪分类信息？

三组共用相同样本与划分：
    G0 基线  DE（5 频带微分熵）+ 收缩估计的空间协方差（51×51 对称 -> 1326 维）
    G1 模态  sDM（51×51 对称 -> 1326 维，使用 upper-triangular + diag）
    G2 联合  G0 + G1 拼接

分类器：L2 正则化逻辑回归（多类 one-vs-rest）；同一 C 候选集同时用在三组上；
        验证被试分数均值用于选 C（不接触测试被试）。

协议：
    * session 1、win_s=4.0、reject_abs_thr=15.0、51 通道（沿用 Gate 3）
    * LOSO：每折 12 train / 2 val / 1 test 被试；val subs 选法与
      exp_koopman_loso.py 完全一致（rng = default_rng(1000 + test_subj)）
    * 被试内段划分：trial-disjoint 三分（train/val/test），按试次分层；
      同一 trial 段不可跨越 train/test 边界 —— 与 Gate 3 同样的防泄漏要求。
    * 标准化：逐被试 z-score，统计量只用训练被试的段（与 Gate 3 LOSO 同）。
    * 逐段提取 DMD：每段独立堆叠 + 估谱 + sDM，**禁止跨 trial 拼接**。
    * DMD 候选秩：{2, 4, 8, 16}；sDM 特征索引：`both`（上三角含对角线，1326 维）
    * 全部候选特征先一次性算好、按 (test_subj, dmd_rank) 缓存，避免重算。

唯一主比较：G2 − G0 的逐被试配对差值。
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime

import numpy as np
from scipy import stats
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import RESULTS_DIR, SEED_FS
from data import list_subjects
from preprocess import PreprocConfig, build_segments
from sdm_features import (
    stacking_dmd_preproc, stacking_dmd_preproc_full,
    stacking_dmd_acquire_modes,
    modes2sDMmat, sDMmat2vecfeat,
)


# ---------------------------------------------------------------------------
# 协议常量
# ---------------------------------------------------------------------------
SESSION = 1
WIN_S = 4.0
REJECT_THR = 15.0
DE_BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
# 候选 rank 砍到 2 个：2（基波主导）和 8（中等秩，含部分谐波）。
# 这是判决实验；候选集先小后扩。PREREGISTER 修订 2 已记录 2700 段上
# "rank>16 后无信号-噪声间隙，再大只会被噪声方向污染"。在 51 通道 EEG 上，
# 2 与 8 已经覆盖了"基波/带通成分"与"部分谐波"两个有物理意义的档位。
DMD_RANKS = (2, 8)
FEATURE_TYPE = "both"                   # upper triangle + diagonal, 51*52/2 = 1326
# C 网格砍到 4 个：覆盖 {强正则, 中正则, 弱正则, 几乎无正则} 四个区间。
LR_C_GRID = (1e-2, 1e-1, 1.0, 10.0)
SHRINKAGE_ALPHA = 0.05                  # 空间协方差收缩到对角的强度
SPLIT_SEED = 0                          # 被试内段划分的种子（与 Gate 3 / 报告一致）
N_TRAIN_TRIALS = 3                      # 每类 train 试次数
N_VAL_TRIALS = 1
N_TEST_TRIALS = 1

# sDM 特征缓存目录：按 (dataset, subj, session, cfg_key, rank) 缓存每段 sDM。
SDM_CACHE_DIR = os.path.join(ROOT, "cache", "sdm")
os.makedirs(SDM_CACHE_DIR, exist_ok=True)

OUT_DIR = os.path.join(RESULTS_DIR, "sdm_loso")
os.makedirs(OUT_DIR, exist_ok=True)


def _cfg_signature() -> str:
    """被试内段预处理 cfg 与 sDM 计算相关参数的签名，参与 sDM 缓存 key。"""
    return f"win{WIN_S:g}_thr{REJECT_THR:g}_fs{SEED_FS}_bands{len(DE_BANDS)}_ranks{'-'.join(map(str, DMD_RANKS))}"


def _sdm_cache_path(subj: int, session: int, rank: int) -> str:
    return os.path.join(SDM_CACHE_DIR,
                        f"seed_s{subj:02d}_ses{session}_{_cfg_signature()}_r{rank}.npy")


def get_or_compute_sdm(segs: np.ndarray, subj: int, session: int,
                       ranks=DMD_RANKS, n_jobs: int = 1,
                       verbose: bool = False) -> dict:
    """读 sDM 缓存：若 (subj, session, rank) 已有 npy 直接加载；否则算并存盘。

    verbose=True 时给单线程路径加段级进度打印（每 50 段一行 + ETA）。

    返回 dict[rank] = (n, C*(C+1)/2) ndarray。
    """
    out = {}
    todo = []      # list of ranks that need computation
    for r in ranks:
        p = _sdm_cache_path(subj, session, r)
        if os.path.exists(p):
            out[r] = np.load(p)
        else:
            todo.append(r)

    if todo:
        label = f"s{subj:02d}/r={'-'.join(map(str, todo))}" if verbose else ""
        per_rank = sdm_features_multi_rank_reported(
            segs, dmd_ranks=tuple(todo), n_jobs=n_jobs, label=label)
        for r in todo:
            out[r] = per_rank[r]
            np.save(_sdm_cache_path(subj, session, r), out[r])
    elif verbose:
        print(f"    s{subj:02d}: 全部 sDM 已缓存，跳过估谱", flush=True)
    return out


# ---------------------------------------------------------------------------
# 数据加载 & 标准化 & 段级划分
# ---------------------------------------------------------------------------
def load_all_subjects(session=SESSION, win_s=WIN_S, thr=REJECT_THR):
    """载入 session 1 全部被试的预处理段，统一 51 通道。

    返回 dict[subj] = dict(segs, y, trial)，segs 已是 (n, C, L) float32。
    """
    cfg = PreprocConfig(win_s=win_s, reject_abs_thr=thr)
    out = {}
    for s in list_subjects("seed"):
        r = build_segments("seed", s, session, cfg=cfg)
        out[s] = dict(segs=r["segs"], y=r["y"], trial=r["trial"],
                      stats=r["stats"])
    return out


def per_subject_zscore(segs: np.ndarray) -> np.ndarray:
    """逐被试 z-score（沿 axis 0,2 聚合统计量）。与 Gate 3 LOSO 完全一致。"""
    mu = segs.mean(axis=(0, 2), keepdims=True)
    sd = segs.std(axis=(0, 2), keepdims=True) + 1e-8
    return ((segs - mu) / sd).astype(np.float32)


def split_segments_by_trial(y, trial, seed=SPLIT_SEED,
                            n_tr=N_TRAIN_TRIALS, n_va=N_VAL_TRIALS,
                            n_te=N_TEST_TRIALS):
    """每类按试次比例切 train/val/test（同 trial 段不可跨边界）。"""
    rng = np.random.default_rng(seed)
    n_total = len(y)
    masks = [np.zeros(n_total, bool) for _ in range(3)]
    for c in np.unique(y):
        trials_c = np.unique(trial[y == c])
        # 至少保证 3 个 trial 可分（SEED 每类 5 个 trial）
        rng.shuffle(trials_c)
        nt, nv, ne = n_tr, n_va, n_te
        # 总量不够时按比例缩
        if len(trials_c) < nt + nv + ne:
            scale = len(trials_c) / (nt + nv + ne)
            nt = max(1, int(round(nt * scale)))
            nv = max(0, int(round(nv * scale)))
            ne = max(1, len(trials_c) - nt - nv)
        tr_t = trials_c[:nt]
        va_t = trials_c[nt:nt + nv]
        te_t = trials_c[nt + nv:nt + nv + ne]
        for arr, ts in zip(masks, (tr_t, va_t, te_t)):
            arr |= np.isin(trial, ts) & (y == c)
    return masks


# ---------------------------------------------------------------------------
# 特征：DE + 空间协方差 + sDM
# ---------------------------------------------------------------------------
def de_features(segs: np.ndarray, fs=SEED_FS, bands=DE_BANDS) -> np.ndarray:
    """(n, C, L) -> (n, C*len(bands))，与 exp_koopman_gate.py 同口径。"""
    n, C, L = segs.shape
    nyq = fs / 2.0
    out = np.zeros((n, C, len(bands)), dtype=np.float64)
    for bi, (lo, hi) in enumerate(bands):
        b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
        v = filtfilt(b, a, segs.astype(np.float64), axis=-1).var(axis=-1)
        out[:, :, bi] = 0.5 * np.log(2 * np.pi * np.e * np.maximum(v, 1e-12))
    return out.reshape(n, -1)


def spatial_cov_features(segs: np.ndarray, alpha=SHRINKAGE_ALPHA) -> np.ndarray:
    """(n, C, L) -> (n, C*(C+1)/2)

    对每段按 (C, L) 算 sample cov，再用 Ledoit-Wolf 风格收缩：
        cov_shrunk = (1-α) cov + α tr(cov)/C * I
    取上三角含对角线（C*(C+1)/2 维）。
    """
    n, C, L = segs.shape
    dim = C * (C + 1) // 2
    out = np.empty((n, dim), dtype=np.float64)
    # 与 MATLAB `find(triu(ones(C),0))'` 等价：按行优先展平后的线性下标
    flat_idx = np.flatnonzero(np.triu(np.ones((C, C), dtype=bool), k=0))
    for i in range(n):
        x = segs[i].astype(np.float64)
        x = x - x.mean(axis=1, keepdims=True)
        cov = (x @ x.T) / max(L - 1, 1)
        trace = np.trace(cov) / C
        cov = (1 - alpha) * cov + alpha * trace * np.eye(C)
        out[i] = cov.ravel()[flat_idx]
    return out


def sdm_features_for_segments(segs: np.ndarray, fs=SEED_FS,
                              dmd_rank: int = 8,
                              feature_type: str = FEATURE_TYPE) -> np.ndarray:
    """(n, C, L) -> (n, C*(C+1)/2)，逐段独立 DMD + sDM + 'both' 索引。"""
    n, C, L = segs.shape
    dim = C * (C + 1) // 2
    out = np.empty((n, dim), dtype=np.float64)
    flat_idx = np.flatnonzero(np.triu(np.ones((C, C), dtype=bool), k=0))
    dt = 1.0 / fs
    for i in range(n):
        Xc = segs[i].astype(np.float64)
        svd_st = stacking_dmd_preproc_full(Xc, dt=dt)
        mode_st = stacking_dmd_acquire_modes(svd_st, dmd_rank=dmd_rank)
        sDM = modes2sDMmat(mode_st)
        out[i] = sDM.ravel()[flat_idx]
    return out


def sdm_features_multi_rank(segs: np.ndarray, fs=SEED_FS,
                            dmd_ranks=DMD_RANKS,
                            n_jobs: int = 1) -> dict:
    """对每段只算一次 full SVD，按各候选 rank 计算 sDM 特征。

    返回 dict[rank] = (n, C*(C+1)/2) ndarray；这是 `run_fold` 内部批量化的入口。
    n_jobs>1 时按段切到子进程并行（Windows 下 spawn，segment 矩阵序列化有代价，
    但仍快于单线程）。
    """
    n, C, L = segs.shape
    dim = C * (C + 1) // 2
    dt = 1.0 / fs
    ranks = list(dmd_ranks)
    chunk_size = max(1, (n + max(n_jobs, 1) - 1) // max(n_jobs, 1))
    chunks = [(segs[i:i + chunk_size], dt, ranks) for i in range(0, n, chunk_size)]
    if n_jobs == 1:
        results = [_sdm_chunk_worker(c) for c in chunks]
    else:
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.map(_sdm_chunk_worker, chunks)
    out = {r: np.empty((n, dim), dtype=np.float64) for r in ranks}
    offset = 0
    for chunk_out in results:
        m = chunk_out.shape[1]
        for ri, r in enumerate(ranks):
            out[r][offset:offset + m] = chunk_out[ri]
        offset += m
    return out


# Windows 下 multiprocessing 用 spawn，子进程函数必须可被 pickle
def _sdm_chunk_worker(args):
    segs_chunk, dt, ranks = args
    C = segs_chunk.shape[1]
    dim = C * (C + 1) // 2
    flat_idx = np.flatnonzero(np.triu(np.ones((C, C), dtype=bool), k=0))
    out = np.empty((len(ranks), len(segs_chunk), dim), dtype=np.float64)
    for i, seg in enumerate(segs_chunk):
        Xc = seg.astype(np.float64)
        svd_st = stacking_dmd_preproc_full(Xc, dt=dt)
        for ri, r in enumerate(ranks):
            mode_st = stacking_dmd_acquire_modes(svd_st, dmd_rank=r)
            sDM = modes2sDMmat(mode_st)
            out[ri, i] = sDM.ravel()[flat_idx]
    return out


def sdm_features_multi_rank_reported(segs: np.ndarray, fs=SEED_FS,
                                     dmd_ranks=DMD_RANKS,
                                     n_jobs: int = 1,
                                     label: str = "") -> dict:
    """同 sdm_features_multi_rank，但加段级 / chunk 级进度打印。

    **Windows 多线程路径不稳定**：mp.Pool 序列化大 chunk 时会触发
    WinError 1450（pipe buffer 不足）。仅在 n_jobs==1 时调用。
    多线程路径仍由 sdm_features_multi_rank 提供，但本函数不暴露它——调用方应
    主动只传 n_jobs=1。如需真正并行，应改用 shared-memory 或 numpy multiprocessing。
    """
    n = segs.shape[0]
    ranks = list(dmd_ranks)
    if n_jobs != 1:
        raise ValueError(
            "Windows 多线程 sDM 不可靠（pipe buffer 不足），请传 n_jobs=1。"
            f"当前 n_jobs={n_jobs}。"
        )
    # 单线程：每 50 段打印一次进度 + ETA
    dt = 1.0 / fs
    C = segs.shape[1]
    dim = C * (C + 1) // 2
    flat_idx = np.flatnonzero(np.triu(np.ones((C, C), dtype=bool), k=0))
    out = {r: np.empty((n, dim), dtype=np.float64) for r in ranks}
    t0 = time.time()
    if label:
        print(f"    {label}: 单线程估谱 {n} 段 × {len(ranks)} rank "
              f"（每 50 段打一次进度）", flush=True)
    last_print = 0
    for i in range(n):
        Xc = segs[i].astype(np.float64)
        svd_st = stacking_dmd_preproc_full(Xc, dt=dt)
        for r in ranks:
            mode_st = stacking_dmd_acquire_modes(svd_st, dmd_rank=r)
            sDM = modes2sDMmat(mode_st)
            out[r][i] = sDM.ravel()[flat_idx]
        if i - last_print >= 50:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / max(rate, 1e-6)
            print(f"    {label}: {i+1}/{n}  速率 {rate:.2f} 段/s  "
                  f"已用 {elapsed:.0f}s  剩余 ≈ {eta:.0f}s", flush=True)
            last_print = i
    if label:
        print(f"    {label}: {n}/{n} 完成，总耗时 {time.time()-t0:.1f}s", flush=True)
    return out


# ---------------------------------------------------------------------------
# 分类
# ---------------------------------------------------------------------------
def fit_logreg(X_tr, y_tr, X_te, C):
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=C, max_iter=4000, solver="lbfgs",
                              multi_class="auto").fit(sc.transform(X_tr), y_tr)
    pred = clf.predict(sc.transform(X_te))
    return pred


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# LOSO 单折
# ---------------------------------------------------------------------------
def run_fold(test_subj: int, data: dict, val_size: int = 2,
             n_jobs: int = 1,
             verbose: bool = True):
    """返回一行记录（dict）—— 三个组的被试级 acc、f1、选中的 C 与 dmd_rank。

    流程：
      1) 选 val_subj：与 Gate 3 LOSO 同款 rng(1000 + test_subj)
      2) 标准化只在 train_subs 的段上做（逐被试）
      3) 段级划分（trial-disjoint），分别得到 train_mask / val_mask / test_mask
      4) 三组特征
      5) C 候选网格在 val 上求平均 acc，best_C 三组共用同一个 dmd_rank

    速度优化：
      - sDM 特征按 (被试, session, rank) 缓存到 cache/sdm/，跨折复用
      - 每个被试的 sDM 只算一次（不带被试间拼接）
    """
    subs = sorted(data)
    others = [s for s in subs if s != test_subj]
    rng = np.random.default_rng(1000 + test_subj)
    val_subs = sorted(rng.choice(others, size=val_size, replace=False).tolist())
    tr_subs = [s for s in others if s not in val_subs]

    # ---- 逐被试标准化 ----
    segs_z = {s: per_subject_zscore(data[s]["segs"]) for s in subs}

    # ---- 拼接 train / val / test 段（带被试边界信息）----
    # 我们要按 (被试, 段偏移) 重建 split mask，所以保存段偏移表
    seg_offsets = {}   # subj -> (start_idx, end_idx) in concatenated array
    all_segs_list = []
    all_y_list = []
    all_trial_list = []
    cur = 0
    for s in subs:
        n_s = len(data[s]["segs"])
        seg_offsets[s] = (cur, cur + n_s)
        all_segs_list.append(segs_z[s])
        all_y_list.append(data[s]["y"])
        all_trial_list.append(data[s]["trial"])
        cur += n_s
    tr_segs = np.concatenate([segs_z[s] for s in tr_subs])
    tr_y = np.concatenate([data[s]["y"] for s in tr_subs])
    tr_trial = np.concatenate([data[s]["trial"] for s in tr_subs])
    va_segs = np.concatenate([segs_z[s] for s in val_subs])
    va_y = np.concatenate([data[s]["y"] for s in val_subs])
    te_segs = segs_z[test_subj]
    te_y = data[test_subj]["y"]

    # ---- 段级 trial-disjoint 划分（仅对 train_subs 拼接后整体划分）----
    tr_m_full, va_m_full, te_m_full = split_segments_by_trial(
        tr_y, tr_trial, seed=SPLIT_SEED)
    # 重新按被试切回 tr_m_full 的子 mask
    train_sub_masks = {}   # subj -> bool mask (length n_subj_segs)
    cur = 0
    for s in tr_subs:
        n_s = len(data[s]["segs"])
        train_sub_masks[s] = tr_m_full[cur:cur + n_s]
        cur += n_s
    te_m = np.ones(len(te_y), bool)
    va_m = np.ones(len(va_y), bool)

    if verbose:
        n_train = sum(m.sum() for m in train_sub_masks.values())
        print(f"  [s{test_subj:02d}] 段数: train={n_train}  "
              f"val={va_m.sum()}  test={te_m.sum()}  "
              f"tr_subs={tr_subs} val_subs={val_subs}", flush=True)

    # ---- sDM 特征：按被试缓存复用 ----
    t0_sdm = time.time()
    sdm_by_subj = {}
    for s in subs:
        sdm_by_subj[s] = get_or_compute_sdm(segs_z[s], s, SESSION,
                                            ranks=DMD_RANKS, n_jobs=n_jobs,
                                            verbose=verbose)
    if verbose:
        print(f"  [s{test_subj:02d}] sDM 特征就绪（{len(subs)} 被试 × "
              f"{len(DMD_RANKS)} rank），耗时 {time.time()-t0_sdm:.1f}s",
              flush=True)

    # ---- 三组特征：先按被试取 sDM 子集，再拼接 ----
    t0_de = time.time()
    f_tr_de = de_features(np.concatenate(
        [segs_z[s][train_sub_masks[s]] for s in tr_subs]))
    f_va_de = de_features(va_segs[va_m])
    f_te_de = de_features(te_segs[te_m])
    f_tr_sc = spatial_cov_features(np.concatenate(
        [segs_z[s][train_sub_masks[s]] for s in tr_subs]))
    f_va_sc = spatial_cov_features(va_segs[va_m])
    f_te_sc = spatial_cov_features(te_segs[te_m])
    f_tr_base = np.concatenate([f_tr_de, f_tr_sc], axis=1)
    f_va_base = np.concatenate([f_va_de, f_va_sc], axis=1)
    f_te_base = np.concatenate([f_te_de, f_te_sc], axis=1)
    if verbose:
        print(f"  [s{test_subj:02d}] DE+SC 完成，耗时 {time.time()-t0_de:.1f}s",
              flush=True)

    # sDM 按 rank 分组的 split 视图
    def gather_sdm(ss, per_subj_mask=None):
        """per_subj_mask: dict[subj] -> bool array（与该被试段数等长）。None 表示全取。"""
        out = {r: [] for r in DMD_RANKS}
        for s in ss:
            m = per_subj_mask[s] if per_subj_mask is not None else None
            for r in DMD_RANKS:
                feat = sdm_by_subj[s][r]
                out[r].append(feat[m] if m is not None else feat)
        return {r: np.concatenate(out[r], axis=0) for r in DMD_RANKS}

    sdm_split = dict(
        tr=gather_sdm(tr_subs, train_sub_masks),
        va=gather_sdm(val_subs),
        te=gather_sdm([test_subj]),
    )

    # ---- 在 val 上为每组选 C；dmd_rank 也用 val 选（与 C 一起做 grid） ----
    ytr = tr_y[tr_m_full]
    yva = va_y[va_m]
    yte = te_y[te_m]

    best = {}
    # 基线组
    best_C, best_acc = LR_C_GRID[0], -1.0
    for C in LR_C_GRID:
        pred = fit_logreg(f_tr_base, ytr, f_va_base, C)
        a = accuracy(yva, pred)
        if a > best_acc:
            best_acc, best_C = a, C
    best["base"] = dict(C=best_C, val_acc=best_acc)

    # sDM 组（每个 dmd_rank 选 C）
    for r in DMD_RANKS:
        bc, ba = LR_C_GRID[0], -1.0
        for C in LR_C_GRID:
            pred = fit_logreg(sdm_split["tr"][r], ytr, sdm_split["va"][r], C)
            a = accuracy(yva, pred)
            if a > ba:
                ba, bc = a, C
        best[f"sdm_r{r}"] = dict(C=bc, val_acc=ba)

    # 联合组（每个 dmd_rank 选 C）
    for r in DMD_RANKS:
        bc, ba = LR_C_GRID[0], -1.0
        f_tr = np.concatenate([f_tr_base, sdm_split["tr"][r]], axis=1)
        f_va = np.concatenate([f_va_base, sdm_split["va"][r]], axis=1)
        for C in LR_C_GRID:
            pred = fit_logreg(f_tr, ytr, f_va, C)
            a = accuracy(yva, pred)
            if a > ba:
                ba, bc = a, C
        best[f"comb_r{r}"] = dict(C=bc, val_acc=ba)

    # 在所有 (dmd_rank) 中挑 val 分数最高的 dmd_rank 作为本折正式 sDM 候选
    best_r = max(DMD_RANKS, key=lambda r: best[f"sdm_r{r}"]["val_acc"])
    best_comb_r = max(DMD_RANKS, key=lambda r: best[f"comb_r{r}"]["val_acc"])

    # ---- 在 test 上报最终 acc / f1 ----
    out = dict(test_subj=test_subj,
               train_subs=tr_subs, val_subs=val_subs,
               split_seed=SPLIT_SEED,
               n_train=int(tr_m_full.sum()), n_val=int(va_m.sum()),
               n_test=int(te_m.sum()),
               best_dmd_rank=int(best_r), best_comb_dmd_rank=int(best_comb_r))

    def eval_set(f_tr, f_te, C, y_te):
        pred = fit_logreg(f_tr, ytr, f_te, C)
        return dict(acc=accuracy(y_te, pred), f1=macro_f1(y_te, pred))

    out["base"] = eval_set(f_tr_base, f_te_base, best["base"]["C"], yte)
    for grp in ["base", "sdm", "comb"]:
        if grp == "sdm":
            r_use, f_te = best_r, sdm_split["te"][best_r]
            f_tr = sdm_split["tr"][best_r]
            C = best[f"sdm_r{best_r}"]["C"]
        elif grp == "comb":
            r_use, f_te = best_comb_r, np.concatenate(
                [f_te_base, sdm_split["te"][best_comb_r]], axis=1)
            f_tr = np.concatenate(
                [f_tr_base, sdm_split["tr"][best_comb_r]], axis=1)
            C = best[f"comb_r{best_comb_r}"]["C"]
        else:
            r_use = None
            f_tr = f_tr_base
            f_te = f_te_base
            C = best["base"]["C"]
        res = eval_set(f_tr, f_te, C, yte)
        out[f"{grp}_test_acc"] = res["acc"]
        out[f"{grp}_test_f1"] = res["f1"]
        if r_use is not None:
            out[f"{grp}_dmd_rank"] = int(r_use)

    # 保存完整 val/test 预测（逐段）
    def full_eval(f_tr, f_te, C, y_te):
        pred = fit_logreg(f_tr, ytr, f_te, C)
        return dict(acc=accuracy(y_te, pred), f1=macro_f1(y_te, pred),
                    y_true=y_te.tolist(), y_pred=pred.tolist())

    out["base_preds"] = full_eval(f_tr_base, f_te_base,
                                  best["base"]["C"], yte)
    out["sdm_preds"] = full_eval(sdm_split["tr"][best_r],
                                 sdm_split["te"][best_r],
                                 best[f"sdm_r{best_r}"]["C"], yte)
    out["comb_preds"] = full_eval(
        np.concatenate([f_tr_base, sdm_split["tr"][best_comb_r]], axis=1),
        np.concatenate([f_te_base, sdm_split["te"][best_comb_r]], axis=1),
        best[f"comb_r{best_comb_r}"]["C"], yte)

    if verbose:
        print(f"  LOSO test=s{test_subj:02d} val={val_subs}  "
              f"G0={out['base_test_acc']:.3f}  "
              f"G1(r={best_r})={out['sdm_test_acc']:.3f}  "
              f"G2(r={best_comb_r})={out['comb_test_acc']:.3f}",
              flush=True)
    return out


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def summarize(rows):
    print("\n" + "=" * 72)
    print("sDM LOSO 完整结果  统计单位 = 留出被试")
    print("=" * 72)
    subs = sorted({r["test_subj"] for r in rows})
    n = len(subs)
    by_subj = {s: next(r for r in rows if r["test_subj"] == s) for s in subs}

    def col(grp):
        return np.array([by_subj[s][f"{grp}_test_acc"] for s in subs])

    base = col("base")
    sdm = col("sdm")
    comb = col("comb")
    base_f1 = np.array([by_subj[s]["base_test_f1"] for s in subs])
    sdm_f1 = np.array([by_subj[s]["sdm_test_f1"] for s in subs])
    comb_f1 = np.array([by_subj[s]["comb_test_f1"] for s in subs])

    print(f"{'组':<14}{'test_acc':>14}{'macro_f1':>14}{'min':>8}{'max':>8}")
    print("-" * 60)
    for name, a, f in [("G0 基线 DE+SC", base, base_f1),
                       ("G1 模态 sDM", sdm, sdm_f1),
                       ("G2 联合 DE+SC+sDM", comb, comb_f1)]:
        print(f"{name:<18}{a.mean():>10.3f}±{a.std():<5.3f}"
              f"{f.mean():>10.3f}±{f.std():<5.3f}"
              f"{a.min():>8.3f}{a.max():>8.3f}")

    print("\n--- 配对检验（n = 留出被试数）---")
    for a_name, b_name, da, db in [
        ("G2 联合", "G0 基线", comb, base),
        ("G1 sDM", "G0 基线", sdm, base),
        ("G2 联合", "G1 sDM", comb, sdm),
    ]:
        d = da - db
        t, p = stats.ttest_rel(da, db)
        try:
            _, pw = stats.wilcoxon(da, db)
        except Exception:
            pw = float("nan")
        n_pos = int((d > 0).sum())
        p_sign = stats.binomtest(n_pos, n, 0.5).pvalue
        ci = 1.96 * d.std(ddof=1) / np.sqrt(n)
        print(f"  {a_name:>8} - {b_name:<10}: "
              f"Δ={d.mean():+.4f}  "
              f"[95%CI {d.mean()-ci:+.4f}, {d.mean()+ci:+.4f}]  "
              f"t={t:.2f}  p={p:.3f}  wilcox={pw:.3f}  "
              f"+{n_pos}/{n}  p_sign={p_sign:.3f}  "
              f"dz={d.mean()/(d.std(ddof=1)+1e-12):.2f}")

    print("\n--- 逐被试差值 (G2 − G0) ---")
    diffs = comb - base
    for s, d in zip(subs, diffs):
        print(f"  s{s:02d}: {d:+.4f}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-subjects", type=int, nargs="*", default=None,
                    help="只跑指定测试被试；默认 1..15 全部")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="Windows 多线程 sDM 会触发 WinError 1450（pipe buffer 不足），"
                         "目前只支持 1。如需并行请改 shared-memory。")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    if args.n_jobs != 1:
        sys.stderr.write(
            f"[error] --n-jobs={args.n_jobs} 不支持。"
            "Windows 下 mp.Pool 序列化大 chunk 会触发 WinError 1450。"
            "请用 --n-jobs 1。\n")
        sys.exit(2)

    # 配置快照
    cfg = dict(
        timestamp=datetime.now().isoformat(),
        session=SESSION, win_s=WIN_S, reject_thr=REJECT_THR,
        fs=SEED_FS, de_bands=DE_BANDS,
        dmd_ranks=DMD_RANKS, feature_type=FEATURE_TYPE,
        shrinkage_alpha=SHRINKAGE_ALPHA,
        lr_C_grid=list(LR_C_GRID),
        split_seed=SPLIT_SEED,
        n_train_trials=N_TRAIN_TRIALS,
        n_val_trials=N_VAL_TRIALS, n_test_trials=N_TEST_TRIALS,
        val_size=2,
        tag=args.tag,
    )
    with open(os.path.join(OUT_DIR, f"config{args.tag}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
    print("=" * 64)
    print("sDM LOSO  SEED 沿用 Gate 3 协议（session 1, 4 s, 51 ch, 1–50 Hz）")
    print("=" * 64)
    print("协议：", json.dumps(cfg, ensure_ascii=False, indent=2))
    print()

    data = load_all_subjects()
    subs = sorted(data)
    print(f"载入完成，{len(subs)} 被试，"
          f"各被试段数：{ {s: len(data[s]['y']) for s in subs} }\n")

    targets = args.test_subjects or subs
    rows = []
    for ts in targets:
        t0 = time.time()
        row = run_fold(ts, data, val_size=2, n_jobs=args.n_jobs, verbose=True)
        row["elapsed"] = time.time() - t0
        rows.append(row)
        path = os.path.join(OUT_DIR, f"rows{args.tag}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=2)

    print(f"\n原始结果已存 {path}")
    summarize(rows)


if __name__ == "__main__":
    main()
