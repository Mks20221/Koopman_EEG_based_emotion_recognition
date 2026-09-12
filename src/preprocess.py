# -*- coding: utf-8 -*-
"""预处理与分段。参数全部来自 docs/PREREGISTER.md 第 2、4 节，已冻结。

管线：载入 -> 带通 -> 切段 -> 段内去均值 -> 伪迹剔除 -> 分层抽样
每一步的产物都可独立缓存，缓存键含全部影响结果的参数 + CODE_VERSION。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import butter, filtfilt

from config import (CACHE_DIR, SEED_BAD_Z_HI, SEED_BAD_Z_LO,
                    SEED_CHANNEL_NAMES, SEED_CHANNEL_PROFILE,
                    SEED_GOOD_CHANNELS, SEED_N_TRIALS)
from data import load_trials

# 影响数值结果的代码版本。改动本文件中任何计算逻辑都必须 +1，否则缓存会串。
# v2: 加入坏导剔除，并把段级幅值统计改到剔除之后
# v3: 坏导判据改为跨录制自比；段级统计改为逐通道归一化（PREREGISTER 修订 1）
CODE_VERSION = 5


@dataclass(frozen=True)
class PreprocConfig:
    """冻结于 docs/PREREGISTER.md，改动需走修订记录。"""
    band_lo: float = 1.0            # Hz
    band_hi: float = 50.0           # Hz
    filt_order: int = 4             # Butterworth 阶数（filtfilt 实际为 2 倍）
    win_s: float = 4.0              # 段长，秒
    overlap: float = 0.0            # 无重叠
    drop_bad_channels: bool = True  # 剔除 config.SEED_BAD_CHANNELS（全体统一）
    reject_abs_thr: float = 15.0    # segment_amplitude > 此值 -> 剔段
    # 用绝对阈值而非"× 被试内中位数"：统计量已被通道期望幅度和本次录制整体水平
    # 双重归一化，量纲无关、跨录制可比，被试整体幅度差异（属 A_s）已被吸收。
    # 相对阈值在坏段超过一半时失效 —— 被试15-ses1 实测 53% 的段有通道故障，
    # 中位数本身就落在坏数据里，"5 倍中位"只剔掉了 0.6%。
    # 15 这个值不敏感：12–30 之间正常被试的剔除率都在 0–2%。

    def key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True) + f"|v{CODE_VERSION}"
        return hashlib.sha1(payload.encode()).hexdigest()[:12]


DEFAULT = PreprocConfig()


# --------------------------------------------------------------------------
# 基本算子
# --------------------------------------------------------------------------
def bandpass(X: np.ndarray, fs: int, cfg: PreprocConfig = DEFAULT) -> np.ndarray:
    """零相位带通。(C,T) -> (C,T)。

    用 filtfilt 而非 lfilter：相位失真会直接扭曲时延嵌入的时序结构，
    而 DMD 估的就是时序算子，非零相位等于给谱注入系统性偏置。
    """
    nyq = fs / 2.0
    b, a = butter(cfg.filt_order, [cfg.band_lo / nyq, cfg.band_hi / nyq], btype="band")
    return filtfilt(b, a, X, axis=-1)


def segment(X: np.ndarray, fs: int, cfg: PreprocConfig = DEFAULT) -> np.ndarray:
    """(C,T) -> (n_seg, C, L)，无重叠，末尾不足一段的部分丢弃。"""
    L = int(round(cfg.win_s * fs))
    step = int(round(L * (1.0 - cfg.overlap)))
    C, T = X.shape
    n = 1 + (T - L) // step if T >= L else 0
    if n <= 0:
        return np.empty((0, C, L))
    idx = np.arange(n) * step
    out = np.stack([X[:, i:i + L] for i in idx])
    return out - out.mean(axis=2, keepdims=True)      # 段内逐通道去均值


def segment_amplitude(segs: np.ndarray, channels) -> np.ndarray:
    """段级幅值统计量：按通道的**跨录制典型幅度**归一化后取通道最大。(n,C,L) -> (n,)

    三个设计点，每一个都是被实测逼出来的：

    1. 不归一化直接取通道最大 -> 被幅度最大的通道垄断。真实数据里 FP1/FPZ/FP2
       （额极，眨眼）常年最大，统计量的中位数被它们抬起来，阈值形同虚设，
       实测剔除率 0.0%。
    2. 按"该通道在本次录制内的中位值"归一化 -> 半场失效的通道基准被压低，
       它正常工作的那半场反被判成伪迹。被试15 的 C1/C2/CB2 实测把统计量中位
       抬到 384。
    3. 按 config.SEED_CHANNEL_PROFILE（跨 45 次录制的典型相对幅度）归一化 ->
       基准不受单次录制故障影响。本次录制的整体幅度水平仍逐录制估计，
       所以被试间的整体幅度差异（属于 A_s，不该被当成伪迹）不会被误判。
    """
    p2p = segs.max(axis=2) - segs.min(axis=2)              # (n,C)
    prof = np.asarray([SEED_CHANNEL_PROFILE[c] for c in channels])
    level = np.median(np.median(p2p, axis=0))              # 本次录制的整体幅度水平
    expected = np.maximum(prof * level, 1e-6)              # 每通道的期望峰峰值
    return (p2p / expected[None, :]).max(axis=1)


def channel_ratio(segs: np.ndarray) -> np.ndarray:
    """每通道的中位峰峰值 / 全通道中位。(n,C,L) -> (C,)  坏导检测的一级统计量。"""
    p2p = segs.max(axis=2) - segs.min(axis=2)
    cm = np.median(p2p, axis=0)
    return cm / np.median(cm)


def scan_bad_channels(dataset: str = "seed", subjects=None, sessions=(1, 2, 3),
                      z_hi: float = SEED_BAD_Z_HI, z_lo: float = SEED_BAD_Z_LO,
                      cfg=None):
    """普查坏导，返回 (并集列表, ratio 矩阵 dict, Z 矩阵 dict)。

    这是 config.SEED_BAD_CHANNELS 的来源，保留于此以便随时复算。
    只用幅值统计，**不接触任何标签**，故不污染预注册判据。

    判据是跨录制比同一通道自己（见 config.SEED_BAD_CHANNELS 的说明）：
    中线电极 CZ/CPZ/PZ 在所有人身上都安静，那是参考电极位置决定的montage 属性，
    不是故障，也不携带被试身份 —— 按录制内跨通道比会错杀它们。
    """
    cfg = cfg or PreprocConfig(drop_bad_channels=False)
    from data import list_subjects
    subjects = list(subjects or list_subjects(dataset))
    keys, mats = [], []
    for subj in subjects:
        for sess in sessions:
            X, _, fs = load_trials(dataset, subj, sess)
            segs = np.concatenate([segment(bandpass(x, fs, cfg), fs, cfg)
                                   for x in X])
            keys.append((subj, sess))
            mats.append(channel_ratio(segs))
    R = np.stack(mats)                                     # (n_rec, C)
    Z = R / np.maximum(np.median(R, axis=0), 1e-12)[None, :]
    bad = (Z > z_hi) | (Z < z_lo)
    union = sorted(np.where(bad.any(axis=0))[0].tolist())
    return union, dict(zip(keys, R)), dict(zip(keys, Z))


# --------------------------------------------------------------------------
# 单个 (被试, session) 的完整流程
# --------------------------------------------------------------------------
def _cache_path(dataset: str, subject: int, session: int, cfg: PreprocConfig) -> str:
    d = os.path.join(CACHE_DIR, "preproc")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{dataset}_s{subject:02d}_ses{session}_{cfg.key()}.npz")


def build_segments(dataset: str, subject: int, session: int,
                   cfg: PreprocConfig = DEFAULT, use_cache: bool = True):
    """返回 dict:
        segs  : (n, C, L) float32   已带通、去均值、剔伪迹
        y     : (n,) int64          情绪标签 0..2
        trial : (n,) int64          试次号 0..14，抽样时要按它分层
        stats : dict                剔除率、逐试次剔除率、被清空的试次，必须报告

    伪迹剔除见 PREREGISTER 修订 1(b)：坏导先按全局黑名单统一剔除，
    再按 profile 归一化的段级统计量做绝对阈值剔段。
    被试间的整体幅度差异（属于 A_s，不该被当伪迹）已被 level 归一化吸收。
    """
    path = _cache_path(dataset, subject, session, cfg)
    if use_cache and os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return dict(segs=z["segs"], y=z["y"], trial=z["trial"],
                    stats=json.loads(str(z["stats"])))

    X, y_trial, fs = load_trials(dataset, subject, session)

    segs_all, y_all, tr_all = [], [], []
    for t, x in enumerate(X):
        s = segment(bandpass(x, fs, cfg), fs, cfg)
        segs_all.append(s)
        y_all.append(np.full(len(s), y_trial[t], dtype=np.int64))
        tr_all.append(np.full(len(s), t, dtype=np.int64))
    segs = np.concatenate(segs_all).astype(np.float32)
    y = np.concatenate(y_all)
    trial = np.concatenate(tr_all)

    # 坏导剔除必须在段级幅值统计之前
    n_ch_before = segs.shape[1]
    channels = list(range(n_ch_before))
    if cfg.drop_bad_channels:
        channels = list(SEED_GOOD_CHANNELS)
        segs = segs[:, channels, :]

    amp = segment_amplitude(segs, channels)
    keep = amp <= cfg.reject_abs_thr
    rate = 1.0 - keep.mean()

    # 逐试次剔除率：整试次被清空是重要信息，不能只报总体比例
    per_trial = {int(t): float(1.0 - keep[trial == t].mean())
                 for t in np.unique(trial)}
    wiped = sorted(t for t, r in per_trial.items() if r > 0.9)

    stats = dict(n_before=int(len(segs)), n_after=int(keep.sum()),
                 reject_rate=float(rate), amp_median=float(np.median(amp)),
                 thr=float(cfg.reject_abs_thr), fs=int(fs),
                 n_ch_before=int(n_ch_before), n_ch_after=int(len(channels)),
                 channels=channels, reject_per_trial=per_trial,
                 wiped_trials=wiped,
                 wiped_labels=sorted({int(y_trial[t]) for t in wiped}))
    segs, y, trial = segs[keep], y[keep], trial[keep]

    np.savez_compressed(path, segs=segs, y=y, trial=trial,
                        stats=json.dumps(stats))
    return dict(segs=segs, y=y, trial=trial, stats=stats)


# --------------------------------------------------------------------------
# 分层抽样（PREREGISTER 4）
# --------------------------------------------------------------------------
def stratified_sample(y: np.ndarray, trial: np.ndarray, n_per_class: int = 60,
                      seed: int = 0) -> np.ndarray:
    """每个情绪抽 n_per_class 段，在该情绪的各试次间尽量均分。

    按试次分层而非直接随机，是因为同一试次内的段高度相关；
    不分层的话某个试次可能贡献绝大多数样本，可分性会被单个试次的特性绑架。
    """
    rng = np.random.default_rng(seed)
    picked = []
    for c in np.unique(y):
        trials_c = np.unique(trial[y == c])
        quota = np.full(len(trials_c), n_per_class // len(trials_c))
        quota[: n_per_class % len(trials_c)] += 1
        got = []
        short = 0
        for t, q in zip(trials_c, quota):
            pool = np.where((y == c) & (trial == t))[0]
            take = min(q, len(pool))
            short += q - take
            got.append(rng.choice(pool, size=take, replace=False))
        got = np.concatenate(got) if got else np.array([], dtype=int)
        if short > 0:                                  # 某试次段数不够，从其余补齐
            rest = np.setdiff1d(np.where(y == c)[0], got)
            if len(rest):
                got = np.concatenate([got, rng.choice(
                    rest, size=min(short, len(rest)), replace=False)])
        picked.append(got)
    out = np.concatenate(picked)
    return np.sort(out)


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import time

    if "--scan-bad" in sys.argv:              # 复算 config.SEED_BAD_CHANNELS
        union, R, Z = scan_bad_channels()
        names = [SEED_CHANNEL_NAMES[c] for c in union]
        print(f"坏导并集 {tuple(union)}  共 {len(union)} 个 -> 剩 {62-len(union)}")
        print(f"  即 {names}")
        for (subj, sess), z in sorted(Z.items()):
            bad = np.where((z > SEED_BAD_Z_HI) | (z < SEED_BAD_Z_LO))[0]
            if len(bad):
                print(f"  被试{subj:>2} ses{sess}: " + "  ".join(
                    f"{SEED_CHANNEL_NAMES[c]}(Z={z[c]:.0f})" for c in bad))
        sys.exit(0)

    cfg = DEFAULT
    print(f"配置 key = {cfg.key()}   CODE_VERSION = {CODE_VERSION}")
    print(f"带通 {cfg.band_lo}-{cfg.band_hi} Hz | 段长 {cfg.win_s}s | "
          f"剔坏导 {cfg.drop_bad_channels}({len(SEED_GOOD_CHANNELS)} 通道) | "
          f"剔段阈值 amp > {cfg.reject_abs_thr}\n")

    from config import SEED_CLASS_NAMES
    from data import list_subjects
    subs = list_subjects() if "--all" in sys.argv else [1, 6, 13, 15]
    sess = int(sys.argv[sys.argv.index("--session") + 1]) \
        if "--session" in sys.argv else 1
    print(f"session {sess}\n")
    print(f"{'被试':<5}{'段数':>12}{'剔除率':>8}{'统计量中位':>11}"
          f"{'清空试次':>22}{'抽样情绪分布':>16}")
    print("-" * 78)
    rates = []
    for subj in subs:
        r = build_segments("seed", subj, sess, cfg, use_cache=False)
        st = r["stats"]
        rates.append(st["reject_rate"])
        idx = stratified_sample(r["y"], r["trial"], 60, seed=0)
        cnt = np.bincount(r["y"][idx], minlength=3)
        wiped = st["wiped_trials"]
        wl = "".join(SEED_CLASS_NAMES[c][:3] for c in st["wiped_labels"])
        print(f"{subj:<5}{st['n_before']:>5}->{st['n_after']:<6}"
              f"{st['reject_rate']*100:>7.1f}%{st['amp_median']:>11.2f}"
              f"{str(wiped) + ('(' + wl + ')' if wiped else ''):>22}"
              f"{str(cnt.tolist()):>16}")
    r_ = np.array(rates)
    print("-" * 78)
    print(f"剔除率: 均值 {r_.mean()*100:.1f}%  最大 {r_.max()*100:.1f}%  "
          f"最小 {r_.min()*100:.1f}%")
