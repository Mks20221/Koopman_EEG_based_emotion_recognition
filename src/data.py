# -*- coding: utf-8 -*-
"""统一数据接口。

对外只暴露 `load_trials(dataset, subject, session) -> (X, y, fs)`。
所有数据集的格式差异（.mat 版本、变量命名、通道顺序、标签编码、采样率）
全部封在本模块内，绝不允许渗进模型代码。

当前只实现 SEED；SEED-IV / DEAP 留了注册位。
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

import numpy as np

from config import (SEED_ROOT, SEED_FS, SEED_N_CHANNELS, SEED_N_TRIALS,
                    SEED_LABEL_MAP)

# 试次变量名形如 djc_eeg1 ... djc_eeg15
_TRIAL_RE = re.compile(r"_eeg(\d+)$")
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


# --------------------------------------------------------------------------
# .mat 读取：双路径。转置只发生在 h5py 分支。
# --------------------------------------------------------------------------
def _is_hdf5(path: str) -> bool:
    """v7.3 的 .mat 就是 HDF5，靠文件头 8 字节判断，不靠扩展名或猜测。"""
    with open(path, "rb") as f:
        return f.read(8) == _HDF5_MAGIC


def _read_mat_trials(path: str) -> dict[int, np.ndarray]:
    """读出 {试次号: (n_channels, n_samples)}。

    关键：MATLAB 按列主序存 (62,T)。
      - scipy.io.loadmat 已还原为 (62,T)  -> 不转置
      - h5py 直接看裸存储，得到 (T,62)    -> 要转置
    转错方向不会抛异常，只会让后续 DMD 出一堆合法但无意义的数，
    所以两条分支各自负责自己的方向，出口统一由 _validate 兜底。
    """
    out: dict[int, np.ndarray] = {}
    if _is_hdf5(path):
        import h5py
        with h5py.File(path, "r") as f:
            for k in f.keys():
                m = _TRIAL_RE.search(k)
                if m:
                    out[int(m.group(1))] = np.array(f[k]).T  # (T,62) -> (62,T)
    else:
        from scipy.io import loadmat
        d = loadmat(path)
        for k, v in d.items():
            m = _TRIAL_RE.search(k)
            if m:
                out[int(m.group(1))] = np.asarray(v)          # 已是 (62,T)
    return out


def _validate(x: np.ndarray, n_ch: int, who: str) -> np.ndarray:
    """出口兜底：通道维必须在 axis 0。

    EEG 里 n_channels << n_samples 永远成立（62 vs 数万），
    所以维度搞反了一定看得出来。宁可在这里炸，也不要让垃圾流进 DMD。
    """
    if x.ndim != 2:
        raise ValueError(f"{who}: 期望 2 维, 得到 {x.shape}")
    if x.shape[0] != n_ch:
        if x.shape[1] == n_ch:
            raise ValueError(
                f"{who}: shape={x.shape}, 通道维在 axis 1 —— 转置方向错了。"
                f"检查 _read_mat_trials 走的是哪条分支。")
        raise ValueError(f"{who}: shape={x.shape}, 通道数不是 {n_ch}")
    if x.shape[1] <= x.shape[0]:
        raise ValueError(f"{who}: shape={x.shape}, 采样点数不该 <= 通道数")
    return x


# --------------------------------------------------------------------------
# SEED
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _seed_file_index() -> dict[int, list[str]]:
    """{被试号: [session1, session2, session3]}，session 按日期升序。

    文件名形如 6_20130712.mat。日期是 YYYYMMDD，字典序即时间序。
    """
    pre = os.path.join(SEED_ROOT, "Preprocessed_EEG")
    if not os.path.isdir(pre):
        raise FileNotFoundError(f"找不到 {pre}，检查 config.SEED_ROOT 或 $SEED_ROOT")
    idx: dict[int, list[str]] = {}
    for fn in os.listdir(pre):
        m = re.fullmatch(r"(\d+)_(\d{8})\.mat", fn)
        if m:
            idx.setdefault(int(m.group(1)), []).append(fn)
    for s in idx:
        idx[s].sort(key=lambda fn: fn.split("_")[1])
    return idx


@lru_cache(maxsize=1)
def _seed_labels() -> np.ndarray:
    """15 个试次的标签，已映射到 0..2。全体被试、全体 session 共用同一序列。"""
    from scipy.io import loadmat
    raw = loadmat(os.path.join(SEED_ROOT, "Preprocessed_EEG", "label.mat"))["label"]
    raw = np.asarray(raw).ravel()
    if raw.size != SEED_N_TRIALS:
        raise ValueError(f"label.mat 应有 {SEED_N_TRIALS} 个标签, 实为 {raw.size}")
    return np.array([SEED_LABEL_MAP[int(v)] for v in raw], dtype=np.int64)


def _load_seed(subject: int, session: int):
    idx = _seed_file_index()
    if subject not in idx:
        raise ValueError(f"SEED 被试号应在 {sorted(idx)}, 得到 {subject}")
    files = idx[subject]
    if not 1 <= session <= len(files):
        raise ValueError(f"被试 {subject} 有 {len(files)} 个 session, 请求了 {session}")
    path = os.path.join(SEED_ROOT, "Preprocessed_EEG", files[session - 1])

    trials = _read_mat_trials(path)
    missing = set(range(1, SEED_N_TRIALS + 1)) - set(trials)
    if missing:
        raise ValueError(f"{files[session-1]}: 缺试次 {sorted(missing)}")

    X = [_validate(np.ascontiguousarray(trials[i], dtype=np.float64),
                   SEED_N_CHANNELS, f"{files[session-1]}:_eeg{i}")
         for i in range(1, SEED_N_TRIALS + 1)]
    return X, _seed_labels(), SEED_FS


# --------------------------------------------------------------------------
# 对外接口
# --------------------------------------------------------------------------
_LOADERS = {"seed": _load_seed}


def load_trials(dataset: str, subject: int, session: int):
    """载入一个 (被试, session) 的全部试次。

    Parameters
    ----------
    dataset : str   目前只支持 "seed"
    subject : int   1-based
    session : int   1-based，按采集日期升序

    Returns
    -------
    X  : list of (n_channels, n_samples) float64，试次长度不定
    y  : (n_trials,) int64，已统一为 0..C-1
    fs : int 采样率 (Hz)
    """
    key = dataset.lower()
    if key not in _LOADERS:
        raise NotImplementedError(f"未实现的数据集 {dataset!r}, 可选 {sorted(_LOADERS)}")
    return _LOADERS[key](subject, session)


def list_subjects(dataset: str = "seed") -> list[int]:
    if dataset.lower() != "seed":
        raise NotImplementedError(dataset)
    return sorted(_seed_file_index())


def n_sessions(dataset: str, subject: int) -> int:
    if dataset.lower() != "seed":
        raise NotImplementedError(dataset)
    return len(_seed_file_index()[subject])


# --------------------------------------------------------------------------
# 自检：抽查几个文件，打印 shape 与标签分布
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from config import SEED_CLASS_NAMES

    picks = [(1, 1), (6, 1), (15, 3)]     # 首个、体积异常的、末个
    print(f"SEED_ROOT = {SEED_ROOT}")
    print(f"被试列表 = {list_subjects()}\n")

    for subj, sess in picks:
        X, y, fs = load_trials("seed", subj, sess)
        fn = _seed_file_index()[subj][sess - 1]
        dur = [x.shape[1] / fs for x in X]
        print(f"--- 被试 {subj} session {sess}  ({fn}) ---")
        print(f"  试次数 {len(X)}   fs {fs} Hz   dtype {X[0].dtype}")
        print(f"  shape  首 {X[0].shape}  末 {X[-1].shape}")
        print(f"  时长   {min(dur):.1f}-{max(dur):.1f} s   合计 {sum(dur)/60:.1f} min")
        cnt = np.bincount(y, minlength=3)
        print(f"  标签   {y.tolist()}")
        print(f"  分布   " + "  ".join(f"{SEED_CLASS_NAMES[i]}={cnt[i]}" for i in range(3)))
        print(f"  幅值   median|x|={np.median(np.abs(X[0])):.2f}  "
              f"max|x|={max(np.abs(x).max() for x in X):.1f}\n")

    # 一致性：全体被试的文件索引是否齐整
    idx = _seed_file_index()
    bad = {s: len(v) for s, v in idx.items() if len(v) != 3}
    print("全体被试 session 数均为 3:", not bad, bad if bad else "")
    print("自检通过。")
    sys.exit(0)
