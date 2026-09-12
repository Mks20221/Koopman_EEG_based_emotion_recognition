# -*- coding: utf-8 -*-
"""sDM（spatial Dynamic Mode）特征提取 —— Python 移植。

完全对应作者仓库 E:/Python/Study/fast-accurate-and-interpretable-decoding-of-electrocorticographic-signals-using-DMD-main
的 func/*.m 与 scr_*.m。区别仅在于：这里不调 LIBLINEAR，分类交给主实验脚本；
另外每段信号**独立**地做 Hankel 堆叠 + DMD + sDM（MATLAB 示例是一次性跑完一组 trial，
但每次 trial 仍单独调用 stacking_dmd_preproc/stacking_dmd_acquire_modes/modes2sDMmat，
所以逐段实现与原文等价）。

本模块提供：
    stack_signal(X, nb_stack)             -- func/stack_signal.m
    stacking_dmd_preproc(X, dt, svd_rank) -- func/stacking_dmd_preproc.m
    stacking_dmd_acquire_modes(svd, rank) -- func/stacking_dmd_acquire_modes.m
    modes2sDMmat(mode_st)                 -- func/modes2sDMmat.m
    sDMmat2vecfeat(sDMmat, feature_type)  -- func/sDMmat2vecfeat.m
    sdm_feature_for_segment(X, fs, ...)   -- 一站式：段 -> sDM 向量（可选 edge/network/both/full）

约定：
    X        shape = (n_channels, n_samples)，float，**未做列归一化**
    dt       采样间隔（秒）
    svd_rank SVD 截断秩；-1 表示全秩
    feature_type  'edge' | 'network' | 'both' | 'full'
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. 信号堆叠（func/stack_signal.m）
# ---------------------------------------------------------------------------
def stack_signal(X: np.ndarray, nb_stack: int) -> np.ndarray:
    """(C, T) -> (C*nb_stack, T-nb_stack+1)。

    MATLAB: Y{i} = X(:, i:end-(nb_stack-i)); Y = cat(1, Y{:});
    用 Python 列表推导 + np.concatenate 保持完全等价的拼接顺序。
    """
    if nb_stack < 1:
        raise ValueError("nb_stack must be >= 1")
    parts = [X[:, i:X.shape[1] - (nb_stack - i)] for i in range(nb_stack)]
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# 2. stacking_dmd_preproc（func/stacking_dmd_preproc.m）
# ---------------------------------------------------------------------------
def stacking_dmd_preproc(X: np.ndarray, dt: float, svd_rank: int = -1) -> dict:
    """对 X 做 Hankel 堆叠、构造 X1/X2、对 X1 做 SVD（截断或全秩）。

    svd_rank=-1 返回完整的 U/S/V（economic 模式）；下游若要按多个 rank 取前 r 列，
    请直接使用 stacking_dmd_preproc_full，**避免每段重算 SVD**。

    Returns dict: U, S, V, X2, params(nb_elec, nb_sample, svd_rank, nb_stack, dt)
    """
    nb_elec, nb_sample = X.shape
    nb_stack = int(np.ceil((nb_sample + 1) / (nb_elec + 1)))
    Xstack = stack_signal(X, nb_stack)
    X1 = Xstack[:, :-1]
    X2 = Xstack[:, 1:]

    # numpy svd 默认 economic 模式（与 MATLAB svd(X,'econ') 等价）。
    # **永远算 full SVD 一次**，下游按 svd_rank 切片（svd_rank=-1 即不切），
    # 避免一段信号上为多个 rank 重复 SVD —— EEG 上 SVD 耗时占比约 80%。
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    V = Vh.T
    if svd_rank != -1:
        U = U[:, :svd_rank]
        S = S[:svd_rank]
        V = V[:, :svd_rank]

    return dict(U=U, S=S, V=V, X2=X2,
                params=dict(nb_elec=nb_elec, nb_sample=nb_sample,
                            svd_rank=svd_rank, nb_stack=nb_stack, dt=dt))


def stacking_dmd_preproc_full(X: np.ndarray, dt: float) -> dict:
    """等价 svd_rank=-1 的 stacking_dmd_preproc，独立名字便于批量代码读。"""
    return stacking_dmd_preproc(X, dt, svd_rank=-1)


# ---------------------------------------------------------------------------
# 3. stacking_dmd_acquire_modes（func/stacking_dmd_acquire_modes.m）
# ---------------------------------------------------------------------------
def stacking_dmd_acquire_modes(svd_st: dict, dmd_rank: int) -> dict:
    """从 SVD 三元组计算 DMD 模式与谱。

    MATLAB 等价公式：
        Atilde = U' * X2 * V / S
        [W, D] = eig(Atilde)
        Phi = X2 * V / S * W          (shape = (n_stack*C, dmd_rank))
        lambda = diag(D)
        omega  = log(lambda) / dt
        freq   = |imag(omega) / (2*pi)|
        r      = |lambda|^(1/dt)
        phi    = Phi(1:nb_elec, :)     # 取原始导联空间对应的第一个 C 行块
    """
    p = svd_st["params"]
    if dmd_rank == -1:
        if p["svd_rank"] != -1:
            raise ValueError("dmd_rank=-1 但 SVD 不是全秩")
        Ur, Sr, Vr = svd_st["U"], svd_st["S"], svd_st["V"]
    else:
        Ur = svd_st["U"][:, :dmd_rank]
        Sr = svd_st["S"][:dmd_rank]
        Vr = svd_st["V"][:, :dmd_rank]
    X2 = svd_st["X2"]
    dt = p["dt"]

    # Atilde = U' X2 V / S  （MATLAB 中 Sr 是 diag 矩阵，V/S 等价于 V * diag(1/S)）
    # NumPy np.linalg.svd 返回的 S 是一维向量，用 Sr[None,:] 沿列广播。
    Atilde = (Ur.T @ X2 @ Vr) / Sr[None, :]
    # numpy.linalg.eig 返回 (values, vectors)，与 MATLAB 的 (vectors, values) 相反
    D, W = np.linalg.eig(Atilde)
    Phi = (X2 @ Vr) / Sr[None, :] @ W         # (n_stack*C, dmd_rank)

    lam = D                                  # 已经是一维（r,）
    omega = np.log(lam) / dt
    freq = np.abs(omega.imag) / (2.0 * np.pi)
    growth = np.abs(lam) ** (1.0 / dt)

    phi_full = Phi
    phi = Phi[:p["nb_elec"], :]                 # 取原始导联对应的块

    return dict(phi=phi, phi_full=phi_full, lam=lam, omega=omega,
                freq=freq, growth=growth,
                params=dict(nb_elec=p["nb_elec"], nb_stack=p["nb_stack"],
                            dmd_rank=dmd_rank, dt=dt))


# ---------------------------------------------------------------------------
# 4. modes2sDMmat（func/modes2sDMmat.m）
# ---------------------------------------------------------------------------
def modes2sDMmat(mode_st: dict) -> np.ndarray:
    """列 L2 归一化后做 Phi @ Phi'（MATLAB 注释明说不是 QR）。

    等价：sDMmat = real(norm_modes @ norm_modes.conj().T)
    """
    phi = mode_st["phi"]
    norms = np.sqrt(np.sum(np.abs(phi) ** 2, axis=0, keepdims=True))
    norms = np.where(norms < 1e-12, 1.0, norms)
    norm_modes = phi / norms
    sDMmat = norm_modes @ norm_modes.conj().T
    # MATLAB 的 real() 把数值误差造成的微小虚部去掉；这里用 np.real
    return np.real(sDMmat)


# ---------------------------------------------------------------------------
# 5. sDMmat2vecfeat（func/sDMmat2vecfeat.m）
# ---------------------------------------------------------------------------
def sDMmat2vecfeat(sDMmat: np.ndarray, feature_type: str = "edge") -> np.ndarray:
    """按 feature_type 抽取 sDM 矩阵的元素。

        'edge'    对角线（seDM 特征）
        'network' 上三角非对角线（snDM 特征）
        'both'    上三角含对角线
        'full'    全部 nb_channel*nb_channel 个元素（行优先）
    """
    if sDMmat.ndim != 2 or sDMmat.shape[0] != sDMmat.shape[1]:
        raise ValueError("sDMmat must be square")
    nb = sDMmat.shape[0]
    if feature_type == "edge":
        idx = np.arange(nb) * (nb + 1)
    elif feature_type == "network":
        idx = np.flatnonzero(np.triu(np.ones((nb, nb), dtype=bool), k=1))
    elif feature_type == "both":
        idx = np.flatnonzero(np.triu(np.ones((nb, nb), dtype=bool), k=0))
    elif feature_type == "full":
        idx = np.arange(nb * nb)
    else:
        raise ValueError(f"unknown feature_type={feature_type!r}")
    return sDMmat.ravel()[idx]


# ---------------------------------------------------------------------------
# 6. 一站式：段 -> sDM 向量
# ---------------------------------------------------------------------------
def sdm_feature_for_segment(X: np.ndarray, fs: float,
                            svd_rank: int, dmd_rank: int,
                            feature_type: str = "edge") -> np.ndarray:
    """对单段信号 X (C, T) 提取 sDM 特征向量。

    与 MATLAB scr_020 的内层循环完全一致（对每个 trial 调用一次）。
    """
    dt = 1.0 / fs
    svd_st = stacking_dmd_preproc(X, dt, svd_rank=svd_rank)
    mode_st = stacking_dmd_acquire_modes(svd_st, dmd_rank=dmd_rank)
    sDMmat = modes2sDMmat(mode_st)
    return sDMmat2vecfeat(sDMmat, feature_type=feature_type)


# ---------------------------------------------------------------------------
# 7. 自检：作者 MATLAB 的合成信号（generate_signal_sech.m）
# ---------------------------------------------------------------------------
def generate_signal_sech(xi: np.ndarray, t: np.ndarray,
                         signal_params: dict, rng: np.random.Generator) -> np.ndarray:
    """作者合成信号：amp * sech(xi) * sin(phase0 + 2π f t) + noise_amp * randn。

    返回 (len(t), len(xi))，**与 MATLAB 形状一致**（时间 × 空间）。
    """
    xi_grid, t_grid = np.meshgrid(xi, t)
    phase0 = (signal_params["signal_phase_range"][1]
              - signal_params["signal_phase_range"][0]) * rng.random(len(xi)) \
        + signal_params["signal_phase_range"][0]
    phase0 = np.broadcast_to(phase0, (len(t), len(xi)))
    sig = signal_params["signal_amplitude"] * (1.0 / np.cosh(xi_grid)) \
        * np.sin(phase0 + 2.0 * np.pi * signal_params["signal_frequnecy"] * t_grid)
    noise = signal_params["noise_amplitude"] * rng.standard_normal(sig.shape)
    return sig + noise
