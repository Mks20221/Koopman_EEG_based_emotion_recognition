# -*- coding: utf-8 -*-
"""Koopman 谱估计 + 两种被试/情绪距离。

DMD 数值逻辑照搬 `koopman_stage0_synth.py`（第 1 周已验证：自治系统频率误差
0.000 Hz），本文件只做两件事：
  1. 把参数封进 `docs/PREREGISTER.md` §3 冻结的 `SpectrumConfig`；
  2. 补两个第 1 周不需要的距离——特征值在归一化 (f,σ) 平面的带权 Wasserstein-2
     （§5.1，CLAUDE.md 约束 3：λ 显含 Δt，不可跨录制直接比较复平面）、
     特征向量空间块的 Grassmann 弦距离（§5.2）。

任何数值逻辑改动都必须先在 PREREGISTER.md 修订记录里写明理由。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import ot


@dataclass(frozen=True)
class SpectrumConfig:
    """PREREGISTER §3，冻结（除 f_lo/f_hi 的 1-30Hz 对照版，见 §11）。"""
    delay: int = 16
    rank: int = 16
    method: str = "fb"          # fbDMD，消除噪声引起的特征值偏置
    f_lo: float = 1.0
    f_hi: float = 45.0
    mod_max: float = 1.02
    top_k: int = 12

    @property
    def f_range(self):
        return (self.f_lo, self.f_hi)

    def key(self) -> str:
        return (f"d{self.delay}r{self.rank}{self.method}"
                f"f{self.f_lo}-{self.f_hi}m{self.mod_max}k{self.top_k}")


DEFAULT = SpectrumConfig()
CONTROL_1_30HZ = SpectrumConfig(f_lo=1.0, f_hi=30.0)   # PREREGISTER §11


def hankel_embed(X: np.ndarray, d: int) -> np.ndarray:
    """时延嵌入：(C,T) -> (C*d, T-d+1)。块 i=0 是最早的窗口，i=d-1 是最新的。"""
    C, T = X.shape
    return np.vstack([X[:, i:T - d + 1 + i] for i in range(d)])


def dmd_spectrum(X: np.ndarray, dt: float, cfg: SpectrumConfig = DEFAULT) -> dict:
    """Hankel-DMD 估谱。与 koopman_stage0_synth.dmd_spectrum 数值逻辑一致。

    返回 dict:
        lam, f, sigma, w : 保留的 top_k 个特征值 / 频率 / 增长率 / 幅度权重
        Phi      : 完整 DMD 模态矩阵 (C*d, r)
        mode_idx : 被保留的 top_k 个模态在 Phi 列中的原始索引
                   （spatial_modes 靠它从 Phi 里取正确的列）
    """
    H = hankel_embed(X, cfg.delay)
    X1, X2 = H[:, :-1], H[:, 1:]
    U, s, Vt = np.linalg.svd(X1, full_matrices=False)
    r = min(cfg.rank, len(s), X1.shape[0])
    U, s, Vt = U[:, :r], s[:r], Vt[:r]
    Sinv = np.diag(1.0 / s)

    Af = U.conj().T @ X2 @ Vt.conj().T @ Sinv
    if cfg.method == "fb":
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
    else:
        A = Af

    lam, W = np.linalg.eig(A)
    Phi = X2 @ Vt.conj().T @ Sinv @ W
    b = np.linalg.lstsq(Phi, H[:, 0], rcond=None)[0]

    f = np.angle(lam) / (2 * np.pi * dt)
    sigma = np.log(np.abs(lam) + 1e-300) / dt

    keep = (f >= cfg.f_range[0]) & (f <= cfg.f_range[1]) & (np.abs(lam) <= cfg.mod_max)
    if keep.sum() == 0:
        keep = (f >= cfg.f_range[0]) & (f <= cfg.f_range[1])
    if keep.sum() == 0:
        keep = np.ones_like(f, dtype=bool)
    keep_idx = np.where(keep)[0]
    amp = np.abs(b[keep_idx])
    order = np.argsort(-amp)[:cfg.top_k]
    mode_idx = keep_idx[order]

    lam_k, f_k, sigma_k, amp_k = lam[mode_idx], f[mode_idx], sigma[mode_idx], amp[order]
    lam_k = np.where(np.abs(lam_k) > 1.0, lam_k / np.abs(lam_k), lam_k)  # 越界投影回单位圆
    w = amp_k / (amp_k.sum() + 1e-12)
    return dict(lam=lam_k, f=f_k, sigma=sigma_k, w=w, Phi=Phi, mode_idx=mode_idx)


def spatial_modes(spectrum: dict, n_channels: int) -> np.ndarray:
    """PREREGISTER §5.2：取 Φ 的空间块（时延嵌入第一个 n_channels 行块，
    对应原始导联空间、也就是 A_s 作用的地方），按幅度排好的 top_k 列做 QR。

    若某段实际保留的模态数 < top_k（罕见，通常整个 rank=16 都在频段内），
    Q 的列数会相应变小；grassmann_distance 用 min(dim) 处理，不视为错误。
    """
    Phi_spatial = spectrum["Phi"][:n_channels, spectrum["mode_idx"]]
    Q, _ = np.linalg.qr(Phi_spatial)
    return Q


def eigval_distance(s1: dict, s2: dict, s_f: float, s_sigma: float) -> float:
    """PREREGISTER §5.1：归一化 (f,σ) 平面上的带权 Wasserstein-2。

    s_f, s_sigma 是全体样本 (f,σ) 的 IQR，只算一次、外部传入，
    不在这里重算——否则每对距离用的尺度不一致。
    """
    a = np.ascontiguousarray(s1["w"])
    b = np.ascontiguousarray(s2["w"])
    df = (s1["f"][:, None] - s2["f"][None, :]) / s_f
    dsig = (s1["sigma"][:, None] - s2["sigma"][None, :]) / s_sigma
    M = df ** 2 + dsig ** 2
    return float(np.sqrt(max(ot.emd2(a, b, M), 0.0)))


def grassmann_distance(Q1: np.ndarray, Q2: np.ndarray) -> float:
    """两个子空间的 Grassmann 弦距离 sqrt(sum sin^2(主角))，主角由
    svd(Q1^H Q2) 的奇异值反余弦得到。用子空间距离而非逐模态配对，
    是因为模态顺序和相位不唯一，逐个配对没有意义（PREREGISTER §5.2）。
    """
    s = np.linalg.svd(Q1.conj().T @ Q2, compute_uv=False)
    s = np.clip(s.real if np.iscomplexobj(s) else s, -1.0, 1.0)
    theta = np.arccos(s)
    return float(np.sqrt(np.sum(np.sin(theta) ** 2)))
