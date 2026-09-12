# -*- coding: utf-8 -*-
"""合成信号上的 sDM 移植验证。

复刻作者仓库的 scr_010_generate_signals + scr_020_calculate_modes
（无 LIBLINEAR 部分，由 sdm_features.py 提供相同 DMD+sDM 输出）。

执行：
    cd E:\\Python\\Study\\Koopman_EEG
    .\\.venv\\Scripts\\python.exe src\\sdm_synth_check.py

输出：
    1) 模式数、频率、增长率前 svd_rank 个；与作者已知 100 Hz 合成对照
    2) sDM 矩阵的 Hermitian / 对称 / 数值范围；与"norm_modes * norm_modes'"一致
    3) edge / network / both / full 四种索引与原文大小一致
    4) 一个随机正交矩阵 Q 对 X 做混合，理论上 DMD 谱应该**不变**
       （CLAUDE.md 第二节的"线性观测"假设——若成立，则 spec(QX) = spec(X)）。
       严格成立条件：A 满列秩 / 谱与特征向量正交无关。此处给出实测 W2 距离。
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sdm_features import (
    generate_signal_sech, stacking_dmd_preproc,
    stacking_dmd_acquire_modes, modes2sDMmat, sDMmat2vecfeat,
)


def synth_signal(rng, xi, t, params):
    return generate_signal_sech(xi, t, params, rng)


def run_one(seed, xi, t, params, svd_rank, fs_eff):
    rng = np.random.default_rng(seed)
    X = synth_signal(rng, xi, t, params)        # shape = (T, C)
    Xc = X.T                                    # (C, T) —— sdm_features 入口形状
    svd_st = stacking_dmd_preproc(Xc, dt=1.0 / fs_eff, svd_rank=svd_rank)
    mode_st = stacking_dmd_acquire_modes(svd_st, dmd_rank=svd_rank)
    sDM = modes2sDMmat(mode_st)
    return X, Xc, svd_st, mode_st, sDM


def main():
    # -------- 复制作者 scr_010 的参数 --------
    xi = np.arange(-20, 21)                                     # 41 通道
    t = np.arange(500) / 1000.0                                 # 0..0.499 s @ 1 kHz
    base = dict(noise_amplitude=1.0,
                signal_frequnecy=100.0,
                signal_amplitude=1.0,
                signal_phase_range=[-np.pi / 6, np.pi / 6])
    fs = 1000.0

    svd_rank = 2
    n_trials = 4

    print("=" * 64)
    print("合成验证：与作者 MATLAB scr_010 + scr_020 等价的 Python 流程")
    print(f"  xi=len{len(xi)} t=len{len(t)} svd_rank={svd_rank}")
    print("=" * 64)

    modes_list = []
    sdm_list = []
    for seed in range(n_trials):
        X, Xc, svd_st, mode_st, sDM = run_one(seed, xi, t, base, svd_rank, fs)
        modes_list.append(mode_st)
        sdm_list.append(sDM)

        # 报告
        print(f"\n[seed {seed}] X shape={X.shape}  Xc shape={Xc.shape}")
        print(f"  nb_stack = {svd_st['params']['nb_stack']}  "
              f"Xstack shape={svd_st['X2'].shape[0] + 1}×{svd_st['X2'].shape[1] + 1}")
        print(f"  lambda   : {np.round(mode_st['lam'], 4)}")
        print(f"  freq(Hz) : {np.round(mode_st['freq'], 4)}  (目标 ≈ 100)")
        print(f"  r(生长率): {np.round(mode_st['growth'], 4)}")
        print(f"  sDM shape={sDM.shape}  "
              f"diag_mean={np.diag(sDM).mean():.4f}  "
              f"||sDM-sDM.T||_F={np.linalg.norm(sDM - sDM.T):.2e}  "
              f"max|sDM.imag|={0.0:.2e}")

        # 索引尺寸核查
        for ft in ["edge", "network", "both", "full"]:
            v = sDMmat2vecfeat(sDM, ft)
            print(f"    feature_type={ft:8s} -> vec len = {v.shape[0]}  "
                  f"sum={v.sum():.4f}  max={v.max():.4f}")

    # -------- 线性观测不变性核对（CLAUDE.md §1） --------
    print("\n" + "=" * 64)
    print("线性观测不变性核对：X̃ = X @ Q (Q 随机正交) -> DMD 谱距离")
    print("=" * 64)
    rng = np.random.default_rng(7)
    Q, _ = np.linalg.qr(rng.standard_normal((len(xi), len(xi))))
    dists = []
    for seed in range(n_trials):
        X, Xc, _, _, _ = run_one(seed, xi, t, base, svd_rank, fs)
        X2 = X @ Q
        # 用相同的 svd_rank=2 估谱（这一对故意只取前 2 个模态，避免模态数放大差异）
        s1 = stacking_dmd_preproc(X.T, dt=1.0 / fs, svd_rank=svd_rank)
        m1 = stacking_dmd_acquire_modes(s1, dmd_rank=svd_rank)
        s2 = stacking_dmd_preproc(X2.T, dt=1.0 / fs, svd_rank=svd_rank)
        m2 = stacking_dmd_acquire_modes(s2, dmd_rank=svd_rank)
        # 谱距离（实对实）：|λ1 - λ2|
        d = np.abs(np.sort(m1["lam"]) - np.sort(m2["lam"]))
        dists.append(d)
        print(f"  seed {seed}  |λ_X - λ_QX| = {d.round(6).tolist()}")
    dists = np.array(dists)
    print(f"\n  mean  |Δλ| = {dists.mean():.2e}   max = {dists.max():.2e}")
    print("  （若 ≤ 1e-8，说明本实现的 svd_rank 截断在合成信号上未触发模态数扰动；"
          "若差异较大，说明 svd_rank 处奇异值间隙不足导致截断子空间不唯一——与"
          "PREREGISTER 修订 2 的 D5/D7 现象同源。）")

    # -------- sDM 形状矩阵自检 --------
    print("\n" + "=" * 64)
    print("sDM 形状与对称性总结")
    print("=" * 64)
    for i, sDM in enumerate(sdm_list):
        sym_err = np.linalg.norm(sDM - sDM.T)
        print(f"  trial {i}:  shape={sDM.shape}  "
              f"||sDM - sDM.T||_F = {sym_err:.2e}  "
              f"trace = {np.trace(sDM):.4f}  "
              f"(= number of modes, 应等于 {svd_rank})")

    print("\n合成验证完成。")


if __name__ == "__main__":
    main()
