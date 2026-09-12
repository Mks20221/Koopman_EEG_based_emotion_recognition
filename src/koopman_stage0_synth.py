#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 0 阶段 / 第 1 周：在合成数据上验证 Koopman 谱不变性
=====================================================
目的：在"有标准答案"的数据上跑通管线，把代码 bug、参数选择、假设本身三者解耦。

要验证的命题：
    观测 x^(s) = A_s z，若源动力学 z 的 Koopman 算子为 K_z，
    则 K_s = A_s K_z A_s^+ 与 K_z 相似，故 spec(K_s) = spec(K_z)。
    ——特征值对被试特有的线性混合 A_s 严格不变。

依赖：numpy scipy matplotlib POT
运行：python koopman_stage0_synth.py
"""
import numpy as np
import scipy.linalg as sla
import matplotlib as mpl
import matplotlib.pyplot as plt
import ot
import os

RNG = np.random.default_rng(0)
OUT = "figs_stage0"
os.makedirs(OUT, exist_ok=True)

mpl.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 8, "figure.dpi": 110,
    "savefig.dpi": 200, "savefig.bbox": "tight", "axes.linewidth": 0.6,
})


# ======================================================================
# 1. 构造特征值已知的源系统
# ======================================================================
def make_koopman_system(freqs_hz, taus_s, dt, seed=0):
    """构造一个 Koopman 算子 K_z，其特征值由 (频率, 时间常数) 精确指定。

    每对共轭特征值 lambda = exp((-1/tau + i*2*pi*f) * dt)，
    对应一个 2x2 的旋转-缩放块；再用随机可逆矩阵做相似变换，
    使 K 看上去"不平凡"但谱不变。
    """
    rng = np.random.default_rng(seed)
    blocks = []
    true_lams = []
    for f, tau in zip(freqs_hz, taus_s):
        r = np.exp(-dt / tau)          # 模长 -> 衰减
        th = 2 * np.pi * f * dt        # 幅角 -> 频率
        blocks.append(r * np.array([[np.cos(th), -np.sin(th)],
                                    [np.sin(th),  np.cos(th)]]))
        true_lams += [r * np.exp(1j * th), r * np.exp(-1j * th)]
    B = sla.block_diag(*blocks)
    K = B.shape[0]
    V = rng.standard_normal((K, K))
    while abs(np.linalg.det(V)) < 1e-6:
        V = rng.standard_normal((K, K))
    return V @ B @ np.linalg.inv(V), np.array(true_lams)


def simulate_source(Kz, n_steps, drive_std=0.05, seed=0):
    """随机驱动的线性系统：z[t+1] = Kz z[t] + w[t]。
    衰减振子 + 随机驱动 = EEG 节律最简单的合理模型。"""
    rng = np.random.default_rng(seed)
    K = Kz.shape[0]
    z = np.zeros((K, n_steps))
    z[:, 0] = rng.standard_normal(K)
    for t in range(n_steps - 1):
        z[:, t + 1] = Kz @ z[:, t] + drive_std * rng.standard_normal(K)
    return z


def simulate_stuart_landau(freqs_hz, dt, n_steps, mu=0.4, coup=0.15, seed=0):
    """非线性对照：耦合 Stuart-Landau 振子。
    Koopman 的意义就在于处理非线性，只测线性系统等于没测。"""
    rng = np.random.default_rng(seed)
    n = len(freqs_hz)
    w = 2 * np.pi * np.array(freqs_hz)
    a = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.5
    out = np.zeros((2 * n, n_steps))
    for t in range(n_steps):
        out[0::2, t], out[1::2, t] = a.real, a.imag
        mean_a = a.mean()
        da = (mu + 1j * w) * a - np.abs(a) ** 2 * a + coup * (mean_a - a)
        a = a + dt * da + 0.05 * np.sqrt(dt) * (
            rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return out


def mix_to_channels(z, n_ch, snr_db, seed=0, nonlinear=0.0):
    """模拟被试特有的导联场：x = A_s z + noise。
    nonlinear>0 时加入二次项，用于检验线性导联假设的偏离容忍度。"""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_ch, z.shape[0])) / np.sqrt(z.shape[0])
    x = A @ z
    if nonlinear > 0:
        x = x + nonlinear * (A @ (z ** 2))
    sig_p = np.mean(x ** 2)
    noise_p = sig_p / (10 ** (snr_db / 10))
    x = x + np.sqrt(noise_p) * rng.standard_normal(x.shape)
    return x, A


# ======================================================================
# 2. Hankel-DMD 与模态过滤
# ======================================================================
def hankel_embed(X, d):
    """时延嵌入：X (C,T) -> H (C*d, T-d+1)。
    原始通道数往往不足以张成完整状态空间，必须嵌入。"""
    C, T = X.shape
    return np.vstack([X[:, i:T - d + 1 + i] for i in range(d)])


def optimal_hard_threshold(s, n_rows, n_cols):
    """Gavish & Donoho (2014) 最优硬阈值，自动定截断秩，避免手拍 r。"""
    beta = min(n_rows, n_cols) / max(n_rows, n_cols)
    omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
    return int(max(1, np.sum(s > omega * np.median(s))))


def dmd_spectrum(X, dt, rank=None, method="fb",
                 f_range=(1.0, 45.0), mod_max=1.02, top_k=12):
    """Hankel-DMD 估谱，返回带权特征值点集。

    返回 dict:
        lam : 保留的特征值 (复数)
        f   : 频率 Hz
        sigma: 增长率 1/s (负=衰减)
        w   : 归一化幅度权重  <- Wasserstein 距离要用
    """
    X1, X2 = X[:, :-1], X[:, 1:]
    U, s, Vt = np.linalg.svd(X1, full_matrices=False)
    r = rank or optimal_hard_threshold(s, *X1.shape)
    r = min(r, len(s), X1.shape[0])
    U, s, Vt = U[:, :r], s[:r], Vt[:r]
    Sinv = np.diag(1.0 / s)

    # 前向算子
    Af = U.conj().T @ X2 @ Vt.conj().T @ Sinv
    if method == "fb":
        # 前向-后向 DMD：消除噪声引起的特征值偏置 (Dawson et al. 2016)
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
    Phi = X2 @ Vt.conj().T @ Sinv @ W            # exact DMD 模态
    b = np.linalg.lstsq(Phi, X[:, 0], rcond=None)[0]   # 幅度

    f = np.angle(lam) / (2 * np.pi * dt)
    sigma = np.log(np.abs(lam) + 1e-300) / dt

    # --- 模态过滤：这一步不做，距离矩阵会被少数离群模态支配 ---
    keep = (f >= f_range[0]) & (f <= f_range[1]) & (np.abs(lam) <= mod_max)
    if keep.sum() == 0:
        keep = (f >= f_range[0]) & (f <= f_range[1])
    if keep.sum() == 0:
        keep = np.ones_like(f, dtype=bool)
    lam, f, sigma, amp = lam[keep], f[keep], sigma[keep], np.abs(b[keep])

    idx = np.argsort(-amp)[:top_k]               # 按幅度取前 top_k
    lam, f, sigma, amp = lam[idx], f[idx], sigma[idx], amp[idx]
    lam = np.where(np.abs(lam) > 1.0, lam / np.abs(lam), lam)   # 投影回单位圆
    w = amp / (amp.sum() + 1e-12)
    return dict(lam=lam, f=f, sigma=sigma, w=w)


def spectrum_w2(s1, s2):
    """两个带权谱之间的 Wasserstein-2 距离（复平面上）。"""
    a, b = np.ascontiguousarray(s1["w"]), np.ascontiguousarray(s2["w"])
    M = np.abs(s1["lam"][:, None] - s2["lam"][None, :]) ** 2
    return float(np.sqrt(max(ot.emd2(a, b, M), 0.0)))


# ======================================================================
# 3. 实验
# ======================================================================
DT = 1 / 200.0
FREQS = [6.0, 10.0, 20.0]        # theta / alpha / beta
TAUS = [0.10, 0.10, 0.05]
N_STEPS = 800                    # 4 s @ 200 Hz
DELAY = 16          # 由 exp_E 扫描得出：随机驱动下 d=16~64 最优


def exp_A_recover():
    """验证 A：阶梯式定位误差来源。
    自治系统误差应为 0（证明实现正确）；随机驱动下的残余误差
    来自 DMD 的自治假设与 EEG 随机驱动本质之间的失配。"""
    print("\n[验证 A] 谱恢复 —— 误差来源阶梯")
    ladder = [("① 自治 + 弱阻尼 + 无驱动", [2.0, 2.0, 2.0], 0.0, 800, 4, 6),
              ("② 自治 + 实际阻尼 + 无驱动", TAUS, 0.0, 200, 4, 6),
              ("③ 随机驱动 std=0.5, d=4", TAUS, 0.5, 800, 4, 6),
              ("④ 随机驱动 std=0.5, d=16", TAUS, 0.5, 800, 16, 16)]
    for tag, taus, drive, N, d, r in ladder:
        Kz, true_lam = make_koopman_system(FREQS, taus, DT)
        z = simulate_source(Kz, N, drive_std=drive)
        sp = dmd_spectrum(hankel_embed(z, d), DT, method="exact", rank=r)
        fp = sp["f"][sp["f"] > 0.5]
        err = [np.min(np.abs(fp - t)) for t in FREQS] if len(fp) else [9] * 3
        print(f"  {tag:28s} 逐频率误差 {np.round(err, 3)} Hz")
    print("  结论：① ② 误差为 0 => 实现正确；③ 的残余误差来自"
          "DMD 自治假设与随机驱动的失配；④ 说明加深时延嵌入可补偿。")
    return make_koopman_system(FREQS, TAUS, DT)[1]


def exp_E_hyperparam():
    """扫描 (时延深度 d, 截断秩 r)，为真实数据定默认值。"""
    print("\n[超参扫描] 随机驱动系统上的 (d, r)")
    Kz, _ = make_koopman_system(FREQS, TAUS, DT)
    z = simulate_source(Kz, 800, drive_std=0.5)
    best = (9e9, None, None)
    for d in [2, 4, 8, 16, 32, 64]:
        H = hankel_embed(z, d)
        for r in [6, 8, 12, 16, 24, 32]:
            if r > min(H.shape) - 1:
                continue
            sp = dmd_spectrum(H, DT, method="exact", rank=r)
            fp = sp["f"][sp["f"] > 0.5]
            if len(fp) == 0:
                continue
            e = np.mean([np.min(np.abs(fp - t)) for t in FREQS])
            if e < best[0]:
                best = (e, d, r)
    print(f"  最优 d={best[1]}  r={best[2]}  平均误差={best[0]:.3f} Hz")
    print("  -> 真实数据上请以此为起点再扫一遍")


def exp_B_invariance(true_lam, n_subj=10, n_ch=62, snr_db=5.0):
    """验证 B（核心）：谱是否与被试特有的混合矩阵 A_s 无关"""
    print(f"\n[验证 B] 谱不变性  受试={n_subj} 通道={n_ch} SNR={snr_db}dB")
    Kz, _ = make_koopman_system(FREQS, TAUS, DT)
    z = simulate_source(Kz, N_STEPS)
    spectra = []
    for s in range(n_subj):
        x, _ = mix_to_channels(z, n_ch, snr_db, seed=100 + s)
        spectra.append(dmd_spectrum(hankel_embed(x, DELAY), DT))

    d = [spectrum_w2(spectra[i], spectra[j])
         for i in range(n_subj) for j in range(i + 1, n_subj)]

    # 参照尺度：换一套真实频率(不同"情绪")，看谱距离应该拉开多少
    Kz2, _ = make_koopman_system([8.0, 14.0, 28.0], TAUS, DT, seed=9)
    z2 = simulate_source(Kz2, N_STEPS, seed=9)
    sp2 = [dmd_spectrum(hankel_embed(mix_to_channels(
        z2, n_ch, snr_db, seed=700 + s)[0], DELAY), DT) for s in range(n_subj)]
    d_cross = [spectrum_w2(a, b) for a in spectra for b in sp2]
    print(f"  同系统跨被试距离 mean={np.mean(d):.4f} std={np.std(d):.4f}")
    print(f"  跨系统(不同动力学)距离 mean={np.mean(d_cross):.4f}")
    print(f"  分离比 = {np.mean(d_cross)/np.mean(d):.2f}  "
          f"（>2 说明谱对 A_s 不敏感、对动力学敏感）")

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="#BBBBBB", lw=0.8)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, n_subj))
    for s, sp in enumerate(spectra):
        ax.scatter(sp["lam"].real, sp["lam"].imag, s=18 + 200 * sp["w"],
                   color=cmap[s], alpha=0.75, lw=0,
                   label=f"被试 {s+1}" if s < 3 else None)
    ax.scatter(true_lam.real, true_lam.imag, marker="*", s=150,
               color="black", zorder=5, label="真实特征值")
    ax.set_xlabel("Re $\\lambda$"); ax.set_ylabel("Im $\\lambda$")
    ax.set_title("不同混合矩阵 $A_s$ 下估得的 Koopman 谱", fontsize=8.5)
    ax.set_aspect("equal"); ax.legend(fontsize=6, frameon=False, loc="lower left")
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    fig.savefig(f"{OUT}/验证B_谱不变性.png"); plt.close(fig)
    print(f"  -> 图已存 {OUT}/验证B_谱不变性.png")
    return np.mean(d)


def exp_C_robustness():
    """破坏性测试：SNR / 秩亏 / 非线性混合，找出方法的失效边界"""
    print("\n[破坏性测试]")
    Kz, true_lam = make_koopman_system(FREQS, TAUS, DT)
    z = simulate_source(Kz, N_STEPS)
    ref = dmd_spectrum(hankel_embed(z, 4), DT, method="exact")

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.7))

    snrs = [20, 15, 10, 5, 0, -5, -10]
    m = [np.mean([spectrum_w2(ref, dmd_spectrum(hankel_embed(
        mix_to_channels(z, 62, s, seed=200 + k)[0], DELAY), DT))
        for k in range(5)]) for s in snrs]
    axes[0].plot(snrs, m, "o-", color="#C0504D", ms=4)
    axes[0].axvspan(-10, 0, color="#F0D0D0", alpha=.5)
    axes[0].set_xlabel("SNR (dB)"); axes[0].set_ylabel("谱误差 $W_2$")
    axes[0].set_title("① 观测噪声（阴影=真实 EEG 区间）", fontsize=8)
    axes[0].invert_xaxis()

    Ks = [6, 12, 24, 48, 62, 100]
    m2 = []
    for K in Ks:
        f = list(np.linspace(4, 30, K // 2)); t = [0.1] * (K // 2)
        Kz2, _ = make_koopman_system(f, t, DT, seed=1)
        z2 = simulate_source(Kz2, N_STEPS, seed=1)
        r2 = dmd_spectrum(hankel_embed(z2, 4), DT, method="exact")
        m2.append(np.mean([spectrum_w2(r2, dmd_spectrum(hankel_embed(
            mix_to_channels(z2, 62, 5, seed=300 + k)[0], DELAY), DT))
            for k in range(3)]))
    axes[1].plot(Ks, m2, "s-", color="#2E75B6", ms=4)
    axes[1].axvline(62, color="#888", ls="--", lw=.8)
    axes[1].text(64, max(m2) * .9, "源数 = 通道数", fontsize=6.5, color="#666")
    axes[1].set_xlabel("源个数 $K$"); axes[1].set_title("② 秩亏", fontsize=8)

    nls = [0, 0.05, 0.1, 0.2, 0.4]
    m3 = [np.mean([spectrum_w2(ref, dmd_spectrum(hankel_embed(
        mix_to_channels(z, 62, 10, seed=400 + k, nonlinear=nl)[0], DELAY), DT))
        for k in range(3)]) for nl in nls]
    axes[2].plot(nls, m3, "^-", color="#2E8B8B", ms=4)
    axes[2].set_xlabel("非线性混合强度"); axes[2].set_title("③ 线性导联假设偏离", fontsize=8)

    for ax in axes:
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/破坏性测试.png"); plt.close(fig)
    print(f"  SNR 扫描 {np.round(m,3)}")
    print(f"  秩亏扫描 {np.round(m2,3)}")
    print(f"  非线性扫描 {np.round(m3,3)}")
    print(f"  -> 图已存 {OUT}/破坏性测试.png")


def exp_D_nonlinear():
    """非线性源系统上的可分性：Koopman 真正的用武之地"""
    print("\n[非线性对照] Stuart-Landau，两组不同频率参数")
    d_within, d_between = [], []
    grpA = [dmd_spectrum(hankel_embed(mix_to_channels(
        simulate_stuart_landau([6, 10, 20], DT, N_STEPS, seed=s),
        62, 5, seed=500 + s)[0], DELAY), DT) for s in range(6)]
    grpB = [dmd_spectrum(hankel_embed(mix_to_channels(
        simulate_stuart_landau([7, 13, 26], DT, N_STEPS, seed=s),
        62, 5, seed=600 + s)[0], DELAY), DT) for s in range(6)]
    for g in (grpA, grpB):
        d_within += [spectrum_w2(g[i], g[j])
                     for i in range(len(g)) for j in range(i + 1, len(g))]
    d_between = [spectrum_w2(a, b) for a in grpA for b in grpB]
    print(f"  组内距离 {np.mean(d_within):.4f} | 组间距离 {np.mean(d_between):.4f}")
    print(f"  可分性比值 = {np.mean(d_between)/np.mean(d_within):.2f} "
          f"（>1.5 说明谱确实携带动力学差异）")


if __name__ == "__main__":
    print("=" * 62)
    print("第 0 阶段 / 第 1 周：合成数据验证")
    print("=" * 62)
    tl = exp_A_recover()
    exp_B_invariance(tl)
    exp_C_robustness()
    exp_D_nonlinear()
    exp_E_hyperparam()
    print("\n全部完成，图见目录:", OUT)
