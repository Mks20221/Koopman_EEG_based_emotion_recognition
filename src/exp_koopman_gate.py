# -*- coding: utf-8 -*-
"""Deep Koopman 判决实验（Go/No-Go），不是创新点一的正式开发。

目的只有一个：回答"存不存在一个 learned Koopman latent space，使 EEG 情绪识别
比没有 Koopman 约束时更好"。**不追求最高准确率**，不加 OT / 跨被试对齐 /
shared-private / RF-SNN —— 任何额外组件都会让归因失效。

三级闸门：
  Gate 1  latent 线性度：learned K 是否明显优于 persistence / 训练集 OLS / 随机线性
  Gate 2  Koopman 贡献：C > B，且 C >= D1（参数量严格匹配的非线性对照）
  Gate 3  跨被试是否保持（LOSO，本脚本暂不做）

实验纪律（每一条都是被具体错误逼出来的）：
  * **按试次划分**，绝不按段随机划分。同试次的段来自同一段影片、时间连续，
    按段划分等于泄漏。旧 kNN 指标正是栽在这里（PREREGISTER 修订 3）。
  * 标准化只用训练折统计量（CLAUDE.md 陷阱 4）。
  * **帧重叠是 Gate 1 的头号混杂**：hop < frame_len 时相邻帧共享原始采样点，
    latent 当然线性可预测。故必须跑 overlap / non-overlap 两种条件，
    只有 non-overlap 下线性度依然成立，Gate 1 才算通过。
  * **线性度不得在测试集上重新拟合 K**：test-set OLS 是 oracle 上界，不是泛化
    指标。真正的基线是训练集拟合的 K_OLS 在测试集上的表现。
  * 非线性对照必须有参数量严格匹配的版本（D1），否则 C<D 无法归因。
  * 划分种子与训练种子分离，否则分不清差异来自"换了测试电影"还是"换了初始化"。
  * 统计单位是**被试**，不是 被试×划分 的乘积。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import RESULTS_DIR, SEED_FS
from data import list_subjects
from koopman_ae import VARIANTS, KoopmanModel
from preprocess import PreprocConfig, build_segments

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DE_BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
OUT_DIR = os.path.join(RESULTS_DIR, "koopman_gate")

# 帧化条件。overlap 是主实验，non_overlap 是 Gate 1 的必要控制：
# hop=frame_len 时相邻帧不共享任何原始采样点，latent 的线性可预测性
# 不可能来自输入重叠。
FRAMINGS = {
    "overlap": dict(frame_len=100, hop=50),      # 0.5s 帧 / 0.25s 跳步，50% 重叠
    "non_overlap": dict(frame_len=100, hop=100),  # 0.5s 帧 / 0.5s 跳步，0% 重叠
}


# ==========================================================================
# 数据
# ==========================================================================
def make_frames(segs, frame_len=100, hop=50):
    """(N, C, L) -> (N, T, C, frame_len)。@200Hz。

    frame_len 决定每个 z 能看到的频率下限（0.5s -> 2Hz），hop 决定 latent 采样率。
    hop < frame_len 会让相邻帧共享原始采样点 —— 见模块 docstring 的警告。
    """
    N, C, L = segs.shape
    T = (L - frame_len) // hop + 1
    idx = np.arange(T) * hop
    return np.stack([segs[:, :, i:i + frame_len] for i in idx], axis=1)


def split_by_trial(y, trial, seed=0):
    """每类 5 个试次 -> 3 train / 1 val / 1 test。返回段级布尔掩码。"""
    rng = np.random.default_rng(seed)
    masks = [np.zeros(len(y), bool) for _ in range(3)]
    for c in np.unique(y):
        perm = rng.permutation(np.unique(trial[y == c]))
        n = len(perm)
        n_tr = max(1, int(round(n * 0.6)))
        n_va = max(1, int(round(n * 0.2))) if n - n_tr >= 2 else 0
        for m, t in zip(masks, (perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:])):
            m |= np.isin(trial, t) & (y == c)
    return masks


def standardize(train_segs, *others):
    """逐通道 z-score，统计量**只从训练折算**（CLAUDE.md 陷阱 4）。"""
    mu = train_segs.mean(axis=(0, 2), keepdims=True)
    sd = train_segs.std(axis=(0, 2), keepdims=True) + 1e-8
    return [(s - mu) / sd for s in (train_segs,) + others]


def de_features(segs, fs=SEED_FS, bands=DE_BANDS):
    n, C, L = segs.shape
    nyq = fs / 2.0
    out = np.zeros((n, C, len(bands)), dtype=np.float64)
    for bi, (lo, hi) in enumerate(bands):
        b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
        v = filtfilt(b, a, segs, axis=-1).var(axis=-1)
        out[:, :, bi] = 0.5 * np.log(2 * np.pi * np.e * np.maximum(v, 1e-12))
    return out.reshape(n, -1)


def fit_logreg(f_tr, y_tr, f_te, y_te):
    sc = StandardScaler().fit(f_tr)
    clf = LogisticRegression(max_iter=3000).fit(sc.transform(f_tr), y_tr)
    return float(clf.score(sc.transform(f_te), y_te))


# ==========================================================================
# 判决指标
# ==========================================================================
def _pairs(z):
    d = z.shape[-1]
    return z[:, :-1, :].reshape(-1, d), z[:, 1:, :].reshape(-1, d)


def linearity_metrics(z_te, z_tr):
    """latent 线性可预测性（越小越线性）。R = ||z+ - P(z-)||²_F / ||z+||²_F。

    四个对照，缺一不可：
      persistence  P = I                  平凡基线，什么都不学
      ols_train    P = 训练集拟合的 K_OLS  **真正的泛化基线**
      test_oracle  P = 测试集拟合的 K_OLS  上界，不是泛化指标，只说明"这段轨迹
                                          原则上有多线性"，绝不可当成绩报告
      random       P = 随机正交矩阵        排除"任何矩阵乘上去都差不多"
    """
    Zm_te, Zp_te = _pairs(z_te)
    Zm_tr, Zp_tr = _pairs(z_tr)
    denom = float(np.sum(Zp_te ** 2)) + 1e-12
    d = z_te.shape[-1]

    K_tr = np.linalg.lstsq(Zm_tr, Zp_tr, rcond=None)[0]
    K_te = np.linalg.lstsq(Zm_te, Zp_te, rcond=None)[0]
    rng = np.random.default_rng(0)
    Qr = np.linalg.qr(rng.standard_normal((d, d)))[0]

    lam = np.linalg.eigvals(K_tr)
    return dict(
        R_lin_persistence=float(np.sum((Zp_te - Zm_te) ** 2)) / denom,
        R_lin_ols_train=float(np.sum((Zp_te - Zm_te @ K_tr) ** 2)) / denom,
        R_lin_test_oracle=float(np.sum((Zp_te - Zm_te @ K_te) ** 2)) / denom,
        R_lin_random=float(np.sum((Zp_te - Zm_te @ Qr) ** 2)) / denom,
        olsK_eig_absmax=float(np.abs(lam).max()),
        olsK_eig_absmean=float(np.abs(lam).mean()))


def latent_health(z):
    """latent 塌缩监控。

    temporal_var_frac -> 0 说明 z 在段内几乎不随时间变化，"动力学"退化成静态
    特征，此时 Koopman 约束被平凡满足（K≈I），没有意义。
    low_freq_frac -> 1 说明编码器为了好预测只保留慢漂移。
    """
    var_t = z.var(axis=1).mean(axis=0)
    var_all = z.reshape(-1, z.shape[-1]).var(axis=0)
    zc = z - z.mean(axis=1, keepdims=True)
    Pm = (np.abs(np.fft.rfft(zc, axis=1)) ** 2).mean(axis=(0, 2))
    ev = np.sort(var_all)[::-1]
    return dict(temporal_var_frac=float(np.mean(var_t / (var_all + 1e-12))),
                low_freq_frac=float(Pm[:2].sum() / (Pm.sum() + 1e-12)),
                effective_dim=float((ev.sum() ** 2) / (np.sum(ev ** 2) + 1e-12)),
                latent_var_min=float(var_all.min()),
                latent_var_max=float(var_all.max()))


# ==========================================================================
# 训练
# ==========================================================================
def train_one(variant, data, z_dim=32, epochs=100, lr=1e-3, batch=32,
              alpha_rec=1.0, beta_koop=1.0, seed_train=0, refit_every=5):
    """训练一个 variant，返回 test 指标 + latent 诊断 + 池化后的 latent。

    三个损失都归一化到 O(1)，使权重可解释、各组可比：
      L_ce   交叉熵
      L_rec  MSE / var(x)              相对重构误差
      L_koop ||z+ - F(z-)||²/||z+||²   相对一步预测误差
    多步预测损失暂设 0：EEG 是随机驱动系统，长 horizon 预测有原理上的天花板，
    硬压会逼 latent 退化成慢漂移（CLAUDE.md 陷阱 1）。
    """
    torch.manual_seed(seed_train)
    np.random.seed(seed_train)
    Xtr, ytr, Xva, yva, Xte, yte = data
    n_ch, frame_len = Xtr.shape[2], Xtr.shape[3]

    model = KoopmanModel(variant, n_ch, frame_len, z_dim=z_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    def to_t(a, d=torch.float32):
        return torch.as_tensor(a, dtype=d, device=DEVICE)

    Xtr_t, ytr_t = to_t(Xtr), to_t(ytr, torch.long)
    Xva_t, yva_t = to_t(Xva), to_t(yva, torch.long)
    Xte_t, yte_t = to_t(Xte), to_t(yte, torch.long)

    best_va, best_state, best_ep = -1.0, None, -1
    n = len(Xtr_t)

    def refit_K():
        """交替优化的另一半：固定 encoder，在全训练集 latent 上闭式重解 ridge K。

        必须用全训练集而不是 minibatch —— K 是全局算子，用 batch 解会让它
        随 batch 抖动，等价于给 encoder 一个噪声目标。
        """
        model.eval()
        with torch.no_grad():
            zs = [model.encode_seq(Xtr_t[i:i + 256]) for i in range(0, n, 256)]
            z_all = torch.cat(zs, dim=0)
            d = z_all.shape[-1]
            model.transition.refit(z_all[:, :-1, :].reshape(-1, d),
                                   z_all[:, 1:, :].reshape(-1, d))
        model.train()

    if variant == "C_cf":
        refit_K()

    for ep in range(epochs):
        if variant == "C_cf" and ep > 0 and ep % refit_every == 0:
            refit_K()
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            z, logits = model(xb)
            loss = ce(logits, yb)
            if model.decoder is not None:
                B, T = xb.shape[0], xb.shape[1]
                xr = model.decoder(z.reshape(B * T, -1)).view(B, T, *xb.shape[2:])
                loss = loss + alpha_rec * (((xr - xb) ** 2).mean() / (xb.var() + 1e-8))
            if model.transition is not None:
                zp, zn = z[:, :-1, :], z[:, 1:, :]
                pred = model.transition(zp.reshape(-1, z.shape[-1])).view_as(zn)
                loss = loss + beta_koop * (((pred - zn) ** 2).sum()
                                           / ((zn ** 2).sum() + 1e-8))
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va_acc = (model(Xva_t)[1].argmax(1) == yva_t).float().mean().item()
        if va_acc > best_va:
            best_va, best_ep = va_acc, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z_te, logit_te = model(Xte_t)
        te_acc = (logit_te.argmax(1) == yte_t).float().mean().item()
        z_tr = model(Xtr_t)[0]
    z_te_np, z_tr_np = z_te.cpu().numpy(), z_tr.cpu().numpy()

    out = dict(variant=variant, seed_train=seed_train, test_acc=te_acc,
               val_acc=best_va, best_epoch=best_ep,
               n_transition_params=model.n_transition_params())
    out.update(linearity_metrics(z_te_np, z_tr_np))
    out.update(latent_health(z_te_np))
    out["linear_probe_acc"] = fit_logreg(z_tr_np.mean(1), ytr, z_te_np.mean(1), yte)

    if model.transition is not None:      # 学到的 transition 自己的一步预测误差
        with torch.no_grad():
            zp = z_te[:, :-1, :].reshape(-1, z_dim)
            zn = z_te[:, 1:, :].reshape(-1, z_dim)
            pred = model.transition(zp)
            out["R_lin_learned"] = float((((pred - zn) ** 2).sum()
                                          / ((zn ** 2).sum() + 1e-12)).item())
    if variant in ("C", "C_cf"):
        K = (model.transition.K.weight.detach().cpu().numpy().T if variant == "C"
             else model.transition.K.detach().cpu().numpy())
        lam = np.linalg.eigvals(K)
        out["learnedK_eig_absmax"] = float(np.abs(lam).max())
        out["learnedK_eig_absmean"] = float(np.abs(lam).mean())
        out["learnedK_n_oscillatory"] = int(np.sum(np.abs(lam.imag) > 1e-6))
    return out, z_tr_np.mean(1), z_te_np.mean(1)


# ==========================================================================
# 每个被试 × 每种帧化条件 跑完整对照
# ==========================================================================
def run_subject(subject, session=1, split_seeds=(0, 1, 2), train_seed=0,
                variants=None,
                framings=("overlap", "non_overlap"), win_s=4.0, thr=15.0,
                z_dim=32, epochs=100, verbose=True):
    cfg = PreprocConfig(win_s=win_s, reject_abs_thr=thr)
    r = build_segments("seed", subject, session, cfg=cfg)
    segs, y, trial = r["segs"], r["y"], r["trial"]

    rows = []
    for sp in split_seeds:
        tr_m, va_m, te_m = split_by_trial(y, trial, seed=sp)
        s_tr, s_va, s_te = standardize(segs[tr_m], segs[va_m], segs[te_m])
        ytr, yva, yte = y[tr_m], y[va_m], y[te_m]

        # DE 基线：与所有神经网络 variant 完全相同的划分、段、评估
        f_tr, f_te = de_features(s_tr), de_features(s_te)
        de_acc = fit_logreg(f_tr, ytr, f_te, yte)

        for fr_name in framings:
            fp = FRAMINGS[fr_name]
            Xtr, Xva, Xte = (make_frames(s, **fp) for s in (s_tr, s_va, s_te))
            data = (Xtr, ytr, Xva, yva, Xte, yte)

            zs = {}
            for variant in (variants or VARIANTS):
                t0 = time.time()
                res, ztr_p, zte_p = train_one(variant, data, z_dim=z_dim,
                                              epochs=epochs, seed_train=train_seed)
                res.update(subject=subject, session=session, split_seed=sp,
                           framing=fr_name, de_acc=de_acc,
                           elapsed=time.time() - t0,
                           n_train=int(tr_m.sum()), n_test=int(te_m.sum()),
                           n_frames=int(Xtr.shape[1]))
                zs[variant] = (ztr_p, zte_p)
                rows.append(res)
                if verbose:
                    print(f"  s{subject:02d} sp{sp} {fr_name[:3]} {variant:>2}: "
                          f"acc={res['test_acc']:.3f} "
                          f"probe={res['linear_probe_acc']:.3f} "
                          f"Rlearn={res.get('R_lin_learned', float('nan')):.3f} "
                          f"Rols={res['R_lin_ols_train']:.3f} "
                          f"Rpers={res['R_lin_persistence']:.3f} "
                          f"tvar={res['temporal_var_frac']:.2f} "
                          f"({res['elapsed']:.0f}s)", flush=True)

            # DE + 每个 variant 的 latent。关键对比是 (DE+z_C) - (DE+z_B)：
            # 只有它能把"Koopman 约束的贡献"与"CNN encoder 本身的贡献"分开。
            for variant, (ztr_p, zte_p) in zs.items():
                acc = fit_logreg(np.concatenate([f_tr, ztr_p], 1), ytr,
                                 np.concatenate([f_te, zte_p], 1), yte)
                rows.append(dict(variant=f"DE+{variant}", subject=subject,
                                 session=session, split_seed=sp, framing=fr_name,
                                 test_acc=acc, de_acc=de_acc))
            rows.append(dict(variant="DE", subject=subject, session=session,
                             split_seed=sp, framing=fr_name, test_acc=de_acc,
                             de_acc=de_acc))
            if verbose:
                print(f"  s{subject:02d} sp{sp} {fr_name[:3]} DE={de_acc:.3f}  "
                      + "  ".join(f"DE+{v}={zs_acc:.3f}" for v, zs_acc in
                                    ((v, [x for x in rows if x['variant'] == f'DE+{v}'
                                          and x['split_seed'] == sp
                                          and x['framing'] == fr_name][-1]['test_acc'])
                                     for v in (variants or VARIANTS))), flush=True)
    return rows


# ==========================================================================
# 汇总：统计单位是被试
# ==========================================================================
def _by_subject(rows, variant, framing, key="test_acc"):
    """先在被试内对多次划分求平均，再返回逐被试的值。统计单位是被试。"""
    per_subj = collections.defaultdict(list)
    for r in rows:
        if r["variant"] == variant and r.get("framing") == framing and key in r:
            per_subj[r["subject"]].append(r[key])
    return np.array([np.mean(v) for _, v in sorted(per_subj.items())])


def summarize(rows, framings):
    for fr in framings:
        print("\n" + "=" * 84)
        print(f"帧化条件：{fr}"
              + ("（相邻帧共享 50% 原始采样点 —— 线性度可能是假象）"
                 if fr == "overlap" else "（相邻帧无共享采样点 —— Gate 1 的真正检验）"))
        print("=" * 84)

        print(f"{'模型':<14}{'test_acc(被试均值±std)':>26}{'linear_probe':>16}"
              f"{'R_learned':>12}{'tvar':>8}")
        print("-" * 84)
        for v in list(VARIANTS) + ["DE"] + [f"DE+{x}" for x in VARIANTS]:
            a = _by_subject(rows, v, fr)
            if len(a) == 0:
                continue
            line = f"{v:<14}{a.mean():>17.3f}±{a.std():<8.3f}"
            p = _by_subject(rows, v, fr, "linear_probe_acc")
            rl = _by_subject(rows, v, fr, "R_lin_learned")
            tv = _by_subject(rows, v, fr, "temporal_var_frac")
            if len(p):
                line += f"{p.mean():>13.3f}   "
                line += f"{rl.mean():>9.3f}" if len(rl) else " " * 12
                line += f"{tv.mean():>8.2f}" if len(tv) else ""
            print(line)

        print(f"\n--- Gate 1 @ {fr}：latent 线性度（越小越线性，n={len(_by_subject(rows,'C',fr))} 被试）---")
        for k in ["R_lin_learned", "R_lin_ols_train", "R_lin_persistence",
                  "R_lin_test_oracle", "R_lin_random"]:
            a = _by_subject(rows, "C", fr, k)
            if len(a):
                note = "  <- oracle 上界，非泛化指标" if k == "R_lin_test_oracle" else ""
                print(f"  C.{k:<22}{a.mean():.4f} ± {a.std():.4f}{note}")

        print(f"\n--- Gate 2 @ {fr}：Koopman 是否有独立贡献（配对差，单位=被试）---")
        b = _by_subject(rows, "B", fr)
        c = _by_subject(rows, "C", fr)
        if len(b) and len(c) and len(b) == len(c):
            dif = c - b
            print(f"  C - B  = {dif.mean():+.4f} ± {dif.std():.4f}"
                  f"  (>0 才说明 Koopman 约束有用)")
        for dv in ["D1", "D2"]:
            d = _by_subject(rows, dv, fr)
            if len(d) and len(c) == len(d):
                dif = c - d
                note = ("参数量严格相等，这才是干净的线性vs非线性"
                        if dv == "D1" else "D2 参数更多，C<D2 不足以归因")
                print(f"  C - {dv} = {dif.mean():+.4f} ± {dif.std():.4f}  ({note})")
        dek_b = _by_subject(rows, "DE+B", fr)
        dek_c = _by_subject(rows, "DE+C", fr)
        if len(dek_b) and len(dek_c) and len(dek_b) == len(dek_c):
            dif = dek_c - dek_b
            print(f"  (DE+C)-(DE+B) = {dif.mean():+.4f} ± {dif.std():.4f}"
                  f"  (Koopman 在 DE 之外的独立信息)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--split-seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--framings", type=str, nargs="*",
                    default=["overlap", "non_overlap"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--z-dim", type=int, default=32)
    ap.add_argument("--variants", type=str, nargs="*", default=None,
                    help="只跑指定 variant（默认全跑）。用于补跑新增对照，"
                         "因给定 (被试,划分种子,训练种子,帧化) 后流程确定，"
                         "补跑结果与既有结果同协议可比。")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    subs = args.subjects if args.subjects else list_subjects("seed")
    print(f"device={DEVICE}  被试={subs}  session={args.session}")
    print(f"划分种子={args.split_seeds}  训练种子={args.train_seed}（已分离）")
    print(f"帧化={args.framings}  epochs={args.epochs}  z_dim={args.z_dim}\n")

    all_rows = []
    for s in subs:
        print(f"=== 被试 {s} ===", flush=True)
        all_rows += run_subject(s, session=args.session,
                                split_seeds=tuple(args.split_seeds),
                                train_seed=args.train_seed,
                                framings=tuple(args.framings),
                                variants=tuple(args.variants) if args.variants else None,
                                z_dim=args.z_dim, epochs=args.epochs)
        path = os.path.join(OUT_DIR, f"gate_rows{args.tag}.json")
        with open(path, "w", encoding="utf-8") as fh:   # 每个被试后落盘，防中断丢失
            json.dump(all_rows, fh, ensure_ascii=False, indent=2)

    print(f"\n原始结果已存 {path}")
    summarize(all_rows, args.framings)


if __name__ == "__main__":
    main()
