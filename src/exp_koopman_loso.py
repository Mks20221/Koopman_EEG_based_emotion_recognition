# -*- coding: utf-8 -*-
"""Gate 3：最小 subject-disjoint（LOSO）判决实验。

被试内的 Gate 2 已判 FAIL，但那**不能**推出整条 Deep Koopman 线死亡 ——
课题的真问题是跨被试，而 A/B/C/D 全部是 subject-specific train / same-subject test。
一个正则项完全可能不提升训练域可分性，却改善未见被试的泛化：

    它的作用不是增加 discriminability，而是削掉一部分 subject-specific 自由度。

因此 Gate 3 的状态是 UNKNOWN 而非 FAIL，必须单独测。

只跑 4 个 variant（A / B / C / D1），只看两个量：

    C - B    Koopman 约束在跨被试下有没有用
    C - D1   这个用处是不是"线性 Koopman"特有的（D1 参数量严格相等）

**不加** OT / shared-private / 对抗 / 多步预测 —— 这是判决实验不是开发。

三种可能的结局（跑之前就写死，避免看完结果再挑解释）：
  结局1  C-B≈0 且 C-D1≈0        -> 被试内没有、跨被试也没有，关闭 Deep Koopman 主线
  结局2  C-B 明显 >0 且 C-D1>0   -> Koopman 是 structural regularizer，改善未见被试泛化，
                                    这才击中课题，值得投入
  结局3  C-B>0 但 C≈D1          -> 时序正则有用，但线性 Koopman 不是关键，
                                    放弃"Koopman 为核心创新"的包装

跨被试特有的两个设计决策：
  * **逐被试标准化**：每个被试用自己的段统计量做 z-score。全程不碰标签，
    不构成标签泄漏；若改用训练被试的全局统计量，测试被试的整体幅度差异
    （即 A_s 的一部分）会直接压垮模型，测到的将是"幅度对不齐"而不是"泛化"。
    这是跨被试 EEG 的通行做法，但必须显式声明它用到了测试被试的无标签数据。
  * **验证集必须也是 subject-disjoint**：从训练被试里再留 2 个做验证选 checkpoint，
    否则 checkpoint 选择本身就在测试被试上过拟合。
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import RESULTS_DIR
from data import list_subjects
from exp_koopman_gate import FRAMINGS, de_features, make_frames
from koopman_ae import KoopmanModel
from preprocess import PreprocConfig, build_segments

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(RESULTS_DIR, "koopman_gate")
LOSO_VARIANTS = ("A", "B", "C", "D1")


def load_all(session=1, win_s=4.0, thr=15.0, n_per_subject=400, seed=0):
    """载入全部被试，逐被试标准化，并按类分层下采样到 n_per_subject 段。

    下采样是为了让 12 被试的训练集控制在几千段量级，使 60 折 LOSO 在可接受
    时间内跑完；分层保证类别比例不被改变。
    """
    rng = np.random.default_rng(seed)
    cfg = PreprocConfig(win_s=win_s, reject_abs_thr=thr)
    out = {}
    for s in list_subjects("seed"):
        r = build_segments("seed", s, session, cfg=cfg)
        segs, y, trial = r["segs"], r["y"], r["trial"]
        # 逐被试标准化：不碰标签
        mu = segs.mean(axis=(0, 2), keepdims=True)
        sd = segs.std(axis=(0, 2), keepdims=True) + 1e-8
        segs = (segs - mu) / sd
        if n_per_subject and len(segs) > n_per_subject:
            keep = []
            per_c = n_per_subject // len(np.unique(y))
            for c in np.unique(y):
                idx_c = np.where(y == c)[0]
                keep.append(rng.choice(idx_c, size=min(per_c, len(idx_c)),
                                       replace=False))
            keep = np.sort(np.concatenate(keep))
            segs, y, trial = segs[keep], y[keep], trial[keep]
        out[s] = dict(segs=segs, y=y, trial=trial)
    return out


def train_loso(variant, tr, va, te, z_dim=32, epochs=40, lr=1e-3, batch=128,
               alpha_rec=1.0, beta_koop=1.0, seed_train=0):
    """数据常驻 CPU、按 batch 搬上 GPU —— 12 被试的帧张量放不进 8GB 显存。"""
    torch.manual_seed(seed_train)
    np.random.seed(seed_train)
    Xtr, ytr = tr
    n_ch, frame_len = Xtr.shape[2], Xtr.shape[3]

    model = KoopmanModel(variant, n_ch, frame_len, z_dim=z_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    Xtr_c = torch.as_tensor(Xtr, dtype=torch.float32)
    ytr_c = torch.as_tensor(ytr, dtype=torch.long)

    @torch.no_grad()
    def evaluate(X, y):
        model.eval()
        correct, zs = 0, []
        for i in range(0, len(X), 256):
            xb = torch.as_tensor(X[i:i + 256], dtype=torch.float32, device=DEVICE)
            z, lg = model(xb)
            correct += (lg.argmax(1).cpu().numpy() == y[i:i + 256]).sum()
            zs.append(z.cpu().numpy())
        return correct / len(y), np.concatenate(zs)

    best_va, best_state, best_ep = -1.0, None, -1
    n = len(Xtr_c)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = Xtr_c[idx].to(DEVICE, non_blocking=True)
            yb = ytr_c[idx].to(DEVICE, non_blocking=True)
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

        va_acc, _ = evaluate(*va)
        if va_acc > best_va:
            best_va, best_ep = va_acc, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    te_acc, z_te = evaluate(*te)
    tr_acc, z_tr = evaluate(Xtr, ytr)

    # 冻结 latent 上的线性探针，看 representation 本身跨被试携带多少情绪信息
    sc = StandardScaler().fit(z_tr.mean(1))
    probe = LogisticRegression(max_iter=2000).fit(sc.transform(z_tr.mean(1)), ytr)
    probe_acc = float(probe.score(sc.transform(z_te.mean(1)), te[1]))

    return dict(variant=variant, test_acc=float(te_acc), val_acc=float(best_va),
                train_acc=float(tr_acc), best_epoch=best_ep,
                linear_probe_acc=probe_acc,
                n_transition_params=model.n_transition_params())


def run_fold(test_subj, data, framing="non_overlap", variants=LOSO_VARIANTS,
             z_dim=32, epochs=40, train_seeds=(0,), n_val=2, verbose=True):
    subs = sorted(data)
    others = [s for s in subs if s != test_subj]
    rng = np.random.default_rng(1000 + test_subj)
    val_subs = list(rng.choice(others, size=n_val, replace=False))
    tr_subs = [s for s in others if s not in val_subs]

    fp = FRAMINGS[framing]

    def pack(ss):
        X = np.concatenate([make_frames(data[s]["segs"], **fp) for s in ss])
        y = np.concatenate([data[s]["y"] for s in ss])
        return X, y

    tr, va, te = pack(tr_subs), pack(val_subs), pack([test_subj])

    # DE 基线，完全相同的被试划分
    de_tr = np.concatenate([de_features(data[s]["segs"]) for s in tr_subs])
    de_te = de_features(data[test_subj]["segs"])
    sc = StandardScaler().fit(de_tr)
    de_clf = LogisticRegression(max_iter=3000).fit(sc.transform(de_tr), tr[1])
    de_acc = float(de_clf.score(sc.transform(de_te), te[1]))

    rows = []
    for st in train_seeds:
        for v in variants:
            t0 = time.time()
            r = train_loso(v, tr, va, te, z_dim=z_dim, epochs=epochs, seed_train=st)
            r.update(test_subject=test_subj, framing=framing, de_acc=de_acc,
                     seed_train=st, n_train=len(tr[0]), n_test=len(te[0]),
                     train_subjects=tr_subs, val_subjects=[int(x) for x in val_subs],
                     elapsed=time.time() - t0)
            rows.append(r)
            if verbose:
                print(f"  LOSO test=s{test_subj:02d} ts{st} {v:>2}: "
                      f"acc={r['test_acc']:.3f} probe={r['linear_probe_acc']:.3f} "
                      f"val={r['val_acc']:.3f} train={r['train_acc']:.3f} "
                      f"({r['elapsed']:.0f}s)", flush=True)
    rows.append(dict(variant="DE", test_subject=test_subj, framing=framing,
                     test_acc=de_acc, de_acc=de_acc))
    if verbose:
        print(f"  LOSO test=s{test_subj:02d} DE={de_acc:.3f}", flush=True)
    return rows


def summarize(rows):
    from scipy import stats
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        acc[r["variant"]][r["test_subject"]].append(r["test_acc"])
    by = {v: {s_: float(np.mean(x)) for s_, x in d.items()} for v, d in acc.items()}

    print("\n" + "=" * 72)
    print("Gate 3：跨被试（LOSO）结果  统计单位=留出被试")
    print("=" * 72)
    print(f"{'模型':<8}{'test_acc':>18}{'linear_probe':>16}")
    print("-" * 72)
    for v in list(LOSO_VARIANTS) + ["DE"]:
        if v not in by:
            continue
        a = np.array(list(by[v].values()))
        line = f"{v:<8}{a.mean():>10.3f}±{a.std():<6.3f}"
        pr = [r["linear_probe_acc"] for r in rows
              if r["variant"] == v and "linear_probe_acc" in r]
        if pr:
            line += f"{np.mean(pr):>10.3f}±{np.std(pr):<5.3f}"
        print(line)

    print("\n--- 配对检验（n = 留出被试数）---")
    for a, b in [("C", "B"), ("C", "D1"), ("C", "A"), ("B", "A"), ("C", "DE")]:
        if a not in by or b not in by:
            continue
        subs = sorted(set(by[a]) & set(by[b]))
        da = np.array([by[a][s] for s in subs])
        db = np.array([by[b][s] for s in subs])
        d = da - db
        t, p = stats.ttest_rel(da, db)
        try:
            _, pw = stats.wilcoxon(da, db)
        except Exception:
            pw = float("nan")
        n_pos = int((d > 0).sum())
        p_sign = stats.binomtest(n_pos, len(d), 0.5).pvalue
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {a} - {b}: {d.mean():+.4f} [95%CI {d.mean()-ci:+.4f},{d.mean()+ci:+.4f}]  "
              f"t={t:.2f} p={p:.3f} wilcox={pw:.3f} {n_pos}/{len(d)} p_sign={p_sign:.3f} "
              f"dz={d.mean()/(d.std(ddof=1)+1e-12):.2f}")

    print("\n--- 结局判定（判据在跑之前已写死，见模块 docstring）---")
    if "C" in by and "B" in by and "D1" in by:
        subs = sorted(by["C"])
        cb = np.mean([by["C"][s] - by["B"][s] for s in subs])
        cd = np.mean([by["C"][s] - by["D1"][s] for s in subs])
        _, p_cb = stats.ttest_rel([by["C"][s] for s in subs],
                                  [by["B"][s] for s in subs])
        if p_cb < 0.05 and cb > 0.02 and cd > 0:
            print("  -> 结局2：Koopman 作为 structural regularizer 改善跨被试泛化")
        elif p_cb < 0.05 and cb > 0.02:
            print("  -> 结局3：时序正则有用，但线性 Koopman 不是关键")
        else:
            print("  -> 结局1：跨被试同样无收益 -> 支持关闭 Deep Koopman 主线")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-per-subject", type=int, default=400)
    ap.add_argument("--framing", type=str, default="non_overlap")
    ap.add_argument("--test-subjects", type=int, nargs="*", default=None)
    ap.add_argument("--variants", type=str, nargs="*", default=list(LOSO_VARIANTS))
    ap.add_argument("--train-seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"device={DEVICE}  session={args.session}  framing={args.framing}")
    print(f"每被试下采样至 {args.n_per_subject} 段  epochs={args.epochs}\n")

    data = load_all(session=args.session, n_per_subject=args.n_per_subject)
    print(f"载入完成，各被试段数：{ {s: len(d['y']) for s, d in data.items()} }\n")

    test_subs = args.test_subjects or sorted(data)
    rows = []
    for ts in test_subs:
        print(f"=== LOSO fold: 留出被试 {ts} ===", flush=True)
        rows += run_fold(ts, data, framing=args.framing,
                         variants=tuple(args.variants), epochs=args.epochs,
                         train_seeds=tuple(args.train_seeds))
        path = os.path.join(OUT_DIR, f"loso_rows{args.tag}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=2)
    print(f"\n结果已存 {path}")
    summarize(rows)


if __name__ == "__main__":
    main()
