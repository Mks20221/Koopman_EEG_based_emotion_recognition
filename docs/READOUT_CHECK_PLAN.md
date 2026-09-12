# 补充读出验证方案（修订版）

**日期**：2026-09-09
**目的**：验证"逐帧编码 + 平均池化"架构是否限制了 Koopman 辅助约束的分类贡献
**性质**：有限补充实验，不改变已报告的 Gate 结论；探索性，不承诺显著结果

---

## 已知现实（基于 `results/koopman_gate/` 实际文件）

| 文件 | 内容 | 是否有 z 序列 | 是否有模型权重 |
|---|---|---|---|
| `gate_rows_fast.json`（660 条） | A/B/C/D1/D2/DE + DE+变体，被试 1-15，overlap+non_overlap，各 3 个 split_seed | ❌ 无 | ❌ 无 |
| `gate_rows_cf.json`（180 条） | C_cf/DE/DE+C_cf，被试 1-15，overlap+non_overlap，各 3 个 split_seed | ❌ 无 | ❌ 无 |
| `loso_rows_full.json`（135 条） | A/B/C/D1/DE，15 折 LOSO，non_overlap，单 seed | ❌ 无 | ❌ 无 |

**结论：所有 JSON 只保存了标量指标，没有保存 latent 序列或模型权重。**
无法做"冻结 encoder + 重新评估读出"的 replay 实验。补充实验必须重新训练。

---

## 核心问题重述

已有 Gate 2/3 在"逐帧编码 → 平均池化 → 线性分类头"架构下未检出 Koopman 贡献。
可能的解释：

1. **平均池化丢弃了帧间顺序信息**，Koopman 对时间结构的约束无法在分类层发挥作用
2. 编码器的 temporal 表示本身就没有被 Koopman 损失有效改变
3. 以上都不是，Koopman 在 EEG 分类上本身没有额外贡献

**针对解释 1 的验证**：用时序读出替代平均池化，看 Koopman 约束（C/C_cf）相对 B/D1 的优势是否出现。

---

## 实验设计

### 读出方式对照

| 读出方式 | 实现 | 是否依赖帧顺序 |
|---|---|---|
| **原始：Mean Pooling** | `z.mean(dim=1)` | ❌ 不依赖（置换不变） |
| **新：单向 GRU 末状态** | `gru(z)[:, -1, :]` | ✅ 依赖帧顺序 |

**attention-weighted mean 被排除**：按每帧内容打分再加权平均，对帧的排列不变（permutation-invariant），无法区分"顺序有用"和"内容有用"两种情况。

### 六组实验（2 读出 × 3 模型）

| 组 | 读出 | 模型 | 主比较 |
|---|---|---|---|
| G1 | Mean Pooling | B | 基线 |
| G2 | Mean Pooling | C | Koopman 线性 transition |
| G3 | Mean Pooling | D1 | 非线性 transition（参数量匹配） |
| G4 | GRU 末状态 | B | 时序读出基线 |
| G5 | GRU 末状态 | C | Koopman + 时序读出 |
| G6 | GRU 末状态 | D1 | 非线性 + 时序读出 |

**主比较**：G5−G4（C−B under GRU）和 G5−G6（C−D1 under GRU）
**辅助比较**：G5−G2（读出改变下 Koopman 贡献的变化）和 G4−G1（读出改变对基线的影响）

**归因规则**：
- 若 G5−G4 > G2−G1 **且** G5−G6 ≈ G2−G3 → 时序读出揭示了 Koopman 的潜在贡献
- 若 G4−G1 ≈ G5−G2 ≈ G6−G3（即所有模型同幅度提升）→ 收益来自时序读出本身，不能归功于 Koopman
- 若所有比较均不显著 → 不能排除解释 1，但不能据此宣称 Koopman 有贡献

---

## 协议名称（修正版）

**被试内实验**（复用原始 `exp_koopman_gate.py` 的划分逻辑）：

| 协议 | 标准化 | 帧化 | 适用 |
|---|---|---|---|
| 被试内 train-fold 标准化 | 仅用训练折统计量 | overlap / non_overlap | G1/G2/G3（Mean Pooling） |
| 被试内 train-fold 标准化 | 仅用训练折统计量 | overlap / non_overlap | G4/G5/G6（GRU，读出需训练） |

**注意**：原始 LOSO 实验使用"逐被试无标签标准化"（`exp_koopman_loso.py:71-74`，用被试全部 session 数据算统计量），与被试内协议不同。本补充实验不涉及 LOSO。

---

## 数据划分

| 项目 | 值 |
|---|---|
| 数据集 | SEED session 1 |
| 被试 | 1–15 |
| 划分 | trial-level 3折（train/val/test），split_seeds=[0,1,2] |
| 训练随机种子 | seed_train=0 |
| 帧化条件 | non_overlap（hop=frame_len=100，0.5 s 帧） |
| 段长 | 4 s |
| 分层 | 按试次分层（`stratified_sample`，与原始实验一致） |

**与原始实验完全相同的数据划分**——只替换读出方式，不改变数据划分。

---

## 模型配置

### GRU 读出头

```python
class GRUReadout(nn.Module):
    """单向 GRU，末状态分类。与 encoder 联合训练（不冻结）。"""
    def __init__(self, z_dim=32, hidden=32, n_classes=3):
        super().__init__()
        self.gru = nn.GRU(z_dim, hidden, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, z):          # z: (B, T, z_dim)
        _, h = self.gru(z)        # h: (1, B, hidden)
        return self.classifier(h.squeeze(0))
```

- **单向**：EEG 帧的物理时间顺序是因果的（当前状态只依赖过去），单向合理
- **末状态**：不用 attention 等复杂设计，避免引入额外超参搜索
- **与 encoder 联合训练**：encoder 参数在训练中也会被 Koopman 损失更新（与原始 C/D1 一致）

### 训练配置

| 项目 | 值 | 说明 |
|---|---|---|
| encoder | 原始 `FrameEncoder`（与原始实验相同结构） | 联合训练 |
| transition | 原始 `LinearKoopman`（C）或 `MLPTransition`（D1）或 None（B） | 与原始实验完全一致 |
| 读出头 | Mean Pooling 或 GRU | 同一实验内，读出头是唯一变量 |
| 分类损失 | CrossEntropyLoss | 与原始一致 |
| Koopman 损失 | `beta_koop * L_koop`（C/D1 有，B 无） | 与原始一致 |
| 重构损失 | `alpha_rec * L_rec`（B/C/D1 有，A 无） | 与原始一致 |
| 优化器 | Adam(lr=1e-3, weight_decay=1e-4) | 与原始一致 |
| epochs | 100 | 与原始一致 |
| batch | 32 | 与原始一致 |
| early stop | 验证集准确率 | 与原始一致 |
| 每被试 × 每划分 × 每模型 × 每读出 | 1 次训练 | 无随机性（seed 固定） |

---

## 分类器拟合与验证流程（一致）

**Mean Pooling 组（G1/G2/G3）**：用 `model.classifier` 的前向结果（`z.mean(1) @ W + b`），与原始实验完全一致。

**GRU 组（G4/G5/G6）**：用 `GRUreadout(model.encode_seq(x))` 的前向结果，与原始实验的验证流程一致。

**两者使用相同的 early stopping 逻辑**（验证集准确率选 best_state），确保比较的是同一优化轨迹下的最优模型。

---

## 冻结 encoder 的 LogReg 探针（探索性，非替代实验）

**用途**：作为已有表示的诊断探针，回答"encoder 学到的 temporal 表示是否包含 Koopman 相关的时间结构"。

**方法**：
1. 加载 G4/G5/G6 训练好的 checkpoint（encoder + GRU 已联合训练）
2. **冻结 encoder**（不更新参数），将 encoder 输出 `z` 序列通过 LogisticRegression 分类
3. 在同一测试集上比较 `acc_GRU_frozen-LR` vs `acc_GRU_trained-head`

**标签**：在文档和报告中显式标记为"探索性诊断探针"，不计入正式六组比较，不替代联合训练实验。

---

## 短程检查（正式实验前必须完成）

在跑完整 6 组实验之前，先用最小配置验证：

| 检查项 | 方法 | 通过标准 |
|---|---|---|
| 输入形状 | 打印 `z.shape`（应为 `(n_frames, T, 32)`） | T=7（4s/0.5s），n_frames 符合预期 |
| 梯度流通 | `loss.backward(); encoder.parameters()` 检查 `grad` 非 None | encoder 参数有梯度 |
| 数据使用范围 | 打印 `Xtr_t.shape`, `Xte_t.shape` | 形状与原始 gate_rows_fast.json 中 `n_train/n_test` 一致 |
| GRU 末状态维度 | 打印 `h.shape`（应为 `(1, B, 32)`） | hidden=32 |
| 损失量级 | 打印 `loss.item()`, `L_koop.item()`, `L_rec.item()` | 三者在同一量级（O(1)） |

**短程配置**：被试 1，split_seed=0，non_overlap，B/C/D1 各训练 5 epoch，GRU hidden=16。

**估计运行时间**：
- 完整实验：15 被试 × 3 split_seeds × 3 models × 2 readouts × 100 epochs ≈ 270,000 个 epoch
- 每 epoch 约 0.5–1 s（CPU），总计约 40–80 小时（CPU）
- **需要 GPU 加速或减少被试/划分数**

---

## 正式实验预算（需在启动前确认资源）

| 配置 | 被试 | 划分 | 模型 | 读出 | 总训练数 |
|---|---|---|---|---|---|
| 全量 | 15 | 3 | B/C/D1 | mean/GRU | 270 组训练 |
| 缩减A | 8 | 3 | B/C/D1 | mean/GRU | 144 组训练 |
| 缩减B | 15 | 1 | B/C/D1 | mean/GRU | 90 组训练 |

**建议**：先跑缩减A（8 被试），确认趋势后再决定是否扩量。

---

## 短程检查结果（已执行）

**配置**：被试 1，split_seed=0，non_overlap，B/C/D1 各 5 epoch，GRU hidden=16，GPU (CUDA)
**日期**：2026-09-09

### 输入形状验证

| 量 | 值 | 通过标准 | 结果 |
|---|---|---|---|
| 段 shape | (838, 51, 800) | — | ✅ |
| 训练/验证/测试段数 | 509 / 166 / 163 | 与 n_train/n_test 一致 | ✅ |
| 帧 T（non_overlap） | 8 | 4s / 0.5s = 8 | ✅ |
| z_tr / z_te | (509, 8, 32) / (163, 8, 32) | T=8, z_dim=32 | ✅ |
| encoder 梯度 | 非 None | encoder 参数有梯度 | ✅ |
| readout 梯度 | 非 None | readout 参数有梯度 | ✅ |
| R_learned（C/D1） | 有效数值 | B 无 transition，NaN 符合预期 | ✅ |
| 所有模型收敛 | best_epoch ≥ 0 | 非 NaN | ✅ |

### 短程数值（n=1 被试，5 epoch，不可用于结论）

| 组 | B | C | D1 | Δ_C-B |
|---|---|---|---|---|
| Mean Pooling | 0.515 | 0.479 | 0.577 | **−0.037** |
| GRU 末状态 | 0.503 | 0.577 | 0.607 | **+0.074** |
| diff(GRU−Mean) | −0.012 | **+0.098** | +0.031 | Δ_diff=**+0.110** |

**初步观察**（n=1，5 epoch，**不是结论**）：
- Mean Pooling 下 C 比 B 差（Koopman 约束反而有害）
- GRU 下 C 与 D1 几乎相同（0.577 vs 0.607），且都优于 B
- GRU 下 Koopman 约束从"有害"变成"与 D1 持平"，方向与原始 Gate 2 的趋势一致（被试内 C−B 约 +0.014，不显著）

**实际运行时间**：
- 9 组训练（3 model × 3 variant × 2 readout）× 5 epoch ≈ 3.8 s（最慢） + 6 × 0.5 s ≈ **7 s**
- 估算完整实验（270 组 × 100 epoch）：约 **30–60 min（GPU）** 或 3–5 h（CPU）

### 短程可执行性判断

✅ **形状、梯度、数据范围全部正确。**
✅ **运行时间可接受（GPU < 1 小时）。**
✅ **没有发现阻止完整实验的问题。**

---

## 主指标与停止条件（先验定义）

| 指标 | 定义 |
|---|---|
| 主指标 | `Δ_C-B_GRU = G5 − G4` 和 `Δ_C-B_mean = G2 − G1` |
| 次指标 | `Δ_diff = Δ_C-B_GRU − Δ_C-B_mean` |

**停止条件**（先验定义，不看到结果后调整）：

| 条件 | 结论 |
|---|---|
| `Δ_C-B_GRU` > 0.02，p < 0.05（配对 t，n≥8）且 `Δ_C-B_GRU > Δ_C-B_mean` | 时序读出可能揭示 Koopman 潜在贡献，值得进一步研究 |
| `Δ_C-B_GRU` 与 `Δ_C-B_mean` 无显著差异，且 G4−G1 ≈ G5−G2 ≈ G6−G3 | 所有模型同幅度变化，收益来自时序读出本身，不能归功于 Koopman |
| `Δ_C-B_GRU` 不显著（p ≥ 0.05）| 不能排除解释 1，但不能据此宣称 Koopman 有贡献 |

**效应量要求**：在 15 被试上检出 d=0.5（Δ=0.05, σ=0.10）需要 n≈27；本实验 n=15，统计力有限，需如实报告。

---

## 交付物

| 文件 | 内容 |
|---|---|
| `results/readout_check/short_run.json` | ✅ 已执行：形状/梯度/损失量级记录 |
| `results/readout_check/readout_rows_{tag}.json` | 完整实验每组 test_acc、训练时间 |
| `results/readout_check/summary.md` | 6 组均值/配对差/p 值/效应量/停止条件判断 |

---

*本方案不承诺获得显著提升。只报告在时序读出下 Koopman 约束是否有额外的分类贡献，以及该贡献是否可以归因于 Koopman 而非时序读出本身。*
