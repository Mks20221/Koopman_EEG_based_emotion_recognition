# Deep Koopman 训练流程核查报告

**日期**：2026-09-09
**范围**：`exp_koopman_gate.py`、`exp_koopman_loso.py` 及相关数据/预处理模块
**目的**：判断已有 Gate 实验是否存在可能改变结论的实现问题，并制定有限补充验证方案
**原则**：不因理论局限称其为 bug；不因没有报错宣布实现正确

---

## 核查 1：数据划分 / 标准化 / 测试信息泄漏

### 1a. trial/被试划分 → 分段 → 标准化顺序

**发现：未发现问题。**

| 步骤 | 代码位置 | 核查结果 |
|---|---|---|
| trial 级别划分（不按段随机） | `exp_koopman_gate.py:74` `split_by_trial` | ✅ 按 trial 分配 mask，同试次的段永远在同一折 |
| 标准化只来自训练折 | `exp_koopman_gate.py:88` `standardize` | ✅ `mu/sd` 仅由 `train_segs` 算，测试段从不参与 |
| 坏导选择用全局常数 | `preprocess.py:180-181` + `config.py:56-59` | ✅ `SEED_GOOD_CHANNELS` 是跨 45 次录制普查得到的固定 list，不依赖折内数据 |
| 段内去均值 | `preprocess.py:76` | ✅ 在 `segment` 内逐段逐通道去均值，不跨段 |

### 1b. LOSO 逐被试标准化的计算范围

**发现：标准化统计量的计算范围是"整个 session 的全部段"，含测试被试。**

`exp_koopman_loso.py:59-85` `load_all()`：
```python
for s in list_subjects("seed"):
    r = build_segments("seed", s, session, cfg=cfg)
    segs, y, trial = r["segs"], r["y"], r["trial"]
    # 逐被试标准化：不碰标签
    mu = segs.mean(axis=(0, 2), keepdims=True)    # axis=(0,2) = 对 batch×time 降维
    sd = segs.std(axis=(0, 2), keepdims=True) + 1e-8
    segs = (segs - mu) / sd
```

- **计算范围**：`mu` / `sd` 的 axis=(0, 2) 对应 (n_seg, C, L) 的 n_seg 和 L 维度，即**跨该被试在本次 session 的全部段、逐通道在时间维上算统计量**。
- **包含测试被试数据**：在 LOSO 划折之前，`load_all()` 已对全部 15 个被试（含测试被试）做了标准化。这意味着测试被试的段统计量（`mu_test`, `sd_test`）是在测试被试自己的全部数据上算的，不是从训练被试传来的。
- **是否使用标签**：没有使用标签（只用 `segs`，`y` 只用于 `pack()` 后的标签分配）。这是无监督标准化。

**这不是"低风险细节"，而是明确的评测协议差异：**

> **协议名称应为"逐被试无标签标准化"（unsupervised per-subject normalization），而非"训练折标准化"。**
> 与被试内实验（`standardize(train_segs, *others)` 用训练折统计量）**不是同一个协议**。
> 两者不能直接比较；任何引用"被试内 vs 跨被试"的差异必须注明用的是不同的标准化协议。
> 论文方法章节须显式声明这一点。

### 1c. 相邻帧是否跨 trial 配对

**发现：未发现问题。**

`make_frames`（`exp_koopman_gate.py:62-71`）在单个 segment 上做帧化，segment 来自 `build_segments`，每个 segment 来自单个 trial。`_pairs`（`exp_koopman_gate.py:115-117`）在 segment 内部做 `z[:, :-1]` / `z[:, 1:]` 配对，不跨 segment。

### 1d. DE 基线是否用相同划分

**发现：已确认。**

`exp_koopman_gate.py:299-301`：DE 特征在 `s_tr` / `s_te` 上算，而 `s_tr` / `s_te` 与神经网络 variant 完全相同的 trial 划分和段集合。

---

## 核查 2：Koopman 损失梯度 / 损失权重 / 相邻帧配对

### 2a. Koopman 损失是否向 encoder 传递梯度

**发现：未发现问题。**

`exp_koopman_gate.py:234-238`：
```python
if model.transition is not None:
    zp, zn = z[:, :-1, :], z[:, 1:, :]
    pred = model.transition(zp.reshape(-1, z.shape[-1])).view_as(zn)
    loss = loss + beta_koop * (((pred - zn) ** 2).sum()
                               / ((zn ** 2).sum() + 1e-8))
```
`z` 来自 `model(xb)` 的前向传播（可微），`transition` 可学，梯度流向 encoder。✅

### 2b. 损失权重配置接口

**发现：配置不完整——`alpha_rec` / `beta_koop` 无法通过命令行独立调节。**

`train_one` 签名（`exp_koopman_gate.py:172`）有 `alpha_rec=1.0, beta_koop=1.0`，但 `run_subject` 调用时（`exp_koopman_gate.py:311-312`）**没有传递这两个参数**，所有 variant 用默认值 1.0。

这意味着：
- 所有 variant（无论有没有 decoder/transition）共享同一套权重
- 如果后续需要调参，必须改 `run_subject`，无法通过 `--alpha-rec` 等命令行参数实现

**不影响历史实验结论**（历史实验用的是默认值 1.0，一致应用于所有 variant）。

### 2c. 相邻帧配对正确性

**发现：已确认。**

non_overlap（`frame_len=100, hop=100`）：相邻帧之间恰好差 100 个采样点（0.5 s），无重叠。✅

---

## 核查 3：训练 / eval 模式 / BatchNorm

### 3a. train/eval 模式切换

**发现：未发现问题。**

每个 epoch 内模式切换交替清晰（`model.train()` / `model.eval()`），验证在 `eval()` 下进行。✅

### 3b. C_cf refit_K 中 BatchNorm 处理

**发现：存在潜在风险——refit 期间 encoder 在 eval 模式收集 latent，但 BatchNorm 行为与训练期不同。**

`refit_K()`（`exp_koopman_gate.py:202-215`）：
```python
model.eval()                         # ← encoder 进入 eval 模式
with torch.no_grad():
    zs = [model.encode_seq(Xtr_t[i:i + 256]) ...]
    model.transition.refit(z_all[:, :-1, :], z_all[:, 1:, :])
model.train()                        # ← 切回 train 继续下一个 epoch
```

在 `eval()` 模式下：
- BatchNorm 的 `running_mean/var` **不会被本次 forward 更新**（因为 `model.training == False`）
- 但 `model.train()` 恢复后，**下一个 minibatch 会用旧的 running stats** 做归一化，然后更新

这会导致 refit 周期（如每 5 epoch）后的第一个 epoch 的 BatchNorm 行为略有偏差。这不是 bug，是 PyTorch BatchNorm 的已知行为，**不影响最终 checkpoint 的 state_dict**（checkpoint 里的 running stats 是 best_state 时刻的值，已经是充分训练后的值）。

### 3c. checkpoint 保存的 BatchNorm 状态

**发现：已确认。**

`best_state` 包含 `model.state_dict()` 的全部内容（含 `BatchNorm.running_mean/var`）。✅

---

## 核查 4：C_cf 的 K 重估与 checkpoint 一致性

### 4a-4c. K 重估范围、频率、buffer 保存

**发现：已确认，均无问题。**

- K 仅在全训练集 latent 上重估（不含 val/te）✅
- 重估频率每 5 epoch，与文档一致 ✅
- K 作为 buffer 随 state_dict 保存/恢复 ✅

### 4d. K 特征值提取方式

**发现：差异存在但不影响结论。**

```python
# exp_koopman_gate.py:272-278
K = (model.transition.K.weight.detach().cpu().numpy().T if variant == "C"
     else model.transition.K.detach().cpu().numpy())
```
- C：`nn.Linear` 权重转置（数学上正确）
- C_cf：直接 buffer（数学上正确）

两者数学等价，不影响分类结果或统计结论。✅

---

## 核查 5：逐被试汇总 / 种子平均 / 统计量可复现性

### 5a. 逐被试汇总方式

**发现：已确认，统计单位是被试。**

`_by_subject`（`exp_koopman_gate.py:354-360`）：先在 split_seed 上平均得到逐被试值，再做配对统计。✅

### 5b. 多 seed 汇总

**发现：默认单 seed，多 seed 时方差可能低估。**

`exp_koopman_loso.py:183-191`：默认 `train_seeds=(0,)`（单 seed）。多种子时 `acc[variant][test_subject]` 包含多个值取平均，但 `by` 字典汇总后只剩下一个值，**seed 间方差在最终报告的 std 中丢失**。

**已知的现实**：历史 LOSO 实验跑了 `train_seeds=(0,)`（单 seed），这个问题在历史结果中不存在。

### 5c. p 值和 CI 可复现性

**发现：JSON 中有足够字段可复算，但需注意统计口径。**

JSON 包含 `test_acc` 和 `variant`/`subject`/`framing`，可从 JSON 完整复算 `_by_subject` → 配对差 → t 检验。⚠️ 注意：`gate_rows_cf.json` 只有 `C_cf`，`gate_rows_fast.json` 有 `A/B/C/D1/D2/DE`，两批实验不是同时跑的，不能混用。

---

## 总体核查结论

| 核查项 | 状态 | 性质 |
|---|---|---|
| 1a. trial 划分、标准化（被试内）、坏导 | ✅ 未发现问题 | |
| 1b. LOSO 逐被试标准化 | ⚠️ **已确认的协议差异** | **不是低风险细节**——测试被试用自己的全部数据算统计量；与被试内协议不同，论文须显式声明并禁止跨协议比较 |
| 1c. 帧配对不跨 trial | ✅ 未发现问题 | |
| 1d. DE 基线用相同划分 | ✅ 未发现问题 | |
| 2a. Koopman 梯度流向 encoder | ✅ 未发现问题 | |
| 2b. 损失权重接口 | ⚠️ 配置不完整 | 无法通过命令行独立调节 alpha_rec/beta_koop（不影响历史结果） |
| 2c. 相邻帧配对 | ✅ 未发现问题 | non_overlap 无重叠 |
| 3a. train/eval 切换 | ✅ 未发现问题 | |
| 3b. C_cf refit 时 BatchNorm | ⚠️ 已知行为 | 每个 refit 周期后第一个 epoch 略有偏差，不影响 checkpoint |
| 3c. checkpoint 状态完整性 | ✅ 未发现问题 | |
| 4a. C_cf 仅用训练数据重估 K | ✅ 未发现问题 | |
| 4b. 重估频率 | ✅ 未发现问题 | 每 5 epoch |
| 4c. K buffer 随 checkpoint 保存 | ✅ 未发现问题 | |
| 4d. K 特征值提取 | ✅ 未发现问题 | C/C_cf 提取方式略有不同但均正确 |
| 5a. 逐被试汇总 | ✅ 未发现问题 | |
| 5b. 多 seed 汇总 | ⚠️ 潜在低估 | 默认单 seed，无实际问题 |
| 5c. p 值 / CI 可复算 | ⚠️ JSON 版本隔离 | gate_rows_cf.json 和 gate_rows_fast.json 不能混用 |

**没有发现实现错误足以推翻 Gate No-Go 判决。**

---

## 尚未验证的事项（需独立审查）

1. **51 通道是否与 `SEED_GOOD_CHANNELS` 一致**：需对照运行日志确认通道剔除数量
2. **LOSO 划分的 val=2 被试**：需确认 `n_val=2` 的默认值与日志一致
3. **统计量是否从 JSON 复算**：部分报告数值（如 C−DE p=0.312）来自日志而非 JSON 复算，需独立核实

---

## 短程可执行性验证（2026-09-09）

执行了 `docs/short_run_check.py`（被试 1，split_seed=0，non_overlap，B/C/D1 各 5 epoch，GRU hidden=16，CUDA）。

### 全部通过项

| 检查项 | 结果 |
|---|---|
| 输入形状 | `z_tr/z_te` = `(n, 8, 32)`，T=8（4s/0.5s）✅ |
| 梯度流通 | encoder 所有参数有非 None 梯度（ep0 batch0）✅ |
| GRU readout 梯度 | `gru.weight_ih_l0` / `classifier.weight` 梯度非 None ✅ |
| 数据范围 | train=509 / val=166 / test=163 段 ✅ |
| R_learned | B（无 transition）=NaN；C/D1 有数值 ✅ |
| 收敛 | 所有模型 best_epoch ≥ 0 ✅ |
| 运行时间 | 9 组 × 5 epoch ≈ 7 s（GPU），完整 270 组 × 100 epoch 估计 30–60 min ✅ |

### 短程数值（n=1，5 epoch，不可用于结论）

| 组 | B | C | D1 | Δ_C-B |
|---|---|---|---|---|
| Mean Pooling | 0.515 | 0.479 | 0.577 | −0.037 |
| GRU 末状态 | 0.503 | 0.577 | 0.607 | +0.074 |
| diff | −0.012 | **+0.098** | +0.031 | Δ_diff=+0.110 |

**初步观察**（n=1，不可外推）：Mean Pooling 下 C 比 B 差；GRU 下 C 的表现与 D1 接近，差异方向与原始 Gate 2 被试内结果（+0.014，不显著）一致。**不足以得出结论，但证明完整实验具有可执行的物理基础。**
