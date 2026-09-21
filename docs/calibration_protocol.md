# Calibration Experiment Protocol

> 本协议定义 SEED 逐 trial 少样本适配实验的设计、流程和报告格式。
> 协议版本：2.0，日期：2026-09-18。

## 数据

- **来源**：`E:\Python\DATA\SEED\ExtractedFeatures`，DE 特征（de_movingAve）
- **Session**：仅使用 session 1（后续可扩展）
- **特征**：逐窗口特征，shape (n_windows, 62 channels × 5 bands = 310)
- **标签**：{-1, 0, 1} → {0, 1, 2}（negative, neutral, positive）
- **trial 内结构**：每个 trial 包含 ~185–265 个窗口，所有窗口共享同一标签
- **trial 分配**：trial 1–9 为标定序列，trial 10–15 为评估集

## 划分协议

### LOSO 划分（15 折）

每折：
- **Test**：1 名被试
- **Val**：下一个循环被试（用于超参数选择）
- **Train**：剩余 13 名被试

### 折内标准化

每折在 13 名训练被试的全部窗口上拟合 StandardScaler。
验证、标定和测试数据使用同一个 scaler。
scaler 参数随 checkpoint 保存。

### 标定与评估

- 标定序列：trial 1–9（共 9 trials）
- 评估集：trial 10–15（共 6 trials）
- **反馈机制**：每完成一个 trial 的预测后，才提供其标签；一次反馈 = 一个 trial
- 反馈 0 次：模型无任何目标被试信息
- 反馈 3/6/9 次：逐步增加标定数据

## 模型

### Source Model

- **结构**：MLP 310 → Dropout(0.3) → Linear(128) → ReLU → Dropout(0.3) → Linear(3)
- **特征**：de_movingAve (62ch × 5bands = 310d)，逐窗口输入
- **训练**：13 名源被试全部窗口，epoch=50，lr=1e-3
- **损失**：逐 trial 等权 CE（先对各 trial 内窗口平均 CE，再对 trial 等权平均）
- **选择**：验证被试上最低等权 CE 的 epoch 快照

### Adaptation Methods

| 方法 | 描述 |
|------|------|
| `none` | 源模型不做任何更新，直接用于评估 |
| `fixed_head` | 冻结特征提取器（fc1），仅更新分类头（fc2），optimizer 跨反馈轮次保持状态 |

### CalibConfig

- `lr`：学习率，从 {1e-4, 1e-3} 选择
- `steps`：每个反馈 trial 的梯度更新步数，从 {5, 20} 选择
- 超参数选择依据：验证被试上第 3、6、9 次反馈后的平均验证 trial 准确率

## 连续更新流程

每名被试：
1. 从同一源 checkpoint 初始化一个 `CalibrationAdapter` 实例
2. 该实例内只初始化一次 optimizer（仅针对 fc2 参数）
3. 逐个处理 trial 1–9，每收到一次反馈，调用一次 `apply_fixed_head_continuous`
4. 调用后模型状态（含 optimizer 动量）被保留，进入下一轮
5. 第 0、3、6、9 次反馈后保存评估结果（不从源模型重新初始化）

## 损失计算（trial 等权）

各 trial 窗口共享标签 `y_trial`，对同一 trial 的窗口 CE 先求平均，再对所有已反馈 trial 等权平均：

```
L = (1/T) * sum_t mean(CE(windows_in_trial_t))
```

其中 T 是已反馈的 trial 数量，不是类别数。

## 报告指标

### 双粒度指标

| 指标 | 说明 |
|------|------|
| `window_accuracy` / `window_macro_f1` | 窗口级 accuracy / Macro-F1 |
| `trial_accuracy` / `trial_macro_f1` | trial 级：同 trial 窗口 logits 概率平均后 argmax |

### 统计单位

- 反馈预算按 trial 计（0/3/6/9）
- 统计单位为留出被试（n=15）
- 差值（fixed_head − none）以被试为配对单位报告探索性置信区间

## 报告格式

每个 run 包含：

```
results/calibration/<run_id>/
├── config.json       # 实际配置（含架构、特征类型）
├── run.log           # 运行日志
├── summary.json      # 聚合指标（trial 级 + window 级）
├── predictions.csv   # 逐被试、逐方法、逐反馈的指标
├── adaptation_curve.png
├── source_s{1..15}.pt  # 各折源模型 checkpoint（含 scaler 参数）
└── run.json          # 完整逐被试记录
```

## 已知问题（旧实现 20260918_090602_6d0a04）

旧实现（`20260918_090602_6d0a04`）存在以下问题，其结果不能与本协议对比：

1. **trial 级聚合偏差**：使用 `load_trial_flat(agg="mean")` 将每个 trial 内所有窗口平均为单一样本，导致训练和评估时一个样本 = 一个 trial（而非窗口），报告的 accuracy 实际为 6 个 trial 上的准确率而非窗口级
2. **无折内标准化**：每折未拟合 StandardScaler，直接在原始特征上训练
3. **不连续更新**：每个反馈时点从源模型重新初始化模型和 optimizer
4. **trial 等权逻辑错误**：使用类别数而非 trial 数计算等权损失
5. **超参数选择偏差**：在全部 9 个标定 trial 全部可见的情况下评估固定_head，然后选择 HP，再在测试时被试上重复此过程，造成信息泄露

本协议（v2.0）修复了以上全部问题。
