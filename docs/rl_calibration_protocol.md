# RL 控制新被试分类头适配：固定预算探索实验

## 路线收尾（2026-09-21）

当前应用可直接采用有监督标定；现有单seed、单session实验未证明RL相对普通微调具有稳定额外收益，因此停止追加RL投入，保留为探索性对照。

应用允许采集、获得标签、事后清洗后再训练或微调；影片和标签预算固定，RL 在本实验中主要控制学习率、更新步数和跳过更新。
停止追加 PPO 训练、奖励修改、动作空间搜索及 RL 扩展实验；不安排新的 RL 实验来证明其无效。
该决策不是理论否证，也不意味着有标签时不能使用 RL。

最终结果目录为 `results/calibration/rl_20260918_1536/`，15 折全部完成。
9 次反馈时的被试平均窗口准确率：

| 方法 | 窗口准确率 |
|---|---:|
| none | 47.27% |
| fixed_head | 55.88% |
| random_action | 56.23% |
| RL_policy | 56.72% |

主要比较 RL−fixed_head 为 **+0.85 个百分点**，探索性配对 95% 区间为 **[−1.64，3.34] 个百分点**。
来源：上述目录的 `summary_from_predictions.json` 与 `report_from_predictions.md`；单 seed=42、单 session=1，未证明稳定额外收益。
所有现有代码、模型、预测、源码快照和结果原位保留，不删除、不覆盖、不为归档大规模移动。
本次仅更新工作文档，结果目录内的冻结文档保持原样。

下一项限定为同标签预算下的事后批量有监督微调，方案见 [标定协议](calibration_protocol.md#batch_head-实施方案2026-09-21待实现未训练)。

## 已完成实验的冻结协议（历史记录）

以下协议固定于首次外折运行前。历史入口 `python -m src.calibration.run_rl`；已完成 15 外折，当前不再启动该路线训练。
SEED session 1，de_movingAve 310 维，MLP hidden=128、dropout=0.3，source_epochs=50，seed=42。
离线利用 SEED 标签模拟反馈；不涉及实时人体交互或在线 EEG 清洗。

## 已确认的历史问题与修复

- 原 `run.py` 每轮只传入当前 trial，未累计反馈；本轮统一累计全部已反馈 trial。
- 源训练 trial ID 在不同被试间重复；本轮用 subject*100+trial 区分。即使同长度条件下等价，也明确统计单位。
- 原 fixed_head 验证重复执行两次轨迹，日志 acc9 与用于选参的轨迹不同；改为一次连续轨迹。
- checkpoint 选择已采用 val CE 最小值、初值正无穷，计算逻辑正确；修复将 zero-based epoch 0 误报为随机初始化的日志字段。
- 现有标准化已在源训练、验证、反馈和评估统一应用，Adam 已持续保留。本轮保留行为。
- 所有方法 head 更新均 train 模式，两个 dropout 均开启；预测和独立奖励 eval 模式。所有旧分数不沿用。

## 隔离与状态

外折测试 s、验证下一个循环被试、其余 13 个训练。每个训练被试轮流作为伪新被试；
下一个循环训练被试作为内层早停验证，其余 11 个训练源 MLP 与拟合 scaler。
伪新被试不用于其源模型拟合、scaler 拟合或早停。外折验证和测试完全不参与策略训练任务。
外折源模型使用 13 个训练被试拟合，外折验证被试选最小 trial 等权 CE checkpoint。

每轮先预测当前 trial，再揭示标签，再构造状态，再选动作并更新。状态为当前 trial 平均概率(3)、
其熵(1)、反馈后计算的更新前窗口 CE(1)、累计 trial 类别比例(3)、反馈次数/9(1)、上一动作 one-hot(6，含 START)。
不含被试、影片、未来标签和独立评估结果。VecNormalize 只更新训练任务状态统计；每个策略 checkpoint 配套
保存其统计量，验证/测试冻结。反馈池 1–9，训练随机排列、验证和测试按 1–9。

## 动作、奖励与 PPO

动作 0=skip，1=(1e-4,5)，2=(1e-4,20)，3=(1e-3,5)，4=(1e-3,20)。
仅更新 fc2；一个 episode 只有一个 Adam，weight_decay=1e-4，学习率变更只修改 param_groups。
每次使用累计反馈，先窗口 CE 在各 trial 内平均，再 trial 等权。使用同一个 apply_action 更新函数。

奖励严格是伪新被试 trial 10–15 的等权窗口 CE：更新前减更新后。奖励不进入状态、不标准化。
训练算法 Stable-Baselines3 2.4.1 PPO，Gymnasium 1.0.0，MLP actor/critic 各 [64,64] Tanh。
每折 **512 个完整 9 步 episode = 4608 transitions**，每 16 个 episode 执行 PPO 更新并验证一次，共 32 次。
n_steps=144，batch_size=72，n_epochs=10，lr=3e-4，gamma=1，GAE lambda=.95，clip=.2，
ent_coef=.01，vf_coef=.5，max_grad_norm=.5，优势标准化开启，无 target_kl 截断。
不比较多算法，不以测试结果修改预算。验证目标是反馈 3/6/9 的平均 trial 准确率，首次并列获胜。
部署使用确定性 argmax 动作，策略参数冻结；评估标签只供评分，不作为测试奖励或动作输入。

## 公平比较与交付

none / fixed_head / random_action / RL_policy 共享外折源模型、scaler、顺序和评估窗口。
fixed_head 在四个非 skip 动作上按同一验证规则选择，随机方法每折固定 numpy.default_rng(42)。
各方法独立从同一 checkpoint 开始，dropout seed=42。随机种子不随验证调用污染策略训练 RNG。

报告反馈 0/3/6/9 的窗口与 trial accuracy/Macro-F1，trial 预测为窗口概率平均的 argmax。
主要比较反馈 9 时 RL−fixed_head 窗口准确率，先每名被试计算再配对汇总。
95% t 区间为未校正探索性区间；区间含零不等于等价。单 seed 不宣称稳定优势。
实际更新步数、目标更新和决策耗时、策略及伪源训练成本分别记录。

结果目录含实际源快照及 SHA256、固定配置、原始数据哈希、每个源模型及 scaler/划分对应关系、
初始/最佳/最终策略、最佳状态标准化、逐反馈状态/动作/步数、训练奖励与选模日志、
真实逐窗口概率和从概率重算的对照表。`complete.json` 只在外折完整验收后写入。

实现依据：[Stable-Baselines3 PPO 官方文档](https://stable-baselines3.readthedocs.io/en/v2.4.1/modules/ppo.html)。
