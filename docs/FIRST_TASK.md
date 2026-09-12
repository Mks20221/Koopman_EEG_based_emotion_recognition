# 交给 Claude Code 的第一个任务

把下面这段直接粘给 Claude Code（它会先自动读 CLAUDE.md）。

---

## 粘贴内容

请先阅读 `CLAUDE.md` 和 `docs/ROADMAP.md` 了解项目背景，
再阅读 `src/koopman_stage0_synth.py` 了解已验证的 DMD 管线。

现在开始第 0 阶段第 2 周的工作。**注意执行顺序**：ROADMAP 里第 5 项
（可分性检验）是风险最高的一项，要最先做，不要按编号顺序。

本次任务分三步，每步做完让我确认后再继续：

### 第 1 步：数据接口

在 `src/data.py` 实现统一接口：

```python
def load_trials(dataset, subject, session):
    """
    返回 (X, y, fs)
      X : list of (n_channels, n_samples)，试次长度不定
      y : (n_trials,) int，统一为 0..C-1
      fs: 采样率
    """
```

先只实现 SEED。要点见 CLAUDE.md 的"数据集"一节，特别注意：
- `.mat` 有 v7.3 和更早两种，需要 scipy + h5py 双路径
- h5py 读出要转置
- 变量名前缀随被试变化，用正则 `_eeg\d+$` 匹配
- 标签 {-1,0,1} 要映射到 {0,1,2}

我的 SEED 路径是：<在这里填你的实际路径>

写完后请抽查 2–3 个文件，打印 shape 和标签分布让我核对。

### 第 2 步：预处理 + 缓存层

在 `src/preprocess.py` 和 `src/cache.py` 实现：
- 带通 1–50 Hz（用 MNE）、必要时重采样、4 s 无重叠切段
- 分层缓存，缓存键含所有影响结果的参数 + code_version
- 批量处理带断点续传

### 第 3 步：可分性检验（最关键）

在 `src/exp_separability.py` 实现，**先只用 session 1**：

1. 复用 `koopman_stage0_synth.py` 里的 `dmd_spectrum`，
   但把距离计算改到 $(f,\sigma)$ 平面（采样率无关，见 CLAUDE.md 约束 3）
2. 逐段估谱，缓存
3. 算带权 Wasserstein 距离矩阵
4. 计算并打印：
   - 同情绪跨被试的谱内离散度
   - 同被试跨情绪的谱内离散度
   - **可分性比值**（对标合成数据的线性 4.0 / 非线性 1.09）
   - silhouette(情绪标签) vs silhouette(被试标签)
   - kNN 预测情绪 vs 预测被试的准确率（基线分别是 1/3 和 1/15）
5. 置换检验给 p 值

出图：距离矩阵热图两版（按情绪排序、按被试排序）并置。

**判据先写死再看结果**，不要事后挑指标。

---

## 后续几个任务的顺序

1. 上面三步
2. 对特征向量做同样的可分性分析（关键对照）
3. DE 特征基线对照
4. 全量 3 个 session
5. 填 ROADMAP 决策表

---

## 与 Claude Code 协作的几个建议

- **一个任务做完就 `/clear`**。上下文累积会显著吃额度（chat 和 Code 共用一个池子）
- 简单活儿用 `/model` 切 Sonnet，省 Opus 额度
- 让它改代码前先说方案，避免大改后发现方向不对
- 每次跑批前确认缓存键包含了当次改动的参数
