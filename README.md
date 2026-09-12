# Koopman-SNN-EEG

跨被试脑电情绪识别：Koopman 谱不变性 + Resonate-and-Fire 脉冲神经网络

## 快速开始

    pip install -r requirements.txt
    python src/koopman_stage0_synth.py     # 复现第 0 阶段第 1 周验证

## 文件说明

- `CLAUDE.md` —— 项目常驻上下文，Claude Code 自动读取
- `docs/ROADMAP.md` —— 阶段路线图与进度追踪
- `docs/FIRST_TASK.md` —— 交给 Claude Code 的首个任务
- `src/koopman_stage0_synth.py` —— 合成数据验证（已跑通）

## 目录

    data/      原始数据，只读
    cache/     分层缓存（预处理/分段/谱/距离矩阵）
    results/   最终结果
    src/       代码
