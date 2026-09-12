# -*- coding: utf-8 -*-
"""最小 Deep Koopman 判决实验的模型定义。

设计原则：本组 variant 共用相同的编码器与分类头结构，只有 transition
模块和损失项不同，用于比较当前架构下的辅助动力学约束。结构相同不表示
初始权重相同；初始化、损失、调参预算与划分由训练脚本控制。结论不外推到所有架构。

  A  : Encoder + CE                       纯监督基线
  B  : Encoder + Decoder + CE             排除重构的贡献
  C  : B + 线性 K，z_{t+1} ≈ K z_t        检验 Koopman 线性
  C_cf: B + **闭式 ridge 解**的 K（不是 SGD 学的），交替优化
  D1 : B + **参数量严格匹配** 的非线性 MLP transition
  D2 : B + 高容量非线性 MLP transition

为什么要 C_cf（2026-08-24 加入）：
C 的 K 是在联合目标 L_CE+αL_rec+βL_Koop 下用 SGD 学的，且 checkpoint 按验证
**分类准确率**选，不是按 Koopman 损失选。因此 learned K 的线性预测不如事后 OLS 可以由目标差异解释，单凭该现象
不能判断实现是否有 bug。C_cf 检验周期性闭式拟合是否改变表示；refit 只对
调用时传入的数据及 ridge 目标给出解，不保证后续每步或未见数据上最优。
注意 K 只通过训练期的 Koopman 损失影响 encoder
（分类走 mean-pooling，不经过 K），所以必须是交替优化，
**训练完再把 K 换成 OLS 是无效的**。

为什么要 D1 和 D2 两个非线性对照（2026-08-24 修订）：
只有一个大容量 D 时，"C < D" 无法区分"非线性本质上更适合"与"D 单纯参数更多"。
默认 z_dim=32 时，D1 与 K 的参数量相等（bias=False，2·z_dim·hidden=z_dim²）。
D1 有隐藏瓶颈，参数量相等不表示表达能力或优化难度相同；D2 提供更高容量对照。
若使用奇数 z_dim，hidden=z_dim//2 会使 D1 参数量不再严格匹配，应由训练配置核对。

2026-09-09 修订：仅修改注释和文档字符串；网络、参数、前向计算及闭式求解不变。
本文件不包含训练循环，不能单独验证损失梯度、数据划分、BatchNorm 模式或 refit 数据来源。
"""
from __future__ import annotations

import torch
import torch.nn as nn

VARIANTS = ("A", "B", "C", "C_cf", "D1", "D2")


class FrameEncoder(nn.Module):
    """(B, C, L) 单帧 -> (B, z_dim)。所有 variant 共用，不含任何 Koopman 相关设计。"""

    def __init__(self, n_channels: int, z_dim: int = 32, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, width, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, width, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(width), nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.head = nn.Linear(width * 4, z_dim)

    def forward(self, x):
        return self.head(self.net(x).flatten(1))


class FrameDecoder(nn.Module):
    """(B, z_dim) -> (B, C, L)。只有 B/C/D* 用得到。"""

    def __init__(self, n_channels: int, frame_len: int, z_dim: int = 32, width: int = 64):
        super().__init__()
        self.width, self.frame_len = width, frame_len
        self.fc = nn.Linear(z_dim, width * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(width, width, kernel_size=5, stride=2, padding=2,
                               output_padding=1),
            nn.BatchNorm1d(width), nn.GELU(),
            nn.ConvTranspose1d(width, width, kernel_size=5, stride=2, padding=2,
                               output_padding=1),
            nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, n_channels, kernel_size=7, padding=3),
        )

    def forward(self, z):
        h = self.net(self.fc(z).view(-1, self.width, 4))
        return nn.functional.interpolate(h, size=self.frame_len, mode="linear",
                                         align_corners=False)


class LinearKoopman(nn.Module):
    """采用无偏置的有限维线性预测 z_{t+1}=K z_t，参数量 z_dim²。

    无偏置是本实验的建模选择，并非仿射模型不能做谱分析：
    z_next=Kz+b 可通过增广常数坐标写成 [[K,b],[0,1]] 的线性模型。
    nn.Linear 的行向量计算为 z @ weight.T；谱到频率/增长率的解释
    还依赖时间步长、观测函数与模型有效性，不能仅由无 bias 保证。
    """

    def __init__(self, z_dim: int = 32):
        super().__init__()
        self.K = nn.Linear(z_dim, z_dim, bias=False)

    def forward(self, z):
        return self.K(z)


class ClosedFormKoopman(nn.Module):
    """K 由 ridge 闭式解给出，不是 SGD 学的参数。

        K = (Z_-^T Z_- + lambda I)^{-1} Z_-^T Z_+

    用 ridge 而非裸伪逆：latent 各维高度共线时 Z_-^T Z_- 接近奇异，
    裸伪逆会放大噪声方向，解出的 K 数值上不可靠。

    K 注册为 buffer 而非 parameter —— 它不参与梯度更新，
    在 refit 所用样本和给定 ridge 目标上求解，再作为固定预测矩阵约束 encoder。
    样本采用行向量约定，forward 为 z @ K；K 对应列向量算子的转置。
    此方法的 no_grad 不影响后续 forward 对 z 求梯度。编码器更新后 K 可能滞后；
    refit 频率、输入数据是否仅来自训练集及模型模式由外部训练脚本负责。
    """

    def __init__(self, z_dim: int = 32, ridge: float = 1e-3):
        super().__init__()
        self.ridge = ridge
        self.register_buffer("K", torch.eye(z_dim))

    @torch.no_grad()
    def refit(self, Zm, Zp):
        """Zm, Zp: (N, z_dim)，训练集上的 (z_t, z_{t+1}) 配对。"""
        d = Zm.shape[1]
        A = Zm.T @ Zm + self.ridge * torch.eye(d, device=Zm.device, dtype=Zm.dtype)
        self.K.copy_(torch.linalg.solve(A, Zm.T @ Zp))

    def forward(self, z):
        return z @ self.K


class MLPTransition(nn.Module):
    """非线性对照 z_{t+1} = F(z_t)。

    matched=True 时 bias=False 且 hidden = z_dim/2，参数量 2·z_dim·hidden = z_dim²，
    在 z_dim 为偶数时与 LinearKoopman 参数量相等；默认 32 维时均为 1024。
    hidden=z_dim//2 是额外瓶颈，匹配参数数目不等于匹配表达能力。
    奇数 z_dim 不严格匹配；本次保留原实现以保持历史实验行为。
    """

    def __init__(self, z_dim: int = 32, hidden: int | None = None, matched: bool = False):
        super().__init__()
        if matched:
            hidden = z_dim // 2
            self.net = nn.Sequential(
                nn.Linear(z_dim, hidden, bias=False), nn.GELU(),
                nn.Linear(hidden, z_dim, bias=False))
        else:
            hidden = hidden or z_dim * 2
            self.net = nn.Sequential(
                nn.Linear(z_dim, hidden), nn.GELU(), nn.Linear(hidden, z_dim))

    def forward(self, z):
        return self.net(z)


class KoopmanModel(nn.Module):
    """按 variant 组装，共用相同 encoder/classifier 结构；训练权重可以不同。"""

    def __init__(self, variant: str, n_channels: int, frame_len: int,
                 z_dim: int = 32, n_classes: int = 3, width: int = 64):
        super().__init__()
        assert variant in VARIANTS, f"未知 variant {variant}"
        self.variant = variant
        self.encoder = FrameEncoder(n_channels, z_dim, width)
        self.classifier = nn.Linear(z_dim, n_classes)      # 作用在时间平均的 z 上
        self.decoder = (FrameDecoder(n_channels, frame_len, z_dim, width)
                        if variant != "A" else None)
        if variant == "C":
            self.transition = LinearKoopman(z_dim)
        elif variant == "C_cf":
            self.transition = ClosedFormKoopman(z_dim)
        elif variant == "D1":
            self.transition = MLPTransition(z_dim, matched=True)
        elif variant == "D2":
            self.transition = MLPTransition(z_dim, hidden=z_dim * 2)
        else:
            self.transition = None

    def n_transition_params(self):
        if self.transition is None:
            return 0
        return sum(p.numel() for p in self.transition.parameters())

    def encode_seq(self, x):
        """(B, T, C, L) -> (B, T, z_dim)。卷积仅处理单帧内部时间结构。

        未设置帧间递归、卷积或位置编码；训练态 BatchNorm 会汇总 B*T 帧的统计量，
        这种统计耦合不等于学习帧间顺序。推理态分类对固定帧集合的排列不变。
        transition 可在外部损失中约束相邻 z，并间接影响编码器表示。
        """
        B, T, C, L = x.shape
        return self.encoder(x.reshape(B * T, C, L)).view(B, T, -1)

    def forward(self, x):
        z = self.encode_seq(x)
        logits = self.classifier(z.mean(dim=1))            # 各组平均读出；K 不参与分类前向
        return z, logits
