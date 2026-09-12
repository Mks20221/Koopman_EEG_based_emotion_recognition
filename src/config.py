# -*- coding: utf-8 -*-
"""全局路径与常量。所有路径只在这里出现一次。"""
import os

# 数据根目录（可用环境变量覆盖，方便换机器）
SEED_ROOT = os.environ.get("SEED_ROOT", r"E:\Python\DATA\SEED")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# 数据集元信息
SEED_FS = 200
SEED_N_CHANNELS = 62
SEED_N_TRIALS = 15
SEED_N_SUBJECTS = 15
SEED_N_SESSIONS = 3

# SEED 标签 {-1,0,1} -> {0,1,2}；0=negative, 1=neutral, 2=positive
SEED_LABEL_MAP = {-1: 0, 0: 1, 1: 2}
SEED_CLASS_NAMES = ["negative", "neutral", "positive"]

# 电极顺序，来自 SEED 附带的 channel-order.xlsx（已去掉表头行，0-based）
SEED_CHANNEL_NAMES = (
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3",
    "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5",
    "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4",
    "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6",
    "PO8", "CB1", "O1", "OZ", "O2", "CB2",
)

# --------------------------------------------------------------------------
# 坏导黑名单（故障电极，非脑信号）
# --------------------------------------------------------------------------
# 由 `python preprocess.py --scan-bad` 在 15 被试 × 3 session 共 45 次录制上普查。
#
# 判据是**跨录制比同一通道自己**，而不是在一次录制内跨通道比：
#     ratio(s,ses,c) = 该通道中位峰峰值 / 该次录制全通道中位
#     Z(s,ses,c)     = ratio(s,ses,c) / median_over_45_recordings(ratio(:,:,c))
#     坏导 <=> Z > 10 或 Z < 0.1
#
# 为什么必须这样比：CZ / CPZ / C1 / PZ 这些中线电极在**所有人**身上幅度都只有
# 典型值的 0.1–0.5 倍（参考电极就在附近），按"一次录制内跨通道比"会把它们全判成
# 失联导剔掉——可它们对所有被试一模一样，根本不携带被试身份信息。
# 换成跨录制自比后，它们的 Z 落在 [0.16, 4.2]，正确地未命中。
#
# 阈值不是拍的：Z 的 99 分位是 3.35、99.5 分位是 10.89，中间是空白，最大值 406。
#
# 全体被试统一剔除同一套通道，绝不逐被试处理。理由见 docs/PREREGISTER.md 修订 1：
# 坏导高度录制特异（PO6 只在被试1-ses1 爆到 Z=406，FC5 只在被试15-ses1 爆到 242），
# 任何"逐被试"的处理都会把电极故障编码成被试指纹，
# 从而在"特征向量是否按被试分离"这条判据上伪造阳性。
SEED_BAD_CHANNELS = (4, 6, 10, 15, 18, 19, 39, 42, 44, 52, 55)
#                  AF4  F5  F2 FC5 FCZ FC2 CP6  P5  P1 PO3 PO6
SEED_GOOD_CHANNELS = tuple(c for c in range(SEED_N_CHANNELS)
                           if c not in SEED_BAD_CHANNELS)   # 51 个
SEED_BAD_Z_HI = 10.0
SEED_BAD_Z_LO = 0.1

# 每个通道的典型相对幅度 = median over 45 次录制 of
#   (该通道中位峰峰值 / 该次录制全通道中位峰峰值)
# 由 `python preprocess.py --scan-bad` 生成。
#
# 用途：段级伪迹剔除的基准。若改用"该通道在本次录制内的中位值"作基准，
# 半场失效的通道会把基准压低，导致它正常工作的那半场反被判成伪迹
# （被试15 的 C1/C2/CB2 实测就是如此，统计量中位被抬到 384）。
# 跨录制的典型值不受单次录制故障影响，是稳定的参照。
#
# 注意这条 profile 本身是有生理意义的：额极 FP1/FPZ/FP2 与 AF4 最大（2.5-3.3，眼动），
# 中线 CPZ 最小（0.11，靠近参考电极）。它不是噪声，是导联场的平均形状。
SEED_CHANNEL_PROFILE = (
    2.6947, 2.5202, 2.5936, 1.7569, 1.7439, 3.2762,
    1.6906, 1.1898, 0.9966, 0.9842, 0.9899, 1.0976,
    1.3647, 1.7954, 1.5890, 1.2197, 0.8542, 0.6524,
    0.5929, 0.6013, 0.7993, 1.0918, 1.4983, 1.3761,
    0.9749, 0.6830, 0.4165, 0.2704, 0.3777, 0.6253,
    0.9226, 1.3517, 1.1627, 0.8552, 0.5814, 0.3762,
    0.1121, 0.3798, 0.5835, 0.8186, 1.0800, 1.0757,
    0.8944, 0.6989, 0.6126, 0.5424, 0.6082, 0.7452,
    0.8793, 1.0820, 1.0588, 1.0196, 0.9323, 0.8258,
    0.9733, 1.0306, 1.1204, 1.1825, 1.0786, 1.0488,
    1.0570, 1.1668,
)
assert len(SEED_CHANNEL_PROFILE) == SEED_N_CHANNELS
assert len(SEED_CHANNEL_NAMES) == SEED_N_CHANNELS
