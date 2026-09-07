#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""Runtime Precision Morphing P0：输入复杂度度量

实现 4 个候选指标，用于动态决定推理时展开到哪一层残差（v[0] / v[0..1] / v[0..D]）。

设计来源：runtime_precision_morphing_design.md §1.4.2

4 个候选指标：
1. 激活值熵 H(a) = -Σ p(a_i) log p(a_i)
   - 高熵 = 特征分散 = 复杂样本
   - 低熵 = 特征集中 = 简单样本
2. 激活值峰度 kurtosis(a)
   - 高峰度 = 稀疏特征 = 简单样本（少数神经元强激活）
   - 低峰度 = 均匀特征 = 复杂样本
3. 预测置信度 max(softmax(z))
   - 高置信度 = 简单样本 → v[0] 足够
   - 低置信度 = 模糊样本 → 需展开
4. 残差能量比 ||v[1]||² / ||v[0]||²
   - 高比例 = 内层球重要 → 需展开
   - 低比例 = v[0] 已捕获主要信息 → 无需展开

运行时变形策略：
    简单样本（如 MNIST 清晰数字）：只用 v[0]（8bit），功耗极低
    模糊样本：展开到 v[0]+v[1]（16bit）
    对抗样本/异常输入：全展开（48bit），高精度防御

关联文档：
    runtime_precision_morphing_design.md（设计文档）
    hc8_net.c hc8_residual_matmul_b（增量计算已实现）

v0.1 P0 指标实现
"""

from __future__ import annotations
import numpy as np
from typing import Optional


# ============================================================
# 指标 1：激活值熵
# ============================================================

def activation_entropy(a: np.ndarray, eps: float = 1e-12) -> float:
    """激活值熵 H(a) = -Σ p(a_i) log p(a_i)

    高熵 = 特征分散 = 复杂样本
    低熵 = 特征集中 = 简单样本

    Args:
        a: 激活值向量（1D）或矩阵（2D，按行计算后取均值）
        eps: 数值稳定项

    Returns:
        熵值（float）
    """
    a_flat = np.abs(a.flatten()).astype(np.float64)
    total = a_flat.sum()
    if total < eps:
        return 0.0
    p = a_flat / total
    p = p[p > eps]  # 去掉零项
    return float(-np.sum(p * np.log(p)))


# ============================================================
# 指标 2：激活值峰度
# ============================================================

def activation_kurtosis(a: np.ndarray) -> float:
    """激活值峰度 kurtosis(a) = E[(a-μ)^4] / σ^4

    高峰度 = 稀疏特征 = 简单样本（少数神经元强激活）
    低峰度 = 均匀特征 = 复杂样本

    使用 Fisher 定义（正态分布峰度 = 0）

    Returns:
        峰度值（float）
    """
    a_flat = a.flatten().astype(np.float64)
    n = a_flat.size
    if n < 2:
        return 0.0
    mean = a_flat.mean()
    var = a_flat.var()
    if var < 1e-12:
        return 0.0
    kurt = float(np.mean((a_flat - mean) ** 4) / (var ** 2) - 3.0)
    return kurt


# ============================================================
# 指标 3：预测置信度
# ============================================================

def prediction_confidence(z: np.ndarray) -> float:
    """预测置信度 max(softmax(z))

    高置信度 = 简单样本 → v[0] 足够
    低置信度 = 模糊样本 → 需展开

    Args:
        z: logits 向量（1D）

    Returns:
        最大 softmax 概率（float，范围 [1/n_classes, 1.0]）
    """
    z_flat = z.flatten().astype(np.float64)
    z_shifted = z_flat - z_flat.max()
    exp_z = np.exp(z_shifted)
    softmax = exp_z / exp_z.sum()
    return float(softmax.max())


# ============================================================
# 指标 4：残差能量比
# ============================================================

def residual_energy_ratio(v0: np.ndarray, v1: np.ndarray,
                           scale0: float, scale1: float,
                           eps: float = 1e-12) -> float:
    """残差能量比 ||v[1]*scale1||² / ||v[0]*scale0||²

    高比例 = 内层球重要 → 需展开
    低比例 = v[0] 已捕获主要信息 → 无需展开

    Args:
        v0: v[0] 残差链第 0 层（int8 量化值）
        v1: v[1] 残差链第 1 层（int8 量化值）
        scale0: v[0] 的 scale
        scale1: v[1] 的 scale
        eps: 数值稳定项

    Returns:
        能量比（float，>= 0）
    """
    energy0 = float(np.sum((v0.astype(np.float64) * scale0) ** 2))
    energy1 = float(np.sum((v1.astype(np.float64) * scale1) ** 2))
    if energy0 < eps:
        return float('inf') if energy1 > eps else 0.0
    return energy1 / energy0


# ============================================================
# 综合度量：一次计算所有指标
# ============================================================

def compute_all_metrics(
    a: np.ndarray,
    z: Optional[np.ndarray] = None,
    v0: Optional[np.ndarray] = None,
    v1: Optional[np.ndarray] = None,
    scale0: float = 1.0,
    scale1: float = 1.0,
) -> dict:
    """一次计算所有可用指标

    Args:
        a: 激活值（用于熵、峰度）
        z: logits（用于置信度，可选）
        v0, v1: 残差链分量（用于能量比，可选）
        scale0, scale1: 残差链 scale

    Returns:
        dict: {
            'entropy': float,
            'kurtosis': float,
            'confidence': float or None,
            'residual_ratio': float or None,
        }
    """
    result = {
        'entropy': activation_entropy(a),
        'kurtosis': activation_kurtosis(a),
    }
    if z is not None:
        result['confidence'] = prediction_confidence(z)
    else:
        result['confidence'] = None
    if v0 is not None and v1 is not None:
        result['residual_ratio'] = residual_energy_ratio(v0, v1, scale0, scale1)
    else:
        result['residual_ratio'] = None
    return result


# ============================================================
# 决策器：基于指标的简单阈值规则
# ============================================================

def decide_depth(
    confidence: float,
    residual_ratio: Optional[float] = None,
    confidence_threshold: float = 0.9,
    residual_threshold: float = 0.1,
    max_depth: int = 5,
) -> int:
    """决定展开深度（P1 阶段的简单决策器）

    基于 runtime_precision_morphing_design.md §1.4.2 的 decide_depth

    策略：
        1. 先用 v[0] 跑前向，得到 confidence
        2. 如果 confidence >= threshold，返回 0（简单样本）
        3. 如果有 residual_ratio 且 >= residual_threshold，返回更深层
        4. 否则返回 1（模糊样本，展开一层）

    Args:
        confidence: v[0] 的预测置信度
        residual_ratio: ||v[1]||²/||v[0]||²（可选）
        confidence_threshold: 置信度阈值（默认 0.9）
        residual_threshold: 残差能量比阈值（默认 0.1）
        max_depth: 最大展开深度（默认 5，即 v[0..5] = 48-bit）

    Returns:
        展开深度（0 = 只用 v[0]，5 = 全展开）
    """
    if confidence >= confidence_threshold:
        return 0
    if residual_ratio is not None and residual_ratio >= residual_threshold:
        # 残差重要，展开到更深层
        # 简单线性映射：ratio 越大，展开越深
        depth = min(int(residual_ratio / residual_threshold), max_depth)
        return max(1, depth)
    return 1  # 默认展开一层


# ============================================================
# 自检
# ============================================================

def _self_test():
    """简单自检：验证 4 个指标在合成数据上的行为"""
    np.random.seed(42)

    print("=" * 70)
    print("Runtime Precision Morphing P0 指标自检")
    print("=" * 70)
    print()

    # 合成数据：简单样本 vs 复杂样本
    # 简单样本：稀疏激活（少数神经元强激活）
    simple_activation = np.zeros(64, dtype=np.float32)
    simple_activation[5] = 10.0  # 单一强激活
    simple_activation[10] = 2.0
    simple_logits = np.array([5.0, 0.1, 0.2, -0.1, 0.0, 0.3, 0.0, -0.2, 0.1, 0.0])

    # 复杂样本：均匀激活
    complex_activation = np.random.randn(64).astype(np.float32) * 2.0
    complex_logits = np.array([0.5, 0.3, 0.8, 0.6, 0.2, 0.7, 0.4, 0.1, 0.3, 0.5])

    # 残差链（模拟 v[0] 和 v[1]）
    v0_simple = np.array([10, 2, 0, 0, 0], dtype=np.int8)
    v1_simple = np.array([1, 0, 0, 0, 0], dtype=np.int8)  # 残差很小
    scale0 = 0.1
    scale1 = 0.001

    v0_complex = np.array([5, -3, 2, 1, -1], dtype=np.int8)
    v1_complex = np.array([8, -5, 3, 2, -2], dtype=np.int8)  # 残差很大
    scale0_c = 0.1
    scale1_c = 0.05

    # 计算指标
    print(f"{'指标':<25} {'简单样本':<20} {'复杂样本':<20} {'预期':<20}")
    print("-" * 85)

    # 1. 熵
    ent_s = activation_entropy(simple_activation)
    ent_c = activation_entropy(complex_activation)
    print(f"{'1. 激活值熵':<25} {ent_s:<20.4f} {ent_c:<20.4f} {'简单<复杂 ✓' if ent_s < ent_c else '简单>复杂 ✗'}")

    # 2. 峰度
    kur_s = activation_kurtosis(simple_activation)
    kur_c = activation_kurtosis(complex_activation)
    print(f"{'2. 激活值峰度':<25} {kur_s:<20.4f} {kur_c:<20.4f} {'简单>复杂 ✓' if kur_s > kur_c else '简单<复杂 ✗'}")

    # 3. 置信度
    conf_s = prediction_confidence(simple_logits)
    conf_c = prediction_confidence(complex_logits)
    print(f"{'3. 预测置信度':<25} {conf_s:<20.4f} {conf_c:<20.4f} {'简单>复杂 ✓' if conf_s > conf_c else '简单<复杂 ✗'}")

    # 4. 残差能量比
    ratio_s = residual_energy_ratio(v0_simple, v1_simple, scale0, scale1)
    ratio_c = residual_energy_ratio(v0_complex, v1_complex, scale0_c, scale1_c)
    print(f"{'4. 残差能量比':<25} {ratio_s:<20.4f} {ratio_c:<20.4f} {'简单<复杂 ✓' if ratio_s < ratio_c else '简单>复杂 ✗'}")

    print()

    # 决策器测试
    print("=" * 70)
    print("决策器测试")
    print("=" * 70)
    print()

    test_cases = [
        ("简单样本（高置信度）", conf_s, ratio_s),
        ("复杂样本（低置信度）", conf_c, ratio_c),
        ("边界样本（置信度=0.9）", 0.9, 0.05),
        ("模糊样本（置信度=0.5，高残差）", 0.5, 0.3),
        ("对抗样本（置信度=0.3，极高残差）", 0.3, 1.0),
    ]

    print(f"{'场景':<35} {'置信度':<10} {'残差比':<10} {'展开深度':<10}")
    print("-" * 65)
    for name, conf, ratio in test_cases:
        depth = decide_depth(conf, ratio)
        print(f"{name:<35} {conf:<10.3f} {ratio:<10.3f} {depth}")

    print()
    print("✓ 自检完成")
    print()
    print("关键发现：")
    print("  - 简单样本（高置信度）→ depth=0（只用 v[0]，省电）")
    print("  - 复杂样本（低置信度 + 高残差）→ depth>0（展开）")
    print("  - 对抗样本（极低置信度 + 极高残差）→ depth=max（全展开防御）")
    print()
    print("下一步（P0 验证）：")
    print("  - 在 MNIST/CIFAR-10 真实数据上验证 4 个指标的判别能力")
    print("  - 对比'v[0] 预测正确'vs'v[0] 预测错误'的样本，看哪个指标能区分")


if __name__ == "__main__":
    _self_test()
