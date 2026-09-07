# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""Task 4.3.5 集成测试 — 在 exp19 ResNet-18 反向结构上复现 exp34 结论

目标:
  使用 Stage 3.0 BitsAllocator (C++ PrecisionBudget) 在 exp19 的 ResNet-18
  反向结构 (10 层) 上进行 bits 分配, 验证复现 exp34 的核心结论:

    1. H_IP_2: BitsAllocator 与拉格朗日数值解 gap < 5%
    2. H_IP_3: M-凸性 (边际增益严格递减)
    3. 成本信号 c_i = grad_l2² × in_dim 正确采集
    4. 总 bits = B_total = 124
    5. 滞后机制: bits 变化限制 ±2

数学依据:
  - exp34 证明贪心边际分配 = 精确 IP 最优 (M-凸性, Ibaraki-Katoh 定理)
  - BitsAllocator 封装 C++ PrecisionBudget::allocate (同一贪心算法)
  - 因此 BitsAllocator 输出应与 exp34 的 allocate_greedy_marginal 完全一致

运行: python test_task_4_3_5_integration.py
"""
import sys
import os
from pathlib import Path

# 添加项目根目录和 engine/ 到 path
# test 文件在 engine/sgn/tests/ 下, parents[3]=项目根(SGN), parents[2]=engine
_PROJ_ROOT = Path(__file__).resolve().parents[3]
_ENGINE_DIR = Path(__file__).resolve().parents[2]
# 参考实现位于 tests/refs/（从 legacy/traditional 复制，见 refs/__init__.py）
_REFS_EXPLORATION = Path(__file__).resolve().parents[0] / "refs" / "stage_2_7_consolidation" / "exploration"
for _p in [_PROJ_ROOT, _ENGINE_DIR, _REFS_EXPLORATION]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

# Stage 3.0 BitsAllocator (被测对象)
from sgn.level import BitsAllocator
from sgn import ValueSpec

# UpgradedLevelContext 尚未在 C++ 端实现（Python 死代码已随剥离删除），
# 且本文件未实际使用，仅保留兼容导入意图（try/except 防 ImportError）。
try:
    from sgn.level import UpgradedLevelContext  # noqa: F401
except ImportError:
    UpgradedLevelContext = None

# 复用 exp34 的实现 (不修改原文件, 通过导入对比)
from exp34_greedy_optimal_proof import (
    RESNET18_BACKWARD_LAYERS,
    measure_layer_signals,
    allocate_greedy_marginal,
    allocate_lagrangian,
    theoretical_cost,
    marginal_gain,
    B_TOTAL,
    BITS_MIN,
    BITS_MAX,
    SEEDS,
)


# ============================================================
# 测试 1: BitsAllocator 输出与 exp34 贪心实现完全一致
# ============================================================

def test_bits_allocator_matches_exp34_greedy():
    """BitsAllocator (C++ PrecisionBudget) == exp34 allocate_greedy_marginal (Python)

    两者实现同一贪心边际分配算法, 输出应逐 bit 一致。
    """
    allocator = BitsAllocator(total_bits=B_TOTAL, b_min=BITS_MIN, b_max=BITS_MAX)
    all_match = True
    for seed in SEEDS:
        signals = measure_layer_signals(RESNET18_BACKWARD_LAYERS, seed)
        c_list = [s["c"] for s in signals]
        # exp34 的成本信号: (grad_l2, in_dim) → c = grad_l2² × in_dim
        costs = [(s["grad_l2"], s["in_dim"]) for s in signals]

        bits_cpp = allocator.allocate(costs)
        bits_py = allocate_greedy_marginal(c_list, B_TOTAL)

        if bits_cpp != bits_py:
            all_match = False
            print(f"  [FAIL] seed={seed}: cpp={bits_cpp} vs py={bits_py}")

    assert all_match, "BitsAllocator 与 exp34 贪心实现不一致"
    print(f"[PASS] test_bits_allocator_matches_exp34_greedy: {len(SEEDS)} seed 全部一致")


# ============================================================
# 测试 2: H_IP_2 — BitsAllocator vs 拉格朗日 gap < 5%
# ============================================================

def test_h_ip_2_bits_allocator_vs_lagrangian():
    """H_IP_2: BitsAllocator 与拉格朗日数值解差距 < 5% (复现 exp34)"""
    allocator = BitsAllocator(total_bits=B_TOTAL, b_min=BITS_MIN, b_max=BITS_MAX)
    max_gap = 0.0
    all_within = True

    for seed in SEEDS:
        signals = measure_layer_signals(RESNET18_BACKWARD_LAYERS, seed)
        c_list = [s["c"] for s in signals]
        costs = [(s["grad_l2"], s["in_dim"]) for s in signals]

        bits_cpp = allocator.allocate(costs)
        bits_lag = allocate_lagrangian(c_list, B_TOTAL)

        cost_cpp = theoretical_cost(c_list, bits_cpp)
        cost_lag = theoretical_cost(c_list, bits_lag)
        gap_pct = abs(cost_cpp - cost_lag) / max(cost_lag, 1e-30) * 100
        max_gap = max(max_gap, gap_pct)

        if gap_pct >= 5.0:
            all_within = False
            print(f"  [FAIL] seed={seed}: gap={gap_pct:.4f}% >= 5%")

    assert all_within, f"H_IP_2 不成立: max gap={max_gap:.4f}% >= 5%"
    print(f"[PASS] test_h_ip_2_bits_allocator_vs_lagrangian: max gap={max_gap:.4f}% < 5%")


# ============================================================
# 测试 3: H_IP_3 — M-凸性验证 (边际增益严格递减)
# ============================================================

def test_h_ip_3_m_convexity():
    """H_IP_3: f_i(b) - f_i(b+1) 严格递减 (M-凸性, Ibaraki-Katoh 定理适用)"""
    signals = measure_layer_signals(RESNET18_BACKWARD_LAYERS, 42)
    c_list = [s["c"] for s in signals]

    all_decreasing = True
    for i, c in enumerate(c_list):
        gains = [marginal_gain(c, b) for b in range(BITS_MIN, BITS_MAX)]
        for k in range(len(gains) - 1):
            if gains[k] <= gains[k + 1]:
                all_decreasing = False
                print(f"  [FAIL] layer {i}: Δ({BITS_MIN+k})={gains[k]:.4e} "
                      f"<= Δ({BITS_MIN+k+1})={gains[k+1]:.4e}")

    assert all_decreasing, "M-凸性不成立: 边际增益非严格递减"
    print(f"[PASS] test_h_ip_3_m_convexity: {len(c_list)} 层边际增益全部严格递减")


# ============================================================
# 测试 4: 总 bits = B_total
# ============================================================

def test_total_bits_equals_budget():
    """BitsAllocator 分配总 bits = B_total = 124"""
    allocator = BitsAllocator(total_bits=B_TOTAL, b_min=BITS_MIN, b_max=BITS_MAX)
    for seed in SEEDS:
        signals = measure_layer_signals(RESNET18_BACKWARD_LAYERS, seed)
        costs = [(s["grad_l2"], s["in_dim"]) for s in signals]
        bits = allocator.allocate(costs)
        total = sum(bits)
        assert total == B_TOTAL, f"seed={seed}: 总 bits={total} != {B_TOTAL}"
    print(f"[PASS] test_total_bits_equals_budget: {len(SEEDS)} seed 总 bits 全部 = {B_TOTAL}")


# ============================================================
# 测试 5: 成本信号 c_i = grad_l2² × in_dim 正确性
# ============================================================

def test_cost_signal_correctness():
    """成本信号 c_i = grad_l2² × in_dim (exp26 确认)"""
    signals = measure_layer_signals(RESNET18_BACKWARD_LAYERS, 42)
    for i, s in enumerate(signals):
        expected_c = s["grad_l2"] ** 2 * s["in_dim"]
        assert abs(s["c"] - expected_c) < 1e-6 * max(expected_c, 1e-30), \
            f"layer {i}: c={s['c']} != grad_l2²×in_dim={expected_c}"
    print(f"[PASS] test_cost_signal_correctness: c_i = grad_l2² × in_dim 全部正确")


# ============================================================
# 测试 6: 滞后机制在 ResNet-18 上的行为
# ============================================================

def test_hysteresis_on_resnet18():
    """滞后机制: bits 变化限制 ±2 (exp23 设计)"""
    allocator = BitsAllocator(
        total_bits=B_TOTAL, b_min=BITS_MIN, b_max=BITS_MAX, hysteresis_delta=2
    )

    # seed 42 的分配作为 prev_bits
    signals_42 = measure_layer_signals(RESNET18_BACKWARD_LAYERS, 42)
    costs_42 = [(s["grad_l2"], s["in_dim"]) for s in signals_42]
    prev_bits = allocator.allocate(costs_42)

    # 用不同 seed 的成本做第二轮 (带滞后)
    signals_123 = measure_layer_signals(RESNET18_BACKWARD_LAYERS, 123)
    costs_123 = [(s["grad_l2"], s["in_dim"]) for s in signals_123]
    new_bits = allocator.allocate_with_hysteresis(costs_123, prev_bits)

    # 每层变化应 ≤ ±2
    for i, (old, new) in enumerate(zip(prev_bits, new_bits)):
        diff = abs(new - old)
        assert diff <= 2, f"层 {i}: 变化 {diff} 超过 ±2 (old={old}, new={new})"

    # 总 bits 仍应等于 B_total
    assert sum(new_bits) == B_TOTAL, f"滞后后总 bits={sum(new_bits)} != {B_TOTAL}"
    print(f"[PASS] test_hysteresis_on_resnet18: 变化全部 ≤ ±2, 总 bits={sum(new_bits)}")


# ============================================================
# 测试 7: UpgradedLevelContext 集成 (bits → max_range → 量化)
# ============================================================

def test_upgraded_context_with_allocator():
    """UpgradedLevelContext 接收 BitsAllocator 输出, bits→max_range 同步"""
    if UpgradedLevelContext is None:
        import pytest
        pytest.skip("UpgradedLevelContext 尚未在 C++ 端实现（随剥离删除 Python 死代码），需 C++ 实现后重启用")
    allocator = BitsAllocator(total_bits=B_TOTAL, b_min=BITS_MIN, b_max=BITS_MAX)
    signals = measure_layer_signals(RESNET18_BACKWARD_LAYERS, 42)
    costs = [(s["grad_l2"], s["in_dim"]) for s in signals]
    bits_list = allocator.allocate(costs)

    # 为每层创建 UpgradedLevelContext, 验证 bits→max_range 同步
    for i, (bits, layer) in enumerate(zip(bits_list, RESNET18_BACKWARD_LAYERS)):
        ctx = UpgradedLevelContext(bits=bits, source=f"resnet18_layer_{i}")
        expected_max_range = (1 << bits) - 1
        assert ctx.bits == bits
        assert ctx.max_range == expected_max_range, \
            f"层 {i}: bits={bits} → max_range={ctx.max_range} != {expected_max_range}"
        # level_f/level_b 默认 None (向后兼容)
        assert not ctx.has_level_f()
        assert not ctx.has_level_b()

    print(f"[PASS] test_upgraded_context_with_allocator: {len(bits_list)} 层 bits→max_range 同步正确")


# ============================================================
# 测试 8: bits 分配方向正确性 (深层高精度)
# ============================================================

def test_bits_allocation_direction():
    """bits 分配方向: 成本高的层获更多 bits (exp19 H_B2 深层高精度)"""
    allocator = BitsAllocator(total_bits=B_TOTAL, b_min=BITS_MIN, b_max=BITS_MAX)

    # 构造极端成本: 层 0 成本极高, 层 9 成本极低
    extreme_costs = [
        (10.0, 512),   # 层 0: 高梯度 × 大 in_dim
        (1.0, 512),
        (1.0, 256),
        (1.0, 256),
        (1.0, 128),
        (1.0, 128),
        (1.0, 64),
        (1.0, 64),
        (1.0, 64),
        (0.01, 3),     # 层 9: 低梯度 × 小 in_dim
    ]
    bits = allocator.allocate(extreme_costs)
    # 层 0 应获最多 bits, 层 9 应获最少
    assert bits[0] == max(bits), f"高成本层 0 应获最多 bits: {bits}"
    assert bits[9] == min(bits), f"低成本层 9 应获最少 bits: {bits}"
    assert bits[0] > bits[9], f"方向错误: bits[0]={bits[0]} <= bits[9]={bits[9]}"
    print(f"[PASS] test_bits_allocation_direction: 层0={bits[0]} > 层9={bits[9]} (方向正确)")


if __name__ == "__main__":
    print("=" * 78)
    print("Task 4.3.5 集成测试 — exp19 ResNet-18 反向结构复现 exp34 结论")
    print(f"结构: {len(RESNET18_BACKWARD_LAYERS)} 层, 总 bits 预算 {B_TOTAL}")
    print(f"seeds: {SEEDS}")
    print("=" * 78)
    print()

    test_bits_allocator_matches_exp34_greedy()
    test_h_ip_2_bits_allocator_vs_lagrangian()
    test_h_ip_3_m_convexity()
    test_total_bits_equals_budget()
    test_cost_signal_correctness()
    test_hysteresis_on_resnet18()
    test_upgraded_context_with_allocator()
    test_bits_allocation_direction()

    print()
    print("=" * 78)
    print("=== All Task 4.3.5 integration tests PASSED ===")
    print("exp34 结论在 Stage 3.0 BitsAllocator 下完整复现:")
    print("  - H_IP_2: BitsAllocator vs 拉格朗日 gap < 5%")
    print("  - H_IP_3: M-凸性 (边际增益严格递减)")
    print("  - BitsAllocator == exp34 贪心实现 (逐 bit 一致)")
    print("=" * 78)
