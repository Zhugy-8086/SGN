# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""PrecisionBudget 单元测试 — 覆盖贪心分配算法 + 拉格朗日对比 + 负向测试

数学依据：
  - exp34: M-凸性证明，贪心边际分配 = 全局最优（gap ~ 1e-14）
  - exp26: 成本信号 c_i = grad_l2² × in_dim（含 in_dim 因子）
  - exp26 B: log₂(var_i) 近似失败（差 29×，缺 in_dim 因子）

运行: python test_precision_budget.py
"""
import sys
import os

# 安全审计 2026-08-16 A2-7：模式 B → 模式 A（import engine.sgn as sgn）
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import engine.sgn as sgn
import math


def test_basic_allocate():
    """基本贪心分配：3 层，预算 30 bits，成本不同的层应获不同 bits"""
    # 成本 c_i 越大，应分配越多 bits
    layers = [
        sgn.LayerCost(c=100.0, b_min=4, b_max=20),
        sgn.LayerCost(c=10.0, b_min=4, b_max=20),
        sgn.LayerCost(c=1.0, b_min=4, b_max=20),
    ]
    pb = sgn.PrecisionBudget(30, layers)
    bits = pb.allocate()
    assert sum(bits) == 30, f"总 bits={sum(bits)}, 预期 30"
    assert len(bits) == 3
    # 成本最大的层应获最多 bits
    assert bits[0] >= bits[1] >= bits[2], f"bits 应递减: {bits}"
    print(f"[PASS] test_basic_allocate: bits={bits}")


def test_budget_constraint():
    """预算约束：总 bits 严格等于 B_total"""
    layers = [sgn.LayerCost(c=float(i + 1), b_min=4, b_max=20) for i in range(10)]
    pb = sgn.PrecisionBudget(124, layers)  # exp26/34 的 124-bit 预算
    bits = pb.allocate()
    assert sum(bits) == 124, f"总 bits={sum(bits)}, 预期 124"
    print(f"[PASS] test_budget_constraint: 10 层 124 bits, bits={bits}")


def test_b_min_floor():
    """b_min 下限：每层至少 b_min bits"""
    layers = [
        sgn.LayerCost(c=100.0, b_min=8, b_max=20),
        sgn.LayerCost(c=1.0, b_min=8, b_max=20),
    ]
    pb = sgn.PrecisionBudget(20, layers)  # b_min 8+8=16，剩余 4 bits
    bits = pb.allocate()
    assert bits[0] >= 8 and bits[1] >= 8, f"每层至少 b_min=8: {bits}"
    # 剩余 4 bits 应全给成本高的层
    assert bits[0] == 12 and bits[1] == 8, f"剩余应给层 0: {bits}"
    print(f"[PASS] test_b_min_floor: bits={bits}")


def test_b_max_ceiling():
    """b_max 上限：层达到 b_max 后不再分配"""
    layers = [
        sgn.LayerCost(c=100.0, b_min=4, b_max=8),  # 上限 8
        sgn.LayerCost(c=1.0, b_min=4, b_max=20),   # 上限 20
    ]
    pb = sgn.PrecisionBudget(30, layers)
    bits = pb.allocate()
    assert bits[0] == 8, f"层 0 应达 b_max=8: {bits}"
    # 层 0 到 b_max=8 后，剩余 22 bits 全给层 1，但层 1 b_max=20，所以 bits[1]=20
    assert bits[1] == 20, f"层 1 应达 b_max=20: {bits}"
    # 总 bits = 8 + 20 = 28 < 30（2 bits 无法分配，两层都达上限）
    assert sum(bits) == 28, f"总 bits 应为 28（两层都达上限）: {bits}"
    print(f"[PASS] test_b_max_ceiling: bits={bits}（两层都达 b_max）")


def test_vs_lagrangian():
    """与拉格朗日数值解对比（复现 exp34，gap < 1e-10）

    拉格朗日法：对每层求 min f_i(b) + λ·b，KKT 条件求解
    exp34 证明贪心 = 拉格朗日最优（M-凸性，gap ~ 1e-14）
    """
    # 10 层，成本呈指数分布（模拟 ResNet-18 梯度方差）
    layers = []
    for i in range(10):
        c = 100.0 * (0.5 ** i)  # 成本递减
        layers.append(sgn.LayerCost(c=c, b_min=4, b_max=20))
    pb = sgn.PrecisionBudget(124, layers)
    greedy_bits = pb.allocate()
    greedy_err = pb.total_error(greedy_bits)

    # 暴力搜索最优（10 层 × 17 档位太多，改用拉格朗日对偶）
    # 对每个 λ，每层求 b_i(λ) = argmin [c_i/(2^b-1) + λ·b]
    # 然后调整 λ 使 Σ b_i(λ) = 124
    best_err = float("inf")
    best_bits = None
    # 二分搜索 λ
    lo, hi = 0.0, 1e6
    for _ in range(200):
        lam = (lo + hi) / 2.0
        bits = []
        for lc in layers:
            bi = lc.b_min
            best_cost = float("inf")
            for b in range(lc.b_min, lc.b_max + 1):
                cost = lc.c / ((1 << b) - 1) + lam * b
                if cost < best_cost:
                    best_cost = cost
                    bi = b
            bits.append(bi)
        total = sum(bits)
        if total > 124:
            lo = lam  # 增大 λ 减 bits
        elif total < 124:
            hi = lam  # 减小 λ 增 bits
        else:
            err = pb.total_error(bits)
            if err < best_err:
                best_err = err
                best_bits = bits
            break
    # 若未精确命中 124，用最接近的
    if best_bits is None:
        # 重新搜索最接近 124 的 λ
        best_err = float("inf")
        for _ in range(1000):
            lam = (lo + hi) / 2.0
            bits = []
            for lc in layers:
                bi = lc.b_min
                best_cost = float("inf")
                for b in range(lc.b_min, lc.b_max + 1):
                    cost = lc.c / ((1 << b) - 1) + lam * b
                    if cost < best_cost:
                        best_cost = cost
                        bi = b
                bits.append(bi)
            total = sum(bits)
            if total > 124:
                lo = lam
            else:
                hi = lam
            if abs(total - 124) == 0:
                err = pb.total_error(bits)
                if err < best_err:
                    best_err = err
                    best_bits = bits

    assert best_bits is not None, "拉格朗日法未找到可行解"
    gap = abs(greedy_err - best_err) / max(best_err, 1e-30)
    assert gap < 1e-10, f"贪心 vs 拉格朗日 gap={gap}, 应 < 1e-10"
    print(f"[PASS] test_vs_lagrangian: greedy_err={greedy_err:.6e}, lagrangian_err={best_err:.6e}, gap={gap:.2e}")


def test_negative_log2_var():
    """负向测试：用 log₂(var_i) 作成本信号应得次优解（复现 exp26 B 失败）

    exp26 B 失败原因：log₂(var_i) 缺 in_dim 因子，差 29×
    正确成本：c_i = grad_l2² × in_dim
    错误成本：c_i = log₂(var_i)（缺 in_dim）
    """
    # 10 层，in_dim 差异大（模拟 conv1 in_dim=3 vs 深层 in_dim=256）
    grad_l2 = [1.0] * 10  # 假设梯度范数相同
    in_dims = [3, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    # 正确成本：c_i = grad_l2² × in_dim
    correct_layers = [
        sgn.LayerCost(c=grad_l2[i] ** 2 * in_dims[i], b_min=4, b_max=20)
        for i in range(10)
    ]
    # 错误成本：c_i = log₂(var_i)（缺 in_dim，仅用 grad_l2²）
    wrong_layers = [
        sgn.LayerCost(c=grad_l2[i] ** 2, b_min=4, b_max=20)  # 缺 in_dim
        for i in range(10)
    ]

    pb_correct = sgn.PrecisionBudget(124, correct_layers)
    pb_wrong = sgn.PrecisionBudget(124, wrong_layers)

    correct_bits = pb_correct.allocate()
    wrong_bits = pb_wrong.allocate()

    # 用正确成本评估两种分配方案
    correct_err = pb_correct.total_error(correct_bits)
    wrong_err_under_correct = pb_correct.total_error(wrong_bits)

    # 正确方案的误差应低于错误方案
    assert correct_err < wrong_err_under_correct, (
        f"正确成本误差 {correct_err} 应低于错误成本方案 {wrong_err_under_correct}"
    )
    # 两种分配应不同（证明 log₂(var) 是次优）
    assert correct_bits != wrong_bits, "两种成本信号应产生不同分配"
    ratio = wrong_err_under_correct / correct_err
    print(f"[PASS] test_negative_log2_var: correct_err={correct_err:.4e}, "
          f"wrong_err={wrong_err_under_correct:.4e}, ratio={ratio:.2f}x")


def test_level_f_level_b_reserved():
    """level_f / level_b 预留字段验证"""
    layers = [sgn.LayerCost(c=100.0, b_min=4, b_max=20)]
    pb = sgn.PrecisionBudget(20, layers)

    # 默认未设置
    assert not pb.has_level_f
    assert not pb.has_level_b

    # 设置 level_f / level_b
    spec_f = sgn.ValueSpec(8)
    spec_b = sgn.ValueSpec(16)
    pb.set_level_f(spec_f)
    pb.set_level_b(spec_b)
    assert pb.has_level_f
    assert pb.has_level_b
    assert pb.level_f.bits == 8
    assert pb.level_b.bits == 16

    # 清除
    pb.clear_level_f()
    pb.clear_level_b()
    assert not pb.has_level_f
    assert not pb.has_level_b
    print("[PASS] test_level_f_level_b_reserved")


def test_total_error_monotonic():
    """总误差单调性：bits 越多误差越小"""
    layers = [sgn.LayerCost(c=100.0, b_min=4, b_max=20)]
    pb = sgn.PrecisionBudget(20, layers)
    prev_err = float("inf")
    for b in range(4, 21):
        err = pb.total_error([b])
        assert err < prev_err, f"bits={b} 误差 {err} 应小于 bits={b-1} 误差 {prev_err}"
        prev_err = err
    print(f"[PASS] test_total_error_monotonic: bits 4→20 误差单调递减")


def test_marginal_gain_decreasing():
    """边际收益递减：M-凸性保证 Δ(b) > Δ(b+1)"""
    c = 100.0
    prev_gain = float("inf")
    for b in range(4, 20):
        gain = sgn.PrecisionBudget.marginal_gain(c, b)
        assert gain < prev_gain, f"b={b} 边际收益 {gain} 应小于 b={b-1} {prev_gain}"
        prev_gain = gain
    print("[PASS] test_marginal_gain_decreasing: M-凸性验证（边际收益递减）")


def test_equal_cost_uniform():
    """等成本均匀分配：所有层成本相同时，bits 应接近均匀"""
    n = 5
    layers = [sgn.LayerCost(c=10.0, b_min=4, b_max=20) for _ in range(n)]
    pb = sgn.PrecisionBudget(100, layers)  # 100/5 = 20 bits/层
    bits = pb.allocate()
    assert sum(bits) == 100
    # 每层应接近 20
    for b in bits:
        assert 18 <= b <= 20, f"等成本应接近均匀: {bits}"
    print(f"[PASS] test_equal_cost_uniform: 5 层 100 bits, bits={bits}")


def test_exp34_resnet18_scenario():
    """复现 exp34 ResNet-18 场景：10 层 124 bits 预算

    exp34 结论：贪心 = 拉格朗日最优，gap ~ 1e-14
    """
    # 模拟 ResNet-18 的成本分布（深层成本高，conv1 末端层成本低）
    # c_i = grad_l2² × in_dim
    grad_l2_sq = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 0.001]
    in_dims = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 3]  # conv1 in_dim=3
    layers = [
        sgn.LayerCost(c=grad_l2_sq[i] * in_dims[i], b_min=4, b_max=20)
        for i in range(10)
    ]
    pb = sgn.PrecisionBudget(124, layers)
    bits = pb.allocate()
    assert sum(bits) == 124
    # conv1（最后一层，in_dim=3）应获较少 bits（exp33-followup 拉格朗日给 12 bits）
    conv1_bits = bits[9]
    assert conv1_bits <= 16, f"conv1 应获较少 bits: {conv1_bits}"
    print(f"[PASS] test_exp34_resnet18_scenario: bits={bits}, conv1={conv1_bits}")


if __name__ == "__main__":
    test_basic_allocate()
    test_budget_constraint()
    test_b_min_floor()
    test_b_max_ceiling()
    test_vs_lagrangian()
    test_negative_log2_var()
    test_level_f_level_b_reserved()
    test_total_error_monotonic()
    test_marginal_gain_decreasing()
    test_equal_cost_uniform()
    test_exp34_resnet18_scenario()
    print("\n=== All PrecisionBudget tests PASSED ===")
