# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""MSint 逐元素异构粒度拆分点积验证（组合落地：1:N 多精度解释 + Level 逐元素精度选择）

对标范式文档 docs/msint_multisplit_paradigm/MSint多精度拆分计算范式.md
  - §1.2/H4: 1:N 多精度解释（MultiScaleView）
  - §3.2/H4: Level 逐元素精度选择（PrecisionSelector）
  - 组合落地：LeveledSplitDot —— 对一批 w/x 元素，每个元素按重要性独立选粒度，
    异构粒度拆分后按粒度分组做多输出点积（1:N），单融合时精确重建原始点积。

核心数学（异构粒度下位拆分仍严格可逆）：
  元素 i 选粒度 b_i，则 w_i = Σ_k w_i[k]·2^(k·b_i)，x_i = Σ_l x_i[l]·2^(l·b_i)，
  w_i·x_i = Σ_{k,l} w_i[k]·x_i[l]·2^((k+l)·b_i)，所有元素贡献直接相加，
  故 y = Σ_i w_i·x_i 位精确等价原始点积（不随粒度选择改变）→ H1 在异构下成立。

验证项：
  - L1 [T] dot_fused_leveled == 原始点积（bit-exact，多 seed + 边界/负数）
  - L2 [T] dot_split_leveled 分组结构：组内 partials 长度 = 2n_b-1，各组独立可重建
  - L3 [S] 按需资源分配：重要性越高粒度越细（parts 数越多），阈值语义
  - L4 [E] 异构/全粗/全细三种策略融合均 == 原始点积（按需分配不损失精度）
  - L5 C++ vs Python 参考实现一致
  - L6 构造校验（长度不一致抛异常）

运行: py -3.14 validate_math_msint_leveled_dot.py
"""
import sys
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parents[3]
_ENGINE_DIR = Path(__file__).resolve().parents[2]
for _p in [_PROJ_ROOT, _ENGINE_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

from sgn.msint.cpp_backend import (
    CppLeveledSplitDot, _PyLeveledSplitDot,
    CppSplitDot, _PySplitDot, USING_CPP,
)

RNG_SEEDS = [0, 1, 2, 3, 4]


def _rand_int32(rng, n):
    return [int(rng.randint(-2**31, 2**31 - 1, dtype=np.int64)) for _ in range(n)]


def _rand_importance(rng, n, hi=2000):
    return [int(rng.randint(0, hi)) for _ in range(n)]


def _signed64(v):
    lo = v & ((1 << 64) - 1)
    return lo - (1 << 64) if lo >= (1 << 63) else lo


def _raw_dot(w, x):
    return _signed64(sum(a * b for a, b in zip(w, x)))


# ============================================================
# L1 [T] 异构粒度单融合 == 原始点积（bit-exact）
# ============================================================

def test_l1_fused_equals_raw():
    edges = [0, 1, -1, 2**31 - 1, -2**31, 2**16 - 1, -2**16,
             2**8 - 1, -2**8, 2**4 - 1, -2**4, 12345, -98765]
    total_cases = 0
    for seed in RNG_SEEDS:
        rng = np.random.RandomState(seed)
        for _ in range(30):
            K = int(rng.randint(1, 500))
            w = _rand_int32(rng, K)
            x = _rand_int32(rng, K)
            imp = _rand_importance(rng, K)
            # 注入边界值
            if K >= 1:
                w[0] = edges[rng.randint(len(edges))]
                x[-1] = edges[rng.randint(len(edges))]
            got = CppLeveledSplitDot.dot_fused_leveled_default(w, x, imp)
            ref = _raw_dot(w, x)
            assert got == ref, f"seed={seed} K={K} 异构融合 {got} != 原始 {ref}"
            total_cases += 1
    print(f"[PASS] L1 [T] dot_fused_leveled == 原始点积：{total_cases} 组（5 seed × 30，含边界/负数）bit-exact")


# ============================================================
# L2 [T] dot_split_leveled 分组结构正确 + 各组独立可重建
# ============================================================

def test_l2_split_group_structure():
    for seed in RNG_SEEDS:
        rng = np.random.RandomState(seed)
        for _ in range(10):
            K = int(rng.randint(2, 300))
            w = _rand_int32(rng, K)
            x = _rand_int32(rng, K)
            imp = _rand_importance(rng, K)
            groups = CppLeveledSplitDot.dot_split_leveled_default(w, x, imp)

            # 出现的粒度都是 {16,8,4} 的子集
            assert set(groups.keys()) <= {16, 8, 4}, f"出现非法粒度 {list(groups.keys())}"

            # 逐组：partials 长度 = 2*(32/b)-1；用 fuse_128 重建该组贡献
            rebuilt = 0
            for b, partials in groups.items():
                assert len(partials) == 2 * (32 // b) - 1, \
                    f"b={b} partials 长度 {len(partials)} != {2*(32//b)-1}"
                hi, lo = CppSplitDot.fuse_128(partials, b)
                rebuilt += (hi << 64) | lo
            # 各组融合（128 位）累加 == 原始点积
            assert _signed64(rebuilt) == _raw_dot(w, x), \
                f"seed={seed} K={K} 各组重建累加 != 原始点积"
    print("[PASS] L2 [T] dot_split_leveled 分组结构：partials 长度=2n-1，各组 fuse_128 累加 bit-exact == 原始点积")


# ============================================================
# L3 [S] 按需资源分配：重要性越高粒度越细（parts 数越多）
# ============================================================

def test_l3_resource_allocation():
    # 构造三段重要性，验证 select_levels 阈值语义 + parts 数单调
    rng = np.random.RandomState(123)
    K = 600
    # 低重要性 0..99，中 100..999，高 1000..2000
    imp = ([int(rng.randint(0, 100)) for _ in range(K // 3)] +
           [int(rng.randint(100, 1000)) for _ in range(K // 3)] +
           [int(rng.randint(1000, 2000)) for _ in range(K - 2 * (K // 3))])
    levels = CppLeveledSplitDot.select_levels_default(imp)

    parts = [32 // b for b in levels]
    low_parts = parts[0:len(imp) // 3]
    mid_parts = parts[len(imp) // 3: 2 * (len(imp) // 3)]
    high_parts = parts[2 * (len(imp) // 3):]

    # 高重要性组平均 parts 数 > 中 > 低
    assert np.mean(high_parts) > np.mean(mid_parts) > np.mean(low_parts), \
        f"parts 数未随重要性单调: low={np.mean(low_parts):.2f} mid={np.mean(mid_parts):.2f} high={np.mean(high_parts):.2f}"

    # 阈值语义精确：<100→b=16(2 parts)，100~999→b=8(4 parts)，>=1000→b=4(8 parts)
    for i, b in enumerate(levels):
        impv = imp[i]
        expect_b = 16 if impv < 100 else (8 if impv < 1000 else 4)
        assert b == expect_b, f"imp={impv} 应选 {expect_b} 实际 {b}"

    print(f"[PASS] L3 [S] 按需资源分配：低={np.mean(low_parts):.2f} 中={np.mean(mid_parts):.2f} "
          f"高={np.mean(high_parts):.2f} parts/元素（重要性越高越细）")


# ============================================================
# L4 [E] 异构/全粗/全细三种策略融合均 == 原始点积
# ============================================================

def test_l4_all_strategies_exact():
    for seed in RNG_SEEDS:
        rng = np.random.RandomState(seed)
        for _ in range(20):
            K = int(rng.randint(5, 400))
            w = _rand_int32(rng, K)
            x = _rand_int32(rng, K)
            imp = _rand_importance(rng, K)
            ref = _raw_dot(w, x)

            # 全粗（全 16 位）
            coarse = CppSplitDot.dot_fused(w, x, 32, 16)
            # 全细（全 4 位）
            fine = CppSplitDot.dot_fused(w, x, 32, 4)
            # 异构（按重要性选粒度）
            leveled = CppLeveledSplitDot.dot_fused_leveled_default(w, x, imp)

            assert coarse == ref, f"全粗 {coarse} != {ref}"
            assert fine == ref, f"全细 {fine} != {ref}"
            assert leveled == ref, f"异构 {leveled} != {ref}"
    print("[PASS] L4 [E] 全粗/全细/异构三种粒度策略融合均 == 原始点积（按需分配不损失精度）")


# ============================================================
# L5 C++ vs Python 参考实现
# ============================================================

def test_l5_cpp_vs_python():
    rng = np.random.RandomState(42)
    n_fail = 0
    for _ in range(200):
        K = int(rng.randint(1, 200))
        w = _rand_int32(rng, K)
        x = _rand_int32(rng, K)
        imp = _rand_importance(rng, K)

        # dot_split_leveled
        c_groups = CppLeveledSplitDot.dot_split_leveled_default(w, x, imp)
        p_groups = _PyLeveledSplitDot.dot_split_leveled_default(w, x, imp) \
            if hasattr(_PyLeveledSplitDot, "dot_split_leveled_default") \
            else _PyLeveledSplitDot.dot_split_leveled(w, x, imp)
        if c_groups != p_groups:
            n_fail += 1

        # dot_fused_leveled
        c_f = CppLeveledSplitDot.dot_fused_leveled_default(w, x, imp)
        p_f = _PyLeveledSplitDot.dot_fused_leveled(w, x, imp)
        if c_f != p_f:
            n_fail += 1

        # select_levels
        if CppLeveledSplitDot.select_levels_default(imp) != \
           _PyLeveledSplitDot.select_levels(imp):
            n_fail += 1

    assert n_fail == 0, f"C++/Python 不一致 {n_fail} 次"
    print(f"[PASS] L5 C++ vs Python 参考：200 组 × (select + split + fused) 全部一致（C++={USING_CPP}）")


# ============================================================
# L6 构造校验
# ============================================================

def test_l6_validation():
    def expect_error(fn, msg):
        try:
            fn()
            return False
        except Exception as e:
            return msg.lower() in str(e).lower()

    assert expect_error(
        lambda: CppLeveledSplitDot.dot_fused_leveled_default([1, 2], [1], [1, 2]),
        "长度"), "w/x 长度不一致应抛异常"
    assert expect_error(
        lambda: CppLeveledSplitDot.dot_fused_leveled_default([1, 2], [1, 2], [1]),
        "长度"), "w/importance 长度不一致应抛异常"
    assert expect_error(
        lambda: CppLeveledSplitDot.select_levels([1, 2], 32, [16, 8, 4], [100]),
        "长度"), "thresholds 长度应为 options-1 应抛异常"
    print("[PASS] L6 构造校验：长度不一致抛异常")


if __name__ == "__main__":
    print("=" * 78)
    print("MSint 逐元素异构粒度拆分点积验证（1:N + Level 逐元素精度选择组合）")
    print(f"C++ 模式: {USING_CPP}")
    print("=" * 78)
    print()

    test_l1_fused_equals_raw()
    test_l2_split_group_structure()
    test_l3_resource_allocation()
    test_l4_all_strategies_exact()
    test_l5_cpp_vs_python()
    test_l6_validation()

    print()
    print("=" * 78)
    print("=== All LeveledSplitDot 验证 PASSED ===")
    print("  - L1 [T] 异构粒度单融合 == 原始点积（bit-exact，H1 在异构下成立）")
    print("  - L2 [T] 按粒度分组 1:N 多输出结构 + 各组独立可重建")
    print("  - L3 [S] 按需资源分配（重要性越高粒度越细）")
    print("  - L4 [E] 全粗/全细/异构均精确等价（按需分配不损失精度）")
    print("  - L5 C++ vs Python 参考一致")
    print("  - L6 构造校验")
    print("=" * 78)
