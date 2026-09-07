# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""MSint 1:N 多精度解释 + Level 逐元素精度选择验证

对标范式文档 docs/msint_multisplit_paradigm/MSint多精度拆分计算范式.md
  - §1.2 / H4: 1:N 多精度解释（MultiScaleView）
  - §3.2 / H4: Level 逐元素精度选择（PrecisionSelector）

验证项：
  - M1: MultiScaleView.interpret 对 int32 同时产出 {16,8,4} 三层次，
        每层 parts 独立重建 == 原值（含负数、边界值）
  - M2: interpret_batch 批量 1:N，形状与每层可逆
  - M3: PrecisionSelector.select 按重要性选择粒度（阈值语义）
  - M4: PrecisionSelector.interpret_batch 逐元素选粒度 + 精确重建（组合）
  - M5: 构造校验（非法 options / thresholds 抛异常）
  - M6: C++ vs Python 参考实现一致

运行: py -3.14 validate_math_msint_multiscale.py
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
    CppMultiScaleView, CppPrecisionSelector, _PyMultiScaleView,
    _PyPrecisionSelector, USING_CPP,
)


def _rand_int32(rng, n):
    return [int(rng.randint(-2**31, 2**31 - 1, dtype=np.int64)) for _ in range(n)]


def reconstruct(parts, split_bits):
    return sum(p << (k * split_bits) for k, p in enumerate(parts))


# ============================================================
# M1: interpret 1:N 多精度解释 + 每层可逆
# ============================================================

def test_m1_interpret_multilevel_reversible():
    rng = np.random.RandomState(20260813)
    edges = [0, 1, -1, 2**31 - 1, -2**31, 2**16 - 1, -2**16, 12345, -98765]
    vals = edges + _rand_int32(rng, 5000)

    for v in vals:
        lv = CppMultiScaleView.interpret(v, 32)
        assert set(lv.keys()) == {16, 8, 4}, f"默认层次应为 {{16,8,4}}，实际 {list(lv.keys())}"
        for b, parts in lv.items():
            assert len(parts) == 32 // b, f"split_bits={b} 应有 {32//b} 部分，实际 {len(parts)}"
            assert reconstruct(parts, b) == v, f"v={v} split_bits={b} 重建不一致"

    print(f"[PASS] M1 interpret 1:N：int32 同时产出 {{16,8,4}}，{len(vals)} 值（含边界/负数）每层精确可逆")


def test_m1_interpret_levels_custom():
    v = -123456789
    lv = CppMultiScaleView.interpret_levels(v, 32, [8, 4])
    assert set(lv.keys()) == {8, 4}
    for b, parts in lv.items():
        assert reconstruct(parts, b) == v
    print("[PASS] M1 interpret_levels：自定义层次 {8,4} 可逆")


def test_m1_default_levels():
    assert CppMultiScaleView.default_levels(32) == [16, 8, 4]
    assert CppMultiScaleView.default_levels(64) == [32, 16, 8, 4]
    assert CppMultiScaleView.default_levels(8) == [4]
    print("[PASS] M1 default_levels：32→[16,8,4]，64→[32,16,8,4]，8→[4]")


# ============================================================
# M2: interpret_batch 批量 1:N
# ============================================================

def test_m2_interpret_batch():
    rng = np.random.RandomState(7)
    K = 200
    vals = _rand_int32(rng, K)

    res = CppMultiScaleView.interpret_batch(vals, 32)
    assert set(res.keys()) == {16, 8, 4}
    for b, matrix in res.items():
        assert len(matrix) == K
        for i, parts in enumerate(matrix):
            assert len(parts) == 32 // b
            assert reconstruct(parts, b) == vals[i]

    # 指定层次
    res8 = CppMultiScaleView.interpret_batch(vals, 32, [8])
    assert set(res8.keys()) == {8}
    print(f"[PASS] M2 interpret_batch：{K} 元素 × {{16,8,4}} 全部可逆（含自定义层次）")


# ============================================================
# M3: PrecisionSelector.select 阈值语义
# ============================================================

def test_m3_select_thresholds():
    ps = CppPrecisionSelector.default_selector()  # options={16,8,4}, thresholds={100,1000}
    assert ps.total_bits == 32
    assert ps.options == [16, 8, 4]
    assert ps.thresholds == [100, 1000]

    cases = [(-5, 16), (0, 16), (99, 16), (100, 8), (500, 8), (999, 8),
             (1000, 4), (10000, 4), (2**31, 4)]
    for imp, expect in cases:
        got = ps.select(imp)
        assert got == expect, f"importance={imp} 应选 {expect}，实际 {got}"

    print("[PASS] M3 select 阈值语义：<100→16，100~999→8，>=1000→4")


def test_m3_custom_selector():
    ps = CppPrecisionSelector(64, [32, 16, 8], [50, 500])
    assert ps.select(0) == 32
    assert ps.select(50) == 16
    assert ps.select(500) == 8
    assert ps.select(10**6) == 8
    print("[PASS] M3 自定义选择器：total_bits=64, options={32,16,8}, thresholds={50,500}")


# ============================================================
# M4: PrecisionSelector 逐元素选择 + 解释组合
# ============================================================

def test_m4_interpret_batch_combo():
    rng = np.random.RandomState(99)
    K = 300
    vals = _rand_int32(rng, K)
    importances = [int(rng.randint(0, 2000)) for _ in range(K)]

    ps = CppPrecisionSelector.default_selector()
    result = ps.interpret_batch(vals, importances)

    assert len(result) == K
    for i, parts in enumerate(result):
        # 每个元素按 select(importance) 粒度拆分，parts 数 = 32/select
        b = ps.select(importances[i])
        assert len(parts) == 32 // b, f"elem{i} 粒度={b} 应有 {32//b} 部分"
        assert reconstruct(parts, b) == vals[i], f"elem{i} 重建不一致"

    print(f"[PASS] M4 组合：{K} 元素按重要性逐元素选粒度并精确重建（粒度数随重要性单调细化）")


# ============================================================
# M5: 构造校验
# ============================================================

def test_m5_constructor_validation():
    def expect_error(fn, msg):
        try:
            fn()
            return False
        except Exception as e:
            return msg.lower() in str(e).lower()

    # options 非从粗到细
    assert expect_error(lambda: CppPrecisionSelector(32, [8, 16], [100]), "递减")
    # thresholds 长度不符
    assert expect_error(lambda: CppPrecisionSelector(32, [16, 8], [100, 200]), "长度")
    # thresholds 非升序
    assert expect_error(lambda: CppPrecisionSelector(32, [16, 8, 4], [500, 100]), "升序")
    # split_bits 不能整除 total_bits
    assert expect_error(lambda: CppPrecisionSelector(32, [12, 4], [100]), "整除")

    # is_exact 反向检查
    assert CppMultiScaleView.is_exact(12345, 32, 8) is True
    assert CppMultiScaleView.is_exact(-1, 32, 16) is True
    assert CppMultiScaleView.is_exact(-99999, 32, 4) is True

    print("[PASS] M5 构造校验：非法 options/thresholds 抛异常，is_exact 边界正确")


# ============================================================
# M6: C++ vs Python 参考实现
# ============================================================

def test_m6_cpp_vs_python():
    rng = np.random.RandomState(42)
    n_fail = 0
    for _ in range(300):
        v = int(rng.randint(-2**31, 2**31 - 1, dtype=np.int64))
        # MultiScaleView
        cpp = CppMultiScaleView.interpret(v, 32)
        py = _PyMultiScaleView.interpret(v, 32)
        if cpp != py:
            n_fail += 1

        # PrecisionSelector 选择 + 解释
        imp = int(rng.randint(0, 2000))
        ps_c = CppPrecisionSelector.default_selector()
        ps_p = _PyPrecisionSelector.default_selector()
        if ps_c.select(imp) != ps_p.select(imp):
            n_fail += 1
        if ps_c.interpret(v, imp) != ps_p.interpret(v, imp):
            n_fail += 1

    assert n_fail == 0, f"C++/Python 不一致 {n_fail} 次"
    print(f"[PASS] M6 C++ vs Python 参考：300 组 × (MultiScaleView + PrecisionSelector) 全部一致（C++={USING_CPP}）")


if __name__ == "__main__":
    print("=" * 78)
    print("MSint 1:N 多精度解释 + Level 逐元素精度选择验证")
    print(f"C++ 模式: {USING_CPP}")
    print("=" * 78)
    print()

    test_m1_interpret_multilevel_reversible()
    test_m1_interpret_levels_custom()
    test_m1_default_levels()
    test_m2_interpret_batch()
    test_m3_select_thresholds()
    test_m3_custom_selector()
    test_m4_interpret_batch_combo()
    test_m5_constructor_validation()
    test_m6_cpp_vs_python()

    print()
    print("=" * 78)
    print("=== All MultiScale/PrecisionSelector 验证 PASSED ===")
    print("  - M1 1:N 多精度解释：int32 → {16,8,4} 每层精确可逆")
    print("  - M2 批量 1:N 解释")
    print("  - M3 Level 逐元素精度选择（阈值语义）")
    print("  - M4 选择 + 解释组合（逐元素粒度 + 精确重建）")
    print("  - M5 构造校验")
    print("  - M6 C++ vs Python 参考一致")
    print("=" * 78)
