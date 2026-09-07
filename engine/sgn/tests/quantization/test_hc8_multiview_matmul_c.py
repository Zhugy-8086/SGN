#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""hc8_multiview_matmul C 扩展对照测试

验证 C 扩展实现（pysgn_net.multiview_matmul）与 Python 参考实现
（hc_tree_unified.matmul_multiview）数学等价（max_diff=0）。

测试项：
  1. C 扩展 vs Python 参考实现（4 视角，C_acc >= 0）
  2. C 扩展 vs Python 参考实现（8 视角，C_acc 有符号）
  3. C 扩展 vs int64 基准（直接 C_acc @ b_int8）
  4. 不同矩阵维度（小/中/大）
  5. n_views=1 退化情况（等价于首层 uint8×int8 matmul）
  6. 错误处理（n_views 越界、形状不匹配）

关联：
  - C 实现：engine/sgn/hc/ext/hc8_net.c（hc8_multiview_matmul）
    （安全审计 2026-08-16：原注释引用 内部档案，已过时）
  - Python 参考：legacy/traditional/stage_2_3_int_path/hc_tree_unified.py（matmul_multiview）
  - 数学框架：内部档案
"""

import sys
import os
import numpy as np

# 添加路径
SGN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
# 参考实现位于 tests/refs/（从 legacy/traditional 复制，见 refs/__init__.py）
STAGE_2_3_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "refs"))

sys.path.insert(0, SGN_ROOT)
sys.path.insert(0, STAGE_2_3_DIR)

import engine.sgn as _sgn
pysgn_net = _sgn._native.hc8_net
from hc_tree_unified import matmul_multiview, relu_parallel, quantize_int8


def test_1_c_vs_python_4views_nonneg():
    """测试 1：C 扩展 vs Python 参考实现（4 视角，C_acc >= 0）"""
    print("\n--- 测试 1：C 扩展 vs Python（4 视角，C_acc >= 0）---")
    np.random.seed(42)
    m, k, n = 8, 16, 12

    # 生成非负 int64 累加值（模拟 ReLU 后的输出）
    C_acc = np.random.randint(0, 2**31, size=(m, k), dtype=np.int64)

    # 生成 int8 权重
    w = np.random.randn(k, n).astype(np.float64) * 0.1
    b_int8, _ = quantize_int8(w)

    # Python 参考实现
    C_py = matmul_multiview(C_acc, b_int8, m, k, n, n_views=4)

    # C 扩展
    C_c = pysgn_net.multiview_matmul(
        C_acc.astype(np.int64),
        b_int8.astype(np.int8),
        m, k, n, 4
    )

    max_diff = np.abs(C_py - C_c).max()
    print(f"  矩阵维度: ({m}, {k}) @ ({k}, {n})")
    print(f"  Python max={C_py.max()}, min={C_py.min()}")
    print(f"  C 扩展 max={C_c.max()}, min={C_c.min()}")
    print(f"  max_diff = {max_diff}")

    assert max_diff == 0, f"C 扩展 vs Python 应完全一致, max_diff={max_diff}"
    print("  ✓ C 扩展与 Python 参考实现完全一致（max_diff=0）")


def test_2_c_vs_python_8views_signed():
    """测试 2：C 扩展 vs Python 参考实现（8 视角，C_acc 有符号）"""
    print("\n--- 测试 2：C 扩展 vs Python（8 视角，C_acc 有符号）---")
    np.random.seed(123)
    m, k, n = 6, 10, 8

    # 生成有符号 int64 累加值（8 视角可覆盖 int64）
    C_acc = np.random.randint(-2**40, 2**40, size=(m, k), dtype=np.int64)

    # 生成 int8 权重
    w = np.random.randn(k, n).astype(np.float64) * 0.1
    b_int8, _ = quantize_int8(w)

    # Python 参考实现
    C_py = matmul_multiview(C_acc, b_int8, m, k, n, n_views=8)

    # C 扩展
    C_c = pysgn_net.multiview_matmul(
        C_acc.astype(np.int64),
        b_int8.astype(np.int8),
        m, k, n, 8
    )

    max_diff = np.abs(C_py - C_c).max()
    print(f"  矩阵维度: ({m}, {k}) @ ({k}, {n})")
    print(f"  C_acc 范围: [{C_acc.min()}, {C_acc.max()}]")
    print(f"  max_diff = {max_diff}")

    assert max_diff == 0, f"C 扩展 vs Python 应完全一致, max_diff={max_diff}"
    print("  ✓ C 扩展与 Python 参考实现完全一致（max_diff=0，8 视角有符号）")


def test_3_c_vs_int64_baseline():
    """测试 3：C 扩展 vs int64 基准（直接 C_acc @ b_int8）"""
    print("\n--- 测试 3：C 扩展 vs int64 基准 ---")
    np.random.seed(456)
    m, k, n = 4, 8, 6

    # 非负 int64（4 视角覆盖 int32）
    C_acc = np.random.randint(0, 2**20, size=(m, k), dtype=np.int64)
    w = np.random.randn(k, n).astype(np.float64) * 0.1
    b_int8, _ = quantize_int8(w)

    # int64 基准：直接 matmul
    C_ref = C_acc.astype(np.int64) @ b_int8.astype(np.int64)

    # C 扩展（4 视角，C_acc >= 0）
    C_c = pysgn_net.multiview_matmul(
        C_acc.astype(np.int64),
        b_int8.astype(np.int8),
        m, k, n, 4
    )

    max_diff = np.abs(C_ref - C_c).max()
    print(f"  矩阵维度: ({m}, {k}) @ ({k}, {n})")
    print(f"  max_diff = {max_diff}")

    assert max_diff == 0, f"C 扩展 vs int64 基准应完全一致, max_diff={max_diff}"
    print("  ✓ C 扩展与 int64 基准完全一致（max_diff=0）")


def test_4_various_dimensions():
    """测试 4：不同矩阵维度"""
    print("\n--- 测试 4：不同矩阵维度 ---")
    np.random.seed(789)

    test_cases = [
        (2, 4, 3, "小矩阵"),
        (16, 32, 24, "中矩阵"),
        (64, 128, 96, "大矩阵"),
    ]

    all_pass = True
    for m, k, n, desc in test_cases:
        C_acc = np.random.randint(0, 2**24, size=(m, k), dtype=np.int64)
        w = np.random.randn(k, n).astype(np.float64) * 0.1
        b_int8, _ = quantize_int8(w)

        C_py = matmul_multiview(C_acc, b_int8, m, k, n, n_views=4)
        C_c = pysgn_net.multiview_matmul(
            C_acc.astype(np.int64),
            b_int8.astype(np.int8),
            m, k, n, 4
        )

        max_diff = np.abs(C_py - C_c).max()
        status = "✓" if max_diff == 0 else "✗"
        print(f"  {desc} ({m},{k})@({k},{n}): max_diff={max_diff} {status}")

        if max_diff != 0:
            all_pass = False

    assert all_pass, "部分维度测试失败"
    print("  ✓ 所有维度测试通过")


def test_5_n_views_1_degenerate():
    """测试 5：n_views=1 退化情况（等价于 uint8 视角 @ b_int8）"""
    print("\n--- 测试 5：n_views=1 退化情况 ---")
    np.random.seed(321)
    m, k, n = 4, 6, 5

    C_acc = np.random.randint(0, 200, size=(m, k), dtype=np.int64)
    w = np.random.randn(k, n).astype(np.float64) * 0.1
    b_int8, _ = quantize_int8(w)

    # n_views=1：只取最低字节
    C_byte0 = (C_acc & 0xFF).astype(np.int64)
    C_ref = C_byte0 @ b_int8.astype(np.int64)

    C_c = pysgn_net.multiview_matmul(
        C_acc.astype(np.int64),
        b_int8.astype(np.int8),
        m, k, n, 1
    )

    max_diff = np.abs(C_ref - C_c).max()
    print(f"  max_diff = {max_diff}")

    assert max_diff == 0, f"n_views=1 退化应等价 uint8 视角 matmul, max_diff={max_diff}"
    print("  ✓ n_views=1 退化正确（等价最低字节视角 matmul）")


def test_6_error_handling():
    """测试 6：错误处理（n_views 越界、形状不匹配）"""
    print("\n--- 测试 6：错误处理 ---")
    np.random.seed(654)
    m, k, n = 4, 8, 6

    C_acc = np.random.randint(0, 1000, size=(m, k), dtype=np.int64)
    b_int8 = np.random.randint(-127, 127, size=(k, n), dtype=np.int8)

    # n_views 越界
    try:
        pysgn_net.multiview_matmul(C_acc, b_int8, m, k, n, 0)
        print("  ✗ n_views=0 应抛出异常")
        assert False
    except Exception as e:
        print(f"  ✓ n_views=0 正确抛出异常: {type(e).__name__}")

    try:
        pysgn_net.multiview_matmul(C_acc, b_int8, m, k, n, 9)
        print("  ✗ n_views=9 应抛出异常")
        assert False
    except Exception as e:
        print(f"  ✓ n_views=9 正确抛出异常: {type(e).__name__}")

    # 形状不匹配
    try:
        bad_b = np.random.randint(-127, 127, size=(k+1, n), dtype=np.int8)
        pysgn_net.multiview_matmul(C_acc, bad_b, m, k, n, 4)
        print("  ✗ b_int8 形状不匹配应抛出异常")
        assert False
    except Exception as e:
        print(f"  ✓ b_int8 形状不匹配正确抛出异常: {type(e).__name__}")

    print("  ✓ 所有错误处理测试通过")


def test_7_multilayer_pipeline():
    """测试 7：多层 pipeline（首层 int8 matmul → ReLU → 多视角 matmul → ReLU → 多视角 matmul）"""
    print("\n--- 测试 7：多层 pipeline（3 层）---")
    np.random.seed(999)
    m = 4
    # 3 层维度：(k_in, k_out) per layer
    dims = [(8, 6), (6, 5), (5, 4)]

    # 生成权重
    weights = []
    for k_in, k_out in dims:
        w = np.random.randn(k_in, k_out).astype(np.float64) * 0.1
        w_int8, w_scale = quantize_int8(w)
        weights.append((w_int8, w_scale))

    # 首层输入: (m, dims[0][0]) = (4, 8)
    x = np.random.randn(m, dims[0][0]).astype(np.float64) * 0.5
    x_int8, x_scale = quantize_int8(x)

    # 首层：标准 int8 × int8 → int64 累加: (4, 8) @ (8, 6) → (4, 6)
    C = x_int8.astype(np.int64) @ weights[0][0].astype(np.int64)

    # 第 2 层：ReLU → 多视角 matmul: (4, 6) @ (6, 5) → (4, 5)
    C = relu_parallel(C)  # C >= 0
    C_py = matmul_multiview(C, weights[1][0], m, dims[1][0], dims[1][1], n_views=4)
    C_c = pysgn_net.multiview_matmul(
        C.astype(np.int64),
        weights[1][0].astype(np.int8),
        m, dims[1][0], dims[1][1], 4
    )
    max_diff_2 = np.abs(C_py - C_c).max()

    # 第 3 层：ReLU → 多视角 matmul: (4, 5) @ (5, 4) → (4, 4)
    C_py = relu_parallel(C_py)
    C_c = relu_parallel(C_c)
    C_py3 = matmul_multiview(C_py, weights[2][0], m, dims[2][0], dims[2][1], n_views=4)
    C_c3 = pysgn_net.multiview_matmul(
        C_c.astype(np.int64),
        weights[2][0].astype(np.int8),
        m, dims[2][0], dims[2][1], 4
    )
    max_diff_3 = np.abs(C_py3 - C_c3).max()

    print(f"  首层: ({m},{dims[0][0]})@({dims[0][0]},{dims[0][1]}) → ({m},{dims[0][1]})")
    print(f"  第 2 层 max_diff = {max_diff_2}")
    print(f"  第 3 层 max_diff = {max_diff_3}")

    assert max_diff_2 == 0, f"第 2 层 C 扩展 vs Python 不一致, max_diff={max_diff_2}"
    assert max_diff_3 == 0, f"第 3 层 C 扩展 vs Python 不一致, max_diff={max_diff_3}"
    print("  ✓ 3 层 pipeline 全部一致（max_diff=0）")


def main():
    print("=" * 70)
    print("hc8_multiview_matmul C 扩展对照测试")
    print(f"pysgn_net 版本: {pysgn_net.__version__}")
    print("=" * 70)

    tests = [
        test_1_c_vs_python_4views_nonneg,
        test_2_c_vs_python_8views_signed,
        test_3_c_vs_int64_baseline,
        test_4_various_dimensions,
        test_5_n_views_1_degenerate,
        test_6_error_handling,
        test_7_multilayer_pipeline,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ 失败: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"结果: {passed}/{passed + failed} 通过")
    if failed == 0:
        print("=== 全部通过 ===")
    else:
        print(f"=== {failed} 项失败 ===")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
