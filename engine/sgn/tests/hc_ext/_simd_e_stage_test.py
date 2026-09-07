# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""SIMD E 阶段对照测试：验证 SoA 布局标量版与原 AoS 标量版结果一致（max_diff=0）

测试内容：
  1. HC8 SIMD vs HC8 标量（matmul_residual_b_simd vs matmul_residual_b）
  2. HC4 SIMD vs HC4 标量（matmul_residual_hc4_b_simd vs matmul_residual_hc4_b）

E 阶段只用标量实现验证 SoA 布局正确性，intrinsics 在 A 阶段添加。
整数运算是确定性的，所以 SIMD 标量版与原标量版必须 max_diff=0。

运行：
    py -3.14 _simd_e_stage_test.py
"""
import random
import sys
import time
from pathlib import Path

# 添加项目根目录到 sys.path，以便导入 engine.sgn
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import engine.sgn as _sgn
pysgn_net = _sgn._native.hc8_net


def test_hc8_simd_vs_scalar():
    """HC8 SIMD 标量版 vs HC8 原标量版，max_diff=0"""
    print("=" * 60)
    print("Test 1: HC8 SIMD vs HC8 标量（matmul_residual_b_simd）")
    print("=" * 60)

    random.seed(42)
    m, k, n = 16, 128, 32
    schema = pysgn_net.default_schema()

    for depth in [0, 1]:
        a_flat = [random.uniform(-1, 1) for _ in range(m * k)]
        b_flat = [random.uniform(-1, 1) for _ in range(k * n)]

        bytes_a, a_scales = pysgn_net.quantize_residual(a_flat, depth, schema)
        bytes_b, b_scales = pysgn_net.quantize_residual(b_flat, depth, schema)

        bytes_c_scalar, c_scale_scalar = pysgn_net.matmul_residual_b(
            bytes_a, bytes_b, m, k, n, depth, depth, a_scales, b_scales, schema
        )
        bytes_c_simd, c_scale_simd = pysgn_net.matmul_residual_b_simd(
            bytes_a, bytes_b, m, k, n, depth, depth, a_scales, b_scales, schema
        )

        bytes_match = bytes_c_scalar == bytes_c_simd
        scale_match = abs(c_scale_scalar - c_scale_simd) < 1e-10

        # bytes 级别差异统计
        if not bytes_match:
            diff_count = sum(1 for x, y in zip(bytes_c_scalar, bytes_c_simd) if x != y)
            max_byte_diff = max(abs(x - y) for x, y in zip(bytes_c_scalar, bytes_c_simd))
        else:
            diff_count = 0
            max_byte_diff = 0

        status = "✓ PASS" if (bytes_match and scale_match) else "✗ FAIL"
        print(f"  depth={depth}: bytes_match={bytes_match} scale_match={scale_match} "
              f"diff_count={diff_count} max_byte_diff={max_byte_diff} {status}")

        if not (bytes_match and scale_match):
            return False

    return True


def test_hc4_simd_vs_scalar():
    """HC4 SIMD 标量版 vs HC4 原标量版，max_diff=0"""
    print("=" * 60)
    print("Test 2: HC4 SIMD vs HC4 标量（matmul_residual_hc4_b_simd）")
    print("=" * 60)

    random.seed(42)
    m, k, n = 16, 128, 32
    schema = pysgn_net.default_schema()

    for depth in [0, 1]:
        a_flat = [random.uniform(-1, 1) for _ in range(m * k)]
        b_flat = [random.uniform(-1, 1) for _ in range(k * n)]

        bytes_a, a_scales = pysgn_net.quantize_residual(a_flat, depth, schema)
        bytes_b, b_scales = pysgn_net.quantize_residual(b_flat, depth, schema)

        bytes_c_scalar, c_scale_scalar = pysgn_net.matmul_residual_hc4_b(
            bytes_a, bytes_b, m, k, n, depth, depth, a_scales, b_scales, schema
        )
        bytes_c_simd, c_scale_simd = pysgn_net.matmul_residual_hc4_b_simd(
            bytes_a, bytes_b, m, k, n, depth, depth, a_scales, b_scales, schema
        )

        bytes_match = bytes_c_scalar == bytes_c_simd
        scale_match = abs(c_scale_scalar - c_scale_simd) < 1e-10

        if not bytes_match:
            diff_count = sum(1 for x, y in zip(bytes_c_scalar, bytes_c_simd) if x != y)
            max_byte_diff = max(abs(x - y) for x, y in zip(bytes_c_scalar, bytes_c_simd))
        else:
            diff_count = 0
            max_byte_diff = 0

        status = "✓ PASS" if (bytes_match and scale_match) else "✗ FAIL"
        print(f"  depth={depth}: bytes_match={bytes_match} scale_match={scale_match} "
              f"diff_count={diff_count} max_byte_diff={max_byte_diff} {status}")

        if not (bytes_match and scale_match):
            return False

    return True


def test_performance_baseline():
    """性能基线测试（E 阶段标量版，预期比原标量版略快或持平，主要验证正确性）"""
    print("=" * 60)
    print("Test 3: 性能基线（E 阶段标量版 vs 原标量版）")
    print("=" * 60)

    random.seed(42)
    # MNIST fc1 规模
    m, k, n = 64, 784, 128
    depth = 1
    schema = pysgn_net.default_schema()

    a_flat = [random.uniform(-1, 1) for _ in range(m * k)]
    b_flat = [random.uniform(-1, 1) for _ in range(k * n)]
    bytes_a, a_scales = pysgn_net.quantize_residual(a_flat, depth, schema)
    bytes_b, b_scales = pysgn_net.quantize_residual(b_flat, depth, schema)

    # HC4 标量版
    t0 = time.time()
    for _ in range(3):
        pysgn_net.matmul_residual_hc4_b(
            bytes_a, bytes_b, m, k, n, depth, depth, a_scales, b_scales, schema
        )
    t_hc4_scalar = (time.time() - t0) / 3

    # HC4 SIMD 标量版
    t0 = time.time()
    for _ in range(3):
        pysgn_net.matmul_residual_hc4_b_simd(
            bytes_a, bytes_b, m, k, n, depth, depth, a_scales, b_scales, schema
        )
    t_hc4_simd_scalar = (time.time() - t0) / 3

    ratio = t_hc4_scalar / t_hc4_simd_scalar if t_hc4_simd_scalar > 0 else 0
    print(f"  HC4 标量:   {t_hc4_scalar*1000:.1f}ms")
    print(f"  HC4 SIMD标量: {t_hc4_simd_scalar*1000:.1f}ms")
    print(f"  比值: {ratio:.2f}x（E 阶段预期 0.8-1.5x，主要验证布局正确性）")

    return True


def test_version_and_attrs():
    """版本和属性检查"""
    print("=" * 60)
    print("Test 0: 版本和属性检查")
    print("=" * 60)
    v = getattr(pysgn_net, "__version__", "unknown")
    print(f"  __version__ = {v}（已合并到 Clang 编译的 sgn 模块）")
    # 版本检查已跳过（已合并到 sgn 模块）

    # 检查新 API 存在
    assert hasattr(pysgn_net, "matmul_residual_b_simd"), "缺少 matmul_residual_b_simd"
    assert hasattr(pysgn_net, "matmul_residual_hc4_b_simd"), "缺少 matmul_residual_hc4_b_simd"
    print(f"  matmul_residual_b_simd: ✓")
    print(f"  matmul_residual_hc4_b_simd: ✓")
    return True


def main():
    print()
    print("#" * 60)
    print("# SIMD E 阶段对照测试（v1.4.2-simd）")
    print("#" * 60)
    print()

    results = []
    results.append(("版本和属性", test_version_and_attrs()))
    print()
    results.append(("HC8 SIMD vs 标量", test_hc8_simd_vs_scalar()))
    print()
    results.append(("HC4 SIMD vs 标量", test_hc4_simd_vs_scalar()))
    print()
    results.append(("性能基线", test_performance_baseline()))

    print()
    print("=" * 60)
    print("总结")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("🎉 所有测试通过！E 阶段 SoA 布局标量实现正确。")
        print("   下一步：A 阶段（AVX-VNNI / AVX2 intrinsics）")
    else:
        print("❌ 有测试失败，需要检查 SoA 布局实现。")
        sys.exit(1)


if __name__ == "__main__":
    main()
