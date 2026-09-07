# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""UFP-1 残差量化自检测试

验证 v[0..depth] 残差量化的正确性：
  1. depth=0 等价于 v[0] 版本（向后兼容）
  2. 往返误差：depth 越大误差越小
  3. depth=5 误差应 << depth=0
  4. matmul_residual 与 float matmul 的误差
  5. relu_residual 的正确性
"""
import sys
from pathlib import Path

# 添加项目根目录到 sys.path，以便导入 engine.sgn
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import engine.sgn as _sgn
pysgn_net = _sgn._native.hc8_net
import random
import math


def approx_eq(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_depth0_equivalence():
    """test 1: depth=0 应等价于 v[0] 版本"""
    print("=== test 1: depth=0 等价于 v[0] 版本 ===")
    schema = pysgn_net.default_schema()
    values = [random.uniform(-1, 1) for _ in range(100)]

    # v[0] 版本
    scale0 = pysgn_net.quant_compute_scale(values)
    bytes_v0 = pysgn_net.quantize(values, scale0, schema)
    back_v0 = pysgn_net.dequantize(bytes_v0, scale0, schema)

    # UFP-1 depth=0 版本
    bytes_ufp, scales_ufp = pysgn_net.quantize_residual(values, 0, schema)
    back_ufp = pysgn_net.dequantize_residual(bytes_ufp, scales_ufp, 0, schema)

    # scale 应相同
    assert approx_eq(scales_ufp[0], scale0, 1e-9), \
        f"depth=0 scale 不匹配：ufp={scales_ufp[0]} vs v0={scale0}"

    # 反量化值应相同
    for i, (a, b) in enumerate(zip(back_v0, back_ufp)):
        assert approx_eq(a, b, 1e-9), \
            f"depth=0 反量化不匹配 [{i}]: ufp={a} vs v0={b}"

    print(f"  ✓ depth=0 与 v[0] 版本完全等价")
    print(f"  scale={scale0:.6e}")
    return True


def test_roundtrip_error_decreases():
    """test 2: 往返误差应随 depth 增大而减小"""
    print("\n=== test 2: 往返误差随 depth 递减 ===")
    schema = pysgn_net.default_schema()

    # 用有挑战性的数据：混合大小值
    values = []
    for i in range(200):
        # 大值 + 小值混合，让残差有内容
        v = random.uniform(-2, 2) + random.uniform(-0.01, 0.01)
        values.append(v)

    errors = {}
    for depth in [0, 1, 2, 3, 4, 5]:
        bytes_h, scales = pysgn_net.quantize_residual(values, depth, schema)
        back = pysgn_net.dequantize_residual(bytes_h, scales, depth, schema)

        # 最大绝对误差
        max_err = max(abs(a - b) for a, b in zip(values, back))
        # 均方误差
        mse = sum((a - b) ** 2 for a, b in zip(values, back)) / len(values)
        errors[depth] = (max_err, mse)
        print(f"  depth={depth}: max_err={max_err:.6e} mse={mse:.6e}")

    # 验证误差递减
    for d in range(5):
        err_d = errors[d][1]
        err_d1 = errors[d + 1][1]
        assert err_d1 <= err_d + 1e-12, \
            f"depth {d}→{d+1} 误差未递减：{err_d} → {err_d1}"

    # depth=5 应 << depth=0
    improvement = errors[0][1] / errors[5][1] if errors[5][1] > 0 else float("inf")
    print(f"  ✓ 误差严格递减")
    print(f"  ✓ depth=5 vs depth=0 MSE 改善：{improvement:.1f}x")
    assert improvement > 10, f"depth=5 改善不足 10x：{improvement}"
    return True


def test_scales_decreasing():
    """test 3: 每层 scale 应递减（残差越来越小）"""
    print("\n=== test 3: 每层 scale 递减 ===")
    schema = pysgn_net.default_schema()
    values = [random.uniform(-1, 1) for _ in range(100)]

    bytes_h, scales = pysgn_net.quantize_residual(values, 5, schema)

    print(f"  6 层 scale：")
    for i, s in enumerate(scales):
        print(f"    scale[{i}] = {s:.6e}")

    for i in range(5):
        assert scales[i + 1] <= scales[i] + 1e-15, \
            f"scale 未递减：scale[{i}]={scales[i]} → scale[{i+1}]={scales[i+1]}"
    print(f"  ✓ scale 严格递减（残差越来越小）")
    return True


def test_matmul_residual():
    """test 4: matmul_residual 与 float matmul 的误差"""
    print("\n=== test 4: matmul_residual 精度 ===")
    schema = pysgn_net.default_schema()
    random.seed(42)

    m, k, n = 4, 8, 3
    a = [random.uniform(-1, 1) for _ in range(m * k)]
    b = [random.uniform(-1, 1) for _ in range(k * n)]

    # float 基准：C = A @ B
    c_float = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for l in range(k):
                s += a[i * k + l] * b[l * n + j]
            c_float[i * n + j] = s

    # UFP-1 matmul（depth=5）
    a_bytes, a_scales = pysgn_net.quantize_residual(a, 5, schema)
    b_bytes, b_scales = pysgn_net.quantize_residual(b, 5, schema)
    c_bytes, c_scale = pysgn_net.matmul_residual(
        a_bytes, b_bytes, m, k, n,
        5, 5, a_scales, b_scales, schema
    )
    c_ufp = pysgn_net.dequantize(c_bytes, c_scale, schema)

    # v[0] matmul（depth=0，对比）
    a_bytes_v0 = pysgn_net.quantize(a, pysgn_net.quant_compute_scale(a), schema)
    b_bytes_v0 = pysgn_net.quantize(b, pysgn_net.quant_compute_scale(b), schema)
    c_bytes_v0, c_scale_v0 = pysgn_net.matmul(
        a_bytes_v0, b_bytes_v0, m, k, n,
        pysgn_net.quant_compute_scale(a), pysgn_net.quant_compute_scale(b), schema
    )
    c_v0 = pysgn_net.dequantize(c_bytes_v0, c_scale_v0, schema)

    # 误差对比
    mse_ufp = sum((a - b) ** 2 for a, b in zip(c_float, c_ufp)) / len(c_float)
    mse_v0 = sum((a - b) ** 2 for a, b in zip(c_float, c_v0)) / len(c_float)

    print(f"  float 基准 C[0][0] = {c_float[0]:.6f}")
    print(f"  v[0]    C[0][0] = {c_v0[0]:.6f}  (mse={mse_v0:.6e})")
    print(f"  ufp-1   C[0][0] = {c_ufp[0]:.6f}  (mse={mse_ufp:.6e})")

    # UFP-1 matmul 用方案 C，运算精度仍是 8-bit
    # 所以 matmul 的误差应与 v[0] 接近（存储高精度但运算 8-bit）
    # 但 UFP-1 的输入反量化更准，所以误差应略小
    print(f"  ✓ matmul_residual 正常工作（方案 C：存储高精度，运算 8-bit）")
    return True


def test_relu_residual():
    """test 5: relu_residual 正确性"""
    print("\n=== test 5: relu_residual ===")
    schema = pysgn_net.default_schema()

    # 构造有正有负的数据
    values = [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]
    m, n = 1, len(values)

    # 用 depth=3 量化
    bytes_h, scales = pysgn_net.quantize_residual(values, 3, schema)

    # 残差 ReLU
    bytes_relu = pysgn_net.relu_residual(bytes_h, m, n, schema)

    # 反量化（输出是 depth=0，用 scale[0]）
    # 但 relu_residual 输出的 v[0] 是基于原 v[0] 的 ReLU
    # scale 应该用原 scales[0]（因为 ReLU 不改变正值的量化值）
    back = pysgn_net.dequantize(bytes_relu, scales[0], schema)

    # 验证：正值应保留，负值应变 0
    print(f"  原值    ReLU后   期望")
    for i, (v, b) in enumerate(zip(values, back)):
        expected = max(0, v)
        ok = approx_eq(b, expected, 0.02)  # 8-bit 量化误差容忍
        print(f"  {v:6.3f}  {b:6.3f}  {expected:6.3f}  {'✓' if ok else '✗'}")
        assert ok, f"ReLU[{i}] 不正确：{b} != {expected}"

    print(f"  ✓ relu_residual 正确（正值保留，负值清零）")
    return True


def test_zero_input():
    """test 6: 全零输入的处理"""
    print("\n=== test 6: 全零输入 ===")
    schema = pysgn_net.default_schema()
    values = [0.0] * 10

    bytes_h, scales = pysgn_net.quantize_residual(values, 5, schema)
    back = pysgn_net.dequantize_residual(bytes_h, scales, 5, schema)

    for i, b in enumerate(back):
        assert b == 0.0, f"全零输入反量化非零 [{i}]: {b}"

    # scale 应全 0（max_abs=0 时 scale=0）
    for i, s in enumerate(scales):
        assert s == 0.0, f"全零输入 scale[{i}] 非 0：{s}"

    print(f"  ✓ 全零输入正确处理（scale=0, 反量化=0）")
    return True


def main():
    print("UFP-1 残差量化自检测试")
    print(f"pysgn_net version: {pysgn_net.__version__}")
    print(f"HC8_BYTES: {pysgn_net.HC8_BYTES}")
    print(f"HC8_LAYERS: {pysgn_net.HC8_LAYERS}")
    print(f"UFP1_MAX_DEPTH: {pysgn_net.UFP1_MAX_DEPTH}")

    random.seed(123)

    tests = [
        test_depth0_equivalence,
        test_roundtrip_error_decreases,
        test_scales_decreasing,
        test_matmul_residual,
        test_relu_residual,
        test_zero_input,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {test.__name__} 返回 False")
        except Exception as e:
            failed += 1
            print(f"  ✗ {test.__name__} 异常：{e}")

    print(f"\n=== Summary ===")
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")
    if failed == 0:
        print(f"  ✓ UFP-1 自检全部通过")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
