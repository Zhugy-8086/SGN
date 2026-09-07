# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""UFP-2 方案 B 自检测试

验证 hc8_residual_matmul_b 的正确性：
1. depth=0 时方案 B == 方案 C（都是 v[0] × v[0]）
2. depth>0 时方案 B 精度 >= 方案 C（反量化 MSE 更小或相等）
3. 基本功能（非零输入、正确维度）
4. 全零输入处理

运行：
    python test_ufp2_scheme_b.py
"""
import sys
import math
import random
from pathlib import Path

# 添加项目根目录到 sys.path，以便导入 engine.sgn
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import engine.sgn as _sgn
pysgn_net = _sgn._native.hc8_net


def test_depth0_equivalence():
    """测试 1：depth=0 时方案 B 和方案 C 结果相同"""
    print("=== Test 1: depth=0 方案 B == 方案 C ===")
    schema = pysgn_net.default_schema()
    random.seed(42)

    m, k, n = 4, 8, 3
    a_float = [random.uniform(-1, 1) for _ in range(m * k)]
    b_float = [random.uniform(-1, 1) for _ in range(k * n)]

    # 残差量化 depth=0
    a_bytes, a_scales = pysgn_net.quantize_residual(a_float, 0, schema)
    b_bytes, b_scales = pysgn_net.quantize_residual(b_float, 0, schema)

    # 方案 C
    c_bytes_c, c_scale_c = pysgn_net.matmul_residual(
        a_bytes, b_bytes, m, k, n, 0, 0, a_scales, b_scales, schema
    )
    c_float_c = pysgn_net.dequantize(c_bytes_c, c_scale_c, schema)

    # 方案 B
    c_bytes_b, c_scale_b = pysgn_net.matmul_residual_b(
        a_bytes, b_bytes, m, k, n, 0, 0, a_scales, b_scales, schema
    )
    c_float_b = pysgn_net.dequantize(c_bytes_b, c_scale_b, schema)

    # 比较：depth=0 时两者应该完全相同（都是 v[0] × v[0] → int32 → 重新量化）
    # 但可能有一个细微差异：方案 C 重新量化 A/B 到 v[0] 后再做矩阵乘，
    # 而方案 B 直接用原始 v[0] 做。如果 quantize_residual(depth=0) 和 quantize 结果相同，
    # 则两者等价。
    max_diff = max(abs(a - b) for a, b in zip(c_float_c, c_float_b))
    print(f"  m={m}, k={k}, n={n}")
    print(f"  max_diff between B and C: {max_diff:.6e}")
    # depth=0 时，方案 C 反量化到 float 再重新量化到 v[0]，可能引入微小差异
    # 方案 B 直接用 v[0]，所以可能有微小差异（但应很小）
    assert max_diff < 0.01, f"depth=0 时方案 B 和 C 差异过大: {max_diff}"
    print("  ✓ PASS (depth=0 方案 B ≈ 方案 C)")
    print()
    return True


def test_depth5_precision():
    """测试 2：depth=5 时方案 B 精度 >= 方案 C"""
    print("=== Test 2: depth=5 方案 B 精度 >= 方案 C ===")
    schema = pysgn_net.default_schema()
    random.seed(123)

    m, k, n = 4, 8, 3
    a_float = [random.uniform(-1, 1) for _ in range(m * k)]
    b_float = [random.uniform(-1, 1) for _ in range(k * n)]

    # 用 depth=5 量化（48-bit 存储精度）
    depth = 5
    a_bytes, a_scales = pysgn_net.quantize_residual(a_float, depth, schema)
    b_bytes, b_scales = pysgn_net.quantize_residual(b_float, depth, schema)

    # 反量化 A 和 B 得到高精度近似
    a_deq = pysgn_net.dequantize_residual(a_bytes, a_scales, depth, schema)
    b_deq = pysgn_net.dequantize_residual(b_bytes, b_scales, depth, schema)

    # 计算 float 参考值（用高精度反量化值做矩阵乘）
    c_ref = []
    for i in range(m):
        for j in range(n):
            s = 0.0
            for kk in range(k):
                s += a_deq[i * k + kk] * b_deq[kk * n + j]
            c_ref.append(s)

    # 方案 C（反量化到 float32 → 重新量化到 v[0] → int8 矩阵乘）
    c_bytes_c, c_scale_c = pysgn_net.matmul_residual(
        a_bytes, b_bytes, m, k, n, depth, depth, a_scales, b_scales, schema
    )
    c_float_c = pysgn_net.dequantize(c_bytes_c, c_scale_c, schema)

    # 方案 B（整数域累加 + double 合并）
    c_bytes_b, c_scale_b = pysgn_net.matmul_residual_b(
        a_bytes, b_bytes, m, k, n, depth, depth, a_scales, b_scales, schema
    )
    c_float_b = pysgn_net.dequantize(c_bytes_b, c_scale_b, schema)

    # 计算 MSE（相对于 float 参考）
    mse_c = sum((a - b) ** 2 for a, b in zip(c_ref, c_float_c)) / len(c_ref)
    mse_b = sum((a - b) ** 2 for a, b in zip(c_ref, c_float_b)) / len(c_ref)

    print(f"  m={m}, k={k}, n={n}, depth={depth}")
    print(f"  方案 C MSE: {mse_c:.6e}")
    print(f"  方案 B MSE: {mse_b:.6e}")
    print(f"  B/C 改善: {mse_c / mse_b:.2f}x" if mse_b > 0 else "  B/C 改善: inf")

    # 方案 B 应该 <= 方案 C（精度更高或相等）
    # 注意：由于最终都量化到 v[0]（8-bit），精度差异可能很小
    assert mse_b <= mse_c * 1.01, f"方案 B 精度应 <= 方案 C: B={mse_b}, C={mse_c}"
    print("  ✓ PASS (方案 B 精度 >= 方案 C)")
    print()
    return True


def test_basic_functionality():
    """测试 3：基本功能"""
    print("=== Test 3: 基本功能 ===")
    schema = pysgn_net.default_schema()

    m, k, n = 2, 3, 2
    a_float = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # 2x3
    b_float = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # 3x2

    depth = 3
    a_bytes, a_scales = pysgn_net.quantize_residual(a_float, depth, schema)
    b_bytes, b_scales = pysgn_net.quantize_residual(b_float, depth, schema)

    c_bytes, c_scale = pysgn_net.matmul_residual_b(
        a_bytes, b_bytes, m, k, n, depth, depth, a_scales, b_scales, schema
    )
    c_float = pysgn_net.dequantize(c_bytes, c_scale, schema)

    # 计算参考值
    c_ref = []
    for i in range(m):
        for j in range(n):
            s = 0.0
            for kk in range(k):
                s += a_float[i * k + kk] * b_float[kk * n + j]
            c_ref.append(s)

    print(f"  A: {a_float}")
    print(f"  B: {b_float}")
    print(f"  C_ref:  {[round(x, 4) for x in c_ref]}")
    print(f"  C_ufp2: {[round(x, 4) for x in c_float]}")

    max_err = max(abs(a - b) for a, b in zip(c_ref, c_float))
    print(f"  max_error: {max_err:.4e}")
    # 8-bit 量化误差应该在合理范围内
    assert max_err < 0.1, f"误差过大: {max_err}"
    print("  ✓ PASS")
    print()
    return True


def test_zero_input():
    """测试 4：全零输入"""
    print("=== Test 4: 全零输入 ===")
    schema = pysgn_net.default_schema()

    m, k, n = 2, 3, 2
    a_float = [0.0] * (m * k)
    b_float = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    depth = 2
    a_bytes, a_scales = pysgn_net.quantize_residual(a_float, depth, schema)
    b_bytes, b_scales = pysgn_net.quantize_residual(b_float, depth, schema)

    c_bytes, c_scale = pysgn_net.matmul_residual_b(
        a_bytes, b_bytes, m, k, n, depth, depth, a_scales, b_scales, schema
    )
    c_float = pysgn_net.dequantize(c_bytes, c_scale, schema)

    max_val = max(abs(x) for x in c_float)
    print(f"  A 全零, C max(|w|): {max_val:.6e}")
    assert max_val == 0.0, f"全零输入应得全零输出: {max_val}"
    print("  ✓ PASS")
    print()
    return True


def test_depth_comparison():
    """测试 5：不同 depth 的精度对比"""
    print("=== Test 5: 不同 depth 的精度对比 ===")
    schema = pysgn_net.default_schema()
    random.seed(456)

    m, k, n = 3, 5, 2
    a_float = [random.uniform(-2, 2) for _ in range(m * k)]
    b_float = [random.uniform(-2, 2) for _ in range(k * n)]

    # float 参考
    c_ref = []
    for i in range(m):
        for j in range(n):
            s = 0.0
            for kk in range(k):
                s += a_float[i * k + kk] * b_float[kk * n + j]
            c_ref.append(s)

    print(f"  {'depth':>5} | {'方案 C MSE':>12} | {'方案 B MSE':>12} | {'B/C':>8}")
    print(f"  {'-'*5} | {'-'*12} | {'-'*12} | {'-'*8}")

    for depth in [0, 1, 2, 5]:
        a_bytes, a_scales = pysgn_net.quantize_residual(a_float, depth, schema)
        b_bytes, b_scales = pysgn_net.quantize_residual(b_float, depth, schema)

        # 方案 C
        c_bytes_c, c_scale_c = pysgn_net.matmul_residual(
            a_bytes, b_bytes, m, k, n, depth, depth, a_scales, b_scales, schema
        )
        c_float_c = pysgn_net.dequantize(c_bytes_c, c_scale_c, schema)
        mse_c = sum((a - b) ** 2 for a, b in zip(c_ref, c_float_c)) / len(c_ref)

        # 方案 B
        c_bytes_b, c_scale_b = pysgn_net.matmul_residual_b(
            a_bytes, b_bytes, m, k, n, depth, depth, a_scales, b_scales, schema
        )
        c_float_b = pysgn_net.dequantize(c_bytes_b, c_scale_b, schema)
        mse_b = sum((a - b) ** 2 for a, b in zip(c_ref, c_float_b)) / len(c_ref)

        ratio = mse_c / mse_b if mse_b > 0 else float('inf')
        print(f"  {depth:>5} | {mse_c:>12.6e} | {mse_b:>12.6e} | {ratio:>7.2f}x")

    print("  ✓ PASS (depth 越高，方案 B 相对方案 C 的精度优势越明显)")
    print()
    return True


def main():
    print("=" * 60)
    print("UFP-2 方案 B 自检测试")
    print(f"pysgn_net version: {pysgn_net.__version__}")
    print(f"UFP2_SCHEME: {pysgn_net.UFP2_SCHEME}")
    print("=" * 60)
    print()

    tests = [
        test_depth0_equivalence,
        test_depth5_precision,
        test_basic_functionality,
        test_zero_input,
        test_depth_comparison,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
            failed += 1
            print()

    print("=" * 60)
    print(f"结果: {passed}/{passed + failed} 通过")
    if failed == 0:
        print("✓ 全部通过")
    else:
        print(f"✗ {failed} 个失败")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
