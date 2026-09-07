# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""HC16 per-channel scale 扩展单元测试

阶段 1: 验证 pysgn_hc16 v1.5.0 per-channel API 的数学正确性

测试矩阵：
  测试 1: per-channel quantize/dequantize 在均匀 scale 下与 per-tensor bit-exact
  测试 2: per-channel quantize 在非均匀 scale 下精度提升验证
  测试 3: per-channel matmul 在均匀 scale 下与 per-tensor matmul bit-exact
  测试 4: per-channel matmul 在非均匀 scale 下与 numpy float32 baseline 对齐
  测试 5: per-channel matmul AVX2 vs 标量 bit-exact
  测试 6: per-channel matmul 与 numpy.float32 @ 对比（精度 < 1e-2）
  测试 7: 边界值（零数组、全相同 scale、单行单列）
  测试 8: 反向场景模拟（grad_output × weight 转置）

关联：
  - spec: .trae/specs/launch-stage-2-6-gef-ef/spec.md §4.1 In-Scope
  - 计划: .trae/documents/stage_2_6_infra_readiness_check.md 阶段 1
  - 实现: hc16_net.h/c v1.5.0 + pysgn_hc16.cpp v1.5.0
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np

# 添加项目根目录到 sys.path，以便导入 engine.sgn
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import engine.sgn as _sgn
    hc16 = _sgn._native.hc16  # Clang 编译的 sgn 模块的 hc16 子模块
except (ImportError, AttributeError) as e:
    # 安全审计 2026-08-16 A2-2：模块级 sys.exit(1) 会在 pytest 收集阶段
    # 杀死整个 pytest 进程——pytest 下改 module-level skip
    try:
        import pytest
        pytest.skip(f"engine.sgn.hc16 不可用: {e}", allow_module_level=True)
    except ImportError:
        print(f"[FAIL] 无法导入 engine.sgn.hc16：{e}")
        print("请先编译：cd engine/sgn && cmake --build build")
        sys.exit(1)


# ============================================================
# 工具函数
# ============================================================

def check_version():
    """检查 hc16 模块可用（已合并到 sgn，跳过版本号检查）"""
    v = getattr(hc16, "__version__", "unknown")
    print(f"[OK] engine.sgn.hc16 版本 {v}（已合并到 Clang 编译的 sgn 模块）")


def check_avx2():
    avx2 = hc16.detect_avx2()
    print(f"[INFO] AVX2 支持：{avx2}")
    return avx2


# ============================================================
# 测试 1: per-channel quantize/dequantize 在均匀 scale 下与 per-tensor bit-exact
# ============================================================

def test_1_per_channel_uniform_vs_per_tensor():
    """均匀 scale（所有行 scale 相同）下 per-channel 应与 per-tensor 完全一致"""
    print("\n=== 测试 1: per-channel uniform scale vs per-tensor bit-exact ===")

    np.random.seed(42)
    rows, per_row_size = 8, 64
    x = np.random.randn(rows, per_row_size).astype(np.float32)

    schema = hc16.default_schema()

    # per-tensor scale
    x_scale = hc16.quant_compute_scale(x.reshape(-1))
    q_per_tensor = hc16.quantize(x.reshape(-1), x_scale, schema).reshape(rows, per_row_size)

    # per-channel scale（所有行用相同 scale = x_scale）
    uniform_scales = np.full(rows, x_scale, dtype=np.float32)
    q_per_channel = hc16.quantize_per_channel(x, uniform_scales, schema, rows, per_row_size)

    assert q_per_tensor.shape == q_per_channel.shape == (rows, per_row_size)
    max_diff = int(np.abs(q_per_tensor.astype(np.int32) - q_per_channel.astype(np.int32)).max())
    assert max_diff == 0, f"per-channel vs per-tensor bit-exact 失败：max_diff={max_diff}"
    print(f"[OK] bit-exact（max_diff=0），shape={q_per_channel.shape}, dtype={q_per_channel.dtype}")


# ============================================================
# 测试 2: per-channel quantize 在非均匀 scale 下精度提升验证
# ============================================================

def test_2_per_channel_non_uniform_precision():
    """非均匀 scale 下 per-channel 应比 per-tensor 精度更高"""
    print("\n=== 测试 2: per-channel non-uniform scale 精度提升 ===")

    np.random.seed(42)
    rows, per_row_size = 4, 32
    # 构造非均匀分布：第 0 行幅值 1.0，第 1 行幅值 100，第 2 行幅值 0.01，第 3 行幅值 10
    x = np.random.randn(rows, per_row_size).astype(np.float32)
    amplitudes = [1.0, 100.0, 0.01, 10.0]
    for i in range(rows):
        x[i] *= amplitudes[i]

    schema = hc16.default_schema()

    # per-tensor（全张量一个 scale，会被幅值 100 的行主导）
    x_scale_tensor = hc16.quant_compute_scale(x.reshape(-1))
    q_tensor = hc16.quantize(x.reshape(-1), x_scale_tensor, schema).reshape(rows, per_row_size)
    x_dequant_tensor = hc16.dequantize(q_tensor, x_scale_tensor).reshape(rows, per_row_size)

    # per-channel（每行独立 scale）
    # 正确的 scale 推导：scales[i] = max(|x[i, :]|) / 32767
    scales_pc = np.array([np.abs(x[i]).max() / 32767.0 for i in range(rows)], dtype=np.float32)
    # pysgn_hc16 的 quant_compute_scale_per_channel 应与手工推导一致
    scales_pc_computed = hc16.quant_compute_scale_per_channel(x, rows, per_row_size)
    max_scale_diff = float(np.abs(scales_pc - scales_pc_computed).max())
    assert max_scale_diff < 1e-6, f"scale 推导不一致：max_diff={max_scale_diff}"
    print(f"[OK] quant_compute_scale_per_channel 推导正确（max_diff={max_scale_diff:.2e}）")

    q_pc = hc16.quantize_per_channel(x, scales_pc, schema, rows, per_row_size)
    x_dequant_pc = hc16.dequantize_per_channel(q_pc, scales_pc, rows, per_row_size)

    # 对比每行的 round-trip 误差
    for i in range(rows):
        err_tensor = float(np.abs(x[i] - x_dequant_tensor[i]).max())
        err_pc = float(np.abs(x[i] - x_dequant_pc[i]).max())
        # per-channel 误差应 <= per-tensor 误差
        ratio = err_tensor / max(err_pc, 1e-12)
        print(f"  行 {i} (amp={amplitudes[i]:>6.2f}): per-tensor err={err_tensor:.4e}, per-channel err={err_pc:.4e}, ratio={ratio:.2f}x")
        assert err_pc <= err_tensor * (1 + 1e-6), \
            f"行 {i} per-channel 误差 {err_pc} 应 <= per-tensor 误差 {err_tensor}"

    print("[OK] per-channel 在非均匀分布下精度 >= per-tensor")


# ============================================================
# 测试 3: per-channel matmul 在均匀 scale 下与 per-tensor matmul bit-exact
# ============================================================

def test_3_matmul_uniform_vs_per_tensor():
    """均匀 scale 下 per-channel matmul 应与 per-tensor matmul 完全一致"""
    print("\n=== 测试 3: per-channel matmul uniform scale vs per-tensor bit-exact ===")

    np.random.seed(42)
    m, k, n = 16, 64, 32
    A = np.random.randn(m, k).astype(np.float32) * 0.1
    B = np.random.randn(k, n).astype(np.float32) * 0.1

    schema = hc16.default_schema()

    # per-tensor: 量化 A 和 B，用单一 scale
    a_scale = hc16.quant_compute_scale(A.reshape(-1))
    b_scale = hc16.quant_compute_scale(B.reshape(-1))
    q_a = hc16.quantize(A.reshape(-1), a_scale, schema).reshape(m, k)
    q_b = hc16.quantize(B.reshape(-1), b_scale, schema).reshape(k, n)
    out_per_tensor = hc16.matmul(q_a, q_b, m, k, n, a_scale, b_scale)

    # per-channel: 所有行用相同 scale（应退化为 per-tensor）
    a_scales = np.full(m, a_scale, dtype=np.float32)
    b_scales = np.full(n, b_scale, dtype=np.float32)
    out_per_channel = hc16.matmul_per_channel(q_a, q_b, m, k, n, a_scales, b_scales)

    max_diff = float(np.abs(out_per_tensor - out_per_channel).max())
    # 浮点累加顺序差异（per-tensor 用 combined_scale = a_scale * b_scale 一次乘，
    # per-channel 用 a_s * b_scales[j] 两次乘），允许 float32 epsilon 级误差
    assert max_diff < 1e-6, f"per-channel vs per-tensor matmul 偏差过大：max_diff={max_diff}"
    print(f"[OK] 数值一致（max_diff={max_diff:.2e} < 1e-6，浮点累加差异）")


# ============================================================
# 测试 4: per-channel matmul 在非均匀 scale 下与 numpy float32 baseline 对齐
# ============================================================

def test_4_matmul_non_uniform_vs_numpy():
    """非均匀 scale 下 per-channel matmul 应与 numpy float32 @ 对齐（精度受量化限制）"""
    print("\n=== 测试 4: per-channel matmul non-uniform scale vs numpy float32 ===")

    np.random.seed(42)
    m, k, n = 16, 64, 32
    A = np.random.randn(m, k).astype(np.float32)
    B = np.random.randn(k, n).astype(np.float32)

    schema = hc16.default_schema()

    # per-channel: 每行/每列独立 scale
    a_scales = hc16.quant_compute_scale_per_channel(A, m, k)
    b_scales = hc16.quant_compute_scale_per_channel(B.T, n, k)  # B 的每列 scale

    q_a = hc16.quantize_per_channel(A, a_scales, schema, m, k)
    q_b = hc16.quantize_per_channel(B.T, b_scales, schema, n, k).T  # 量化后转回 k×n

    out_pc = hc16.matmul_per_channel(q_a, q_b, m, k, n, a_scales, b_scales)

    # numpy baseline（float32，无量化）
    out_ref = A @ B

    rel_err = float(np.abs(out_pc - out_ref).max() / np.abs(out_ref).max())
    print(f"  m={m}, k={k}, n={n}: max_rel_err={rel_err:.4e}")
    assert rel_err < 1e-2, f"per-channel matmul 与 numpy baseline 偏差过大：rel_err={rel_err}"
    print(f"[OK] rel_err={rel_err:.4e} < 1e-2")


# ============================================================
# 测试 5: per-channel matmul AVX2 vs 标量 bit-exact
# ============================================================

def test_5_matmul_avx2_vs_scalar_bit_exact():
    """AVX2 路径与标量路径必须 bit-exact"""
    print("\n=== 测试 5: per-channel matmul AVX2 vs 标量 bit-exact ===")

    if not hc16.detect_avx2():
        print("[SKIP] AVX2 不可用，跳过")
        return

    np.random.seed(42)
    m, k, n = 16, 64, 32
    A = np.random.randn(m, k).astype(np.float32) * 0.1
    B = np.random.randn(k, n).astype(np.float32) * 0.1

    schema = hc16.default_schema()
    a_scales = hc16.quant_compute_scale_per_channel(A, m, k)
    b_scales = hc16.quant_compute_scale_per_channel(B.T, n, k)
    q_a = hc16.quantize_per_channel(A, a_scales, schema, m, k)
    q_b = hc16.quantize_per_channel(B.T, b_scales, schema, n, k).T

    out_scalar = hc16.matmul_per_channel_scalar(q_a, q_b, m, k, n, a_scales, b_scales)
    out_avx2 = hc16.matmul_per_channel_avx2(q_a, q_b, m, k, n, a_scales, b_scales)

    max_diff = float(np.abs(out_scalar - out_avx2).max())
    assert max_diff == 0.0, f"AVX2 vs 标量 bit-exact 失败：max_diff={max_diff}"
    print(f"[OK] bit-exact（max_diff=0.0）")


# ============================================================
# 测试 6: per-channel matmul 与 numpy.float32 @ 对比（精度 < 1e-2）
# ============================================================

def test_6_matmul_vs_numpy_large():
    """较大矩阵 per-channel matmul 精度验证"""
    print("\n=== 测试 6: per-channel matmul vs numpy float32 @ (512×512) ===")

    np.random.seed(42)
    m, k, n = 128, 256, 128
    A = np.random.randn(m, k).astype(np.float32) * 0.05
    B = np.random.randn(k, n).astype(np.float32) * 0.05

    schema = hc16.default_schema()
    a_scales = hc16.quant_compute_scale_per_channel(A, m, k)
    b_scales = hc16.quant_compute_scale_per_channel(B.T, n, k)
    q_a = hc16.quantize_per_channel(A, a_scales, schema, m, k)
    q_b = hc16.quantize_per_channel(B.T, b_scales, schema, n, k).T

    out_pc = hc16.matmul_per_channel(q_a, q_b, m, k, n, a_scales, b_scales)
    out_ref = A @ B

    rel_err = float(np.abs(out_pc - out_ref).max() / np.abs(out_ref).max())
    print(f"  m={m}, k={k}, n={n}: max_rel_err={rel_err:.4e}")
    assert rel_err < 1e-2, f"大矩阵 per-channel matmul 偏差过大：rel_err={rel_err}"
    print(f"[OK] rel_err={rel_err:.4e} < 1e-2")


# ============================================================
# 测试 7: 边界值（零数组、全相同 scale、单行单列）
# ============================================================

def test_7_edge_cases():
    """边界值测试"""
    print("\n=== 测试 7: 边界值测试 ===")
    schema = hc16.default_schema()

    # 7a: 零数组（scale 推导应返回 1.0，量化结果全 0）
    x_zero = np.zeros((4, 8), dtype=np.float32)
    scales_zero = hc16.quant_compute_scale_per_channel(x_zero, 4, 8)
    assert np.all(scales_zero == 1.0), f"零数组 scale 应为 1.0，实际 {scales_zero}"
    q_zero = hc16.quantize_per_channel(x_zero, scales_zero, schema, 4, 8)
    assert np.all(q_zero == 0), f"零数组量化结果应全 0，实际 max={q_zero.max()}"
    print("[OK] 7a: 零数组处理正确")

    # 7b: 全相同 scale（应正常工作，退化为 per-tensor）
    x_uniform = np.ones((4, 8), dtype=np.float32) * 0.5
    scales_uniform = hc16.quant_compute_scale_per_channel(x_uniform, 4, 8)
    assert np.allclose(scales_uniform, scales_uniform[0]), "全相同幅值的 scale 应全部相同"
    print("[OK] 7b: 全相同 scale 正常")

    # 7c: 单行单列（m=1, n=1）
    A = np.random.randn(1, 16).astype(np.float32)
    B = np.random.randn(16, 1).astype(np.float32)
    a_scales = hc16.quant_compute_scale_per_channel(A, 1, 16)
    b_scales = hc16.quant_compute_scale_per_channel(B.T, 1, 16)
    q_a = hc16.quantize_per_channel(A, a_scales, schema, 1, 16)
    q_b = hc16.quantize_per_channel(B.T, b_scales, schema, 1, 16).T
    out = hc16.matmul_per_channel(q_a, q_b, 1, 16, 1, a_scales, b_scales)
    out_ref = A @ B
    rel_err = float(np.abs(out[0, 0] - out_ref[0, 0]) / abs(out_ref[0, 0]))
    assert rel_err < 1e-2, f"单行单列 matmul 偏差过大：rel_err={rel_err}"
    print(f"[OK] 7c: 单行单列 matmul 正常（rel_err={rel_err:.2e}）")


# ============================================================
# 测试 8: 反向场景模拟（grad_output × weight 转置）
# ============================================================

def test_8_backward_scenario():
    """模拟反向传播场景：grad_w = grad_output.T @ x_col"""
    print("\n=== 测试 8: 反向场景模拟 grad_w = grad_output.T @ x_col ===")

    np.random.seed(42)
    # 典型反向场景：B*L=64 个样本，C_out=32 个输出通道，K=C_in*kh*kw=27
    M, N, K = 64, 32, 27  # grad_output: (M, N), x_col: (M, K), grad_w: (N, K) = grad_output.T @ x_col

    grad_output = np.random.randn(M, N).astype(np.float32) * 0.1
    x_col = np.random.randn(M, K).astype(np.float32) * 0.1

    schema = hc16.default_schema()

    # grad_output 按 N 维度（列）per-channel scale，转置后按行
    go_scales = hc16.quant_compute_scale_per_channel(grad_output.T, N, M)  # (N,)
    # x_col 按 M 维度（行）per-channel scale
    xc_scales = hc16.quant_compute_scale_per_channel(x_col, M, K)  # (M,)

    q_go = hc16.quantize_per_channel(grad_output.T, go_scales, schema, N, M)  # (N, M)
    q_xc = hc16.quantize_per_channel(x_col, xc_scales, schema, M, K)  # (M, K)

    # grad_w = grad_output.T @ x_col = q_go @ q_xc, shape (N, K)
    # matmul 语义：C[i][j] = sum_l(A[i][l] * B[l][j]) * a_scales[i] * b_scales[j]
    # 这里 A=q_go (N×M), B=q_xc (M×K), a_scales=go_scales (N,), b_scales=xc_scales (K,)
    # 但 b_scales 应该是 B 的每列 scale，xc_scales 是 B 的每行 scale，不匹配！
    # 解决方案：b_scales 应该是 K 维度上的 scale，但 x_col 的 per-channel 是按 M（行）
    # 所以这里需要 per-tensor scale for x_col（K 维度上无 per-channel 需求）
    # 或者：对 x_col 按 K 维度做 per-channel（但 K=27 是 C_in*kh*kw，不是 channel）

    # 简化：对 x_col 用 per-tensor scale
    xc_scale_tensor = hc16.quant_compute_scale(x_col.reshape(-1))
    q_xc_tensor = hc16.quantize(x_col.reshape(-1), xc_scale_tensor, schema).reshape(M, K)
    b_scales_uniform = np.full(K, xc_scale_tensor, dtype=np.float32)

    grad_w_pc = hc16.matmul_per_channel(q_go, q_xc_tensor, N, M, K, go_scales, b_scales_uniform)

    # numpy baseline
    grad_w_ref = grad_output.T @ x_col

    rel_err = float(np.abs(grad_w_pc - grad_w_ref).max() / np.abs(grad_w_ref).max())
    print(f"  grad_w shape={grad_w_pc.shape}, max_rel_err={rel_err:.4e}")
    assert rel_err < 1e-2, f"反向场景 grad_w 偏差过大：rel_err={rel_err}"
    print(f"[OK] 反向场景 grad_w 精度正常（rel_err={rel_err:.4e}）")


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("HC16 per-channel scale 扩展单元测试（pysgn_hc16 v1.5.0）")
    print("=" * 70)

    check_version()
    check_avx2()

    tests = [
        test_1_per_channel_uniform_vs_per_tensor,
        test_2_per_channel_non_uniform_precision,
        test_3_matmul_uniform_vs_per_tensor,
        test_4_matmul_non_uniform_vs_numpy,
        test_5_matmul_avx2_vs_scalar_bit_exact,
        test_6_matmul_vs_numpy_large,
        test_7_edge_cases,
        test_8_backward_scenario,
    ]

    passed, failed = 0, 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"总计：{passed} 通过，{failed} 失败")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
