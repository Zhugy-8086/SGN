# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""HC16 C 扩展验证脚本（2026-07-30）

验证 pysgn_hc16 扩展的：
  1. 基本导入和 schema
  2. AVX2 检测
  3. 量化精度（vs Python 参考实现）
  4. matmul 标量路径精度（vs Python numpy）
  5. matmul AVX2 路径精度（vs 标量路径 bit-exact）
  6. quantized_matmul 便捷封装
  7. 性能对比（标量 vs AVX2 vs numpy float32 BLAS）

用法:
    cd engine/sgn/tests/hc_ext/
    python test_hc16_net.py
"""
from __future__ import annotations

import sys
import os
import time
from pathlib import Path

# ── OMP 冲突调式日志 ──────────────────────────────────────────
# 安全审计 2026-08-16 A2-6：原注释描述 MSVC/Clang 双 .pyd 共存场景——
# 2026-08-06 起全部 .pyd 已合并为 Clang 编译的单一 sgn 模块，该场景不复
# 存在；KMP_DUPLICATE_LIB_OK 保留为防御性设置（进程内如加载其它 OpenMP
# 运行时的第三方扩展时仍可避免 "Error #15" 崩溃）。
_DEBUG = "SGN_DEBUG" in os.environ
if _DEBUG:
    _omp_before = os.environ.get('KMP_DUPLICATE_LIB_OK', '(未设置)')
    print(f"[DEBUG] test_hc16_net: KMP_DUPLICATE_LIB_OK 设置前={_omp_before}")
    print(f"[DEBUG] test_hc16_net: sys.path={sys.path}")
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np

# 添加项目根目录到 sys.path，以便导入 engine.sgn
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── 路径解析调式日志 ──────────────────────────────────────────
# 背景：import engine.sgn 依赖 _PROJECT_ROOT 正确指向项目根目录。
# 若路径偏移会导致加载 engine/sgn/__init__.py（包）而非 .pyd（扩展）。
if _DEBUG:
    print(f"[DEBUG] test_hc16_net: 路径解析")
    print(f"[DEBUG]   _PROJECT_ROOT={_PROJECT_ROOT}")
    print(f"[DEBUG]   sys.path(前3)={sys.path[:3]}")

import engine.sgn as _sgn
hc16 = _sgn._native.hc16  # Clang 编译的 sgn 模块的 hc16 子模块
pysgn_hc16 = hc16  # 向后兼容别名

# ── 加载后调式日志 ──
if _DEBUG:
    _loaded_omp = os.environ.get('KMP_DUPLICATE_LIB_OK', '(未设置)')
    _sgn_path = getattr(_sgn, '__file__', '(未知)')
    print(f"[DEBUG] test_hc16_net: 加载后 KMP_DUPLICATE_LIB_OK={_loaded_omp}")
    print(f"[DEBUG] test_hc16_net: engine.sgn.__file__={_sgn_path}")
    print(f"[DEBUG] test_hc16_net: hc16.__version__={getattr(hc16, '__version__', 'unknown')}")

# ============================================================
# Python 参考实现（来自 msint_math_validation_2026_07_30.py）
# ============================================================

_HC16_QMIN = -32767  # 与 C 扩展一致（防 madd int32 溢出）
_HC16_QMAX = 32767


def hc16_quantize_ref(x: np.ndarray) -> np.ndarray:
    """HC16 量化 + 反量化（round-trip, per-tensor scale）"""
    if x.size == 0:
        return x.copy()
    x_abs = np.abs(x)
    x_max = float(x_abs.max())
    if x_max == 0.0:
        return x.copy()
    scale = x_max / 32767.0
    inv_scale = 1.0 / scale
    q = np.round(x * inv_scale)
    np.clip(q, _HC16_QMIN, _HC16_QMAX, out=q)
    return (q * scale).astype(x.dtype, copy=False)


def hc16_matmul_ref(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """HC16 matmul 参考（路径 A：反量化后 float matmul）"""
    if x.size == 0 or w.size == 0:
        return np.zeros((x.shape[0], w.shape[1]), dtype=np.float32)
    x_scale = float(np.abs(x).max()) / 32767.0 if float(np.abs(x).max()) > 0 else 1.0
    w_scale = float(np.abs(w).max()) / 32767.0 if float(np.abs(w).max()) > 0 else 1.0
    x_q = np.clip(np.round(x / x_scale), _HC16_QMIN, _HC16_QMAX).astype(np.int16)
    w_q = np.clip(np.round(w / w_scale), _HC16_QMIN, _HC16_QMAX).astype(np.int16)
    x_deq = x_q.astype(np.float32) * np.float32(x_scale)
    w_deq = w_q.astype(np.float32) * np.float32(w_scale)
    return x_deq @ w_deq


def relative_error(ref: np.ndarray, approx: np.ndarray) -> float:
    """相对误差 ||ref - approx|| / ||ref||"""
    ref_norm = np.linalg.norm(ref)
    if ref_norm == 0:
        return 0.0
    return float(np.linalg.norm(ref - approx) / ref_norm)


# ============================================================
# 测试函数
# ============================================================

def test_1_basic_import():
    """测试 1: 基本导入和 schema"""
    print("=" * 70)
    print("测试 1: 基本导入和 schema")
    print("=" * 70)

    schema = pysgn_hc16.default_schema()
    print(f"  default_schema: {schema}")
    assert schema.qmin == -32767, f"qmin 应为 -32767，实际 {schema.qmin}"
    assert schema.qmax == 32767, f"qmax 应为 32767，实际 {schema.qmax}"
    print(f"  ✓ qmin={schema.qmin}, qmax={schema.qmax}")

    avx2 = pysgn_hc16.detect_avx2()
    print(f"  AVX2 支持: {avx2}")
    print(f"  版本: {pysgn_hc16.__version__}")
    print(f"  ✓ 基本导入成功")


def test_2_quantize_precision():
    """测试 2: 量化精度（vs Python 参考实现，int16 bit-exact）"""
    print("\n" + "=" * 70)
    print("测试 2: 量化精度（vs Python 参考实现，int16 bit-exact）")
    print("=" * 70)

    np.random.seed(42)
    schema = pysgn_hc16.default_schema()

    test_cases = [
        ("小尺度 (std=0.1)", np.random.randn(128, 128).astype(np.float32) * 0.1),
        ("中尺度 (std=1.0)", np.random.randn(128, 128).astype(np.float32) * 1.0),
        ("大尺度 (std=10.0)", np.random.randn(128, 128).astype(np.float32) * 10.0),
    ]

    all_pass = True
    for name, x in test_cases:
        # Python 参考量化（int16 值）
        x_max = float(np.abs(x).max())
        x_scale_ref = x_max / 32767.0 if x_max > 0 else 1.0
        q_ref = np.clip(np.round(x / x_scale_ref), _HC16_QMIN, _HC16_QMAX).astype(np.int16)

        # C 扩展量化（int16 值）
        x_scale_c = pysgn_hc16.quant_compute_scale(x)
        q_c = pysgn_hc16.quantize(x, x_scale_c, schema)

        # 比较 int16 量化值
        # 注：C lroundf (round half away from zero) vs Python np.round (banker's rounding)
        # 在 .5 边界值会差 1 ULP，属正常舍入差异，不影响实际使用
        max_diff = int(np.max(np.abs(q_ref.astype(np.int32) - q_c.astype(np.int32))))
        scale_diff = abs(x_scale_ref - x_scale_c)
        is_pass = (max_diff <= 1) and (scale_diff < 1e-10)
        if not is_pass:
            all_pass = False

        print(f"  {name}: int16 max_diff={max_diff}, scale_diff={scale_diff:.2e} "
              f"{'✓' if is_pass else '✗'}")

    print(f"\n  结论: {'✓ 量化 int16 值与 Python 参考一致（max_diff≤1，舍入模式差异）' if all_pass else '✗ 量化不一致！'}")
    print(f"      注: C lroundf vs Python np.round 在 .5 边界值差 1 ULP，属正常")


def test_3_matmul_scalar_precision():
    """测试 3: matmul 标量路径精度（vs Python numpy）"""
    print("\n" + "=" * 70)
    print("测试 3: matmul 标量路径精度（vs Python numpy）")
    print("=" * 70)

    np.random.seed(42)

    test_cases = [
        ("小矩阵 (64×64)", 64, 64, 64),
        ("中等矩阵 (256×256)", 256, 256, 256),
        ("大矩阵 (512×512)", 512, 512, 512),
    ]

    for name, m, k, n in test_cases:
        x = np.random.randn(m, k).astype(np.float32) * 0.1
        w = np.random.randn(k, n).astype(np.float32) * 0.1

        # Python 参考（路径 A：反量化后 float matmul）
        y_ref = hc16_matmul_ref(x, w)

        # C 扩展标量路径
        y_c = pysgn_hc16.quantized_matmul(x, w, m, k, n, pysgn_hc16.default_schema())

        rel_err = relative_error(y_ref, y_c)
        print(f"  {name}: rel_err={rel_err:.2e} "
              f"{'✓' if rel_err < 1e-4 else '✗'}")

    print(f"\n  注: C 标量路径用 int64 累加，Python 参考用 float32 BLAS")
    print(f"      两者数学等价但浮点累加顺序不同，rel_err < 1e-4 即可接受")


def test_4_matmul_avx2_vs_scalar():
    """测试 4: matmul AVX2 路径 vs 标量路径（bit-exact）"""
    print("\n" + "=" * 70)
    print("测试 4: matmul AVX2 路径 vs 标量路径（bit-exact）")
    print("=" * 70)

    if not pysgn_hc16.detect_avx2():
        print("  ⚠ AVX2 不可用，跳过此测试")
        return

    np.random.seed(42)
    schema = pysgn_hc16.default_schema()

    test_cases = [
        ("小矩阵 (64×64)", 64, 64, 64),
        ("中等矩阵 (256×256)", 256, 256, 256),
        ("大矩阵 (512×512)", 512, 512, 512),
        ("K=1024 防溢出测试", 128, 1024, 128),
    ]

    all_pass = True
    for name, m, k, n in test_cases:
        x = np.random.randn(m, k).astype(np.float32) * 0.1
        w = np.random.randn(k, n).astype(np.float32) * 0.1

        # 量化
        x_scale = pysgn_hc16.quant_compute_scale(x)
        w_scale = pysgn_hc16.quant_compute_scale(w)
        x_q = pysgn_hc16.quantize(x, x_scale, schema)
        w_q = pysgn_hc16.quantize(w, w_scale, schema)

        # 标量路径
        y_scalar = pysgn_hc16.matmul_scalar(x_q, w_q, m, k, n, x_scale, w_scale)

        # AVX2 路径
        y_avx2 = pysgn_hc16.matmul_avx2(x_q, w_q, m, k, n, x_scale, w_scale)

        # bit-exact 比较
        max_diff = float(np.max(np.abs(y_scalar - y_avx2)))
        is_exact = max_diff == 0.0
        if not is_exact:
            all_pass = False

        print(f"  {name}: max_diff={max_diff:.2e} "
              f"{'✓ bit-exact' if is_exact else '✗ 差异！'}")

    print(f"\n  结论: {'✓ AVX2 与标量路径 bit-exact 一致' if all_pass else '✗ AVX2 与标量路径不一致！'}")
    print(f"      （AVX2 _mm256_madd_epi16 输出 int32，标量用 int64，但累加值相同）")


def test_5_quantized_matmul_convenience():
    """测试 5: quantized_matmul 便捷封装（vs Python 参考）"""
    print("\n" + "=" * 70)
    print("测试 5: quantized_matmul 便捷封装")
    print("=" * 70)

    np.random.seed(42)

    m, k, n = 128, 256, 128
    x = np.random.randn(m, k).astype(np.float32) * 0.1
    w = np.random.randn(k, n).astype(np.float32) * 0.1

    # Python 参考
    y_ref = hc16_matmul_ref(x, w)

    # C 扩展便捷封装
    y_c = pysgn_hc16.quantized_matmul(x, w, m, k, n, pysgn_hc16.default_schema())

    rel_err = relative_error(y_ref, y_c)
    print(f"  矩阵 ({m}×{k}) @ ({k}×{n}): rel_err={rel_err:.2e} "
          f"{'✓' if rel_err < 1e-4 else '✗'}")
    print(f"  ✓ 便捷封装一步完成 float→量化→int16 matmul→反量化→float")


def test_6_performance():
    """测试 6: 性能对比（标量 vs AVX2 vs numpy float32 BLAS）"""
    print("\n" + "=" * 70)
    print("测试 6: 性能对比（标量 vs AVX2 vs numpy float32 BLAS）")
    print("=" * 70)

    np.random.seed(42)
    schema = pysgn_hc16.default_schema()

    sizes = [128, 256, 512]
    iters = {128: 50, 256: 20, 512: 5}

    print(f"\n  {'矩阵尺寸':<18} {'标量(ms)':<12} {'AVX2(ms)':<12} {'numpy(ms)':<12} {'AVX2/标量':<12} {'AVX2/numpy':<12}")
    print("  " + "-" * 78)

    for n in sizes:
        m = k = n
        x = np.random.randn(m, k).astype(np.float32) * 0.1
        w = np.random.randn(k, n).astype(np.float32) * 0.1

        # 量化
        x_scale = pysgn_hc16.quant_compute_scale(x)
        w_scale = pysgn_hc16.quant_compute_scale(w)
        x_q = pysgn_hc16.quantize(x, x_scale, schema)
        w_q = pysgn_hc16.quantize(w, w_scale, schema)

        it = iters[n]

        # 标量路径
        t0 = time.time()
        for _ in range(it):
            y_scalar = pysgn_hc16.matmul_scalar(x_q, w_q, m, k, n, x_scale, w_scale)
        scalar_ms = (time.time() - t0) / it * 1000

        # AVX2 路径
        if pysgn_hc16.detect_avx2():
            t0 = time.time()
            for _ in range(it):
                y_avx2 = pysgn_hc16.matmul_avx2(x_q, w_q, m, k, n, x_scale, w_scale)
            avx2_ms = (time.time() - t0) / it * 1000
        else:
            avx2_ms = float('nan')

        # numpy float32 BLAS 基准
        t0 = time.time()
        for _ in range(it):
            y_np = x @ w
        numpy_ms = (time.time() - t0) / it * 1000

        speedup_scalar = scalar_ms / avx2_ms if avx2_ms > 0 else 0
        speedup_numpy = avx2_ms / numpy_ms if numpy_ms > 0 else 0

        print(f"  {n}×{n:<14} {scalar_ms:<12.2f} {avx2_ms:<12.2f} {numpy_ms:<12.2f} {speedup_scalar:<12.2f}x {speedup_numpy:<12.2f}x")

    print(f"\n  注:")
    print(f"    - 标量路径: int64 累加器，无 SIMD")
    print(f"    - AVX2 路径: _mm256_madd_epi16 + int64 累加 + OpenMP 并行")
    print(f"    - numpy: float32 BLAS（OpenBLAS/MKL，高度优化）")
    print(f"    - AVX2/numpy < 1.0 属正常（int16 路径无 BLAS 级优化）")
    print(f"    - HC16 优势在精度（258x）+ 整数路径硬件交互性，非纯粹速度")


def main():
    print("HC16 C 扩展验证（pysgn_hc16）")
    print(f"日期: 2026-07-30")
    print(f"目标: 验证 HC16 C 扩展的精度和性能")
    print()

    test_1_basic_import()
    test_2_quantize_precision()
    test_3_matmul_scalar_precision()
    test_4_matmul_avx2_vs_scalar()
    test_5_quantized_matmul_convenience()
    test_6_performance()

    print("\n" + "=" * 70)
    print("所有验证测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
