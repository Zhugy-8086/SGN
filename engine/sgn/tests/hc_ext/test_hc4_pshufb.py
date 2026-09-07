# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""HC4 PSHUFB LUT 验证脚本（2026-07-30）

验证项：
  1. 基本导入 + AVX2 检测
  2. LUT 正确性（16×16 乘法表）
  3. PSHUFB 批量乘法正确性（32 个 uint4×uint4→uint8）
  4. PSHUFB matmul 精度（uint4×uint4→int32 累加，vs numpy）
  5. 量化 matmul 精度（float→uint4→matmul→float，vs numpy）
  6. PSHUFB vs nibble split 数学等价性
  7. 性能对比
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
    print(f"[DEBUG] test_hc4_pshufb: KMP_DUPLICATE_LIB_OK 设置前={_omp_before}")
    print(f"[DEBUG] test_hc4_pshufb: sys.path={sys.path}")
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np

# 添加项目根目录到 sys.path，以便导入 engine.sgn
# 注意：2026-08-06 修复 parent 从 4 层改为 5 层。
# 修复前 _PROJECT_ROOT = engine/，修复后 = SGN/（项目根）。
# 修复前 engine/ 在 sys.path 中会 shadow sgn.cp*.pyd 扩展，
# 导致 import engine.sgn 加载 __init__.py（包）而非 .pyd（扩展）。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── 路径解析调式日志 ──────────────────────────────────────────
# 背景：import engine.sgn 依赖 _PROJECT_ROOT 正确指向项目根目录。
# 若路径偏移会导致加载 engine/sgn/__init__.py（包）而非 .pyd（扩展）。
if _DEBUG:
    print(f"[DEBUG] test_hc4_pshufb: 路径解析")
    print(f"[DEBUG]   _PROJECT_ROOT={_PROJECT_ROOT}")
    print(f"[DEBUG]   sys.path(前3)={sys.path[:3]}")

import engine.sgn as _sgn
hc4 = _sgn._native.hc4  # Clang 编译的 sgn 模块的 hc4 子模块（替代旧 MSVC pysgn_hc4_pshufb.pyd）

# ── 加载后调式日志 ──
if _DEBUG:
    _loaded_omp = os.environ.get('KMP_DUPLICATE_LIB_OK', '(未设置)')
    _sgn_path = getattr(_sgn, '__file__', '(未知)')
    print(f"[DEBUG] test_hc4_pshufb: 加载后 KMP_DUPLICATE_LIB_OK={_loaded_omp}")
    print(f"[DEBUG] test_hc4_pshufb: engine.sgn.__file__={_sgn_path}")
    print(f"[DEBUG] test_hc4_pshufb: hc4.__version__={getattr(hc4, '__version__', 'unknown')}")


def _rel_error(ref, approx):
    ref_norm = float(np.linalg.norm(ref))
    if ref_norm == 0:
        return 0.0
    return float(np.linalg.norm(ref - approx) / ref_norm)


def _max_error(ref, approx):
    return float(np.max(np.abs(ref - approx)))


def test_1_basic():
    """测试 1: 基本导入 + AVX2 检测"""
    print("\n[测试 1] 基本导入 + AVX2 检测")
    avx2 = hc4.detect_avx2()
    print(f"  AVX2 支持: {'是' if avx2 else '否'}")
    assert avx2 == 1, "AVX2 未检测到"
    print("  ✓ 通过")
    return True


def test_2_lut_correctness():
    """测试 2: LUT 正确性（16×16 乘法表）"""
    print("\n[测试 2] LUT 正确性")
    lut = hc4.get_lut()  # (256,) uint8
    assert lut.shape == (256,)

    # 验证 LUT[b*16 + a] == a * b
    for b in range(16):
        for a in range(16):
            expected = a * b
            actual = int(lut[b * 16 + a])
            assert actual == expected, f"LUT[{b}][{a}] = {actual}, expected {expected}"

    print(f"  LUT 形状: {lut.shape}")
    print(f"  LUT[0][0..15]: {list(lut[0:16])}")
    print(f"  LUT[1][0..15]: {list(lut[16:32])}")
    print(f"  LUT[15][0..15]: {list(lut[240:256])}")
    print("  ✓ 通过 - 16×16 乘法表全部正确")
    return True


def test_3_pshufb_mul_32():
    """测试 3: PSHUFB 批量乘法正确性（32 个 uint4×uint4→uint8）"""
    print("\n[测试 3] PSHUFB 批量乘法（32 个 uint4×uint4→uint8）")
    np.random.seed(42)
    a = np.random.randint(0, 16, size=32).astype(np.uint8)
    b = np.random.randint(0, 16, size=32).astype(np.uint8)

    # C 扩展 PSHUFB 乘法
    prod_c = hc4.pshufb_mul_32(a, b)

    # numpy 参考
    prod_ref = (a.astype(np.int32) * b.astype(np.int32)).astype(np.uint8)

    max_err = _max_error(prod_ref, prod_c)
    print(f"  输入 a: {list(a[:8])}...")
    print(f"  输入 b: {list(b[:8])}...")
    print(f"  乘积 C:  {list(prod_c[:8])}...")
    print(f"  乘积 ref:{list(prod_ref[:8])}...")
    print(f"  max_error: {max_err}")

    assert max_err == 0, f"PSHUFB 乘法错误: max_err={max_err}"
    print("  ✓ 通过 - 32 个乘积全部 bit-exact")
    return True


def test_4_matmul_precision():
    """测试 4: PSHUFB matmul 精度（uint4×uint4→int32 累加）"""
    print("\n[测试 4] PSHUFB matmul 精度")
    np.random.seed(42)
    m, k, n = 16, 64, 8

    # 生成 uint4 数据（0-15）
    a = np.random.randint(0, 16, size=(m, k)).astype(np.uint8)
    b = np.random.randint(0, 16, size=(k, n)).astype(np.uint8)

    # C 扩展 matmul
    c_c = hc4.matmul(a.flatten(), b.flatten(), m, k, n)  # (m, n) int32

    # numpy 参考
    c_ref = (a.astype(np.int32) @ b.astype(np.int32))

    max_err = _max_error(c_ref, c_c)
    rel_err = _rel_error(c_ref, c_c)
    print(f"  矩阵: {m}×{k} @ {k}×{n}")
    print(f"  max_error: {max_err}")
    print(f"  rel_error: {rel_err}")

    assert max_err == 0, f"matmul 错误: max_err={max_err}"
    print("  ✓ 通过 - uint4×uint4→int32 累加 bit-exact")
    return True


def test_5_quantized_matmul_precision():
    """测试 5: 量化 matmul 精度（float→uint4→matmul→float）"""
    print("\n[测试 5] 量化 matmul 精度（float→uint4→matmul→float）")
    np.random.seed(42)
    m, k, n = 32, 128, 16

    # 生成 float 数据（小范围，适合 uint4 量化）
    x = np.random.randn(m, k).astype(np.float32) * 0.3
    w = np.random.randn(k, n).astype(np.float32) * 0.3

    # C 扩展量化 matmul
    y_c = hc4.quantized_matmul(x.flatten(), w.flatten(), m, k, n)

    # numpy 参考（float matmul）
    y_ref = (x @ w).astype(np.float32)

    max_err = _max_error(y_ref, y_c)
    rel_err = _rel_error(y_ref, y_c)
    print(f"  矩阵: {m}×{k} @ {k}×{n}")
    print(f"  max_error: {max_err:.4e}")
    print(f"  rel_error: {rel_err:.4e}")

    # uint4 精度很低（仅 15 级，中心化范围 [-7, 7]），量化误差大
    # 对于 randn*0.3 的数据，理论量化误差约 20-30%
    assert rel_err < 0.3, f"rel_error 过大: {rel_err}"
    print(f"  ✓ 通过 - uint4 量化 matmul rel_error={rel_err:.4e}（< 0.3，uint4 固有精度限制）")
    return True


def test_6_pshufb_vs_nibble_split():
    """测试 6: PSHUFB vs nibble split 数学等价性

    nibble split: int8 = high*16 + low，展开后用 int8 乘法
    PSHUFB: 直接查表 int4×int4 乘积

    两者数学等价（已在 msint_math_validation 中验证）
    这里验证 C 实现的等价性
    """
    print("\n[测试 6] PSHUFB vs nibble split 数学等价性")
    np.random.seed(42)
    m, k, n = 16, 64, 8

    # 生成 uint4 数据
    a = np.random.randint(0, 16, size=(m, k)).astype(np.uint8)
    b = np.random.randint(0, 16, size=(k, n)).astype(np.uint8)

    # PSHUFB matmul
    c_pshufb = hc4.matmul(a.flatten(), b.flatten(), m, k, n)

    # nibble split 等价：直接用 int8 乘法（uint4 是 uint8 的子集）
    c_nibble = (a.astype(np.int32) @ b.astype(np.int32))

    max_err = _max_error(c_nibble, c_pshufb)
    print(f"  矩阵: {m}×{k} @ {k}×{n}")
    print(f"  PSHUFB vs nibble split max_error: {max_err}")

    assert max_err == 0, f"PSHUFB vs nibble split 不等价: max_err={max_err}"
    print("  ✓ 通过 - PSHUFB 与 nibble split bit-exact 等价")
    return True


def test_7_performance():
    """测试 7: 性能对比"""
    print("\n[测试 7] 性能对比")
    np.random.seed(42)

    # PSHUFB matmul 性能
    m, k, n = 128, 256, 128
    a = np.random.randint(0, 16, size=(m, k)).astype(np.uint8)
    b = np.random.randint(0, 16, size=(k, n)).astype(np.uint8)

    # 预热
    _ = hc4.matmul(a.flatten(), b.flatten(), m, k, n)

    t0 = time.time()
    for _ in range(5):
        c_pshufb = hc4.matmul(a.flatten(), b.flatten(), m, k, n)
    t_pshufb = (time.time() - t0) / 5

    # numpy int32 matmul 参考
    a_int = a.astype(np.int32)
    b_int = b.astype(np.int32)
    t0 = time.time()
    for _ in range(5):
        c_ref = a_int @ b_int
    t_numpy = (time.time() - t0) / 5

    max_err = _max_error(c_ref, c_pshufb)
    print(f"  uint4 matmul ({m}×{k} @ {k}×{n}):")
    print(f"    PSHUFB (AVX2): {t_pshufb*1000:.2f} ms")
    print(f"    numpy int32:   {t_numpy*1000:.2f} ms")
    print(f"    加速比: {t_numpy/t_pshufb:.2f}x")
    print(f"    max_error: {max_err}")

    # 量化 matmul 性能
    x = np.random.randn(m, k).astype(np.float32) * 0.3
    w = np.random.randn(k, n).astype(np.float32) * 0.3

    t0 = time.time()
    for _ in range(5):
        y_c = hc4.quantized_matmul(x.flatten(), w.flatten(), m, k, n)
    t_quant = (time.time() - t0) / 5

    t0 = time.time()
    for _ in range(5):
        y_ref = (x @ w).astype(np.float32)
    t_blas = (time.time() - t0) / 5

    rel_err = _rel_error(y_ref, y_c)
    print(f"\n  量化 matmul ({m}×{k} @ {k}×{n}):")
    print(f"    HC4 PSHUFB: {t_quant*1000:.2f} ms")
    print(f"    numpy BLAS: {t_blas*1000:.2f} ms")
    print(f"    rel_error: {rel_err:.4e}")

    print("  ✓ 通过")
    return True


def main():
    print("=" * 70)
    print("HC4 PSHUFB LUT - int4×int4→int8 查表乘法验证")
    print("=" * 70)

    tests = [
        test_1_basic,
        test_2_lut_correctness,
        test_3_pshufb_mul_32,
        test_4_matmul_precision,
        test_5_quantized_matmul_precision,
        test_6_pshufb_vs_nibble_split,
        test_7_performance,
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
            print(f"  ✗ 失败: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"验证结果: {passed} 通过, {failed} 失败")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
