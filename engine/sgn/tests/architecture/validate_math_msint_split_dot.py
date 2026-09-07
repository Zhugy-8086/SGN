# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""MSint 前向多精度拆分点积验证（SplitDot）

对标范式文档 docs/msint_multisplit_paradigm/MSint多精度拆分计算范式.md §2

验证项：
  - H1 单融合数值等价：fused == Σ w_i*x_i（任意精度），多 seed + 大 K + 边界值
  - 多输出模式：dot_split 产出 [fine, cross, coarse]（n=2），fuse_128 精确还原
  - 一般化：split_bits=8（4 部分 → 7 个 partial），验证 1:N 多尺度
  - C++ vs Python 参考实现（_PySplitDot）一致性
  - 位拆分可逆性：split_parts 后 reconstruct == value（含负数）

运行: py -3.14 validate_math_msint_split_dot.py
"""
import sys
from pathlib import Path

# 添加项目根目录和 engine/ 到 path
_PROJ_ROOT = Path(__file__).resolve().parents[3]
_ENGINE_DIR = Path(__file__).resolve().parents[2]
for _p in [_PROJ_ROOT, _ENGINE_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

from sgn.msint.cpp_backend import CppSplitDot, _PySplitDot, USING_CPP


def _rand_int32(rng, n):
    """生成 n 个 int32 范围内的随机有符号值（含边界极端值）。"""
    vals = []
    for _ in range(n):
        v = int(rng.randint(-2**31, 2**31 - 1, dtype=np.int64))
        vals.append(v)
    return vals


def reconstruct(parts, split_bits):
    """用位拆分结果重建原值（Python 任意精度）。"""
    return sum(p << (k * split_bits) for k, p in enumerate(parts))


def _signed64(x):
    """把任意精度整数 x 截断为有符号 64 位（保留低 64 位位模式）。"""
    lo = x & ((1 << 64) - 1)
    return lo - (1 << 64) if lo >= (1 << 63) else lo


# ============================================================
# H1: 单融合数值等价
# ============================================================

def test_h1_fused_equals_exact():
    """H1: dot_fused / fuse_128 数值等价于原始点积（任意精度）。"""
    total_fail = 0
    rng = np.random.RandomState(20260813)

    for seed in range(5):
        rng2 = np.random.RandomState(seed)
        K = int(rng2.randint(1, 200))
        w = _rand_int32(rng2, K)
        x = _rand_int32(rng2, K)

        exact = sum(a * b for a, b in zip(w, x))

        # C++ fuse_128（128 位精确）
        partials = CppSplitDot.dot_split(w, x, 32, 16)
        hi, lo = CppSplitDot.fuse_128(partials, 16)
        # 128 位有符号重建：(hi:int64)<<64 | lo:uint64。
        # hi 为有符号高字，Python 负数左移自动做无限符号扩展，
        # OR 上低 64 位即得正确的 128 位二补数，无需再减 2^128。
        fused_128 = (hi << 64) | lo

        # dot_fused 低 64 位（有符号）应与 fuse_128 低 64 位位模式一致
        fused_64 = CppSplitDot.dot_fused(w, x, 32, 16)
        assert fused_64 == _signed64(fused_128), \
            f"seed{seed}: dot_fused 低64位不一致 {fused_64} vs {_signed64(fused_128)}"

        # 128 位结果应精确等于 exact（只要 exact 落在 128 位范围内）
        if not (-(1 << 127) <= exact < (1 << 127)):
            total_fail += 1
            print(f"  [SKIP] seed{seed}: exact={exact} 超出 128 位范围，跳过严格比对")
            continue
        if fused_128 != exact:
            total_fail += 1
            print(f"  [FAIL] seed{seed}: fused_128={fused_128} vs exact={exact}")

    assert total_fail == 0, f"H1 失败 {total_fail} 次"
    print(f"[PASS] H1 单融合数值等价：dot_fused/fuse_128 与精确点积一致（5 seed × K 随机）")


def test_h1_boundary_values():
    """H1 边界值：0、±1、±(2^31-1)、-2^31、±2^16-1。"""
    rng = np.random.RandomState(7)
    edges = [0, 1, -1, 2**31 - 1, -2**31, 2**16 - 1, -2**16, 2**15, -2**15]

    K = 50
    for _ in range(50):
        # 构造含边界值的向量（其余随机）
        w = [edges[rng.randint(len(edges))] for _ in range(K)]
        x = [edges[rng.randint(len(edges))] for _ in range(K)]

        exact = sum(a * b for a, b in zip(w, x))
        partials = CppSplitDot.dot_split(w, x, 32, 16)
        hi, lo = CppSplitDot.fuse_128(partials, 16)
        fused_128 = (hi << 64) | lo

        assert fused_128 == exact, f"边界值失败: fused={fused_128} exact={exact}"
    print("[PASS] H1 边界值：±1, ±2^31-1, -2^31, ±2^16-1 全部等价")


# ============================================================
# 多输出模式：n=2 → [fine, cross, coarse]
# ============================================================

def test_multio_output_fine_cross_coarse():
    """多输出模式：n=2（int32→int16）产出 [fine, cross, coarse]。"""
    rng = np.random.RandomState(99)
    K = 64
    w = _rand_int32(rng, K)
    x = _rand_int32(rng, K)

    partials = CppSplitDot.dot_split(w, x, 32, 16)
    assert len(partials) == 3, f"n=2 应产出 3 个 partial，实际 {len(partials)}"

    fine, cross, coarse = partials

    # 独立重新计算三个尺度（用拆分后的高低位）
    w_l = [v & 0xFFFF for v in w]
    w_h = [v >> 16 for v in w]          # 符号扩展
    x_l = [v & 0xFFFF for v in x]
    x_h = [v >> 16 for v in x]

    exp_fine = sum(a * b for a, b in zip(w_l, x_l))
    exp_cross = sum(a * b + c * d for a, b, c, d in zip(w_h, x_l, w_l, x_h))
    exp_coarse = sum(a * b for a, b in zip(w_h, x_h))

    assert fine == exp_fine, f"fine={fine} exp={exp_fine}"
    assert cross == exp_cross, f"cross={cross} exp={exp_cross}"
    assert coarse == exp_coarse, f"coarse={coarse} exp={exp_coarse}"

    # 三尺度应能精确还原融合值（多输出 → 单融合可逆）
    hi, lo = CppSplitDot.fuse_128(partials, 16)
    fused_128 = (hi << 64) | lo
    exact = sum(a * b for a, b in zip(w, x))
    assert fused_128 == exact

    print(f"[PASS] 多输出 n=2: [fine={fine}, cross={cross}, coarse={coarse}] 独立计算一致且可逆")


# ============================================================
# 一般化多尺度：split_bits=8 → 4 部分 → 7 个 partial（1:N）
# ============================================================

def test_general_n_part_multiscale():
    """一般化：split_bits=8（int32→4×int8）产出 2n-1=7 个 partial。"""
    rng = np.random.RandomState(1234)
    K = 80
    w = _rand_int32(rng, K)
    x = _rand_int32(rng, K)

    partials = CppSplitDot.dot_split(w, x, 32, 8)
    assert len(partials) == 7, f"n=4 应产出 7 个 partial，实际 {len(partials)}"

    # fuse_128 精确还原
    hi, lo = CppSplitDot.fuse_128(partials, 8)
    fused_128 = (hi << 64) | lo
    exact = sum(a * b for a, b in zip(w, x))
    assert fused_128 == exact, f"8bit 拆分融合 {fused_128} != {exact}"
    print(f"[PASS] 一般化 n=4: 7 个 partial，fuse_128 精确还原（1:N 多尺度成立）")


def test_general_n8_nibble_multiscale():
    """一般化：split_bits=4（int32→8×nibble）产出 2n-1=15 个 partial。"""
    rng = np.random.RandomState(20260814)
    K = 160
    w = _rand_int32(rng, K)
    x = _rand_int32(rng, K)

    partials = CppSplitDot.dot_split(w, x, 32, 4)
    assert len(partials) == 15, f"n=8 应产出 15 个 partial，实际 {len(partials)}"

    # fuse_128 精确还原
    hi, lo = CppSplitDot.fuse_128(partials, 4)
    fused_128 = (hi << 64) | lo
    exact = sum(a * b for a, b in zip(w, x))
    assert fused_128 == exact, f"4bit 拆分融合 {fused_128} != {exact}"
    print(f"[PASS] 一般化 n=8: 15 个 partial，fuse_128 精确还原（4-bit nibble 路径）")


# ============================================================
# 位拆分可逆性
# ============================================================

def test_split_parts_reversible():
    """split_parts 位拆分可逆：reconstruct(parts) == value（含负数）。"""
    rng = np.random.RandomState(555)
    n_fail = 0
    for _ in range(20000):
        v = int(rng.randint(-2**31, 2**31 - 1, dtype=np.int64))
        for split_bits in (4, 8, 16):
            parts = CppSplitDot.split_parts(v, 32, split_bits)
            if reconstruct(parts, split_bits) != v:
                n_fail += 1
                print(f"  [FAIL] v={v} split_bits={split_bits} parts={parts}")
    assert n_fail == 0, f"位拆分可逆失败 {n_fail} 次"
    print("[PASS] split_parts 位拆分可逆：20K 随机 int32 × {4,8,16} 全部重建一致")


# ============================================================
# C++ vs Python 参考实现
# ============================================================

def test_cpp_vs_python_reference():
    """C++ SplitDot 与 Python 参考实现 _PySplitDot 结果一致。"""
    rng = np.random.RandomState(42)
    n_fail = 0
    for _ in range(500):
        K = int(rng.randint(1, 100))
        w = _rand_int32(rng, K)
        x = _rand_int32(rng, K)
        for split_bits in (4, 8, 16):
            cpp_partials = CppSplitDot.dot_split(w, x, 32, split_bits)
            py_partials = _PySplitDot.dot_split(w, x, 32, split_bits)
            if cpp_partials != py_partials:
                n_fail += 1
                print(f"  [FAIL] split_bits={split_bits}: cpp={cpp_partials} py={py_partials}")
    assert n_fail == 0, f"C++/Python 不一致 {n_fail} 次"
    print(f"[PASS] C++ vs Python 参考实现：500 组 × {4,8,16} 全部一致（C++={USING_CPP}）")


def test_dot_fused_cpp_vs_python():
    """dot_fused（低 64 位）C++ 与 Python 一致。"""
    rng = np.random.RandomState(3)
    for _ in range(500):
        K = int(rng.randint(1, 100))
        w = _rand_int32(rng, K)
        x = _rand_int32(rng, K)
        cpp = CppSplitDot.dot_fused(w, x, 32, 16)
        py = _PySplitDot.dot_fused(w, x, 32, 16)
        assert cpp == py, f"dot_fused 不一致: cpp={cpp} py={py}"
    print("[PASS] dot_fused（低 64 位）C++ 与 Python 参考一致")


if __name__ == "__main__":
    print("=" * 78)
    print("MSint 前向多精度拆分点积验证（SplitDot）")
    print(f"C++ 模式: {USING_CPP}")
    print("=" * 78)
    print()

    test_split_parts_reversible()
    test_h1_fused_equals_exact()
    test_h1_boundary_values()
    test_multio_output_fine_cross_coarse()
    test_general_n_part_multiscale()
    test_general_n8_nibble_multiscale()
    test_cpp_vs_python_reference()
    test_dot_fused_cpp_vs_python()

    print()
    print("=" * 78)
    print("=== All SplitDot 验证 PASSED ===")
    print("  - H1 单融合数值等价（fused == Σ w_i*x_i）")
    print("  - 多输出模式 [fine, cross, coarse] 多尺度 1:N")
    print("  - 一般化 split_bits=8 → 7 个 partial")
    print("  - 一般化 split_bits=4（nibble 路径）→ 15 个 partial")
    print("  - 位拆分可逆（含负数）")
    print("  - C++ vs Python 参考一致")
    print("=" * 78)
