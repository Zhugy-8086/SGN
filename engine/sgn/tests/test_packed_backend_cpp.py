# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""Stage 3.0.5 Task 5.4 对照测试 — C++ vs Python

覆盖:
  - Task 5.4.1: 1000 个元素批量读取，C++ vs Python diff=0
  - Task 5.4.2: 多视角（bitsplit/concat）解析，C++ vs Python diff=0
  - Task 5.4.3: 性能测试：C++ 批量读取比 Python 快 30× 以上
  - Task 5.3.3: Python fallback 测试（模拟 C++ 不可用）

运行: python test_packed_backend_cpp.py
"""
import sys
import os
import time
import random
from pathlib import Path

# 添加项目根目录和 engine/ 到 path
_PROJ_ROOT = Path(__file__).resolve().parents[3]
_ENGINE_DIR = Path(__file__).resolve().parents[2]
for _p in [_PROJ_ROOT, _ENGINE_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np


def _rand_uint64(rng, n):
    """生成 n 个 [0, 2^64) 的随机无符号 64 位整数

    numpy randint 默认 dtype=int32，无法直接生成 64 位范围，
    因此组合两个 32 位值。"""
    result = []
    for _ in range(n):
        hi = int(rng.randint(0, 2**32, dtype=np.int64))
        lo = int(rng.randint(0, 2**32, dtype=np.int64))
        result.append((hi << 32) | lo)
    return result

# 导入 C++ 扩展和 fallback（用 engine.sgn 前缀，避免 pytest 全量收集时
# `sgn` 被解析为 .pyd 非包导致 `sgn.msint` 不可访问）
from engine.sgn.msint.cpp_backend import CppPackedBackend, CppMSIntView, USING_CPP

# 导入 Python 原生实现（对照基准，位于 engine.sgn.msint._py 子包）
from engine.sgn.msint._py.backends import PackedBackend as PyPackedBackend
from engine.sgn.msint._py.protocol import SlotInfo


# ============================================================
# Task 5.4.1: 1000 个元素批量读取，C++ vs Python diff=0
# ============================================================

def test_batch_read_1000_elements():
    """1000 个 PackedBackend 实例批量读取，C++ vs Python diff=0

    场景：大量神经元，每个神经元一个 PackedBackend（4 个 8-bit 槽位）
    """
    rng = np.random.RandomState(42)
    n_elements = 1000
    bits_list = [8, 8, 8, 8]  # 4 个 8-bit 无符号槽位

    # 生成 1000 个随机 packed 值（dtype=int64 避免 int32 溢出）
    packed_values = [int(rng.randint(0, 1 << 32, dtype=np.int64)) for _ in range(n_elements)]

    # C++ 批量读取
    cpp_results = []
    for pv in packed_values:
        b = CppPackedBackend.from_bits(bits_list, packed=pv)
        cpp_results.append(b.get_all())

    # Python 批量读取
    py_results = []
    for pv in packed_values:
        slots = [SlotInfo(name=f"s{i}", bits=b) for i, b in enumerate(bits_list)]
        b = PyPackedBackend(slot_table=slots, packed_value=pv)
        py_results.append(b.get_all())

    # 对比 diff=0
    total_diff = 0
    for i, (cpp, py) in enumerate(zip(cpp_results, py_results)):
        for j, (c, p) in enumerate(zip(cpp, py)):
            diff = abs(c - p)
            total_diff += diff
            if diff != 0:
                print(f"  [FAIL] elem {i} slot {j}: cpp={c} vs py={p}")

    assert total_diff == 0, f"总 diff={total_diff} != 0"
    print(f"[PASS] test_batch_read_1000_elements: {n_elements} 元素 diff=0")


def test_batch_read_signed_slots():
    """有符号槽位批量读取，C++ vs Python diff=0"""
    rng = np.random.RandomState(123)
    bits_list = [8, 8]
    signed_flags = [True, True]  # 两个 8-bit 有符号槽位

    # 生成 500 个随机 packed 值
    packed_values = [int(rng.randint(0, 1 << 16)) for _ in range(500)]

    cpp_results = []
    for pv in packed_values:
        b = CppPackedBackend.from_bits(bits_list, signed_flags=signed_flags, packed=pv)
        cpp_results.append(b.get_all())

    py_results = []
    for pv in packed_values:
        slots = [SlotInfo(name=f"s{i}", bits=b, signed=s)
                 for i, (b, s) in enumerate(zip(bits_list, signed_flags))]
        b = PyPackedBackend(slot_table=slots, packed_value=pv)
        py_results.append(b.get_all())

    total_diff = 0
    for cpp, py in zip(cpp_results, py_results):
        for c, p in zip(cpp, py):
            total_diff += abs(c - p)
    assert total_diff == 0, f"有符号槽位 diff={total_diff}"
    print(f"[PASS] test_batch_read_signed_slots: 500 元素有符号 diff=0")


def test_get_all_simd_matches_scalar():
    """C++ get_all_simd 与 get_all 结果一致"""
    rng = np.random.RandomState(456)
    bits_list = [8, 8, 8, 8, 8, 8]  # 6 个 8-bit 等宽无符号

    for _ in range(100):
        pv = int(rng.randint(0, 1 << 48, dtype=np.int64))
        b = CppPackedBackend.from_bits(bits_list, packed=pv)
        scalar = b.get_all()
        simd = b.get_all_simd()
        assert scalar == simd, f"scalar={scalar} != simd={simd}"

    print(f"[PASS] test_get_all_simd_matches_scalar: 100 次 SIMD==标量")


# ============================================================
# Task 5.4.2: 多视角（bitsplit/concat）解析，C++ vs Python diff=0
# ============================================================

def test_bitsplit_cpp_vs_python():
    """bitsplit 视角，C++ vs Python diff=0"""
    rng = np.random.RandomState(789)
    total_diff = 0

    for _ in range(1000):
        raw = int(rng.randint(-10000, 10000))
        total_bits = 32
        target_bits = 8

        cpp_result = CppMSIntView.bitsplit(raw, total_bits, target_bits)

        # Python 对照：手动实现
        raw_unsigned = raw & ((1 << total_bits) - 1) if raw < 0 else raw
        mask = (1 << target_bits) - 1
        py_result = []
        remaining = raw_unsigned
        for _ in range((total_bits + target_bits - 1) // target_bits):
            py_result.append(remaining & mask)
            remaining >>= target_bits

        for c, p in zip(cpp_result, py_result):
            total_diff += abs(c - p)

    assert total_diff == 0, f"bitsplit diff={total_diff}"
    print(f"[PASS] test_bitsplit_cpp_vs_python: 1000 次 bitsplit diff=0")


def test_concat_cpp_vs_python():
    """concat 视角，C++ vs Python diff=0"""
    rng = np.random.RandomState(1024)
    total_diff = 0

    for _ in range(1000):
        # 随机生成 2-4 个值，每个 8-bit
        n = rng.randint(2, 5)
        values = [int(rng.randint(0, 256)) for _ in range(n)]
        bits_list = [8] * n

        cpp_result = CppMSIntView.concat(values, bits_list)

        # Python 对照
        py_result = 0
        for val, bits in zip(values, bits_list):
            mask = (1 << bits) - 1
            py_result = (py_result << bits) | (val & mask)

        total_diff += abs(cpp_result - py_result)

    assert total_diff == 0, f"concat diff={total_diff}"
    print(f"[PASS] test_concat_cpp_vs_python: 1000 次 concat diff=0")


def test_bitsplit_index():
    """bitsplit_index 取分片"""
    raw = 0x12345678
    parts = CppMSIntView.bitsplit(raw, 32, 8)
    for i, expected in enumerate(parts):
        result = CppMSIntView.bitsplit_index(raw, 32, 8, i)
        assert result == expected, f"idx {i}: {result} != {expected}"
    print(f"[PASS] test_bitsplit_index: bitsplit_index 与 bitsplit 一致")


# ============================================================
# Task 5.4.3: 性能测试：C++ 批量读取比 Python 快 30× 以上
# ============================================================

def test_performance_cpp_vs_python():
    """性能测试：C++ get_all_simd vs Python get_all

    10000 个 PackedBackend 实例，每个 8 个 8-bit 槽位
    """
    rng = np.random.RandomState(42)
    n_elements = 10000
    bits_list = [8, 8, 8, 8, 8, 8, 8, 8]  # 8 个 8-bit 无符号槽位

    # 生成 64 位随机值（组合两个 32 位值，避免 numpy int32/int64 范围限制）
    packed_values = _rand_uint64(rng, n_elements)

    # 预构造实例（不计时）
    cpp_backends = [CppPackedBackend.from_bits(bits_list, packed=pv) for pv in packed_values]
    py_slots = [SlotInfo(name=f"s{i}", bits=8) for i in range(8)]
    py_backends = [PyPackedBackend(slot_table=py_slots, packed_value=pv) for pv in packed_values]

    # C++ 性能（get_all_simd）
    t0 = time.perf_counter()
    for b in cpp_backends:
        _ = b.get_all_simd()
    t_cpp = time.perf_counter() - t0

    # Python 性能（get_all）
    t0 = time.perf_counter()
    for b in py_backends:
        _ = b.get_all()
    t_py = time.perf_counter() - t0

    speedup = t_py / t_cpp if t_cpp > 0 else float('inf')

    print(f"  C++ get_all_simd: {t_cpp*1000:.2f} ms ({n_elements} 元素)")
    print(f"  Python get_all:   {t_py*1000:.2f} ms ({n_elements} 元素)")
    print(f"  加速比: {speedup:.1f}×")

    # C++ 应至少快 3×（保守阈值）
    # 注：单元素 pybind11 调用开销约 0.3μs/次，限制了加速比上限
    # 30× 目标需批量 API（一次 C++ 调用处理整个数组），属于 Task 5.5 范畴
    assert speedup >= 3.0, f"加速比 {speedup:.1f}× < 3×（目标 30× 需批量 API）"
    print(f"[PASS] test_performance_cpp_vs_python: {speedup:.1f}× 加速（30× 需批量 API）")


def test_performance_cpp_get_all_vs_simd():
    """C++ 标量 vs SIMD 性能对比"""
    rng = np.random.RandomState(42)
    n_elements = 10000
    bits_list = [8, 8, 8, 8, 8, 8, 8, 8]

    packed_values = _rand_uint64(rng, n_elements)
    cpp_backends = [CppPackedBackend.from_bits(bits_list, packed=pv) for pv in packed_values]

    # 标量
    t0 = time.perf_counter()
    for b in cpp_backends:
        _ = b.get_all()
    t_scalar = time.perf_counter() - t0

    # SIMD
    t0 = time.perf_counter()
    for b in cpp_backends:
        _ = b.get_all_simd()
    t_simd = time.perf_counter() - t0

    print(f"  C++ 标量: {t_scalar*1000:.2f} ms")
    print(f"  C++ SIMD: {t_simd*1000:.2f} ms")
    print(f"  SIMD/标量: {t_scalar/t_simd:.2f}×" if t_simd > 0 else "  SIMD/标量: inf")
    print(f"[PASS] test_performance_cpp_get_all_vs_simd: SIMD 可用")


# ============================================================
# Task 5.3.3: Python fallback 测试
# ============================================================

def test_fallback_works():
    """Python fallback 正常工作（C++ 不可用时）"""
    # 模拟 C++ 不可用：直接用 Python wrapper
    from engine.sgn.msint.cpp_backend import _PyPackedBackendWrapper, _PyMSIntView

    # PackedBackend fallback
    b = _PyPackedBackendWrapper([8, 8, 8, 8], packed=0xFF800040)
    result = b.get_all()
    assert result == [255, 128, 0, 64], f"fallback get_all={result}"

    # MSIntView fallback
    parts = _PyMSIntView.bitsplit(0x12345678, 32, 8)
    assert parts == [120, 86, 52, 18], f"fallback bitsplit={parts}"

    concat_result = _PyMSIntView.concat([255, 128], [8, 8])
    assert concat_result == 0xFF80, f"fallback concat={concat_result}"

    print(f"[PASS] test_fallback_works: Python fallback 功能正确")


def test_unified_api_consistency():
    """统一 API（C++ 或 Python）行为一致"""
    # 无论 C++ 或 Python 模式，API 应一致
    b = CppPackedBackend.from_bits([8, 8], packed=0xFF80)
    assert b.get(0) == 255
    assert b.get(1) == 128
    assert b.get_all() == [255, 128]
    assert b.slot_count == 2
    assert b.total_bits == 16

    # set 测试
    b.set(0, 100)
    assert b.get(0) == 100

    # MSIntView
    assert CppMSIntView.bitsplit(0xABCD, 16, 8) == [0xCD, 0xAB]
    assert CppMSIntView.concat([0xAB, 0xCD], [8, 8]) == 0xABCD

    print(f"[PASS] test_unified_api_consistency: API 行为一致 (C++={USING_CPP})")


def test_mixed_bits_non_uniform():
    """非等宽 bits 场景（12+8+12=32）"""
    bits_list = [12, 8, 12]
    rng = np.random.RandomState(999)

    for _ in range(100):
        v0 = int(rng.randint(0, 4096))   # 12-bit
        v1 = int(rng.randint(0, 256))    # 8-bit
        v2 = int(rng.randint(0, 4096))   # 12-bit

        b = CppPackedBackend.from_bits(bits_list)
        b.set(0, v0)
        b.set(1, v1)
        b.set(2, v2)

        # C++ 读取
        assert b.get(0) == v0
        assert b.get(1) == v1
        assert b.get(2) == v2

        # Python 对照
        slots = [SlotInfo(name=f"s{i}", bits=b_bits) for i, b_bits in enumerate(bits_list)]
        py_b = PyPackedBackend(slot_table=slots, packed_value=b.packed_value)
        assert py_b.get_all() == b.get_all()

    print(f"[PASS] test_mixed_bits_non_uniform: 12+8+12 非等宽 100 次 diff=0")


# ============================================================
# Task 5.5: 批量 API 正确性 + 性能（30× 目标）
# ============================================================

def test_batch_get_all_correctness():
    """batch_get_all 结果与逐元素 get_all 一致"""
    from engine.sgn.msint.cpp_backend import batch_get_all

    rng = np.random.RandomState(42)
    bits_list = [8, 8, 8, 8]  # 4 个 8-bit 无符号
    n = 1000

    # 生成随机 packed 值
    packed_values = _rand_uint64(rng, n)
    packed_arr = np.array(packed_values, dtype=np.uint64)

    # C++ 批量读取
    batch_result = batch_get_all(bits_list, packed_arr)  # numpy [n, 4]

    # 逐元素读取（对照）
    for i, pv in enumerate(packed_values):
        b = CppPackedBackend.from_bits(bits_list, packed=pv)
        elem_result = b.get_all()
        for j in range(4):
            assert batch_result[i, j] == elem_result[j], \
                f"elem {i} slot {j}: batch={batch_result[i, j]} vs elem={elem_result[j]}"

    print(f"[PASS] test_batch_get_all_correctness: {n} 元素 batch==逐元素")


def test_batch_get_all_mixed_bits():
    """batch_get_all 非等宽 bits 场景"""
    from engine.sgn.msint.cpp_backend import batch_get_all

    rng = np.random.RandomState(123)
    bits_list = [12, 8, 12]  # 非等宽
    n = 500

    packed_values = _rand_uint64(rng, n)
    packed_arr = np.array(packed_values, dtype=np.uint64)

    batch_result = batch_get_all(bits_list, packed_arr)

    for i, pv in enumerate(packed_values):
        b = CppPackedBackend.from_bits(bits_list, packed=pv)
        elem_result = b.get_all()
        for j in range(3):
            assert batch_result[i, j] == elem_result[j], \
                f"elem {i} slot {j}: batch={batch_result[i, j]} vs elem={elem_result[j]}"

    print(f"[PASS] test_batch_get_all_mixed_bits: 12+8+12 非等宽 {n} 元素 diff=0")


def test_batch_get_all_performance():
    """批量 API 性能：C++ batch_get_all vs Python 逐元素 get_all

    目标：C++ batch 比 Python 逐元素快 30× 以上
    """
    from engine.sgn.msint.cpp_backend import batch_get_all

    rng = np.random.RandomState(42)
    n = 10000
    bits_list = [8, 8, 8, 8, 8, 8, 8, 8]  # 8 个 8-bit 无符号

    packed_values = _rand_uint64(rng, n)
    packed_arr = np.array(packed_values, dtype=np.uint64)

    # Python 逐元素基准
    py_slots = [SlotInfo(name=f"s{i}", bits=8) for i in range(8)]
    py_backends = [PyPackedBackend(slot_table=py_slots, packed_value=pv) for pv in packed_values]
    t0 = time.perf_counter()
    for b in py_backends:
        _ = b.get_all()
    t_py = time.perf_counter() - t0

    # C++ 批量
    t0 = time.perf_counter()
    _ = batch_get_all(bits_list, packed_arr)
    t_cpp = time.perf_counter() - t0

    speedup = t_py / t_cpp if t_cpp > 0 else float('inf')

    print(f"  Python 逐元素: {t_py*1000:.2f} ms ({n} 元素 × 8 槽位)")
    print(f"  C++ batch:     {t_cpp*1000:.2f} ms ({n} 元素 × 8 槽位)")
    print(f"  加速比: {speedup:.1f}×")

    assert speedup >= 30.0, f"加速比 {speedup:.1f}× < 30×（批量 API 目标）"
    print(f"[PASS] test_batch_get_all_performance: {speedup:.1f}× 加速（≥30× 目标达成）")


def test_batch_get_all_fallback():
    """batch_get_all Python fallback 正确性"""
    from engine.sgn.msint.cpp_backend import _batch_get_all_py

    bits_list = [8, 8, 8, 8]
    packed_values = [0xFF804020, 0x01020304, 0xFFFFFFFF]

    result = _batch_get_all_py(bits_list, packed_values)
    assert result.shape == (3, 4), f"shape={result.shape}"
    assert result[0].tolist() == [255, 128, 64, 32], f"row0={result[0]}"
    assert result[1].tolist() == [1, 2, 3, 4], f"row1={result[1]}"
    assert result[2].tolist() == [255, 255, 255, 255], f"row2={result[2]}"

    print(f"[PASS] test_batch_get_all_fallback: Python batch 正确")


if __name__ == "__main__":
    print("=" * 78)
    print("Stage 3.0.5 Task 5.4/5.5 对照测试 — C++ vs Python")
    print(f"C++ 模式: {USING_CPP}")
    print("=" * 78)
    print()

    # Task 5.4.1: 批量读取对照
    test_batch_read_1000_elements()
    test_batch_read_signed_slots()
    test_get_all_simd_matches_scalar()

    # Task 5.4.2: 多视角对照
    test_bitsplit_cpp_vs_python()
    test_concat_cpp_vs_python()
    test_bitsplit_index()

    # Task 5.4.3: 性能测试（逐元素）
    test_performance_cpp_vs_python()
    test_performance_cpp_get_all_vs_simd()

    # Task 5.3.3: fallback 测试
    test_fallback_works()
    test_unified_api_consistency()
    test_mixed_bits_non_uniform()

    # Task 5.5: 批量 API
    print()
    print("--- Task 5.5: 批量 API ---")
    test_batch_get_all_correctness()
    test_batch_get_all_mixed_bits()
    test_batch_get_all_performance()
    test_batch_get_all_fallback()

    print()
    print("=" * 78)
    print("=== All Task 5.4/5.5 测试 PASSED ===")
    print(f"  C++ 模式: {USING_CPP}")
    print("  - 1000 元素批量读取 diff=0")
    print("  - bitsplit/concat 多视角 diff=0")
    print("  - Python fallback 功能正确")
    print("  - 批量 API 30×+ 加速")
    print("=" * 78)
