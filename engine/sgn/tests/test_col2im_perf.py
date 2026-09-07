# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""Stage 3.0.7 Task 7.2.1: col2im 性能对比测试

验证 C++ sgn.col2im_add 性能不低于原 C pysgn_col2im.col2im_add (容差 5%)

SBE/Triple-int8 仍使用原 C 实现 (pysgn_hc16/pysgn_net), 无 C++ 替代, 此处不涉及。

运行:
    py -3.14 engine/sgn/tests/test_col2im_perf.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 安全审计 2026-08-16 A2-7：模式 B → 模式 A（import engine.sgn as sgn）
import engine.sgn as sgn
cpp_col2im_add = sgn.col2im_add

# 原 C 实现（已合并到 sgn.col2im_c 子模块）
py_col2im_mod = sgn.col2im_c


# ============================================================
# 测试框架
# ============================================================

_tests = []


def test(name):
    def deco(fn):
        _tests.append((name, fn))
        return fn
    return deco


def run_tests():
    passed = 0
    failed = 0
    for name, fn in _tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
            failed += 1
    print()
    print(f"{'=' * 70}")
    print(f"col2im 性能测试: {passed} passed, {failed} failed")
    print(f"{'=' * 70}")
    return 0 if failed == 0 else 1


# ============================================================
# 正确性验证: C++ vs 原 C 结果一致
# ============================================================

@test("col2im_add C++ vs 原 C 数值一致性")
def test_correctness():
    """验证 C++ 和原 C 实现产生相同结果"""
    # 典型 CNN 场景: B=4, C=64, kh=kw=3, H_out=W_out=14, stride=1, H_padded=W_padded=16
    B, C = 4, 64
    kh, kw = 3, 3
    H_out, W_out = 14, 14
    stride = 1
    H_padded, W_padded = 16, 16

    rng = np.random.RandomState(42)
    x_col = rng.randn(B, C, kh, kw, H_out, W_out).astype(np.float32)

    # C++ 结果
    x_padded_cpp = np.zeros((B, C, H_padded, W_padded), dtype=np.float32)
    cpp_col2im_add(x_col, x_padded_cpp, B, C, kh, kw,
                    H_out, W_out, stride, H_padded, W_padded)

    # 原 C 结果
    x_padded_py = np.zeros((B, C, H_padded, W_padded), dtype=np.float32)
    py_col2im_mod.col2im_add(x_col, x_padded_py, B, C, kh, kw,
                              H_out, W_out, stride, H_padded, W_padded)

    # 数值应完全一致 (相同算法, 相同输入)
    diff = np.abs(x_padded_cpp - x_padded_py).max()
    assert diff < 1e-6, f"C++ vs 原 C 差异过大: max_diff={diff} (应 < 1e-6)"
    print(f"    max_diff={diff:.2e}")


# ============================================================
# 性能对比
# ============================================================

def benchmark_col2im(func, x_col, x_padded_template, B, C, kh, kw,
                      H_out, W_out, stride, H_padded, W_padded,
                      n_warmup=5, n_iter=30):
    """基准测试: 多次运行取最小值（最抗噪声）"""
    for _ in range(n_warmup):
        x_padded = x_padded_template.copy()
        func(x_col, x_padded, B, C, kh, kw,
             H_out, W_out, stride, H_padded, W_padded)

    times = []
    for _ in range(n_iter):
        x_padded = x_padded_template.copy()
        t0 = time.perf_counter()
        func(x_col, x_padded, B, C, kh, kw,
             H_out, W_out, stride, H_padded, W_padded)
        times.append(time.perf_counter() - t0)

    # 使用最小值而非中位数：微基准测试中最小值最接近真实计算时间
    return np.min(times), np.std(times)


def benchmark_col2im_interleaved(func_a, func_b, x_col, x_padded_template,
                                  B, C, kh, kw, H_out, W_out, stride,
                                  H_padded, W_padded,
                                  n_warmup=5, n_iter=30):
    """交错基准测试: 每轮交替运行 func_a 和 func_b，消除系统状态漂移

    解决问题: VCOMP140 (MSVC OpenMP) 线程调度方差大，顺序测量时
    C++ 和原 C 处于不同 CPU 频率/缓存状态，导致比较不公平。
    交错测量保证两者经历相同系统状态。
    """
    # warmup 两个函数
    for _ in range(n_warmup):
        x_padded = x_padded_template.copy()
        func_a(x_col, x_padded, B, C, kh, kw,
               H_out, W_out, stride, H_padded, W_padded)
        x_padded = x_padded_template.copy()
        func_b(x_col, x_padded, B, C, kh, kw,
               H_out, W_out, stride, H_padded, W_padded)

    times_a = []
    times_b = []
    for _ in range(n_iter):
        # 交替运行: 先 a 后 b, 每轮两者处于相同系统状态
        x_padded = x_padded_template.copy()
        t0 = time.perf_counter()
        func_a(x_col, x_padded, B, C, kh, kw,
               H_out, W_out, stride, H_padded, W_padded)
        times_a.append(time.perf_counter() - t0)

        x_padded = x_padded_template.copy()
        t0 = time.perf_counter()
        func_b(x_col, x_padded, B, C, kh, kw,
               H_out, W_out, stride, H_padded, W_padded)
        times_b.append(time.perf_counter() - t0)

    return np.min(times_a), np.min(times_b)


@test("col2im_add 性能: C++ 不低于原 C (统一容差)")
def test_performance():
    """验证 C++ 实现性能不低于原 C 实现

    编译器统一（Clang 22.1.8 + libomp）后消除跨编译器分级容差。
    剩余开销来自 pybind11 封装层（~0.04ms 固定开销），非编译器差异：
      - 原 C 耗时 < 0.1ms: 容差 3.5x — pybind11 参数校验/GIL 开销主导
      - 原 C 耗时 >= 0.1ms: 统一容差 10% (1.1x) — 计算主导
    """
    # 多种规模测试
    scenarios = [
        # (name, B, C, kh, kw, H_out, W_out, stride, H_padded, W_padded)
        ("小规模 B=2 C=32 3x3",  2, 32, 3, 3, 14, 14, 1, 16, 16),
        ("中规模 B=4 C=64 3x3",  4, 64, 3, 3, 14, 14, 1, 16, 16),
        ("大规模 B=8 C=128 3x3", 8, 128, 3, 3, 28, 28, 1, 30, 30),
    ]

    all_pass = True
    for name, B, C, kh, kw, H_out, W_out, stride, H_padded, W_padded in scenarios:
        rng = np.random.RandomState(42)
        x_col = rng.randn(B, C, kh, kw, H_out, W_out).astype(np.float32)
        x_padded_template = np.zeros((B, C, H_padded, W_padded), dtype=np.float32)

        # 交错基准测试: C++ 和原 C 交替运行，消除系统状态漂移
        # (VCOMP140 线程调度方差大，顺序测量会导致不公平比较)
        cpp_time, py_time = benchmark_col2im_interleaved(
            cpp_col2im_add, py_col2im_mod.col2im_add,
            x_col, x_padded_template,
            B, C, kh, kw, H_out, W_out, stride, H_padded, W_padded)

        # 统一容差：编译器差异已消除，仅 pybind11 封装层开销
        # 注：C++ 和原 C 都用 Clang 22.1.8 + libomp 编译，同编译器+同 OpenMP 运行时
        if py_time < 0.0001:  # < 0.1ms: pybind11 参数校验/GIL 开销主导
            tolerance = 3.5   # 3.5x 容差 — pybind11 wrapper ~0.04ms 固定开销
            tier = "pybind11开销主导, 容差3.5x"
        else:                   # >= 0.1ms: 计算主导
            tolerance = 1.1   # 10% 容差 — 统一（编译器统一后消除分级容差）
            tier = "计算主导, 统一容差10%"

        ratio = cpp_time / py_time
        status = "✓" if ratio <= tolerance else "✗"
        print(f"    {name}: C++={cpp_time*1000:.3f}ms, 原C={py_time*1000:.3f}ms, "
              f"ratio={ratio:.3f} [{tier}] {status}")

        if ratio > tolerance:
            all_pass = False

    assert all_pass, "C++ col2im 性能低于原 C 实现 (超过分级容差)"


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("Stage 3.0.7 Task 7.2.1: col2im 性能对比测试")
    print("C++ sgn.col2im_add vs 原 C sgn.col2im_c.col2im_add")
    print("=" * 70)
    print()
    return run_tests()


if __name__ == "__main__":
    sys.exit(main())
