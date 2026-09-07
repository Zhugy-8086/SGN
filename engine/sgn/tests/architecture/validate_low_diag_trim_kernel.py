# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""低位对角裁剪——内核路径验证（选项二：trim_high_diag 内核参数）

背景：n 位拆分（total_bits=32，split_bits 整除 32）时，n² 个 (a,c) 点积
聚合出 2n-1 个 partial[m]。对「int32 截断」目标（只保留结果低 32 位）：
  - m>=n 的项 partial[m]·2^(4m) 均为 2^32 的倍数 → 对低 32 位无贡献 → 可裁剪
  - m<n 的项贡献低 32 位 → 必须保留
  → 裁剪后 partials[m>=n] 保持 0，4 位档（n=8）省 28/64 = 43.75% ALU。

本实验直接验证 C++ 内核路径（区别于 validate_low_diag_trim_int32.py 的
Python 层模拟）：
  1. CppSplitDot.dot_split(w, x, 32, sb, trim_high_diag=True)
     - partials 长度 = 2n-1
     - partials[m>=n] == 0（高位对角被裁剪）
     - partials[m<n] 与全量 dot_split 对应项 bit-exact 一致
  2. CppSplitDot.dot_fused_i32(w, x, sb)（消费端接口）
     - 与精确点积 int32 截断 bit-exact 一致
     - 与 fuse_128(全量 partials) 低 32 位 bit-exact 一致
  3. C++ 与 Python fallback（_PySplitDot）在裁剪路径上逐位一致

split_bits 覆盖 {4, 8, 16}（32 的因数，对应 n=8/4/2）。
分布覆盖：全幅均匀 int32 / 高斯定标 / 稀疏 / 边界值，多 seed × 多 K。

运行: py -3.14 validate_low_diag_trim_kernel.py
"""
import sys
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parents[4]
_ENGINE_DIR = Path(__file__).resolve().parents[3]
for _p in [_PROJ_ROOT, _ENGINE_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

from sgn.msint.cpp_backend import CppSplitDot, USING_CPP
from sgn.msint.cpp_backend import _PySplitDot  # Python fallback 参考实现

TOTAL_BITS = 32


def _to_i32(x):
    """任意精度整数 → 有符号 int32（保留低 32 位位模式）。"""
    lo = x & 0xFFFFFFFF
    return lo - 0x100000000 if lo >= 0x80000000 else lo


def _fuse_i32(partials, split_bits):
    """按 m<n 融合并截断 int32（Python 任意精度参考）。"""
    n = TOTAL_BITS // split_bits
    return _to_i32(sum(partials[m] << (m * split_bits) for m in range(n)))


# ============================================================
# 数据分布生成（与 validate_low_diag_trim_int32.py 一致）
# ============================================================

def _uniform_int32(rng, n):
    return [int(rng.randint(-2**31, 2**31 - 1, dtype=np.int64)) for _ in range(n)]


def _gauss_quant(rng, n, scale=512):
    return [int(np.clip(round(rng.normal(0, scale)), -2**31, 2**31 - 1)) for _ in range(n)]


def _sparse(rng, n, p=0.9):
    vals = []
    for _ in range(n):
        vals.append(0 if rng.random() < p
                    else int(rng.randint(-2**31, 2**31 - 1, dtype=np.int64)))
    return vals


# ============================================================
# 单组验证
# ============================================================

def _run_case(w, x, sb, label, stats):
    n = TOTAL_BITS // sb
    n_partials = 2 * n - 1
    exact = sum(int(a) * int(b) for a, b in zip(w, x))
    exact_i32 = _to_i32(exact)

    # 全量 dot_split（trim_high_diag=False，回归护栏）
    full = CppSplitDot.dot_split(w, x, TOTAL_BITS, sb, False)
    assert len(full) == n_partials, f"{label}: full 长度 {len(full)} != {n_partials}"
    assert _to_i32(sum(full[m] << (m * sb) for m in range(n_partials))) == exact_i32, \
        f"{label}: 全量融合 != 精确 int32"

    # 1) 内核裁剪路径
    tri = CppSplitDot.dot_split(w, x, TOTAL_BITS, sb, True)
    assert len(tri) == n_partials, f"{label}: trim 长度 {len(tri)} != {n_partials}"
    hi_ok = all(tri[m] == 0 for m in range(n, n_partials))
    lo_ok = all(tri[m] == full[m] for m in range(n))
    # 2) 消费端接口 dot_fused_i32
    cpp_i32 = CppSplitDot.dot_fused_i32(w, x, sb)
    ref_i32 = _fuse_i32(full, sb)

    # 3) Python fallback 裁剪路径逐位一致
    py_full = _PySplitDot.dot_split(w, x, TOTAL_BITS, sb, False)
    py_tri = _PySplitDot.dot_split(w, x, TOTAL_BITS, sb, True)
    py_i32 = _PySplitDot.dot_fused_i32(w, x, sb)
    py_lo_ok = all(py_tri[m] == py_full[m] for m in range(n))
    py_hi_ok = all(py_tri[m] == 0 for m in range(n, n_partials))

    ok = (hi_ok and lo_ok and cpp_i32 == exact_i32 and cpp_i32 == ref_i32
          and py_lo_ok and py_hi_ok and py_i32 == exact_i32 and cpp_i32 == py_i32)
    stats["total"] += 1
    stats["ok"] += 1 if ok else 0
    if not ok:
        stats["fail"].append(
            f"{label}(sb={sb}) hi={hi_ok} lo={lo_ok} "
            f"cpp_i32={cpp_i32} exact={exact_i32} ref={ref_i32} py={py_i32}")
    return ok


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 78)
    print("低位对角裁剪——内核路径验证（选项二：trim_high_diag）")
    print(f"C++ 后端: {USING_CPP}")
    print("=" * 78)
    print()

    stats = {"total": 0, "ok": 0, "fail": []}
    rng = np.random.RandomState(20260814)

    for sb in (4, 8, 16):
        n = TOTAL_BITS // sb
        n_partials = 2 * n - 1
        total_pairs = n * n
        keep_pairs = sum(n - a for a in range(n))
        print(f"split_bits={sb}  n={n}  partials={n_partials}  "
              f"保留 {keep_pairs}/{total_pairs} 对（裁剪 {total_pairs - keep_pairs}，"
              f"{-100 * (total_pairs - keep_pairs) / total_pairs:.2f}%）")

    # 1) 合成分布：多 seed × 多 K
    for seed in range(5):
        r2 = np.random.RandomState(seed)
        for K in (64, 256, 1024, 4096):
            for sb in (4, 8, 16):
                w = _uniform_int32(r2, K)
                x = _uniform_int32(r2, K)
                _run_case(w, x, sb, f"均匀 seed{seed} K{K}", stats)

                w = _gauss_quant(r2, K)
                _run_case(w, x, sb, f"高斯w seed{seed} K{K}", stats)

                w = _sparse(r2, K)
                _run_case(w, x, sb, f"稀疏w seed{seed} K{K}", stats)

    # 2) 边界值组合（0 / ±1 / ±2^31-1 / -2^31 / 满幅 nibble 边界）
    edges = [0, 1, -1, 2**31 - 1, -2**31, 2**16 - 1, -2**16, 2**15, -2**15,
             15, 16, -16, 2**28 - 1, -2**28]
    for seed in range(3):
        r2 = np.random.RandomState(seed)
        K = 64
        w = [edges[r2.randint(len(edges))] for _ in range(K)]
        x = [edges[r2.randint(len(edges))] for _ in range(K)]
        for sb in (4, 8, 16):
            _run_case(w, x, sb, f"边界 seed{seed} K{K}", stats)

    print()
    print(f"统计: 共 {stats['total']} 组，内核裁剪路径一致 {stats['ok']}/{stats['total']}")
    if stats["fail"]:
        print("失败组:")
        for f in stats["fail"][:15]:
            print(f"  {f}")
    print()
    ok = stats["ok"] == stats["total"] and stats["total"] > 0
    print("结论:")
    print(f"  [{'PASS' if ok else 'FAIL'}] 内核 trim_high_diag + dot_fused_i32："
          f"{'bit-exact（含 16/8/4 三档）' if ok else '存在不一致'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
