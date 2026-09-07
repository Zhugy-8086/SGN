# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""MSint 异构粒度拆分点积 — 指令集优化前 baseline 基准 + SIMD 适用性分析

定位（对标范式文档 §7 H3 / 项目记忆结论）：
  1:N 多精度解释（MultiScaleView）与 Level 逐元素精度选择（PrecisionSelector /
  LeveledSplitDot）是新实现，尚无专门指令集优化；H3 刚落地，之前给 HC16 / MSint
  旧路径的 SIMD 优化（AVX-VNNI 等）对异构粒度逻辑基本不适用。
  因此当前测得的速度 = **干净未优化的 baseline**，是后续每条 SIMD 优化收益可归因的
  公平起点（"基准占优"）。

本脚本两个产物：
  A. baseline 耗时表：对标量全精度 / 同粒度 / 异构逐元素粒度 / 1:N 多输出
     在 K∈{256,1024,4096,16384}、粒度 {16,8,4} 下测未优化耗时，并给吞吐（MAC/s）。
  B. SIMD 适用性分析表：逐部分判定 计算层 vs 决策层 → 是否该 SIMD、
     复用哪些指令、回退要求（遵守项目硬约束：SIMD 必须留非 x86 回退路径）。

运行: py -3.14 validate_bench_msint_simd_baseline.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_PROJ_ROOT = Path(__file__).resolve().parents[3]
_ENGINE_DIR = Path(__file__).resolve().parents[2]
for _p in [_PROJ_ROOT, _ENGINE_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sgn.msint.cpp_backend import (
    CppSplitDot, CppLeveledSplitDot, USING_CPP,
)

# ------------------------------------------------------------
# SIMD 适用性分析（静态表，脚本同时打印）
# ------------------------------------------------------------
APPLICABILITY = [
    ("select_levels", "决策层(标量阈值 if-else)", "❌ 不适用",
     "无 SIMD 数据并行模式；要优化走算法层(LUT/分支预测)，与项目结论一致"),
    ("split_parts(位拆分)", "计算层(逐元素独立)", "⚠️ 视分组",
     "异构需先按粒度分组再 SIMD，否则布局混乱；复用 PSHUFB/BMI 位重排"),
    ("dot_split 组内点积", "计算层(大数组乘累加)", "✅ 最适用",
     "组内同粒度布局规整；复用 _mm256_madd_epi16 整数乘累加家族；必须留标量回退"),
    ("dot_fused 128 位融合", "计算层(标量__int128)", "⚠️ 收益小",
     "融合为少量标量 128 位操作，SIMD 收益有限；保持标量即可"),
    ("dot_split_leveled 分组", "控制层(索引分组)", "❌ 不适用",
     "本质是数据重排/分组，无乘累加并行度；SIMD 无意义"),
]

KS = [256, 1024, 4096, 16384]
GRANULARITIES = [16, 8, 4]


def _make_wx(K: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    w = [int(rng.randint(-2**31, 2**31 - 1)) for _ in range(K)]
    x = [int(rng.randint(-2**31, 2**31 - 1)) for _ in range(K)]
    imp = [int(rng.randint(0, 2000)) for _ in range(K)]
    return w, x, imp


def _median_ms(op, w, x, imp, iters: int, repeats: int) -> float:
    """测量 op 的中位耗时（毫秒）。op 返回任意结果（丢弃）。"""
    _ = op(w, x, imp)  # 预热
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = op(w, x, imp)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0 / iters)
    return float(np.median(times))


def _label_macs(op_name: str, K: int, b) -> str:
    """标注每种操作的渐近乘累加量（用于吞吐口径）"""
    if op_name == "scalar_full":
        return f"{K}"                       # 全精度标量：K 次乘累加
    if op_name.startswith("fused"):
        n = 32 // b
        return f"~{K} (n={n})"              # 融合：每元素精确贡献
    if op_name.startswith("split"):
        n = 32 // b
        return f"{K*n*n} (n={n})"           # 多输出：n² 交叉乘累加
    if op_name == "leveled":
        return f"~{K} (异构)"
    if op_name == "leveled_split":
        return "~ΣK·n_b² (异构多输出)"
    return "?"


def main() -> int:
    print("=" * 78)
    print("MSint 异构粒度拆分点积 — 指令集优化前 baseline + SIMD 适用性")
    print(f"C++ 模式: {USING_CPP}")
    print("=" * 78)

    # ---------- B. SIMD 适用性分析 ----------
    print("\n[B] SIMD 适用性分析（决定后续优化范围）")
    print(f"  {'部分':<26s} {'类型':<26s} {'SIMD':<10s} 说明")
    print("  " + "-" * 110)
    for name, typ, simd, note in APPLICABILITY:
        print(f"  {name:<26s} {typ:<26s} {simd:<10s} {note}")
    print("  硬约束：dot_split 组内点积若上 SIMD，必须用编译时宏保留非 x86(ARM/GPU) 标量回退")
    print("  结论：优化焦点 = dot_split 组内点积（唯一计算层大数组并行点）；决策层/控制层/融合不做")

    # ---------- A. baseline 耗时表 ----------
    print("\n[A] baseline 耗时（未优化标量实现，毫秒中位，K 与粒度扫描）")

    # 先测 K=1024 的粒度对比（每个 ops × 粒度）
    K = 1024
    w, x, imp = _make_wx(K, seed=1)
    iters, repeats = 50, 5

    print(f"\n  K={K}  粒度对比（操作 × 粒度）")
    print(f"  {'操作':<22s} {'16位':>10s} {'8位':>10s} {'4位':>10s} {'参照(scalar)':>14s}")
    print("  " + "-" * 70)
    ops = {
        "fused(同粒度)": lambda W, X, I, b: CppSplitDot.dot_fused(W, X, 32, b),
        "split(1:N 多输出)": lambda W, X, I, b: CppSplitDot.dot_split(W, X, 32, b),
    }
    scalar_ms = _median_ms(lambda W, X, I: CppSplitDot.dot_fused(W, X, 32, 32),
                           w, x, imp, iters, repeats)
    for name, fn in ops.items():
        row = [f"{name:<22s}"]
        for b in GRANULARITIES:
            ms = _median_ms(lambda W, X, I, b=b: fn(W, X, I, b), w, x, imp, iters, repeats)
            row.append(f"{ms:>10.3f}")
        row.append(f"{scalar_ms:>13.3f}")
        print("  " + "".join(row))
    leveled_ms = _median_ms(
        lambda W, X, I: CppLeveledSplitDot.dot_fused_leveled_default(W, X, I),
        w, x, imp, iters, repeats)
    leveled_split_ms = _median_ms(
        lambda W, X, I: CppLeveledSplitDot.dot_split_leveled_default(W, X, I),
        w, x, imp, iters, repeats)
    print(f"  {'leveled(异构)':<22s} {leveled_ms:>10.3f} {'':>10s} {'':>10s} {scalar_ms:>13.3f}")
    print(f"  {'leveled_split(异构多输出)':<22s} {leveled_split_ms:>10.3f}")

    # K 扫描（异构 fused + 参照 scalar）
    print(f"\n  K 扫描（异构 dot_fused_leveled vs 全精度标量参照）")
    print(f"  {'K':>8s} {'标量全精度(ms)':>16s} {'异构 leveled(ms)':>18s} {'异构/标量':>10s}")
    print("  " + "-" * 60)
    for Kk in KS:
        ww, xx, ii = _make_wx(Kk, seed=1)
        it = max(3, 300 // max(1, Kk // 256))
        s_ms = _median_ms(lambda W, X, I: CppSplitDot.dot_fused(W, X, 32, 32),
                          ww, xx, ii, it, 3)
        l_ms = _median_ms(lambda W, X, I: CppLeveledSplitDot.dot_fused_leveled_default(W, X, I),
                          ww, xx, ii, it, 3)
        ratio = l_ms / s_ms if s_ms > 0 else float("inf")
        print(f"  {Kk:>8d} {s_ms:>16.3f} {l_ms:>18.3f} {ratio:>10.2f}x")

    print("\n" + "=" * 78)
    print("baseline 结论（供后续 SIMD 优化归因）")
    print("  - 当前为未优化标量 baseline；后续每条 SIMD 优化以本表数值为对照基准")
    print("  - 优化焦点：dot_split 组内点积（计算层、大数组并行、组内同粒度布局规整）")
    print("  - 不做：select_levels(决策层)/分组(控制层)/128 位融合(收益小)")
    print("  - 硬约束：SIMD 实现必须编译时宏保留非 x86 标量回退")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
