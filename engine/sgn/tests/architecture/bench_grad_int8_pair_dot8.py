# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""int8 对（h,l）梯度载体 — 引擎侧 dot8 消费路径吞吐基准

对应 msint_int8_pair_grad_carrier_2026_08_31.md §四.3 剩余项：
  数学层已给出成本模型（fine 2 次/coarse 1 次 dot8，V1-V7 7/7 PASS），
  本脚本在真实 .pyd 上实测消费端吞吐。

口径与对照（每次调用 = 1 个 K 元素行的点积消费）：
  - pair-8bit : sgn.SplitDot.dot_split(total_bits=16, split_bits=8)
                现有引擎唯一 8 位消费入口，n²=4 次 simd::dot8/调用
                （w 侧 2 肢 × x 侧 2 肢；对 int8 载体×int8 权重场景，
                 x 侧高位肢是符号扩展（0/-1），2 次为纯浪费 → 理论 2 次）
  - int16     : dot_split(total_bits=16, split_bits=16) = 1 次 simd::dot16/调用
                （int16 载体消费对照，同为 2B/元素）
  - float32   : numpy np.dot（Q16-SR 现状写回消费基线，4B/元素）
  - scalar    : SGN_KERNEL_BACKEND=scalar 强制标量子进程重跑（回退回归 + SIMD 加速比）

派生口径（对照 §七 成本模型）：
  pair-fine  理论 ≈ dot_split-8bit 的 1/2（2/4 次 dot8）
  pair-coarse 理论 ≈ dot_split-8bit 的 1/4（1/4 次 dot8）

正确性（routine）：dot_fused(16,8) 与 numpy int64 点积 bit-exact 全档校验。

运行:
  python engine/sgn/tests/architecture/bench_grad_int8_pair_dot8.py
  python engine/sgn/tests/architecture/bench_grad_int8_pair_dot8.py --scalar
      （内部以 SGN_KERNEL_BACKEND=scalar 重启子进程）
"""

import subprocess
import sys
import os
import time
from pathlib import Path

import numpy as np

_PROJ_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

import engine.sgn as sgn

KS = [1024, 4096, 16384, 65536]
K_OVERHEAD = 16          # 标定绑定/转换固定开销的小 K
REPEATS = 5


def make_inputs(K, seed=0):
    rng = np.random.RandomState(seed)
    w = rng.randint(-32768, 32767, size=K).tolist()   # int16 梯度（pair 全解码值域）
    x = rng.randint(-128, 127, size=K).tolist()       # int8 权重
    return w, x


def median_ms(op, iters, repeats=REPEATS):
    op()  # 预热
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            op()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0 / iters)
    return float(np.median(times))


def correctness():
    """routine 校验：8 位消费路径 bit-exact（对标量 int64 参考）。"""
    for K in (1, 255, 1024, 65537):
        w, x = make_inputs(K, seed=K)
        ref = sum(a * b for a, b in zip(w, x))
        got = sgn.SplitDot.dot_fused(w, x, 16, 8)
        assert got == ref, f"dot_fused(16,8) mismatch at K={K}: {got} != {ref}"
    print("正确性: dot_fused(16,8) == int64 标量参考，K∈{1,255,1024,65537} bit-exact ✓")


def iters_for(K):
    return {16: 2000, 1024: 400, 4096: 200, 16384: 60, 65536: 15}[K]


def main():
    scalar_mode = "--scalar" in sys.argv
    tag = "强制标量" if scalar_mode else "运行时调度"

    print("=" * 78)
    print(f"int8 对 dot8 消费路径吞吐基准（{tag}）")
    print(f"环境: Python {sys.version.split()[0]}, numpy {np.__version__}")
    print(f"口径: 每调用=1 行 K 元素点积消费；median of {REPEATS}×iters")
    print("=" * 78)

    correctness()
    print()

    # ---------- 标定绑定/转换固定开销 ----------
    w0, x0 = make_inputs(K_OVERHEAD)
    t_over = median_ms(lambda: sgn.SplitDot.dot_split(w0, x0, 16, 8),
                       iters_for(K_OVERHEAD))
    print(f"固定开销标定 (K={K_OVERHEAD}): {t_over*1000:.1f} µs/调用"
          f"（Python list→C++ 转换 + 拆分打包常数项）\n")

    header = (f"{'K':>6} | {'8bit 4×dot8':>12} | {'int16 1×dot16':>13} | "
              f"{'f32 np.dot':>12} | {'8bit Mops/s':>11} | {'f32/8bit':>8}")
    print(header)
    print("-" * len(header))

    results = {}
    for K in KS:
        w, x = make_inputs(K)
        it = iters_for(K)

        t8 = median_ms(lambda: sgn.SplitDot.dot_split(w, x, 16, 8), it)
        t16 = median_ms(lambda: sgn.SplitDot.dot_split(w, x, 16, 16), it)

        wf = np.array(w, dtype=np.float32)
        xf = np.array(x, dtype=np.float32)
        tf = median_ms(lambda: np.dot(wf, xf), max(it, 100))

        # 8bit 路径 MAC 吞吐（1 次 dot_split = K MACs）
        mops8 = K / (t8 * 1e-3) / 1e6
        results[K] = (t8, t16, tf, mops8)

        print(f"{K:>6} | {t8:>10.3f} ms | {t16:>11.3f} ms | "
              f"{tf:>10.3f} ms | {mops8:>10.1f} | {tf/t8:>7.2f}x")

    # ---------- 派生口径：理论 pair 消费成本 ----------
    print("\n--- 派生：理论 pair 消费成本（对照 §七 成本模型）---")
    print(f"{'K':>6} | {'现有入口(4 dot8)':>15} | {'pair-fine(2 dot8)':>17} | "
          f"{'pair-coarse(1 dot8)':>19} | {'f32(4B) vs coarse(1B)':>21}")
    for K in KS:
        t8, t16, tf, _ = results[K]
        # 扣固定开销后按 dot8 次数线性折算（拆分打包随次数同比例减少）
        t_adj = t8 - t_over
        t_fine = t_over + t_adj * 0.5
        t_coarse = t_over + t_adj * 0.25
        print(f"{K:>6} | {t8:>12.3f} ms | {t_fine:>15.3f} ms | "
              f"{t_coarse:>17.3f} ms | {tf/t_coarse:>19.2f}x")

    # ---------- int16 对照 ----------
    print("\n--- 对照：int16 载体（1×dot16，2B/元素）vs 8bit 现有入口 ---")
    for K in KS:
        t8, t16, _, _ = results[K]
        print(f"K={K:>6}: dot_split(16,16)/dot_split(16,8) = {t8/t16:.2f}x"
              f"（理论原语比 ≈ 2×，偏差来自拆分/修正开销）")

    print("\n结论: " + ("强制标量回退正常" if scalar_mode
          else "运行时调度完成；SIMD/标量加速比见两轮输出对比"))
    return 0


if __name__ == "__main__":
    if "--scalar" not in sys.argv:
        env = dict(os.environ, SGN_KERNEL_BACKEND="scalar")
        script = str(Path(__file__).resolve())
        r = subprocess.run([sys.executable, script, "--scalar"], env=env)
        print()
        if r.returncode != 0:
            sys.exit(r.returncode)
    sys.exit(main())
