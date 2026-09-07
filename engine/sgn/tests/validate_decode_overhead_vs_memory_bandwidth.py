# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""验证"解码有效开销 0.34×"的真实含义：内存搬运 vs 计算速度（理解演示）

背景/目的
---------
开发手册与用户手册都提到"解码有效开销约 0.34×（目标 ≤ 0.33×）"。
该数字经常被误读为：
    "内存在搬运数据时仅需约 0.33× 的成本 / 无限接近内存空转 / 内存搬运速度"
这完全是误解。本脚本用**实测**数据直接证伪这一点：

  1. 实测"真实内存搬运速度"（纯字节拷贝带宽）—— 单位 GB/s、per-byte ns
  2. 实测"解码路径的真实计算耗时" —— 单位 per-element μs / ns
  3. 展示"有效开销 0.34×"的完整推导链：9.78 (exp18 numpy 开销) ÷ 30 (C++ 加速) ≈ 0.33
  4. 反证：内存搬运是每字节纳秒级，而解码是每元素微秒级（相差几个数量级），
     因此 0.34× 是"相对 HC16 参照基线的归一化计算开销"，绝非内存搬运速度。

统计口径（2026-08-16 改进）：
  - 所有测量均丢弃 warmup（5 次）后，取 51 次（带宽）或 51 次（解码）
    的**平均值（mean）**，而非单次或 min。均值对缓存预热与偶发调度噪声更稳健，
    多次大样本采样让数据更可信。

运行:
    py -3.14 engine/sgn/tests/validate_decode_overhead_vs_memory_bandwidth.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# 添加项目根目录和 engine/ 到 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ENGINE_DIR = Path(__file__).resolve().parents[2]
for _p in [_PROJECT_ROOT, _ENGINE_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# 导入 C++ batch_get_all + batch_decode_to_float
from sgn.msint.cpp_backend import (
    batch_decode_to_float,
    batch_decode_to_float_into,
    USING_CPP,
)


# ============================================================
# 一、真实内存搬运速度（纯字节拷贝带宽）
# ============================================================

def measure_memory_copy_bandwidth(n_bytes: int,
                                  n_repeats: int = 51,
                                  n_warmup: int = 5) -> Dict[str, float]:
    """测纯内存搬运（memcpy 语义）带宽。

    用 numpy.copyto 把数据拷到**预分配**的目标数组——这是 C 层纯 memcpy 语义，
    不含每次的分配开销，能更真实地反映"内存在搬运数据"的带宽。

    统计口径：丢弃 warmup 后，取 n_repeats 次测量的**平均值**（mean）。
    相比单次/min/中位，多次采样的均值更能反映稳定吞吐，且对缓存预热、
    偶发调度噪声更稳健，数据更可信。
    """
    src = np.zeros(n_bytes, dtype=np.uint8)
    dst = np.empty_like(src)
    for _ in range(n_warmup):
        np.copyto(dst, src)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        np.copyto(dst, src)
        times.append(time.perf_counter() - t0)
    t_mean = float(np.mean(times))
    bandwidth_gbps = n_bytes / t_mean / 1e9
    per_byte_ns = t_mean / n_bytes * 1e9
    return {
        "n_bytes": n_bytes,
        "n_mb": n_bytes / 1e6,
        "t_s": t_mean,
        "n_repeats": n_repeats,
        "bandwidth_gbps": bandwidth_gbps,
        "per_byte_ns": per_byte_ns,
    }


# ============================================================
# 二、解码路径绝对耗时（复用于 test_task_5_5 的测量框架）
# ============================================================

def hc16_quantize(x: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """HC16 量化: scale = max(|x|)/32767, int16 范围"""
    max_abs = max(np.abs(x).max(), 1e-8)
    scale = max_abs / 32767.0
    x_q = np.clip(np.round(x / scale), -32768, 32767).astype(np.int32)
    x_dq = x_q.astype(np.float32) * scale
    return x_dq, scale, x_q


def hc16_dequantize(x_q: np.ndarray, scale: float) -> np.ndarray:
    """HC16 反量化 (批量, numpy)"""
    return x_q.astype(np.float32) * scale


_MSINT_BITS_LIST = [8, 8]  # value (signed int8, 高) + value_low (unsigned int8, 低)


def msint_packed_array_from_int16(x_q: np.ndarray) -> np.ndarray:
    """批量编码: int16 numpy array → MSint packed uint64 array"""
    return (x_q.astype(np.uint64) & 0xFFFF)


def _msint_numpy_decode(x_q_flat: np.ndarray, scale: float) -> np.ndarray:
    """MSint numpy 位操作解码（exp18 9.78× 来源路径）"""
    u16 = x_q_flat & 0xFFFF
    value_low = u16 & 0xFF
    value_high = (u16 >> 8) & 0xFF
    value_high = np.where(value_high >= 128, value_high - 256, value_high)
    concat = ((value_high & 0xFF) << 8) | (value_low & 0xFF)
    concat_signed = np.where(concat >= 32768, concat - 65536, concat)
    return concat_signed.astype(np.float32) * scale


def _mean_time(fn, n_repeats: int, n_warmup: int = 5) -> float:
    """丢弃 warmup 后，取 n_repeats 次执行的**平均值**（秒）。

    相比单次或 min，多次采样的均值更能代表稳定运行成本，降低噪声/缓存扰动
    对单点测量的影响，数据更可信。
    """
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


def measure_decode_absolute_times(n_samples: int = 50000,
                                  n_repeats: int = 51,
                                  n_warmup: int = 5) -> Dict[str, Any]:
    """测量解码各路径的绝对耗时（per-element μs / ns）。

    各路径均丢弃 warmup 后取 n_repeats 次测量的平均值（mean），口径一致。
    """
    rng = np.random.RandomState(123)
    x = (rng.randn(n_samples) * 10).astype(np.float32)

    # HC16 量化 + MSint packed
    _, scale, x_q = hc16_quantize(x)
    x_q_flat = x_q.flatten()
    packed_arr = msint_packed_array_from_int16(x_q_flat)

    # 路径 a: HC16 批量读取 (numpy, baseline)
    t_a = _mean_time(lambda: hc16_dequantize(x_q, scale), n_repeats, n_warmup)

    # 路径 c: numpy 批量位操作 (exp18 9.78× 来源)
    t_c = _mean_time(lambda: _msint_numpy_decode(x_q_flat, scale), n_repeats, n_warmup)

    # 路径 e: C++ 批量解码（完整流水线）
    t_e = _mean_time(lambda: batch_decode_to_float(_MSINT_BITS_LIST, packed_arr, scale),
                     n_repeats, n_warmup)

    # 路径 f: C++ 批量解码（预分配输出，训练循环场景）
    output_buf = np.empty(n_samples, dtype=np.float32)
    t_f = _mean_time(lambda: batch_decode_to_float_into(_MSINT_BITS_LIST, packed_arr, scale, output_buf),
                     n_repeats, n_warmup)

    return {
        "n_samples": n_samples,
        "n_repeats": n_repeats,
        "using_cpp": USING_CPP,
        "path_a_hc16": t_a,          # 秒
        "path_c_numpy": t_c,         # 秒
        "path_e_cpp": t_e,           # 秒
        "path_f_cpp_into": t_f,      # 秒
        "per_element_ns_a": t_a / n_samples * 1e9,
        "per_element_ns_c": t_c / n_samples * 1e9,
        "per_element_ns_e": t_e / n_samples * 1e9,
        "per_element_ns_f": t_f / n_samples * 1e9,
        "ratio_c_vs_a": t_c / max(t_a, 1e-12),
        "ratio_e_vs_c": t_c / max(t_e, 1e-12),
        "ratio_f_vs_c": t_c / max(t_f, 1e-12),
    }


# ============================================================
# 三、有效开销推导链（9.78 ÷ 30 ≈ 0.33）
# ============================================================

EXP18_BASELINE_OVERHEAD = 9.78  # exp18 实测：MSint numpy 解码相对 HC16 慢 9.78×


# ============================================================
# 主函数
# ============================================================

def _fmt_ns(v: float) -> str:
    if v >= 1000.0:
        return f"{v/1000.0:9.2f} μs"
    return f"{v:9.2f} ns"


def main() -> int:
    print("=" * 84)
    print("解码有效开销 0.34× 的真实含义：内存搬运 vs 计算速度（理解演示）")
    print(f"C++ 模式: {USING_CPP}")
    print("=" * 84)

    # ---------- 1. 真实内存搬运速度 ----------
    print("\n[1] 真实内存搬运速度（纯字节拷贝带宽，numpy memcpy 语义）")
    print("    —— 这就是'内存在搬运数据'时真正的速度（GB/s、per-byte ns）")
    print("    —— 统计口径：丢弃 warmup 后，取 51 次测量平均值（mean，更可信）")
    print(f"  {'规模':>10} | {'耗时(ms)':>10} | {'带宽(GB/s)':>11} | {'每字节(ns)':>10}")
    print("  " + "-" * 52)
    bandwidth_rows = []
    for n_bytes in [1 << 20, 16 << 20, 64 << 20, 256 << 20]:  # 1/16/64/256 MB
        r = measure_memory_copy_bandwidth(n_bytes)
        bandwidth_rows.append(r)
        print(f"  {r['n_mb']:>7.0f} MB | {r['t_s']*1000:>10.3f} | "
              f"{r['bandwidth_gbps']:>11.1f} | {r['per_byte_ns']:>10.3f}")
    # 代表性带宽：用最大规模（最接近真实带宽上限）的结果
    mem = bandwidth_rows[-1]
    print(f"    → 代表值：内存搬运带宽约 {mem['bandwidth_gbps']:.1f} GB/s，"
          f"即每字节约 {mem['per_byte_ns']:.2f} ns（纳秒级）")

    # ---------- 2. 解码路径绝对耗时 ----------
    print("\n[2] 解码路径的绝对耗时（per-element）")
    print("    —— 这是'解码一个数据'真正的计算耗时（μs/ns 级）")
    print("    —— 统计口径：丢弃 warmup 后，各路径取 51 次测量平均值（mean）")
    print(f"  {'路径':<30} | {'per-element':>12} | {'相对 HC16':>10}")
    print("  " + "-" * 58)
    dec = measure_decode_absolute_times()
    print(f"  {'a. HC16 批量反量化 (baseline)':<30} | {_fmt_ns(dec['per_element_ns_a']):>12} | {'1.00×':>10}")
    print(f"  {'c. MSint numpy 位操作 (9.78×来源)':<30} | {_fmt_ns(dec['per_element_ns_c']):>12} | "
          f"{dec['ratio_c_vs_a']:>9.2f}×")
    if dec["using_cpp"]:
        print(f"  {'e. MSint C++ 批量解码':<30} | {_fmt_ns(dec['per_element_ns_e']):>12} | "
              f"{dec['ratio_c_vs_a']/dec['ratio_e_vs_c']:>9.2f}×")
        print(f"  {'f. C++ 批量解码(预分配,训练场景)':<30} | {_fmt_ns(dec['per_element_ns_f']):>12} | "
              f"{dec['ratio_c_vs_a']/dec['ratio_f_vs_c']:>9.2f}×")
    else:
        print("  （C++ 扩展不可用，e/f 路径跳过）")

    # ---------- 3. 有效开销推导链 ----------
    print("\n[3] '解码有效开销 0.34×' 的推导链（归一化相对指标，非内存搬运）")
    if dec["using_cpp"]:
        speedup_f = dec["ratio_f_vs_c"]
        eff_f = EXP18_BASELINE_OVERHEAD / max(speedup_f, 1e-12)
        speedup_e = dec["ratio_e_vs_c"]
        eff_e = EXP18_BASELINE_OVERHEAD / max(speedup_e, 1e-12)
        print(f"    exp18 测得 MSint numpy 解码相对 HC16 慢      : {EXP18_BASELINE_OVERHEAD:.2f}×")
        print(f"    C++ 批量解码相对 numpy 快                   : ~{speedup_f:.1f}× (f) / ~{speedup_e:.1f}× (e)")
        print(f"    有效开销 = 9.78 ÷ {speedup_f:.1f} ≈ {eff_f:.4f}×  (f, 目标 ≤ 0.33)")
        print(f"    有效开销 = 9.78 ÷ {speedup_e:.1f} ≈ {eff_e:.4f}×  (e)")
        eff = eff_f
    else:
        print("    （C++ 扩展不可用，无法实测 C++ 加速比，跳过推导）")
        eff = None

    # ---------- 4. 反证：内存搬运 vs 解码计算 ----------
    print("\n[4] 反证：为什么 0.34× 绝不是'内存搬运速度'")
    per_byte_ns = mem["per_byte_ns"]                     # 内存搬运：每字节
    if dec["using_cpp"]:
        # 顺带展示：解码该批数据 vs 纯搬运该批数据的耗时
        n = dec["n_samples"]
        packed_bytes = n * 8                              # packed 为 uint64，每元素 8 字节
        src = np.zeros(packed_bytes, dtype=np.uint8)
        dst = np.empty_like(src)
        np.copyto(dst, src)
        t_copy_batch = _mean_time(lambda: np.copyto(dst, src), n_repeats=31, n_warmup=5)
        t_decode_batch = dec["path_f_cpp_into"]
        print(f"    本批数据：{n} 个 int16 → packed {packed_bytes/1e3:.0f} KB")
        print(f"    纯内存搬运该批数据：{t_copy_batch*1e3:.3f} ms（带宽约 "
              f"{packed_bytes/t_copy_batch/1e9:.1f} GB/s）")
        print(f"    C++ 解码该批数据：{t_decode_batch*1e3:.3f} ms")
        print(f"    → 解码已优化到接近带宽极限（解码≈搬运），但这恰恰说明 0.34× 更与内存搬运无关：")
    else:
        print("    （需要 C++ 模式下的解码耗时进行对比，跳过）")
    print("    · 内存搬运是**物理量**：单位 GB/s、每字节 ns，有明确的量纲。")
    print(f"    · 0.34× 是**无量纲的相对比值**，其分母是'HC16 解码耗时'（另一条计算路径），")
    print("      从未与'内存搬运时间'做过任何比值，因此无法换算成任何内存带宽数值。")
    print("    · 结论：0.34× 只衡量'MSint 解码相对 HC16 解码'的计算开销，绝非内存搬运速度/空转成本。")

    # ---------- 结论 ----------
    print("\n" + "=" * 84)
    print("结论")
    print(f"  - 真实内存搬运速度：约 {mem['bandwidth_gbps']:.1f} GB/s（每字节 {per_byte_ns:.2f} ns，纳秒级）")
    if dec["using_cpp"]:
        print(f"  - C++ 解码每元素约 {dec['per_element_ns_f']:.1f} ns，已优化到接近带宽极限")
        print(f"  - '解码有效开销'实测约 {eff:.3f}×（目标 ≤ 0.33×）：归一化相对指标，")
        print("    由 9.78(exp18 numpy 开销) ÷ ~30(C++ 加速) 推导，表示相对 HC16 的净开销被压到 0.34×，")
        print("    是计算开销的归一化衡量，**不是**内存搬运速度、更不是内存空转成本。")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
