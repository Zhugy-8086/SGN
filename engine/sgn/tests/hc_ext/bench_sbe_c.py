# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""
SBE C 标量路径 vs Python numpy 路径性能基准

运行：
    cd engine/sgn/tests/hc_ext/
    py -3.14 bench_sbe_c.py

注意（方法论修正 v1.5.2+）：
  sbe_conv2d.sbe_matmul / quantize_weight_sbe 内部检查 _HAS_SBE_C 标志，
  当 True 时直接调用 C 扩展。若直接 import sbe_matmul 当 "Python 路径"，
  实际两条路径都走 C，测出 ~1.00x 假象。

  修正：在调用 Python 路径前 monkey-patch sbe_conv2d._HAS_SBE_C = False，
  强制走 _compute_scale_np / _quantize_to_bytes_np / _c_matmul_int_no_requant
  （纯 numpy int32 matmul）的真实 Python 基线。
"""
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
    _omp_conflict_fix = os.environ.get('KMP_DUPLICATE_LIB_OK', '(未设置)')
    print(f"[DEBUG] bench_sbe_c: KMP_DUPLICATE_LIB_OK={_omp_conflict_fix}")
    print(f"[DEBUG] bench_sbe_c: sys.path={sys.path}")
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np

HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = HERE.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import engine.sgn as _sgn
pysgn_net = _sgn._native.hc8_net

# ── 加载后调式日志 ──
if _DEBUG:
    _loaded_omp = os.environ.get('KMP_DUPLICATE_LIB_OK', '(未设置)')
    _pyd_path = _sgn.__file__ if hasattr(_sgn, '__file__') else '(未知)'
    print(f"[DEBUG] bench_sbe_c: 加载后 KMP_DUPLICATE_LIB_OK={_loaded_omp}")
    print(f"[DEBUG] bench_sbe_c: sgn.pyd 路径={_pyd_path}")
    print(f"[DEBUG] bench_sbe_c: pysgn_net 版本={getattr(pysgn_net, '__version__', 'unknown')}")

SBE_DIR = HERE.parent / "refs"
sys.path.insert(0, str(SBE_DIR))
import sbe_conv2d  # 以模块方式导入，便于 monkey-patch _HAS_SBE_C


def _sbe_matmul_py(x_np, w_blocks_py, groups, k_block, m, k, n):
    """强制走 Python/numpy 路径（临时关闭 _HAS_SBE_C）"""
    saved = sbe_conv2d._HAS_SBE_C
    sbe_conv2d._HAS_SBE_C = False
    try:
        return sbe_conv2d.sbe_matmul(x_np, w_blocks_py, groups, k_block, m, k, n)
    finally:
        sbe_conv2d._HAS_SBE_C = saved


def _quantize_weight_sbe_py(w_np, groups, k_block, m, k, n):
    """强制走 Python/numpy 路径生成 list of (bytes, float) 格式权重"""
    saved = sbe_conv2d._HAS_SBE_C
    sbe_conv2d._HAS_SBE_C = False
    try:
        return sbe_conv2d.quantize_weight_sbe(w_np, groups, k_block, m, k, n)
    finally:
        sbe_conv2d._HAS_SBE_C = saved


def bench_one(name, m, k, n, groups, k_block, seed=42, N=20):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((m, k)).astype(np.float32) * 0.5
    w = rng.standard_normal((k, n)).astype(np.float32) * 0.3

    # 预计算量化权重（不计时）
    # C 路径：返回 (w_signed, w_sum_b, w_scales) 三元组
    w_signed, w_sum_b, w_scales = pysgn_net.sbe_quantize_weight_blocks(
        w, groups, k_block, k, n
    )
    # Python 路径：返回 list of (bytes, float)（必须 _HAS_SBE_C=False 时生成）
    w_blocks_py = _quantize_weight_sbe_py(w, groups, k_block, m, k, n)

    # warmup
    pysgn_net.sbe_matmul(x, w_signed, w_sum_b, w_scales, groups, k_block, m, k, n)
    _sbe_matmul_py(x, w_blocks_py, groups, k_block, m, k, n)

    # C 路径
    t0 = time.perf_counter()
    for _ in range(N):
        pysgn_net.sbe_matmul(x, w_signed, w_sum_b, w_scales, groups, k_block, m, k, n)
    t_c = (time.perf_counter() - t0) / N

    # Python 路径（强制 numpy）
    t0 = time.perf_counter()
    for _ in range(N):
        _sbe_matmul_py(x, w_blocks_py, groups, k_block, m, k, n)
    t_py = (time.perf_counter() - t0) / N

    speedup = t_py / t_c if t_c > 0 else float("inf")
    shape_str = f"({m},{k})x({k},{n})"
    print(f"{name:<14} {shape_str:<22} {t_c*1000:>8.2f} {t_py*1000:>8.2f} {speedup:>7.2f}x")
    return t_c, t_py, speedup


def main():
    print("=" * 66)
    print("SBE 性能基准: C 路径 vs Python numpy 路径")
    print(f"pysgn_net 版本: {pysgn_net.__version__}")
    vnni = pysgn_net.has_avx_vnni()
    if vnni:
        path_c = "AVX-VNNI (_mm256_dpbusd_epi32)"
    else:
        path_c = "AVX2 nibble split (_mm256_maddubs_epi16)"
    print(f"AVX-VNNI: {'支持' if vnni else '不支持'}")
    print(f"C 路径: {path_c}")
    eax, ebx, ecx, edx = pysgn_net.cpuid_7_1_raw()
    print(f"CPUID 7.1: EAX=0x{eax:08X} EBX=0x{ebx:08X} ECX=0x{ecx:08X} EDX=0x{edx:08X}")
    print("=" * 66)
    print()

    header = f"{'case':<14} {'shape':<22} {'C(ms)':>8} {'Py(ms)':>8} {'speedup':>8}"
    print(header)
    print("-" * 66)

    cases = [
        ("CIFAR conv",   256,  27,  32,  3,   9),
        ("MNIST FC",      64, 784, 128, 28,  28),
        ("CIFAR FC1",    256, 768, 256, 12,  64),
        ("CIFAR FC2",    256, 256,  10,  4,  64),
        ("Large FC",     128,1024, 512, 16,  64),
    ]

    results = []
    for name, m, k, n, g, kb in cases:
        t_c, t_py, sp = bench_one(name, m, k, n, g, kb)
        results.append((name, t_c, t_py, sp))

    print("-" * 66)
    print()
    avg_speedup = sum(r[3] for r in results) / len(results)
    print(f"平均加速比: {avg_speedup:.2f}x")
    print()
    if vnni:
        print("注: 当前 CPU 支持 AVX-VNNI，C 走 _mm256_dpbusd_epi32 路径。")
    else:
        print("注: 当前 CPU 不支持 AVX-VNNI（CPUID errata，SEH 探针确认）。")
        print("    C 走 AVX2 nibble split 路径（_mm256_maddubs_epi16 + _mm256_madd_epi16）。")
        print("    如 CPU 真正支持 VNNI，C 路径预期再加速 2-3x。")


if __name__ == "__main__":
    main()
