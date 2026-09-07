# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""hc(B,L) 分层编解码通道验证（R3-B 落地）

验证 engine 落地的可复用解码通道 sgn.hc_decode（hc/hc_decode.cpp）：
  V1  全层 encode→decode == R0 网格量化（quant_hc_layered）——档位守门（kRounding）
  V2  部分解码（L' < L，A 路径）== 截断数字数组全解——带宽/计算双省的正确性
  V3  SNR 表：窄 SIMD 通道解码宽精度的重建质量（与 R2a 原型一致）

对照 Python 参照（validate_hc_rewrite_r2_proto.py 的 hc_encode/hc_decode/quant_hc_layered），
C 端 float32 累加 → 允许数 ULP 级差异（档位 kRounding）。

运行：
    cd engine/sgn/tests/hc_ext/
    py -3.14 test_hc_decode_channel.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent.parent))

import numpy as np  # noqa: E402

import engine.sgn as _sgn  # noqa: E402
hc = _sgn._native.hc_decode


# ============================================================================
# Python 参照（与 validate_hc_rewrite_r2_proto.py 逐位一致）
# ============================================================================
def hc_encode(x, B, L):
    x = np.asarray(x, dtype=np.float64)
    b = 2 ** B
    top = 2 ** (B - 1) - 1
    amax = np.max(np.abs(x))
    if amax == 0:
        return np.zeros(x.shape + (L,), dtype=np.int64), 0.0
    unit = 1.0 / (b ** (L - 1))
    q = np.round(x / amax * top / unit) * unit
    N = np.round(q * (b ** (L - 1))).astype(np.int64)
    digits = np.zeros(x.shape + (L,), dtype=np.int64)
    rem = N
    for i in range(L):
        p = b ** (L - 1 - i)
        digits[..., i] = rem // p
        rem = rem - digits[..., i] * p
    return digits, amax / top


def hc_decode(digits, B):
    b = 2 ** B
    L = digits.shape[-1]
    acc = digits[..., 0].astype(np.float64)
    for i in range(1, L):
        acc = acc + digits[..., i] / (b ** i)
    return acc


def quant_hc_layered(x, B, L):
    x = np.asarray(x, dtype=np.float64)
    b = 2.0 ** B
    top = b / 2.0 - 1.0
    amax = np.max(np.abs(x))
    if amax == 0:
        return np.zeros_like(x)
    unit = 1.0 / (b ** (L - 1))
    q = np.round(x / amax * top / unit) * unit
    q = np.clip(q, -top, top)
    return q / top * amax


def snr_db(x, dq):
    sig = np.sum(x ** 2)
    noise = np.sum((x - dq) ** 2) + 1e-30
    return 10.0 * np.log10(sig / noise)


# ============================================================================
# C 通道封装（layers 为 SoA 展平字节；B=16 每元素 2 字节）
# 注：C 端输入 float32（窄通道存储维度），参照须用同一 x32 对拍
# ============================================================================
def c_encode(x32, B, L):
    layers, scale = hc.hc_encode(np.ascontiguousarray(x32, np.float32), B, L)
    return layers, scale


def c_decode(layers, n, B, Lprime, scale):
    return hc.hc_decode(layers, n, B, Lprime, scale)


def c_full(x32, B, L):
    """C 通道全层重建 float32 输出。"""
    layers, scale = c_encode(x32, B, L)
    return c_decode(layers, x32.size, B, L, scale)


# ============================================================================
# 验证
# ============================================================================
def _distributions(rng):
    n = 20_000
    return {
        "均匀[-1,1]": rng.uniform(-1, 1, n),
        "高斯N(0,0.3)": rng.normal(0, 0.3, n),
        "重尾log-normal": np.exp(rng.normal(0, 2, n)) * rng.choice([-1, 1], n),
        "混合异常值": np.where(rng.random(n) < 0.95, rng.uniform(-1, 1, n),
                              rng.uniform(-50, 50, n)),
    }


def main():
    print("#" * 76)
    print("# hc(B,L) 分层编解码通道验证（R3-B 落地，2026-08-19）")
    print("# 模块: sgn.hc_decode（hc/hc_decode.cpp，C 端 float32 累加 → kRounding）")
    print("#" * 76)

    rng = np.random.default_rng(2026)
    dists = _distributions(rng)
    cases = [(8, 1), (8, 2), (8, 3), (8, 6), (16, 1), (16, 2), (16, 3)]
    failures = []

    # ---- V1：C 通道全层重建 == R0 网格量化（档位守门）----
    # 参照用同一 float32 输入（C 端输入经 forcecast 为 float32），
    # 二者差异仅来自 C 端 float32 累加 vs 参照 float64 累加 → 数 ULP（kRounding）
    print("\nV1  C 通道全层重建 == R0 网格量化（quant_hc_layered，同 float32 输入）")
    print(f"  {'(B,L)':>8s} | " + " | ".join(f"{n:>20s}" for n in dists))
    v1_ok = {}
    for B, L in cases:
        row = []
        for dname, x in dists.items():
            x32 = x.astype(np.float32)
            xh = c_full(x32, B, L)
            ref = quant_hc_layered(x32, B, L)
            max_diff = float(np.max(np.abs(xh.astype(np.float64) - ref)))
            ok = bool(np.allclose(xh, ref, atol=0.0, rtol=2e-6))
            row.append(f"{'OK' if ok else 'FAIL'}({max_diff:.1e})")
            v1_ok[(B, L)] = v1_ok.get((B, L), True) and ok
        print(f"  (B={B},L={L}) | " + " | ".join(f"{s:>20s}" for s in row))
        if not v1_ok[(B, L)]:
            failures.append(f"V1 B={B} L={L}")

    # ---- V2：部分解码（A 路径）== Python 参照截断数字数组全解 ----
    # 对照 R2a 原型 hc_decode_partial：只取前 Lp 层数字再全解。
    # C 端部分解只读 Lp·n 字节（A 路径带宽省）；与参照差 ≤ 数 ULP。
    print("\nV2  部分解码(L') == Python 截断数字数组全解（A 路径，只读 L'·n 字节）")
    print(f"  {'(B,L)→L\'':>10s} | " + " | ".join(f"{n:>20s}" for n in dists))
    v2_ok = True
    for B, L in [(8, 6), (8, 3), (16, 3)]:
        for Lp in range(1, L):
            row = []
            for dname, x in dists.items():
                x32 = x.astype(np.float32)
                layers, scale = c_encode(x32, B, L)
                a = c_decode(layers, x32.size, B, Lp, scale)         # C 端 A 路径
                digits, _ = hc_encode(x32, B, L)                     # 同输入 Python 拆层
                ref_p = hc_decode(digits[..., :Lp], B) * scale       # 截断全解
                ok = bool(np.allclose(a, ref_p, atol=0.0, rtol=2e-6))
                row.append("OK" if ok else "FAIL")
                v2_ok = v2_ok and ok
            print(f"  (B={B},L={L})→L'={Lp} | " + " | ".join(f"{s:>20s}" for s in row))
            if not ok:
                failures.append(f"V2 B={B} L={L} L'={Lp}")

    # ---- V3：SNR 表（重建质量，与 R2a 一致）----
    print("\nV3  SNR 表（dB）：C 通道全层重建")
    print(f"  {'(B,L)':>8s} | " + " | ".join(f"{n:>20s}" for n in dists))
    for B, L in cases:
        row = []
        for dname, x in dists.items():
            x32 = x.astype(np.float32)
            xh = c_full(x32, B, L)
            row.append(f"{snr_db(x32, xh):.2f}")
        print(f"  (B={B},L={L}) | " + " | ".join(f"{s:>20s}" for s in row))

    # ---- 档位守门总结 ----
    print("\n" + "#" * 76)
    if failures:
        print(f"FAIL: {failures}")
        return 1
    print("PASS: V1（档位 kRounding，≤ 数 ULP）✓  V2（A 路径部分解码）✓")
    print("  → hc(B,L) 宽精度解码通道落地完成，可作为窄 SIMD 通道解码宽精度数据路径。")
    print("#" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
