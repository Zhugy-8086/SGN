# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""profile_conv2d_split.py - 定位 conv2d 前向瓶颈：im2col/拷贝 vs matmul

把每个 conv 层的前向拆成两部分量化：
  - 全量 conv2d（im2col + matmul + bias/转置）
  - 仅 matmul（W_col @ x_col，x_col 用 numpy 预构建）
  → im2col+拷贝 ≈ 全量 − matmul（bias/转置是轻量 pass，忽略）

运行：cd engine/sgn/build && python ..\\autograd\\profile_conv2d_split.py
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests", "refs", "baseline"))
import sgn
from test_phase5 import make_tensor

ag = sgn.autograd


def im2col_np(x, kh, kw, stride, pad):
    """(B,C,H,W) -> x_col (C*kh*kw, B*out_h*out_w)，b 主序 + (oh,ow) 序"""
    B, C, H, W = x.shape
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    cols = np.empty((C * kh * kw, B * out_h * out_w), dtype=np.float32)
    r = 0
    for ic in range(C):
        for ki in range(kh):
            for kj in range(kw):
                patch = xp[:, ic, ki:ki + out_h * stride:stride, kj:kj + out_w * stride:stride]
                cols[r] = patch.reshape(-1)
                r += 1
    return cols


def bench(fn, warm=5, rep=51):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _T(x):
    """numpy → sgn.Tensor（不要求 grad）"""
    return ag.Tensor.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def main() -> int:
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    print("=" * 70)
    print(f"conv2d 前向瓶颈定位：im2col/拷贝 vs matmul  B={B}")
    print("口径：丢弃 warmup 取中位；ag.conv2d 不录制 = 纯前向内核")
    print("=" * 70)

    layers = [("conv1", 3, 32, 32), ("conv2", 32, 64, 16), ("conv3", 64, 128, 8)]
    for name, in_c, out_c, hw in layers:
        X = np.random.randn(B, in_c, hw, hw).astype(np.float32)
        W = np.random.randn(out_c, in_c, 3, 3).astype(np.float32) * 0.1
        bb = np.random.randn(out_c).astype(np.float32) * 0.01

        X_t, W_t, bb_t = _T(X), _T(W), _T(bb)
        ag.clear()
        full = bench(lambda: ag.conv2d(X_t, W_t, bb_t, 1, 1))

        # matmul 部分：W_col (out_c, in_c*9) @ x_col (in_c*9, B*out_h*out_w)
        out_h = out_w = hw  # stride=1, pad=1, k=3
        x_col = im2col_np(X, 3, 3, 1, 1)
        w_col = W.reshape(out_c, -1)
        mm = bench(lambda: ag.matmul_forward(_T(w_col), _T(x_col)))

        im2col_share = full - mm
        print(f"\n{name} (B={B}, {in_c}->{out_c}, {hw}x{hw}):")
        print(f"  conv2d 全量 : {full*1000:8.3f} ms")
        print(f"  matmul      : {mm*1000:8.3f} ms  ({mm/full*100:5.1f}%)")
        print(f"  im2col+拷贝 : {im2col_share*1000:8.3f} ms  ({im2col_share/full*100:5.1f}%)")
        print(f"  x_col 规模  : {w_col.shape[1]} x {x_col.shape[1]} = {w_col.shape[1]*x_col.shape[1]/1e6:.2f}M 元素")

    return 0


if __name__ == "__main__":
    sys.exit(main())
