# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""bench_conv2d_fwd.py - 单层 conv2d forward 计时（隔离 P1-E im2col 改动效果）

仅测 forward（im2col + matmul + bias）：P1-E 只改前向 im2col 路径，
fwd-only 计时可精确量化 im2col 行条带化的收益，避免 fwd+bwd 中反向噪声干扰。
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


def bench(fn, warm=5, rep=31):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main() -> int:
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    print("=" * 60)
    print(f"单层 conv2d forward  B={B}（im2col+matmul+bias）")
    print("=" * 60)
    for name, in_c, out_c, hw in [("conv1", 3, 32, 32), ("conv2", 32, 64, 16), ("conv3", 64, 128, 8)]:
        X = np.random.randn(B, in_c, hw, hw).astype(np.float32)
        W = np.random.randn(out_c, in_c, 3, 3).astype(np.float32) * 0.1
        b = np.random.randn(out_c).astype(np.float32) * 0.01
        X_t = make_tensor(X, False)
        W_t = make_tensor(W, True)
        b_t = make_tensor(b, True)

        def run():
            ag.clear()
            ag.start_recording()
            ag.conv2d(X_t, W_t, b_t, 1, 1)
            ag.stop_recording()

        t = bench(run)
        print(f"  {name} (B={B}, {in_c}->{out_c}, {hw}x{hw}): fwd = {t*1000:8.3f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
