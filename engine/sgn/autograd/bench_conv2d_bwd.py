# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""bench_conv2d_bwd.py - 单层 conv2d fwd+bwd 计时（隔离 conv2d 反向改动效果）

对每个 conv 层单独测 conv2d fwd+bwd（前向代码未变，故前后差值 = 反向改动效果）。
用于 A/B 验证：col2im 零填充 + transpose+连续 GEMM 是否真的加速 conv2d 反向。
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
    print(f"单层 conv2d fwd+bwd  B={B}（前向未变 → 差值即反向改动效果）")
    print("=" * 60)
    for name, in_c, out_c, hw in [("conv1", 3, 32, 32), ("conv2", 32, 64, 16), ("conv3", 64, 128, 8)]:
        X = np.random.randn(B, in_c, hw, hw).astype(np.float32)
        W = np.random.randn(out_c, in_c, 3, 3).astype(np.float32) * 0.1
        b = np.random.randn(out_c).astype(np.float32) * 0.01
        dY = np.random.randn(B, out_c, hw, hw).astype(np.float32)
        X_t = make_tensor(X, False)
        W_t = make_tensor(W, True)
        b_t = make_tensor(b, True)

        def run():
            ag.clear()
            ag.start_recording()
            y = ag.conv2d(X_t, W_t, b_t, 1, 1)
            ag.stop_recording()
            y.backward(dY)

        t = bench(run)
        print(f"  {name} (B={B}, {in_c}->{out_c}, {hw}x{hw}): fwd+bwd = {t*1000:8.3f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
