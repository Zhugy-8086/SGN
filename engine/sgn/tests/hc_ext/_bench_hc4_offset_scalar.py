#!/usr/bin/env python
"""A1-4 基准：hc4_residual_matmul_b 标量版端到端耗时（偏移修正 O(mkn)→O(mk+kn)）。"""
import sys
import time
import random

sys.path.insert(0, r"C:\code\mimo-test\SGN")
import engine.sgn as sgn  # noqa: E402

pn = sgn._native.hc8_net
schema = pn.default_schema()


def bench(m, k, n, depth, rep=3):
    random.seed(1)
    af = [random.uniform(-1, 1) for _ in range(m * k)]
    bf = [random.uniform(-1, 1) for _ in range(k * n)]
    a_bytes, a_scales = pn.quantize_residual(af, depth, schema)
    b_bytes, b_scales = pn.quantize_residual(bf, depth, schema)
    args = (a_bytes, b_bytes, m, k, n, depth, depth, a_scales, b_scales, schema)
    pn.matmul_residual_hc4_b(*args)          # 预热
    t0 = time.perf_counter()
    for _ in range(rep):
        pn.matmul_residual_hc4_b(*args)
    return (time.perf_counter() - t0) / rep * 1000


print("m    k    n    depth   ms/iter")
for (m, k, n) in ((64, 64, 64), (128, 256, 128), (256, 512, 256)):
    for depth in (0, 2):
        t = bench(m, k, n, depth)
        print(f"{m:<4} {k:<4} {n:<4} {depth:<7} {t:8.2f}")
