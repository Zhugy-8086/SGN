# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""benchmark_ste_fusion.py - STE 策略 + conv2d_relu 融合算子速度基准测试

测试内容：
  1. FLOAT32 vs STE 策略：linear / conv2d 前向+反向速度对比
  2. conv2d+relu 分开 vs conv2d_relu 融合：速度对比
  3. conv2d_relu 融合算子正确性验证

运行：cd engine/sgn/build && $env:KMP_DUPLICATE_LIB_OK='TRUE' ; python ../autograd/benchmark_ste_fusion.py
"""

import sys
import os
import time
import numpy as np

# ── OMP 冲突调式日志 ──────────────────────────────────────────
# 安全审计 2026-08-16 A2-6：原注释描述 MSVC/Clang 双 .pyd 共存场景——
# 2026-08-06 起全部 .pyd 已合并为 Clang 编译的单一 sgn 模块，该场景不复
# 存在；KMP_DUPLICATE_LIB_OK 保留为防御性设置（进程内如加载其它 OpenMP
# 运行时的第三方扩展时仍可避免 "Error #15" 崩溃）。
_DEBUG = "SGN_DEBUG" in os.environ
if _DEBUG:
    _omp_before = os.environ.get('KMP_DUPLICATE_LIB_OK', '(未设置)')
    print(f"[DEBUG] benchmark_ste_fusion: KMP_DUPLICATE_LIB_OK 设置前={_omp_before}")
    print(f"[DEBUG] benchmark_ste_fusion: sys.path={sys.path}")
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
# 安全审计 2026-08-16 A2-3：删除指向 traditional/baseline 的死路径插入——
# 该目录已迁入 legacy/traditional/，且 test_phase5.py 与本脚本同目录，
# import 由脚本目录自动解析，此行从未生效。

import sgn
from test_phase5 import make_tensor


def timeit(fn, warmup=3, n_iter=10):
    """运行 fn() n_iter 次，返回 (median_ms, min_ms)"""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000, np.min(times) * 1000


# ============================================================================
# 1. FLOAT32 vs STE 策略：linear
# ============================================================================
def benchmark_linear_strategy():
    print("=" * 70)
    print("1. FLOAT32 vs STE 策略：linear 前向+反向")
    print("=" * 70)

    ag = sgn.autograd

    for feat_in, feat_out, batch in [(256, 128, 64), (512, 256, 128), (2048, 256, 4)]:
        print(f"\n  linear({feat_in}, {feat_out}) x batch={batch}")

        x_np = np.random.randn(batch, feat_in).astype(np.float32)
        w_np = np.random.randn(feat_out, feat_in).astype(np.float32) * 0.1
        b_np = np.random.randn(feat_out).astype(np.float32) * 0.1
        dY_np = np.random.randn(batch, feat_out).astype(np.float32)

        for strategy_name, strategy in [("FLOAT32", ag.BackwardStrategy.FLOAT32),
                                         ("STE", ag.BackwardStrategy.STE)]:
            ag.set_backward_strategy(strategy)
            if strategy == ag.BackwardStrategy.STE:
                ag.set_ste_quant_config(bits=8, clip_sigma=4.0)

            def make_run(_x, _w, _b, _dY):
                def run():
                    ag.clear()
                    x = make_tensor(_x, requires_grad=False)
                    w = make_tensor(_w, requires_grad=True)
                    b = make_tensor(_b, requires_grad=True)
                    ag.start_recording()
                    y = ag.linear(x, w, b)
                    ag.stop_recording()
                    y.backward(_dY)
                    _ = w.grad, b.grad
                return run

            med, mn = timeit(make_run(x_np, w_np, b_np, dY_np), warmup=3, n_iter=10)
            print(f"    {strategy_name:>8s}: median={med:8.3f}ms  min={mn:8.3f}ms")

        ag.set_backward_strategy(ag.BackwardStrategy.FLOAT32)


# ============================================================================
# 2. FLOAT32 vs STE 策略：conv2d
# ============================================================================
def benchmark_conv2d_strategy():
    print("\n" + "=" * 70)
    print("2. FLOAT32 vs STE 策略：conv2d 前向+反向")
    print("=" * 70)

    ag = sgn.autograd

    # (C_in, C_out, H, W, batch, kernel, stride, padding)
    configs = [
        (3, 32, 32, 32, 4, 3, 1, 1),
        (32, 64, 16, 16, 4, 3, 1, 1),
        (64, 128, 8, 8, 4, 3, 1, 1),
    ]

    for C_in, C_out, H, W, B, K, S, P in configs:
        print(f"\n  conv2d({C_in}, {C_out}, {H}x{W}) x batch={B}, k={K}")

        x_np = np.random.randn(B, C_in, H, W).astype(np.float32)
        w_np = np.random.randn(C_out, C_in, K, K).astype(np.float32) * 0.1
        b_np = np.random.randn(C_out).astype(np.float32) * 0.1
        H_out = (H + 2 * P - K) // S + 1
        W_out = (W + 2 * P - K) // S + 1
        dY_np = np.random.randn(B, C_out, H_out, W_out).astype(np.float32)

        for strategy_name, strategy in [("FLOAT32", ag.BackwardStrategy.FLOAT32),
                                         ("STE", ag.BackwardStrategy.STE)]:
            ag.set_backward_strategy(strategy)
            if strategy == ag.BackwardStrategy.STE:
                ag.set_ste_quant_config(bits=8, clip_sigma=4.0)

            def make_run(_x, _w, _b, _dY, _S, _P):
                def run():
                    ag.clear()
                    x = make_tensor(_x, requires_grad=False)
                    w = make_tensor(_w, requires_grad=True)
                    b = make_tensor(_b, requires_grad=True)
                    ag.start_recording()
                    y = ag.conv2d(x, w, b, stride=_S, padding=_P)
                    ag.stop_recording()
                    y.backward(_dY)
                    _ = w.grad, b.grad
                return run

            med, mn = timeit(make_run(x_np, w_np, b_np, dY_np, S, P), warmup=3, n_iter=10)
            print(f"    {strategy_name:>8s}: median={med:8.3f}ms  min={mn:8.3f}ms")

        ag.set_backward_strategy(ag.BackwardStrategy.FLOAT32)


# ============================================================================
# 3. conv2d+relu 分开 vs conv2d_relu 融合
# ============================================================================
def benchmark_conv2d_relu_fusion():
    print("\n" + "=" * 70)
    print("3. conv2d+relu 分开 vs conv2d_relu 融合")
    print("=" * 70)

    ag = sgn.autograd
    ag.set_backward_strategy(ag.BackwardStrategy.FLOAT32)

    configs = [
        (3, 32, 32, 32, 4, 3, 1, 1),
        (32, 64, 16, 16, 4, 3, 1, 1),
        (64, 128, 8, 8, 4, 3, 1, 1),
    ]

    for C_in, C_out, H, W, B, K, S, P in configs:
        print(f"\n  conv2d({C_in}, {C_out}, {H}x{W}) x batch={B}, k={K}")

        x_np = np.random.randn(B, C_in, H, W).astype(np.float32)
        w_np = np.random.randn(C_out, C_in, K, K).astype(np.float32) * 0.1
        b_np = np.random.randn(C_out).astype(np.float32) * 0.1
        H_out = (H + 2 * P - K) // S + 1
        W_out = (W + 2 * P - K) // S + 1
        dY_np = np.random.randn(B, C_out, H_out, W_out).astype(np.float32)

        # --- 分开：conv2d + relu ---
        def make_run_separate(_x, _w, _b, _dY, _S, _P):
            def run():
                ag.clear()
                x = make_tensor(_x, requires_grad=False)
                w = make_tensor(_w, requires_grad=True)
                b = make_tensor(_b, requires_grad=True)
                ag.start_recording()
                y = ag.conv2d(x, w, b, stride=_S, padding=_P)
                y = ag.relu(y)
                ag.stop_recording()
                y.backward(_dY)
                _ = w.grad, b.grad
            return run

        med_sep, mn_sep = timeit(make_run_separate(x_np, w_np, b_np, dY_np, S, P), warmup=3, n_iter=10)
        print(f"    conv2d+relu 分开:  median={med_sep:8.3f}ms  min={mn_sep:8.3f}ms")

        # --- 融合：conv2d_relu ---
        def make_run_fused(_x, _w, _b, _dY, _S, _P):
            def run():
                ag.clear()
                x = make_tensor(_x, requires_grad=False)
                w = make_tensor(_w, requires_grad=True)
                b = make_tensor(_b, requires_grad=True)
                ag.start_recording()
                y = ag.conv2d_relu(x, w, b, stride=_S, padding=_P)
                ag.stop_recording()
                y.backward(_dY)
                _ = w.grad, b.grad
            return run

        med_fus, mn_fus = timeit(make_run_fused(x_np, w_np, b_np, dY_np, S, P), warmup=3, n_iter=10)
        print(f"    conv2d_relu 融合:   median={med_fus:8.3f}ms  min={mn_fus:8.3f}ms")

        speedup = med_sep / med_fus if med_fus > 0 else 0
        print(f"    融合加速比:          {speedup:.2f}x")


# ============================================================================
# 4. conv2d_relu 融合算子正确性验证
# ============================================================================
def verify_conv2d_relu_correctness():
    print("\n" + "=" * 70)
    print("4. conv2d_relu 融合算子正确性验证")
    print("=" * 70)

    ag = sgn.autograd
    ag.set_backward_strategy(ag.BackwardStrategy.FLOAT32)

    np.random.seed(42)
    B, C_in, C_out, H, W, K, S, P = 4, 3, 32, 32, 32, 3, 1, 1

    x_np = np.random.randn(B, C_in, H, W).astype(np.float32)
    w_np = np.random.randn(C_out, C_in, K, K).astype(np.float32) * 0.1
    b_np = np.random.randn(C_out).astype(np.float32) * 0.1
    H_out = (H + 2 * P - K) // S + 1
    W_out = (W + 2 * P - K) // S + 1
    dY_np = np.random.randn(B, C_out, H_out, W_out).astype(np.float32)

    # --- 分开 ---
    ag.clear()
    x1 = make_tensor(x_np.copy(), requires_grad=False)
    w1 = make_tensor(w_np.copy(), requires_grad=True)
    b1 = make_tensor(b_np.copy(), requires_grad=True)
    ag.start_recording()
    y1 = ag.conv2d(x1, w1, b1, stride=S, padding=P)
    y1 = ag.relu(y1)
    ag.stop_recording()
    y1.backward(dY_np.copy())
    y1_np = y1.to_numpy()
    dw1 = w1.grad.copy()
    db1 = b1.grad.copy()

    # --- 融合 ---
    ag.clear()
    x2 = make_tensor(x_np.copy(), requires_grad=False)
    w2 = make_tensor(w_np.copy(), requires_grad=True)
    b2 = make_tensor(b_np.copy(), requires_grad=True)
    ag.start_recording()
    y2 = ag.conv2d_relu(x2, w2, b2, stride=S, padding=P)
    ag.stop_recording()
    y2.backward(dY_np.copy())
    y2_np = y2.to_numpy()
    dw2 = w2.grad.copy()
    db2 = b2.grad.copy()

    # 前向对比
    fwd_diff = float(np.max(np.abs(y1_np - y2_np)))
    fwd_ok = np.allclose(y1_np, y2_np, atol=1e-5)
    print(f"  前向输出 max_diff: {fwd_diff:.2e}  {'OK' if fwd_ok else 'FAIL'}")

    # 梯度对比
    dw_diff = float(np.max(np.abs(dw1 - dw2)))
    dw_ok = np.allclose(dw1, dw2, atol=1e-5)
    print(f"  dW max_diff:        {dw_diff:.2e}  {'OK' if dw_ok else 'FAIL'}")

    db_diff = float(np.max(np.abs(db1 - db2)))
    db_ok = np.allclose(db1, db2, atol=1e-5)
    print(f"  db max_diff:        {db_diff:.2e}  {'OK' if db_ok else 'FAIL'}")

    all_ok = fwd_ok and dw_ok and db_ok
    print(f"  融合算子正确性:     {'PASS' if all_ok else 'FAIL'}")

    return all_ok


# ============================================================================
# main
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("STE 策略 + conv2d_relu 融合算子 速度基准测试")
    print("=" * 70)

    benchmark_linear_strategy()
    benchmark_conv2d_strategy()
    benchmark_conv2d_relu_fusion()
    verify_conv2d_relu_correctness()

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)