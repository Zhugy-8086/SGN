# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""benchmark_phase5.py - 6 层 CNN 训练速度对比（Phase 5 Task A5.3 / Task 6.3）

Stage 3.2 Phase 5 Task A5.3 / 当前阶段 Task 6.3：对比 C++ Autograd 和 PyTorch 的前向+反向速度。

注意：C++ Autograd 已启用 AVX2 FMA（float32 matmul）和 OpenMP 并行化
      （2026-08-16 起含 batchnorm2d 零拷贝 + 8 个 GEMM 内核 OpenMP 并行化），
      PyTorch 使用 MKL 高度优化的 BLAS 和 conv 算法。
      C++ 比 PyTorch 慢 3.3-4.8x（fwd+bwd，B=4/8/16），主要瓶颈：
      - backward 占总时间 ~65%（B=16）
      - conv2d 反向 im2col/col2im 内存 gather/scatter
      - gather 型 transpose matmul（_mm256_i32gather_ps）

运行：cd engine/sgn/build && python ../autograd/benchmark_phase5.py
"""

import sys
import os
import time
import ctypes
import numpy as np
import torch
import torch.nn as nn


# ---- 峰值内存测量（Windows API，零依赖） ----
_HAS_PEAK_MEM = True
try:
    _kernel32 = ctypes.windll.kernel32
    _psapi = ctypes.windll.psapi

    class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    def _get_rss_mb():
        hproc = _kernel32.GetCurrentProcess()
        counters = _PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        if _psapi.GetProcessMemoryInfo(ctypes.c_void_p(hproc), ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / (1024.0 * 1024.0)
        return 0.0
except Exception:
    _HAS_PEAK_MEM = False
    def _get_rss_mb():
        return 0.0


def measure_peak_mem(fn, *args, **kwargs):
    """运行 fn(*args, **kwargs) 并返回 (结果, 峰值内存增量 MB)"""
    base = _get_rss_mb()
    peak = base
    result = fn(*args, **kwargs)
    # 运行后测一次
    after = _get_rss_mb()
    peak = max(peak, after)
    delta = peak - base
    return result, max(0.0, delta)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
# 2026-08-16 legacy 独立：baseline 参考实现迁至 tests/refs/baseline/（原 ../traditional/baseline 断链）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests', 'refs', 'baseline'))

import sgn
from cnn6_cifar10 import CNN6Cifar10
from test_phase5 import cpp_cnn6_forward_backward, make_tensor


def benchmark_torch(model, x_torch, dY_torch, n_iter=10):
    """PyTorch 前向+反向 benchmark，返回 (median, min, peak_mem_MB)"""
    # warmup
    for _ in range(3):
        model.zero_grad()
        y = model(x_torch)
        y.backward(dY_torch)

    times = []
    base_rss = _get_rss_mb()
    peak_rss = base_rss
    for _ in range(n_iter):
        model.zero_grad()
        t0 = time.perf_counter()
        y = model(x_torch)
        y.backward(dY_torch)
        times.append(time.perf_counter() - t0)
        # 每次迭代后刷新峰值
        cur = _get_rss_mb()
        if cur > peak_rss:
            peak_rss = cur

    return np.median(times), np.min(times), max(0.0, peak_rss - base_rss)


def benchmark_cpp(x_np, weights, bn_params, dY_np, n_iter=10):
    """C++ Autograd 前向+反向 benchmark，返回 (median, min, peak_mem_MB)"""
    # warmup
    for _ in range(3):
        cpp_cnn6_forward_backward(x_np, weights, bn_params, dY_np)

    times = []
    base_rss = _get_rss_mb()
    peak_rss = base_rss
    for _ in range(n_iter):
        t0 = time.perf_counter()
        cpp_cnn6_forward_backward(x_np, weights, bn_params, dY_np)
        times.append(time.perf_counter() - t0)
        cur = _get_rss_mb()
        if cur > peak_rss:
            peak_rss = cur

    return np.median(times), np.min(times), max(0.0, peak_rss - base_rss)


def benchmark_torch_forward_only(model, x_torch, n_iter=10):
    """PyTorch 仅前向 benchmark"""
    model.eval()
    for _ in range(3):
        with torch.no_grad():
            model(x_torch)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x_torch)
        times.append(time.perf_counter() - t0)
    model.train()
    return np.median(times), np.min(times)


def benchmark_cpp_forward_only(x_np, weights, bn_params, n_iter=10):
    """C++ Autograd 仅前向 benchmark（录制 tape 但不 backward）"""
    ag = sgn.autograd
    B = x_np.shape[0]

    def forward_with_tape_no_backward():
        ag.clear()
        w_conv1 = make_tensor(weights['conv1_w'])
        b_conv1 = make_tensor(weights['conv1_b'])
        w_conv2 = make_tensor(weights['conv2_w'])
        b_conv2 = make_tensor(weights['conv2_b'])
        w_conv3 = make_tensor(weights['conv3_w'])
        b_conv3 = make_tensor(weights['conv3_b'])
        w_fc1 = make_tensor(weights['fc1_w'])
        b_fc1 = make_tensor(weights['fc1_b'])
        w_fc2 = make_tensor(weights['fc2_w'])
        b_fc2 = make_tensor(weights['fc2_b'])
        w_fc3 = make_tensor(weights['fc3_w'])
        b_fc3 = make_tensor(weights['fc3_b'])
        bn_g, bn_b, bn_rm, bn_rv = {}, {}, {}, {}
        for i in range(1, 6):
            bn_g[i] = make_tensor(bn_params[f'bn{i}_gamma'])
            bn_b[i] = make_tensor(bn_params[f'bn{i}_beta'])
            bn_rm[i] = make_tensor(bn_params[f'bn{i}_running_mean'])
            bn_rv[i] = make_tensor(bn_params[f'bn{i}_running_var'])
        x = make_tensor(x_np)
        m, e = 0.1, 1e-5
        ag.start_recording()
        y = ag.conv2d(x, w_conv1, b_conv1, 1, 1)
        y = ag.batchnorm2d(y, bn_g[1], bn_b[1], bn_rm[1], bn_rv[1], m, e)
        y = ag.relu(y)
        y = ag.maxpool2d(y, 2, 2)
        y = ag.conv2d(y, w_conv2, b_conv2, 1, 1)
        y = ag.batchnorm2d(y, bn_g[2], bn_b[2], bn_rm[2], bn_rv[2], m, e)
        y = ag.relu(y)
        y = ag.maxpool2d(y, 2, 2)
        y = ag.conv2d(y, w_conv3, b_conv3, 1, 1)
        y = ag.batchnorm2d(y, bn_g[3], bn_b[3], bn_rm[3], bn_rv[3], m, e)
        y = ag.relu(y)
        y = ag.maxpool2d(y, 2, 2)
        y = ag.reshape(y, [B, -1])
        y = ag.linear(y, w_fc1, b_fc1)
        y = ag.bn_train(y, bn_g[4], bn_b[4], bn_rm[4], bn_rv[4], m, e, 0)
        y = ag.relu(y)
        y = ag.linear(y, w_fc2, b_fc2)
        y = ag.bn_train(y, bn_g[5], bn_b[5], bn_rm[5], bn_rv[5], m, e, 0)
        y = ag.relu(y)
        y = ag.linear(y, w_fc3, b_fc3)
        ag.stop_recording()
        return y.to_numpy()

    # warmup
    for _ in range(3):
        forward_with_tape_no_backward()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        forward_with_tape_no_backward()
        times.append(time.perf_counter() - t0)
    return np.median(times), np.min(times)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("Phase 5 Task A5.3: 6 层 CNN 训练速度对比 (C++ Autograd vs PyTorch)")
    print("=" * 70)

    # 测试不同 batch size
    for B in [4, 8, 16]:
        print(f"\n{'='*70}")
        print(f"Batch Size = {B}")
        print(f"{'='*70}")

        model = CNN6Cifar10(num_classes=10)
        model.train()

        x_np = np.random.randn(B, 3, 32, 32).astype(np.float32)
        x_torch = torch.from_numpy(x_np)
        dY_np = np.random.randn(B, 10).astype(np.float32)
        dY_torch = torch.from_numpy(dY_np)

        # 提取权重
        weights = {}
        for name in ['conv1', 'conv2', 'conv3', 'fc1', 'fc2', 'fc3']:
            layer = getattr(model, name)
            weights[f'{name}_w'] = layer.weight.detach().numpy().copy()
            weights[f'{name}_b'] = layer.bias.detach().numpy().copy()

        bn_layers = [model.bn1, model.bn2, model.bn3, model.bn4, model.bn5]
        bn_params = {}
        for i, bn in enumerate(bn_layers, 1):
            bn_params[f'bn{i}_gamma'] = bn.weight.detach().numpy().copy()
            bn_params[f'bn{i}_beta'] = bn.bias.detach().numpy().copy()
            bn_params[f'bn{i}_running_mean'] = bn.running_mean.detach().numpy().copy()
            bn_params[f'bn{i}_running_var'] = bn.running_var.detach().numpy().copy()

        n_iter = 10 if B <= 8 else 5

        # PyTorch forward+backward
        torch_med, torch_min, torch_mem = benchmark_torch(model, x_torch, dY_torch, n_iter=n_iter)
        print(f"  PyTorch  fwd+bwd:  median={torch_med*1000:.2f}ms  min={torch_min*1000:.2f}ms  "
              f"peak_mem={torch_mem:.1f}MB")

        # C++ forward+backward
        cpp_med, cpp_min, cpp_mem = benchmark_cpp(x_np, weights, bn_params, dY_np, n_iter=n_iter)
        print(f"  C++      fwd+bwd:  median={cpp_med*1000:.2f}ms  min={cpp_min*1000:.2f}ms  "
              f"peak_mem={cpp_mem:.1f}MB")

        # 加速比
        speedup_med = torch_med / cpp_med
        speedup_min = torch_min / cpp_min
        print(f"  加速比 (PyTorch/C++):  median={speedup_med:.3f}x  min={speedup_min:.3f}x")
        if speedup_med < 1:
            print(f"  ⚠ C++ 当前比 PyTorch 慢 {1/speedup_med:.1f}x（AVX2 FMA + OpenMP 已启用，瓶颈在 backward）")

        # 峰值内存对比
        if torch_mem > 0 and cpp_mem > 0:
            mem_ratio = cpp_mem / torch_mem if torch_mem > 0 else 0
            print(f"  峰值内存 (C++/PyTorch): {mem_ratio:.2f}x  (C++ {cpp_mem:.1f}MB vs PyTorch {torch_mem:.1f}MB)")

        # PyTorch forward-only
        torch_fwd_med, torch_fwd_min = benchmark_torch_forward_only(model, x_torch, n_iter=n_iter)
        print(f"  PyTorch  fwd only: median={torch_fwd_med*1000:.2f}ms  min={torch_fwd_min*1000:.2f}ms")

        # C++ forward-only
        cpp_fwd_med, cpp_fwd_min = benchmark_cpp_forward_only(x_np, weights, bn_params, n_iter=n_iter)
        print(f"  C++      fwd only: median={cpp_fwd_med*1000:.2f}ms  min={cpp_fwd_min*1000:.2f}ms")

        speedup_fwd = torch_fwd_med / cpp_fwd_med
        print(f"  加速比 (PyTorch/C++ fwd): median={speedup_fwd:.3f}x")

    print(f"\n{'='*70}")
    print("说明：")
    print("  - C++ Autograd 已启用 AVX2 FMA（float32 matmul）和 OpenMP")
    print("  - C++ 优化包括：P0 linear/conv2d 调 matmul, P1 cache tiling + register blocking, P2 微优化")
    print("  - PyTorch 使用 MKL 高度优化的 BLAS 和 conv 算法")
    print("  - 主要瓶颈：backward 占总时间 ~65%（B=16），conv2d 反向 im2col/col2im、gather 型 transpose matmul")
    print("=" * 70)


if __name__ == "__main__":
    main()
