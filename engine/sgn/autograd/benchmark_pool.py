# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""benchmark_pool.py - 通用内存池 ON vs OFF 多 workload 对比基准

对比 sgn 通用内存池（`sgn.autograd.set_pool_allocator`）开启/关闭两种状态下的
C++ Autograd 耗时。提速 = (OFF − ON) / OFF，正值 = 开池后变快。

  - OFF = stdlib 分配器（默认，与 benchmark_phase5.py 基线一致）
  - ON  = 通用内存池（size-class 分桶 + thread_local 无锁 free-list，
          `common/allocator.h` + `common/pool_allocator.h`）

覆盖 workload（不止 6 层 CNN，反向传播/小分配/量化路径/长循环均有）：
  1. CNN fwd+bwd   B=4/8/16 —— 大缓冲（im2col/dW）+ 反向密集（复用 benchmark_phase5）
  2. CNN fwd-only  B=16     —— 纯前向
  3. MLP fwd+bwd   3 层 Linear+ReLU —— 无 conv 大缓冲，小/中尺寸 + 反向密集
  4. CNN fwd+bwd @ STE 策略  B=16 —— 量化反向路径（分配模式与 FLOAT32 不同）
  5. 小算子密集链  bn+relu ×16 —— 纯小尺寸分配，测簿记开销下限
  6. MLP 长循环 ×100 迭代 —— 模拟真实训练连续分配/释放摊销

方法：同一进程内交替测量 OFF / ON 各 ROUNDS 轮取中位（抑制环境漂移）。
每轮测量前 set_pool_allocator() 自动 clear_pool()，保证公平。
离群过滤：剔除相对中位偏离 >25% 的离群轮次（负载尖峰防护，见 _filter_outliers）。

运行：
    cd engine/sgn/build
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
    python ..\autograd\benchmark_pool.py
"""

import os

# libomp/libiomp5md 冲突防御（需在 import torch/sgn 之前设置，见手册 §13.1.4）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
# baseline 参考实现（2026-08-16 legacy 独立后迁至 tests/refs/baseline/）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests", "refs", "baseline"))

import sgn
from cnn6_cifar10 import CNN6Cifar10
from test_phase5 import make_tensor
from benchmark_phase5 import benchmark_cpp, benchmark_cpp_forward_only

ROUNDS = 3

ag = sgn.autograd


def set_pool(enabled: bool) -> None:
    """启用/禁用通用内存池（C++ 绑定切换时会自动 clear_pool）。"""
    ag.set_pool_allocator(bool(enabled))


# ============================================================================
# workload 构造
# ============================================================================

def build_cnn_case(B: int):
    """构造 6 层 CNN 输入/权重/BN 参数（与 benchmark_phase5.main 一致）。"""
    torch.manual_seed(42)
    np.random.seed(42)
    model = CNN6Cifar10(num_classes=10)
    model.train()
    x_np = np.random.randn(B, 3, 32, 32).astype(np.float32)
    dY_np = np.random.randn(B, 10).astype(np.float32)
    weights = {}
    for name in ["conv1", "conv2", "conv3", "fc1", "fc2", "fc3"]:
        layer = getattr(model, name)
        weights[f"{name}_w"] = layer.weight.detach().numpy().copy()
        weights[f"{name}_b"] = layer.bias.detach().numpy().copy()
    bn_layers = [model.bn1, model.bn2, model.bn3, model.bn4, model.bn5]
    bn_params = {}
    for i, bn in enumerate(bn_layers, 1):
        bn_params[f"bn{i}_gamma"] = bn.weight.detach().numpy().copy()
        bn_params[f"bn{i}_beta"] = bn.bias.detach().numpy().copy()
        bn_params[f"bn{i}_running_mean"] = bn.running_mean.detach().numpy().copy()
        bn_params[f"bn{i}_running_var"] = bn.running_var.detach().numpy().copy()
    return x_np, weights, bn_params, dY_np


class MlpCase:
    """3 层 MLP（784→256→128→10）前向+反向，无 conv 大缓冲。"""

    def __init__(self, B: int = 64):
        np.random.seed(7)
        self.B = B
        self.x = np.random.randn(B, 784).astype(np.float32)
        self.dY = np.random.randn(B, 10).astype(np.float32)
        # sgn linear 权重布局为 (out, in)（与 PyTorch nn.Linear.weight 一致）
        shapes = [(256, 784), (128, 256), (10, 128)]
        self.ws = [np.random.randn(*s).astype(np.float32) * 0.05 for s in shapes]
        self.bs = [np.zeros(s[0], dtype=np.float32) for s in shapes]

    def run_once(self):
        ag.clear()
        x = make_tensor(self.x)
        ps = [(make_tensor(w, requires_grad=True), make_tensor(b, requires_grad=True))
              for w, b in zip(self.ws, self.bs)]
        ag.start_recording()
        y = x
        for i, (w, b) in enumerate(ps):
            y = ag.linear(y, w, b)
            if i < len(ps) - 1:
                y = ag.relu(y)
        ag.stop_recording()
        out = y.to_numpy()
        y.backward(self.dY)
        return out

    def bench(self, n_iter: int = 50):
        for _ in range(3):
            self.run_once()  # warmup
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            self.run_once()
            times.append(time.perf_counter() - t0)
        return float(np.median(times))


class SmallOpsCase:
    """小算子密集链：batchnorm2d + relu ×16，纯小尺寸分配。"""

    def __init__(self, B: int = 16, C: int = 64, HW: int = 16):
        np.random.seed(11)
        self.x = np.random.randn(B, C, HW, HW).astype(np.float32)
        self.g = np.ones(C, dtype=np.float32)
        self.b = np.zeros(C, dtype=np.float32)
        self.rm = np.zeros(C, dtype=np.float32)
        self.rv = np.ones(C, dtype=np.float32)

    def run_once(self):
        ag.clear()
        x = make_tensor(self.x)
        g = make_tensor(self.g)
        b_ = make_tensor(self.b)
        rm = make_tensor(self.rm)
        rv = make_tensor(self.rv)
        ag.start_recording()
        y = x
        for _ in range(16):
            y = ag.batchnorm2d(y, g, b_, rm, rv, 0.1, 1e-5)
            y = ag.relu(y)
        ag.stop_recording()
        return y.to_numpy()

    def bench(self, n_iter: int = 50):
        for _ in range(3):
            self.run_once()
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            self.run_once()
            times.append(time.perf_counter() - t0)
        return float(np.median(times))


def bench_mlp_long_run(case: MlpCase, n_iter: int = 100) -> float:
    """长循环：连续 n_iter 次分配/释放不停顿，测摊销后单次耗时。"""
    case.run_once()  # warmup
    t0 = time.perf_counter()
    for _ in range(n_iter):
        case.run_once()
    total = time.perf_counter() - t0
    return total / n_iter


# ============================================================================
# 测量框架
# ============================================================================

def _filter_outliers(vals: list) -> list:
    """离群轮次过滤：剔除相对中位偏离 >25% 的测量值（负载尖峰防护）。

    背景：长循环类测量用均值口径时，偶发负载尖峰（如系统后台任务）会把一轮
    ON/OFF 整体抬高，造成"负收益"假象（2026-08-16 实测：MLP 长循环某轮
    ON 侧被尖峰抬高 → 误判 ↓26.6%，复跑为噪声）。这里按轮次粒度过滤：
    与整组中位偏差 >25% 的轮次视为离群剔除，避免尖峰污染结论。
    """
    if len(vals) < 2:
        return list(vals)
    med = float(np.median(vals))
    if med <= 0:
        return list(vals)
    kept = [v for v in vals if abs(v - med) <= 0.25 * med]
    # 至少保留 1 个值（极端情况全离群时退回原始中位）
    return kept if kept else [med]


def measure_pair(fn, *args, **kwargs) -> tuple:
    """交替测量 OFF/ON 各 ROUNDS 轮，剔除离群轮次后取中位。

    返回 (off_median_s, on_median_s)。每轮切换自动 clear_pool()。
    """
    offs, ons = [], []
    for _ in range(ROUNDS):
        set_pool(False)
        offs.append(fn(*args, **kwargs))
        set_pool(True)
        ons.append(fn(*args, **kwargs))
    offs = _filter_outliers(offs)
    ons = _filter_outliers(ons)
    return float(np.median(offs)), float(np.median(ons))


def measure_pair_mean(fn, *args, **kwargs) -> tuple:
    """长循环类：交替测量 OFF/ON 各 ROUNDS 轮（每轮已是 n_iter 次均值），
    剔除离群轮次后取中位。均值口径对单轮内尖峰敏感，须按轮次过滤。"""
    offs, ons = [], []
    for _ in range(ROUNDS):
        set_pool(False)
        offs.append(fn(*args, **kwargs))
        set_pool(True)
        ons.append(fn(*args, **kwargs))
    offs = _filter_outliers(offs)
    ons = _filter_outliers(ons)
    return float(np.median(offs)), float(np.median(ons))


def report(name: str, off_s: float, on_s: float, unit: str = "ms") -> None:
    """打印一行对比结果。统一口径：正数 = 提速（开池变快），负数 = 变慢。

    提速 % = (OFF − ON) / OFF × 100
    """
    speedup = (off_s - on_s) / off_s * 100.0 if off_s > 0 else 0.0
    scale = 1000.0 if unit == "ms" else 1.0
    tag = "提速" if speedup >= 0 else "变慢"
    print(f"  {name:<28s} OFF={off_s*scale:8.3f}{unit}  ON={on_s*scale:8.3f}{unit}  "
          f"{tag} {abs(speedup):6.2f}%")


def main() -> None:
    print("=" * 78)
    print("通用内存池 ON vs OFF 多 workload 对比（正数=提速/开池变快，负数=变慢）")
    print(f"  交替测量 {ROUNDS} 轮，剔除离群轮次后取中位；每轮切换自动 clear_pool()")
    print("=" * 78)

    # ---- 1/2. CNN fwd+bwd / fwd-only（B=4/8/16）----
    print("\n[1] 6 层 CNN fwd+bwd（大缓冲 + 反向密集）")
    for B in [4, 8, 16]:
        x_np, weights, bn_params, dY_np = build_cnn_case(B)
        n_iter = 10 if B <= 8 else 5

        def cnn_bwd(x=x_np, w=weights, b=bn_params, d=dY_np, n=n_iter):
            return benchmark_cpp(x, w, b, d, n_iter=n)[0]

        off, on = measure_pair(cnn_bwd)
        report(f"B={B}", off, on)

    print("\n[2] 6 层 CNN fwd-only（B=16）")
    x_np, weights, bn_params, _ = build_cnn_case(16)

    def cnn_fwd(w=weights, b=bn_params, x=x_np):
        return benchmark_cpp_forward_only(x, w, b, n_iter=5)[0]

    off, on = measure_pair(cnn_fwd)
    report("B=16", off, on)

    # ---- 3. MLP fwd+bwd ----
    print("\n[3] MLP fwd+bwd 3 层（B=64, 784→256→128→10，无 conv 大缓冲，反向密集）")
    mlp = MlpCase(B=64)
    off, on = measure_pair(mlp.bench, n_iter=50)
    report("B=64", off, on)

    # ---- 4. CNN fwd+bwd @ STE 策略 ----
    print("\n[4] 6 层 CNN fwd+bwd @ STE 量化反向（B=16）")
    x_np, weights, bn_params, dY_np = build_cnn_case(16)
    ag.set_backward_strategy(ag.BackwardStrategy.STE)

    def cnn_bwd_ste(x=x_np, w=weights, b=bn_params, d=dY_np):
        return benchmark_cpp(x, w, b, d, n_iter=5)[0]

    off_ste, on_ste = measure_pair(cnn_bwd_ste)
    ag.set_backward_strategy(ag.BackwardStrategy.FLOAT32)  # 恢复默认
    report("B=16 STE", off_ste, on_ste)

    # ---- 5. 小算子密集链 ----
    print("\n[5] 小算子密集链 batchnorm2d+relu ×16（B=16,C=64,16×16，纯小分配）")
    small = SmallOpsCase()
    off, on = measure_pair(small.bench, n_iter=50)
    report("小尺寸×32 算子", off, on)

    # ---- 6. MLP 长循环摊销 ----
    print("\n[6] MLP 长循环 ×100 迭代（模拟真实训练连续分配/释放，均值+离群过滤）")
    off, on = measure_pair_mean(bench_mlp_long_run, mlp, n_iter=100)
    report("B=64 ×100 iter", off, on)

    # 收尾：恢复默认分配器
    set_pool(False)
    print(f"\n{'='*78}")
    print("注：环境未关闭省电模式，绝对数值受负载波动；OFF/ON 同进程对比更具参考意义。")
    print("=" * 78)


if __name__ == "__main__":
    main()
