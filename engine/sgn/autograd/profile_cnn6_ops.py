# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""profile_cnn6_ops.py - 6 层 CNN 逐算子计时（fwd / bwd）

目的：定位 benchmark_phase5 的 8.1x 差距花在哪些算子（conv2d / batchnorm / relu /
maxpool / linear / reshape），不靠猜、用数据说话。

方法与口径（2026-08-16）：
  - 前向：逐算子包计时（含 tape 录制，与 benchmark_phase5 同路径），丢弃 warmup 后取中位。
  - 反向：增量子图法——对每个前缀（到第 i 个算子为止）单独建图并 backward 计时，
    第 i 个算子的反向代价 ≈ 前缀(i) 反向耗时 − 前缀(i−1) 反向耗时。
  - 总开销 = 总耗时 − Σ(算子耗时)：前向侧为 tape 录制/张量创建/分配，反向侧为图销毁/叶梯度归集。
  - 反向总量 = (fwd+bwd) − fwd_only，与逐算子反向代价之和的差 = 反向侧固定开销。

运行（先设 OMP 环境变量，与 benchmark_phase5 一致）：
  cd engine/sgn/build
  $env:KMP_DUPLICATE_LIB_OK = "TRUE"
  python ..\\autograd\\profile_cnn6_ops.py [B]
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, List, Tuple

# 必须在导入 torch 之前设置（libomp/libiomp5md 冲突，同 benchmark_phase5）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
# 2026-08-16 legacy 独立：baseline 参考实现迁至 tests/refs/baseline/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests", "refs", "baseline"))

import sgn
from cnn6_cifar10 import CNN6Cifar10
from test_phase5 import make_tensor

ag = sgn.autograd

# 步骤：每个元素 (名称, 作用于 (y, tensors) -> y)
Step = Tuple[str, Callable]


class Tensors:
    pass


def build_data(B: int):
    """从 CNN6Cifar10 提取权重/BN 参数，与 benchmark_phase5 完全一致（seed 固定）。"""
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
    bn_params = {}
    for i, bn in enumerate([model.bn1, model.bn2, model.bn3, model.bn4, model.bn5], 1):
        bn_params[f"bn{i}_gamma"] = bn.weight.detach().numpy().copy()
        bn_params[f"bn{i}_beta"] = bn.bias.detach().numpy().copy()
        bn_params[f"bn{i}_running_mean"] = bn.running_mean.detach().numpy().copy()
        bn_params[f"bn{i}_running_var"] = bn.running_var.detach().numpy().copy()
    return weights, bn_params, x_np, dY_np


def build_tensors(weights, bn_params, x_np) -> Tensors:
    t = Tensors()
    t.w_conv1 = make_tensor(weights["conv1_w"], True)
    t.b_conv1 = make_tensor(weights["conv1_b"], True)
    t.w_conv2 = make_tensor(weights["conv2_w"], True)
    t.b_conv2 = make_tensor(weights["conv2_b"], True)
    t.w_conv3 = make_tensor(weights["conv3_w"], True)
    t.b_conv3 = make_tensor(weights["conv3_b"], True)
    t.w_fc1 = make_tensor(weights["fc1_w"], True)
    t.b_fc1 = make_tensor(weights["fc1_b"], True)
    t.w_fc2 = make_tensor(weights["fc2_w"], True)
    t.b_fc2 = make_tensor(weights["fc2_b"], True)
    t.w_fc3 = make_tensor(weights["fc3_w"], True)
    t.b_fc3 = make_tensor(weights["fc3_b"], True)
    t.bn_g, t.bn_b, t.bn_rm, t.bn_rv = {}, {}, {}, {}
    for i in range(1, 6):
        t.bn_g[i] = make_tensor(bn_params[f"bn{i}_gamma"], True)
        t.bn_b[i] = make_tensor(bn_params[f"bn{i}_beta"], True)
        t.bn_rm[i] = make_tensor(bn_params[f"bn{i}_running_mean"], False)
        t.bn_rv[i] = make_tensor(bn_params[f"bn{i}_running_var"], False)
    t.x = make_tensor(x_np, False)
    return t


def make_steps(B: int) -> List[Step]:
    m, e = 0.1, 1e-5
    steps: List[Step] = []

    def add(name: str, fn: Callable):
        steps.append((name, fn))

    add("conv2d_1", lambda y, t: ag.conv2d(y, t.w_conv1, t.b_conv1, 1, 1))
    add("bn_1", lambda y, t: ag.batchnorm2d(y, t.bn_g[1], t.bn_b[1], t.bn_rm[1], t.bn_rv[1], m, e))
    add("relu_1", lambda y, t: ag.relu(y))
    add("pool_1", lambda y, t: ag.maxpool2d(y, 2, 2))
    add("conv2d_2", lambda y, t: ag.conv2d(y, t.w_conv2, t.b_conv2, 1, 1))
    add("bn_2", lambda y, t: ag.batchnorm2d(y, t.bn_g[2], t.bn_b[2], t.bn_rm[2], t.bn_rv[2], m, e))
    add("relu_2", lambda y, t: ag.relu(y))
    add("pool_2", lambda y, t: ag.maxpool2d(y, 2, 2))
    add("conv2d_3", lambda y, t: ag.conv2d(y, t.w_conv3, t.b_conv3, 1, 1))
    add("bn_3", lambda y, t: ag.batchnorm2d(y, t.bn_g[3], t.bn_b[3], t.bn_rm[3], t.bn_rv[3], m, e))
    add("relu_3", lambda y, t: ag.relu(y))
    add("pool_3", lambda y, t: ag.maxpool2d(y, 2, 2))
    add("reshape", lambda y, t: ag.reshape(y, [B, -1]))
    add("linear_4", lambda y, t: ag.linear(y, t.w_fc1, t.b_fc1))
    add("bn_4", lambda y, t: ag.bn_train(y, t.bn_g[4], t.bn_b[4], t.bn_rm[4], t.bn_rv[4], m, e, 0))
    add("relu_4", lambda y, t: ag.relu(y))
    add("linear_5", lambda y, t: ag.linear(y, t.w_fc2, t.b_fc2))
    add("bn_5", lambda y, t: ag.bn_train(y, t.bn_g[5], t.bn_b[5], t.bn_rm[5], t.bn_rv[5], m, e, 0))
    add("relu_5", lambda y, t: ag.relu(y))
    add("linear_6", lambda y, t: ag.linear(y, t.w_fc3, t.b_fc3))
    return steps


def median(xs) -> float:
    return float(np.median(xs))


def run_full_forward(steps: List[Step], tensors: Tensors):
    ag.clear()
    ag.start_recording()
    y = tensors.x
    try:
        for _, fn in steps:
            y = fn(y, tensors)
    finally:
        ag.stop_recording()
    return y


def forward_prefix(steps: List[Step], tensors: Tensors, upto: int):
    ag.clear()
    ag.start_recording()
    y = tensors.x
    try:
        for i in range(upto + 1):
            y = steps[i][1](y, tensors)
    finally:
        ag.stop_recording()
    return y


def profile(B: int, n_warmup: int = 5, n_iter: int = 21) -> dict:
    weights, bn_params, x_np, dY_np = build_data(B)
    steps = make_steps(B)
    n_steps = len(steps)

    # ---- 0. 总量基线（复现 benchmark 口径）----
    fwd_bwd = []
    fwd_only = []
    for it in range(n_warmup + n_iter):
        t = build_tensors(weights, bn_params, x_np)
        t0 = time.perf_counter()
        y = run_full_forward(steps, t)
        y.backward(dY_np)
        fwd_bwd.append(time.perf_counter() - t0)

        t = build_tensors(weights, bn_params, x_np)
        t0 = time.perf_counter()
        run_full_forward(steps, t)
        fwd_only.append(time.perf_counter() - t0)
    fwd_bwd_med = median(fwd_bwd[n_warmup:])
    fwd_only_med = median(fwd_only[n_warmup:])

    # ---- 1. 前向逐算子 ----
    fwd_ops = [[] for _ in range(n_steps)]
    for it in range(n_warmup + n_iter):
        t = build_tensors(weights, bn_params, x_np)
        ag.clear()
        ag.start_recording()
        y = t.x
        try:
            for i, (_, fn) in enumerate(steps):
                t0 = time.perf_counter()
                y = fn(y, t)
                fwd_ops[i].append(time.perf_counter() - t0)
        finally:
            ag.stop_recording()
    fwd_med = [median(x[n_warmup:]) for x in fwd_ops]

    # ---- 2. 反向逐算子（增量子图法）----
    bwd_cum = []
    for i in range(n_steps):
        times = []
        for it in range(n_warmup + n_iter):
            t = build_tensors(weights, bn_params, x_np)
            y = forward_prefix(steps, t, i)
            grad = np.ones(list(y.shape), dtype=np.float32)
            t0 = time.perf_counter()
            y.backward(grad)
            times.append(time.perf_counter() - t0)
        bwd_cum.append(median(times[n_warmup:]))

    bwd_ops = [bwd_cum[0]]
    for i in range(1, n_steps):
        bwd_ops.append(bwd_cum[i] - bwd_cum[i - 1])

    return {
        "B": B,
        "n_steps": n_steps,
        "fwd_bwd_med": fwd_bwd_med,
        "fwd_only_med": fwd_only_med,
        "bwd_total_med": fwd_bwd_med - fwd_only_med,
        "fwd_med": fwd_med,
        "bwd_med": bwd_ops,
        "names": [n for n, _ in steps],
    }


def main() -> int:
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    print("=" * 78)
    print(f"6 层 CNN 逐算子计时  B={B}  （fwd 含 tape 录制 / bwd 增量子图归因）")
    print(f"口径：丢弃 warmup 后取中位；环境未控，需 KMP_DUPLICATE_LIB_OK=TRUE")
    print("=" * 78)

    r = profile(B)
    names = r["names"]
    fwd = r["fwd_med"]
    bwd = r["bwd_med"]
    n = r["n_steps"]

    print(f"\n总量基线：")
    print(f"  fwd+bwd : {r['fwd_bwd_med']*1000:8.3f} ms  (benchmark_phase5 参考 ~80.56ms @B16)")
    print(f"  fwd only: {r['fwd_only_med']*1000:8.3f} ms")
    print(f"  bwd 总量: {r['bwd_total_med']*1000:8.3f} ms  ({(r['bwd_total_med']/max(r['fwd_bwd_med'],1e-12)*100):.0f}% of fwd+bwd)")

    sum_fwd = sum(fwd)
    sum_bwd = sum(bwd)
    print(f"\n逐算子（fwd / bwd）：")
    print(f"  {'算子':<12} | {'fwd(ms)':>9} | {'fwd%':>6} | {'bwd(ms)':>9} | {'bwd%':>6}")
    print("  " + "-" * 50)
    for i in range(n):
        print(f"  {names[i]:<12} | {fwd[i]*1000:9.3f} | {fwd[i]/max(sum_fwd,1e-12)*100:5.1f}% | "
              f"{bwd[i]*1000:9.3f} | {bwd[i]/max(sum_bwd,1e-12)*100:5.1f}%")
    print("  " + "-" * 50)
    print(f"  {'Σ 算子':<12} | {sum_fwd*1000:9.3f} | {100.0:5.1f}% | {sum_bwd*1000:9.3f} | {100.0:5.1f}%")
    fwd_ovh = r["fwd_only_med"] - sum_fwd
    bwd_ovh = r["bwd_total_med"] - sum_bwd
    print(f"\n固定/框架开销：")
    print(f"  前向 开销（tape 录制/张量创建/分配）= {fwd_ovh*1000:.3f} ms"
          f"（占 fwd only {fwd_ovh/max(r['fwd_only_med'],1e-12)*100:.0f}%）")
    print(f"  反向 开销（图销毁/叶梯度归集/分配）= {bwd_ovh*1000:.3f} ms"
          f"（占 bwd 总量 {bwd_ovh/max(r['bwd_total_med'],1e-12)*100:.0f}%）")

    print("\n按类型汇总（fwd 占比最高者优先）：")
    from collections import defaultdict
    agg_f = defaultdict(float)
    agg_b = defaultdict(float)
    for i, nm in enumerate(names):
        typ = nm.split("_")[0]
        agg_f[typ] += fwd[i]
        agg_b[typ] += bwd[i]
    for typ in sorted(agg_f, key=lambda k: -agg_f[k]):
        print(f"  {typ:<12}: fwd {agg_f[typ]*1000:7.3f} ms | bwd {agg_b[typ]*1000:7.3f} ms")

    print("\n说明：反向为增量子图归因（近似，层间可能有串扰）；")
    print("     前向为同进程逐算子计时（相对值可信，绝对值受环境负载影响）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
