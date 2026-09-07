# bench_fair_compare.py - 公平对比基准：MLP 前向+反向（三方同任务）
#
# 性能白皮书 §3 重写配套脚本（2026-09-07）。对比原则（用户拍板）：
#   - 只比"大家都有"的功能：同任务（MLP 784-256-128-10 + ReLU + CE）、
#     同 batch、同精度（f32）、同硬件、同口径（fwd+bwd 墙钟，含梯度，
#     不含优化器 step/数据搬运）；
#   - 三方：numpy 手写（无框架基线——用户不引框架时的参照）、
#     SGN C++ autograd（本框架）、PyTorch MKL（工业参照）；
#   - 计时：预热 5 轮 + 30 轮取中位。
#
# 运行：SGN 根目录 py -3.14 engine/sgn/tests/bench_fair_compare.py

import os
import sys

# 已知 OMP 运行时冲突（用户操作手册 §12.6）：SGN(libomp) + PyTorch(MKL
# libiomp5md) 同进程加载需该开关；正式发布口径建议分进程计时
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
import statistics

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(ROOT, "..", "..")))

LAYERS = (784, 256, 128, 10)
ROUNDS, WARMUP = 30, 5


def make_weights(seed=42):
    rng = np.random.default_rng(seed)
    sizes = list(zip(LAYERS[:-1], LAYERS[1:]))
    Ws = [rng.normal(0, np.sqrt(2.0 / a), (a, b)).astype(np.float32)
          for a, b in sizes]
    bs = [np.zeros(b, dtype=np.float32) for _, b in sizes]
    return Ws, bs


def numpy_mlp(x, y, Ws, bs):
    """numpy 手写 MLP fwd+bwd（无框架基线：manual 前向/反向）。"""
    t0 = time.perf_counter()
    h = x
    hs = [x]
    for i, (W, b) in enumerate(zip(Ws, bs)):
        h = h @ W + b
        if i < len(Ws) - 1:
            h = np.maximum(h, 0.0)               # 末层线性（logits）
        hs.append(h)
    logits = hs[-1]
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    loss = -logp[np.arange(len(y)), y].mean()
    dlogits = np.exp(logp)
    dlogits[np.arange(len(y)), y] -= 1.0
    dlogits /= len(y)
    g = dlogits
    gWs = [None] * len(Ws)
    gbs = [None] * len(bs)
    for i in range(len(Ws) - 1, -1, -1):
        gWs[i] = hs[i].T @ g
        gbs[i] = g.sum(axis=0)
        if i > 0:
            g = (g @ Ws[i].T) * (hs[i] > 0)
    dt = time.perf_counter() - t0
    return loss, dt


def main():
    rng = np.random.default_rng(0)
    B = 256
    x = rng.normal(0, 1, (B, LAYERS[0])).astype(np.float32)
    y = rng.integers(0, LAYERS[-1], size=B)
    Ws, bs = make_weights()

    rows = {}

    # ---- numpy 手写 ----
    for _ in range(WARMUP):
        numpy_mlp(x, y, Ws, bs)
    rows["numpy 手写（无框架基线）"] = statistics.median(
        numpy_mlp(x, y, Ws, bs)[1] for _ in range(ROUNDS)) * 1000

    # ---- SGN C++ autograd ----
    import engine.sgn as sgn
    ag, nn = sgn.autograd, sgn.nn
    model = nn.Sequential(
        nn.Linear(LAYERS[0], LAYERS[1]), nn.ReLU(),
        nn.Linear(LAYERS[1], LAYERS[2]), nn.ReLU(),
        nn.Linear(LAYERS[2], LAYERS[-1]))
    x_t = ag.Tensor(np.ascontiguousarray(x))
    for _ in range(WARMUP):
        with ag.record_scope(clear=True):
            out = model([x_t])
            out.backward(np.ones_like(out.to_numpy()) / B)
    times = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        with ag.record_scope(clear=True):
            out = model([x_t])
            out.backward(np.ones_like(out.to_numpy()) / B)
        times.append((time.perf_counter() - t0) * 1000)
    rows["SGN C++ autograd"] = statistics.median(times)

    # ---- PyTorch MKL（工业参照）----
    try:
        import torch
        import torch.nn as tnn
        tm = tnn.Sequential(
            tnn.Linear(LAYERS[0], LAYERS[1]), tnn.ReLU(),
            tnn.Linear(LAYERS[1], LAYERS[2]), tnn.ReLU(),
            tnn.Linear(LAYERS[2], LAYERS[-1]))
        tx = torch.from_numpy(x)
        ty = torch.from_numpy(y)
        crit = tnn.CrossEntropyLoss()
        for _ in range(WARMUP):
            crit(tm(tx), ty).backward()
        times = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            loss = crit(tm(tx), ty)
            loss.backward()
            times.append((time.perf_counter() - t0) * 1000)
        rows["PyTorch（MKL，工业参照）"] = statistics.median(times)
    except ImportError:
        rows["PyTorch（MKL，工业参照）"] = None

    print(f"=== 公平对比基准：MLP {LAYERS} fwd+bwd，B={B}，{ROUNDS} 轮中位 ===\n")
    base = rows["numpy 手写（无框架基线）"]
    for k, v in rows.items():
        if v is None:
            print(f"  {k:28s}  （未安装，跳过）")
        else:
            print(f"  {k:28s}  {v:8.3f} ms/iter   （vs numpy {v / base:.2f}×）")
    return rows


if __name__ == "__main__":
    main()
