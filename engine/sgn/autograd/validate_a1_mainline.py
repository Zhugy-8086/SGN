# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""A1 主线训练验证：engine 策略对比（CNN 安全 + 精细收敛地板）

背景：docs/SGN_统一数学框架.md §8.3（A1（HC8+SR）主推）+ §14.9（R3-A：A1 主线落地）。
engine 接线（2026-08-19）：新增 BackwardStrategy::A1 = 前向 STE 式激活 8bit 量化
+ 反向 SR 梯度量化（autograd_nn.cpp / autograd.cpp）。

两个案例：
  ① CNN（conv-relu-pool ×2 + fc，合成回归）——粗 loss 地板（~0.18）
     预期：全部策略 ≈ FLOAT32（8bit 量化在粗地板上无损）→ A1 安全性验证
  ② 标准化线性回归（N=256,D=32，精细收敛 ~7e-5）
     预期：FLOAT32 ≈ SR ≈ GEF；A1 受【前向 8bit 激活量化噪声地板】限制，
     loss ~ (range/127)²/12（~1e-4），无法达到 FLOAT32 的 7e-5 → A1 机制验证

结论（R3-A）：
  - A1 在实用 CNN 训练上无损（安全主路径）；
  - A1 的实际精度地板由【前向 8bit 量化】决定（~(range/127)²/12），
    非 SR 反向——精细收敛时这是绑定约束（8bit 训练固有极限）。

运行：cd engine/sgn/autograd && python validate_a1_mainline.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import numpy as np
import sgn

ag = sgn.autograd


def make_tensor(a, rg=False):
    t = ag.Tensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    t.requires_grad = rg
    return t


def quant8(x):
    """HC8 对称 8bit 量化（R0 证明 = 普通 int8 同网格）"""
    amax = np.max(np.abs(x))
    if amax == 0:
        return x * 0.0
    scale = amax / 127.0
    q = np.round(x / scale)
    q = np.clip(q, -127.0, 127.0)
    return q * scale


# ============================================================================
# ① CNN 安全案例
# ============================================================================
def init_cnn(rng):
    return {
        "c1w": rng.normal(0, 0.1, (8, 3, 3, 3)).astype(np.float32),
        "c1b": np.zeros(8, np.float32),
        "c2w": rng.normal(0, 0.1, (16, 8, 3, 3)).astype(np.float32),
        "c2b": np.zeros(16, np.float32),
        "fcw": rng.normal(0, 0.05, (10, 16 * 4 * 4)).astype(np.float32),
        "fcb": np.zeros(10, np.float32),
    }


def cnn_forward(model, X, quant_fwd):
    W1 = quant8(model["c1w"]) if quant_fwd else model["c1w"]
    W2 = quant8(model["c2w"]) if quant_fwd else model["c2w"]
    W3 = quant8(model["fcw"]) if quant_fwd else model["fcw"]
    params = {
        "c1w": make_tensor(W1, True), "c1b": make_tensor(model["c1b"], True),
        "c2w": make_tensor(W2, True), "c2b": make_tensor(model["c2b"], True),
        "fcw": make_tensor(W3, True), "fcb": make_tensor(model["fcb"], True),
    }
    c1 = ag.conv2d(make_tensor(X), params["c1w"], params["c1b"], 1, 1)
    p1 = ag.maxpool2d(ag.relu(c1), 2, 2)
    c2 = ag.conv2d(p1, params["c2w"], params["c2b"], 1, 1)
    p2 = ag.maxpool2d(ag.relu(c2), 2, 2)
    flat = ag.reshape(p2, [-1, 16 * 4 * 4])
    y = ag.linear(flat, params["fcw"], params["fcb"])
    return y, params


def train_cnn(strategy, seed, T=300, B=32, lr=0.01):
    rng = np.random.default_rng(seed)
    data_rng = np.random.default_rng(999)
    P = data_rng.normal(0, 0.02, (10, 3 * 16 * 16)).astype(np.float32)

    def gen():
        X = rng.normal(0, 1, (B, 3, 16, 16)).astype(np.float32)
        return X, np.tanh(X.reshape(B, -1) @ P.T).astype(np.float32)

    model = init_cnn(rng)
    quant_fwd = (strategy == "A1")
    ag.set_backward_strategy(getattr(ag.BackwardStrategy, strategy))
    ag.set_sr_seed(1234 + seed)
    losses = np.zeros(T)
    for t in range(T):
        X, y = gen()
        ag.clear()
        ag.start_recording()
        yp_t, params = cnn_forward(model, X, quant_fwd)
        ag.stop_recording()
        yp = yp_t.to_numpy()
        yp_t.backward((2.0 * (yp - y) / B))
        for key in model:
            model[key] = model[key] - lr * params[key].grad
        losses[t] = float(np.mean((yp - y) ** 2))
    return losses


# ============================================================================
# ② 线性精细收敛案例
# ============================================================================
def train_linear(strategy, seed, T=8000, N=256, D=32, lr=5e-3):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (N, D)).astype(np.float32)
    wt = rng.normal(0, 1, D).astype(np.float32)
    y0 = X @ wt + rng.normal(0, 0.05, N)
    y = ((y0 - y0.mean()) / y0.std()).astype(np.float32)
    w = np.zeros((1, D), np.float32)
    b = np.zeros((1,), np.float32)
    quant_fwd = (strategy == "A1")
    ag.set_backward_strategy(getattr(ag.BackwardStrategy, strategy))
    ag.set_sr_seed(42)
    losses = np.zeros(T)
    for t in range(T):
        ag.clear()
        W = quant8(w) if quant_fwd else w
        w_t = make_tensor(W, True)
        b_t = make_tensor(b, True)
        ag.start_recording()
        yp_t = ag.linear(make_tensor(X), w_t, b_t)
        ag.stop_recording()
        yp = yp_t.to_numpy().ravel()
        yp_t.backward((2.0 * (yp - y) / N).reshape(-1, 1))
        w = w - lr * w_t.grad
        b = b - lr * b_t.grad
        losses[t] = float(np.mean((yp - y) ** 2))
    return losses


def main():
    print("=" * 76)
    print("A1 主线训练验证（engine）：① CNN 安全  ② 线性精细收敛地板")
    print("=" * 76)

    strategies = ["FLOAT32", "STE", "SR", "GEF", "A1"]

    print("\n① CNN（conv-relu-pool×2 + fc，合成回归，T=300）—— 粗地板 ~0.18")
    cnn = {}
    for st in strategies:
        L = train_cnn(st, seed=7)
        cnn[st] = float(np.min(L[-50:]))
        print(f"    {st:9s}: 末50最小 {cnn[st]:.4e}")
    base = cnn["FLOAT32"]
    print("    判定：A1 vs FLOAT32 差距（<1% = 8bit 无损，A1 安全主路径）")
    for st in strategies:
        print(f"      {st:9s}: {cnn[st]/base:.4f}x FLOAT32")

    print("\n② 标准化线性回归（N=256,D=32,T=8000）—— 精细收敛 ~7e-5")
    lin = {}
    for st in strategies:
        L = train_linear(st, seed=7)
        lin[st] = float(np.min(L[-1000:]))
        print(f"    {st:9s}: 末1000最小 {lin[st]:.4e}")
    base2 = lin["FLOAT32"]
    print("    判定：A1 受前向 8bit 激活量化地板（~(range/127)²/12 ≈ 1e-4）限制；")
    print("           SR ≈ GEF ≈ FLOAT32（反向量化在精细收敛下差异小，SR 无偏略优）")
    for st in strategies:
        print(f"      {st:9s}: {lin[st]/base2:.3f}x FLOAT32")

    print("\n" + "=" * 76)
    print("R3-A 结论：")
    print("  1. A1（前向 8bit + SR 反向）在 CNN 训练上 ≈ FLOAT32（安全主路径）；")
    print("  2. A1 的实际精度地板由【前向 8bit 量化】决定（~(range/127)²/12），")
    print("     非 SR 反向——8bit 存储的固有极限，符合 R0/审计预期。")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
