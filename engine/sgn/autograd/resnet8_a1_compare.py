# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""ResNet-8 + CIFAR-10：FLOAT32 vs A1 对比（R4，A1 深网安全性验证）

背景
----
R3-A 验证 A1（前向 8bit 激活 STE 量化 + 反向 SR 梯度量化）在 2 层 CNN 上 ≈ FLOAT32。
R4 把它升级到真实深网：CIFAR-10 版 ResNet-8（3 残差 stage × 1 block，channel 16/32/64，
全局池化 + fc），验证 A1 在【残差结构 + 深度网络】上是否仍安全。

engine 新增算子（R4 前置，2026-08-19）：
  - ag.add（残差连接，backward dA=dY/dB=dY）
  - ag.avgpool2d（全局平均池化 GAP，标准 ResNet 分类头，替代 maxpool 峰值放大）

A1 语义（与 validate_a1_mainline 一致）：
  - 权重：每步手动 quant8（numpy 侧）
  - 激活：conv2d/linear 走 engine 的 STE 前向（量化输出）
  - 反向：engine BackwardStrategy::A1（SR 梯度量化）
  - BN：gamma/beta 量化（不量化），running stats 持久 Tensor 跨步累积

数据：CIFAR-10（torchvision 下载到 SGN/data/cifar10），32x32x3，one-hot + MSE loss。

运行：cd engine/sgn/autograd && python resnet8_a1_compare.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import numpy as np
import sgn

ag = sgn.autograd

DATA_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data'))


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
# CIFAR-10 数据（torchvision → numpy）
# ============================================================================
def load_cifar10(subset=None, seed=42):
    import torchvision
    ds = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=False)
    X = ds.data.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465], np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], np.float32).reshape(1, 3, 1, 1)
    X = (X - mean) / std
    y = np.array(ds.targets)
    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    if subset is not None and subset < n:
        idx = idx[:subset]
    return X[idx], y[idx]


def to_onehot(y, K=10):
    oh = np.zeros((y.shape[0], K), np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh


# ============================================================================
# ResNet-8（CIFAR 版）：conv1 + 3×(残差块) + 全局池化 + fc
# ============================================================================
def init_resnet8(seed=7):
    rng = np.random.default_rng(seed)

    def kaiming(shape, fan_in):
        b = np.sqrt(6.0 / fan_in) * np.sqrt(2.0)
        return rng.uniform(-b, b, shape).astype(np.float32)

    p = {}
    p["c1w"] = kaiming((16, 3, 3, 3), 3 * 9)
    p["c1b"] = np.zeros(16, np.float32)
    p["s1a_w"] = kaiming((16, 16, 3, 3), 16 * 9)
    p["s1a_b"] = np.zeros(16, np.float32)
    p["s1b_w"] = kaiming((16, 16, 3, 3), 16 * 9)
    p["s1b_b"] = np.zeros(16, np.float32)
    p["s2a_w"] = kaiming((32, 16, 3, 3), 16 * 9)
    p["s2a_b"] = np.zeros(32, np.float32)
    p["s2b_w"] = kaiming((32, 32, 3, 3), 32 * 9)
    p["s2b_b"] = np.zeros(32, np.float32)
    p["s2sc_w"] = kaiming((32, 16, 1, 1), 16)
    p["s2sc_b"] = np.zeros(32, np.float32)
    p["s3a_w"] = kaiming((64, 32, 3, 3), 32 * 9)
    p["s3a_b"] = np.zeros(64, np.float32)
    p["s3b_w"] = kaiming((64, 64, 3, 3), 64 * 9)
    p["s3b_b"] = np.zeros(64, np.float32)
    p["s3sc_w"] = kaiming((64, 32, 1, 1), 32)
    p["s3sc_b"] = np.zeros(64, np.float32)
    p["fcw"] = kaiming((10, 64), 64)
    p["fcb"] = np.zeros(10, np.float32)
    # BN gamma/beta（参数，需 grad）
    for name, c in [("c1", 16), ("s1a", 16), ("s1b", 16), ("s2a", 32),
                    ("s2b", 32), ("s2sc", 32), ("s3a", 64), ("s3b", 64),
                    ("s3sc", 64)]:
        p[f"bn_{name}_g"] = np.ones(c, np.float32)
        p[f"bn_{name}_b"] = np.zeros(c, np.float32)
    return p


def resnet8_forward(X, p, quant_fwd, bn_stats):
    """X: (N,3,32,32) numpy。返回 (logits Tensor, params_tensors 注册表)。

    params_tensors: {name: Tensor}（权重/bias/gamma/beta 需 grad），train 用它更新。
    bn_stats: {name: (rm_Tensor, rv_Tensor)} 持久 running stats（跨步累积，不需 grad）。
    """
    params = {}

    def conv(x, wname, bname, s, pad):
        W = quant8(p[wname]) if quant_fwd else p[wname]
        wt = make_tensor(W, True)
        bt = make_tensor(p[bname], True)
        params[wname] = wt
        params[bname] = bt
        return ag.conv2d(x, wt, bt, s, pad)

    def bn(x, name):
        gt = make_tensor(p[f"bn_{name}_g"], True)
        bt = make_tensor(p[f"bn_{name}_b"], True)
        params[f"bn_{name}_g"] = gt
        params[f"bn_{name}_b"] = bt
        rm_t, rv_t = bn_stats[name]
        return ag.batchnorm2d(x, gt, bt, rm_t, rv_t, 0.1, 1e-5)

    def fc(x):
        W = quant8(p["fcw"]) if quant_fwd else p["fcw"]
        wt = make_tensor(W, True)
        bt = make_tensor(p["fcb"], True)
        params["fcw"] = wt
        params["fcb"] = bt
        return ag.linear(x, wt, bt)

    # conv1: 3->16
    c1 = conv(make_tensor(X), "c1w", "c1b", 1, 1)
    c1 = ag.relu(bn(c1, "c1"))
    # stage1（16->16，残差同形）
    h = c1
    s1 = ag.relu(bn(conv(h, "s1a_w", "s1a_b", 1, 1), "s1a"))
    s1 = bn(conv(s1, "s1b_w", "s1b_b", 1, 1), "s1b")
    h = ag.relu(ag.add(s1, h))
    # stage2（16->32，s2 downsample）
    s2 = ag.relu(bn(conv(h, "s2a_w", "s2a_b", 2, 1), "s2a"))
    s2 = bn(conv(s2, "s2b_w", "s2b_b", 1, 1), "s2b")
    sc2 = bn(conv(h, "s2sc_w", "s2sc_b", 2, 0), "s2sc")
    h = ag.relu(ag.add(s2, sc2))
    # stage3（32->64，s2 downsample）
    s3 = ag.relu(bn(conv(h, "s3a_w", "s3a_b", 2, 1), "s3a"))
    s3 = bn(conv(s3, "s3b_w", "s3b_b", 1, 1), "s3b")
    sc3 = bn(conv(h, "s3sc_w", "s3sc_b", 2, 0), "s3sc")
    h = ag.relu(ag.add(s3, sc3))
    # 全局平均池化（GAP，标准 ResNet 分类头）+ fc
    g = ag.avgpool2d(h, 8, 8)
    flat = ag.reshape(g, [-1, 64])
    logits = fc(flat)
    return logits, params


def make_bn_stats(p):
    """持久 BN running stats Tensor（跨步复用，engine 原地更新累积）。"""
    stats = {}
    for name in ["c1", "s1a", "s1b", "s2a", "s2b", "s2sc", "s3a", "s3b", "s3sc"]:
        c = p[f"bn_{name}_g"].shape[0]
        rm = make_tensor(np.zeros(c, np.float32), False)
        rv = make_tensor(np.ones(c, np.float32), False)
        stats[name] = (rm, rv)
    return stats


# ============================================================================
# 训练
# ============================================================================
def train(strategy, X, y_oh, T, B, lr, seed=7):
    p = init_resnet8(seed)
    bn_stats = make_bn_stats(p)
    quant_fwd = (strategy == "A1")
    ag.set_backward_strategy(getattr(ag.BackwardStrategy, strategy))
    ag.set_sr_seed(1234 + seed)
    N = X.shape[0]
    rng = np.random.default_rng(seed + 1)
    losses = np.zeros(T)
    accs = np.zeros(T)
    for t in range(T):
        idx = rng.integers(0, N, B)
        xb, yb = X[idx], y_oh[idx]
        ag.clear()
        ag.start_recording()
        logits, params = resnet8_forward(xb, p, quant_fwd, bn_stats)
        ag.stop_recording()
        logits_np = logits.to_numpy()
        grad = 2.0 * (logits_np - yb) / B
        logits.backward(grad)
        # 更新（BN running stats 在 engine 内部已原地更新，跳过）
        for key, t_ in params.items():
            p[key] = p[key] - lr * t_.grad
        losses[t] = float(np.mean((logits_np - yb) ** 2))
        accs[t] = float(np.mean(np.argmax(logits_np, 1) == np.argmax(yb, 1)))
    return losses, accs


def main():
    print("#" * 76)
    print("# ResNet-8 + CIFAR-10：FLOAT32 vs A1（R4，A1 深网安全性）")
    print("#" * 76)
    X, y = load_cifar10(subset=10000)
    y_oh = to_onehot(y)
    print(f"  数据: {X.shape}, y={len(np.unique(y))} 类")
    for st in ["FLOAT32", "A1"]:
        losses, accs = train(st, X, y_oh, T=600, B=32, lr=0.01)
        print(f"  {st:8s}: last20 loss {np.mean(losses[-20:]):.4e}  "
              f"acc {np.mean(accs[-20:]):.3f}  (first {losses[0]:.4e})")
    print("=" * 76)


if __name__ == "__main__":
    main()
