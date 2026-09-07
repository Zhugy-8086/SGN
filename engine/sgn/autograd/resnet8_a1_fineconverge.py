# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""ResNet-8 + 合成回归：FLOAT32 vs A1 精细收敛地板测量（R5，可控回归地板）

背景
----
R4（resnet8_a1_compare.py）在 CIFAR-10 上验证 A1 深网安全：loss 8.9e-2，但该 loss 远高于
8bit 量化地板（~1e-4），看不见地板在哪儿。R5 采用【可控回归地板】方案：

方案（可控回归地板，R5-A 主线）
----
- 输入：低频光滑场 X (N,3,32,32)——每通道叠加 4 个低频正弦模式，值域 ~[-1,1]。
  有效自由度低 → 深度回归可收敛到极小 loss（无优化地板）。
- 目标：冻结教师 ResNet-8（同架构、独立种子）的 float32 输出 y (N,10)。
  学生与教师同架构 → 目标必然落在可表示空间内（学生可精确复制教师权重）→
  FLOAT32 可收敛到 ~1e-6，与 A1 的 8bit 地板形成干净对照。
- 训练：全批量（B=N=128，BN 批量统计稳定）+ SGD 动量 0.9 + 阶梯 LR 衰减。
- 判据：FLOAT32 last loss < 1e-3（建立"无优化地板"）；A1 是否停在 ~1e-4（前向 8bit 地板）。
- 多种子（R5-B）：冻结教师（seed_teacher=99 固定），学生换 init+SR 种子，测地板抖动/均值。

A1 语义（与 resnet8_a1_compare.py / validate_a1_mainline.py 一致）：
  - 权重：每步手动 quant8（numpy 侧）
  - 激活：conv2d/linear 走 engine 的 STE 前向（量化输出）
  - 反向：engine BackwardStrategy::A1（SR 梯度量化）
  - BN：gamma/beta 不量化，running stats 持久 Tensor 跨步累积（训练用批量统计）

运行：cd engine/sgn/autograd && python resnet8_a1_fineconverge.py [steps]
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
# 合成回归数据：低频光滑场 + 冻结教师目标
# ============================================================================
def make_lowfreq_fields(N=512, seed=11):
    """低频光滑场：每通道 4 个低频正弦模式叠加，值域 ~[-1,1]。"""
    rng = np.random.default_rng(seed)
    hh = np.arange(32, dtype=np.float32)[:, None]   # (32,1)
    ww = np.arange(32, dtype=np.float32)[None, :]   # (1,32)
    X = np.zeros((N, 3, 32, 32), np.float32)
    for n in range(N):
        for c in range(3):
            f = np.zeros((32, 32), np.float32)
            for _ in range(4):
                fx = int(rng.integers(1, 5))
                fy = int(rng.integers(1, 5))
                a = float(rng.uniform(0.5, 1.0))
                ph = float(rng.uniform(0.0, 2 * np.pi))
                f += a * np.sin(2 * np.pi * fx * hh / 32 + 2 * np.pi * fy * ww / 32 + ph)
            X[n, c] = f / 4.0
    return X


def teacher_targets(X, seed_teacher=99):
    """冻结教师（同架构、独立种子）float32 前向输出作为回归目标 y (N,10)。"""
    p_t = init_resnet8(seed_teacher)
    bn_t = make_bn_stats(p_t)
    logits, _ = resnet8_forward(X, p_t, quant_fwd=False, bn_stats=bn_t)
    return logits.to_numpy()


# ============================================================================
# ResNet-8（CIFAR 版，同 R4）：conv1 + 3×(残差块) + 全局池化 + fc
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
    for name, c in [("c1", 16), ("s1a", 16), ("s1b", 16), ("s2a", 32),
                    ("s2b", 32), ("s2sc", 32), ("s3a", 64), ("s3b", 64),
                    ("s3sc", 64)]:
        p[f"bn_{name}_g"] = np.ones(c, np.float32)
        p[f"bn_{name}_b"] = np.zeros(c, np.float32)
    return p


def resnet8_forward(X, p, quant_fwd, bn_stats):
    """X: (N,3,32,32) numpy。返回 (logits Tensor, params_tensors 注册表)。"""
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
    # 全局平均池化（GAP）+ fc
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
# 训练（全批量 + 动量 + 阶梯 LR）
# ============================================================================
def train(strategy, X, y, T, lr0=0.02, momentum=0.9, decay_every=500, seed=7):
    p = init_resnet8(seed)
    bn_stats = make_bn_stats(p)
    quant_fwd = (strategy == "A1")
    ag.set_backward_strategy(getattr(ag.BackwardStrategy, strategy))
    ag.set_sr_seed(1234 + seed)
    N = X.shape[0]
    vel = {k: np.zeros_like(v) for k, v in p.items()}
    lr = lr0
    losses = np.zeros(T)
    import time
    t0 = time.time()
    for t in range(T):
        if t % decay_every == 0 and t > 0:
            lr *= 0.5
        ag.clear()
        ag.start_recording()
        logits, params = resnet8_forward(X, p, quant_fwd, bn_stats)
        ag.stop_recording()
        logits_np = logits.to_numpy()
        grad = 2.0 * (logits_np - y) / N
        logits.backward(grad)
        for key, t_ in params.items():
            vel[key] = momentum * vel[key] - lr * t_.grad
            p[key] = p[key] + vel[key]
        losses[t] = float(np.mean((logits_np - y) ** 2))
        if (t + 1) % 100 == 0:
            el = time.time() - t0
            print(f"      [{strategy}] step {t+1:5d}  loss {losses[t]:.4e}  "
                  f"lr {lr:.5f}  ({el:.0f}s)", flush=True)
    return losses


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [7]
    N, K = 128, 10
    print("#" * 76)
    print("# ResNet-8 + 合成回归：FLOAT32 vs A1 精细收敛地板（R5，可控回归地板）")
    print("#" * 76)
    print(f"  输入: ({N},3,32,32) 低频光滑场  目标: ({N},{K}) 冻结教师输出")
    print(f"  种子: {seeds}（教师 seed=99 固定，学生 init+SR 换种子）")
    X = make_lowfreq_fields(N)
    y = teacher_targets(X)
    print(f"  目标 y: min {y.min():.4f} max {y.max():.4f} std {y.std():.4f}")
    for st in ["FLOAT32", "A1"]:
        mins, lasts = [], []
        for sd in seeds:
            losses = train(st, X, y, T=T, seed=sd)
            mn = float(np.min(losses))
            last = float(np.mean(losses[-50:]))
            mins.append(mn)
            lasts.append(last)
            print(f"  [{st} seed={sd:2d}] min {mn:.4e}  last50 {last:.4e}")
        if len(seeds) > 1:
            print(f"  {st:8s} {len(seeds)} seeds: "
                  f"min {np.mean(mins):.3e}±{np.std(mins):.1e}  "
                  f"last50 {np.mean(lasts):.3e}±{np.std(lasts):.1e}")
    print("=" * 76)


if __name__ == "__main__":
    main()
