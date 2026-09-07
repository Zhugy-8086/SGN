# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""benchmark_mnist8.py - 8 层经典 CNN（ResNet-8）MNIST 训练速度对比

背景（2026-09-03，mkern 微内核层 R1-R3 落地后的基线刷新）：
    性能白皮书原基线 benchmark_phase5.py 为 6 层 CNN（CIFAR-10），模型规模偏小；
    本脚本把基线升级为仓库既有经典 8 层架构 ResNet-8（conv1 + 3 残差 stage
    × 2 conv + 1×1 shortcut + GAP + fc，7 conv + 1 fc = 8 加权层，全程 BN），
    数据集改用 MNIST（SGN/data/MNIST/raw，28x28x1， ResNet-8 按输入通道=1 适配）。
    计时口径与 benchmark_phase5.py 一致：fwd+bwd 中位数 / 最小值，B=4/8/16。

注意（归因声明）：mkern R1（浮点归约固定树）/R2（dot4_packed）/R3（mkern/gemm）
    均不在本 float 训练主路径上（R1 只改变 sum 系累加语义、性能持平；R2/R3 无
    Python 可见消费方），本脚本用于确认基线未回归并刷新白皮书的模型规模基线。

运行：cd engine/sgn/build && python ../autograd/benchmark_mnist8.py
"""

import os
import sys
import time

# 线程口径（先于 torch/sgn 导入设置）：小 batch（B≤16）下 torch 默认 10 线程的
# 线程池同步开销主导且受宿主省电状态影响剧烈波动（实测同负载 20-145ms），
# 1 线程稳定（~29ms）；SGN 侧 OpenMP 在此规模近似串行。两侧钉单线程 = 同口径。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import sgn

ag = sgn.autograd

DATA_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data'))


# ============================================================================
# MNIST 数据（torchvision 读取 SGN/data/MNIST/raw，无下载）
# ============================================================================
def load_mnist(subset=None, seed=42):
    import torchvision
    ds = torchvision.datasets.MNIST(root=DATA_ROOT, train=True, download=False)
    X = ds.data.numpy().astype(np.float32)[:, None, :, :] / 255.0   # (N,1,28,28)
    mean = np.array([0.1307], np.float32).reshape(1, 1, 1, 1)
    std = np.array([0.3081], np.float32).reshape(1, 1, 1, 1)
    X = (X - mean) / std
    y = ds.targets.numpy()
    idx = np.arange(X.shape[0])
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    if subset is not None and subset < X.shape[0]:
        idx = idx[:subset]
    return X[idx], y[idx]


def make_tensor(a, rg=False):
    t = ag.Tensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    t.requires_grad = rg
    return t


# ============================================================================
# ResNet-8 权重初始化（MNIST 版：输入通道 1，28x28；s2/s3 stride2 下采样）
# ============================================================================
def init_resnet8_mnist(seed=7):
    rng = np.random.default_rng(seed)

    def kaiming(shape, fan_in):
        b = np.sqrt(6.0 / fan_in) * np.sqrt(2.0)
        return rng.uniform(-b, b, shape).astype(np.float32)

    p = {}
    p["c1w"] = kaiming((16, 1, 3, 3), 1 * 9)
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


def sgn_resnet8_forward_backward(x_np, p, dY_np):
    """SGN autograd：ResNet-8（MNIST）前向+反向一次。

    p 为预初始化的权重 dict（调用方一次生成、复用传入）——Tensor 包装属于
    SGN 每次调用的固有 API 开销（与 test_phase5 口径一致），但权重数值初始化
    （Kaiming RNG）不属于训练路径，必须移出计时区，否则对 SGN 不公平。
    """
    ag.clear()
    B = x_np.shape[0]
    bn_stats = {n: (make_tensor(np.zeros(c, np.float32)),
                    make_tensor(np.ones(c, np.float32)))
                for n, c in [("c1", 16), ("s1a", 16), ("s1b", 16), ("s2a", 32),
                             ("s2b", 32), ("s2sc", 32), ("s3a", 64), ("s3b", 64),
                             ("s3sc", 64)]}

    def conv(x, wname, bname, s, pad):
        wt = make_tensor(p[wname], True)
        bt = make_tensor(p[bname], True)
        return ag.conv2d(x, wt, bt, s, pad)

    def bn(x, name):
        gt = make_tensor(p[f"bn_{name}_g"], True)
        bt = make_tensor(p[f"bn_{name}_b"], True)
        rm_t, rv_t = bn_stats[name]
        return ag.batchnorm2d(x, gt, bt, rm_t, rv_t, 0.1, 1e-5)

    x = make_tensor(x_np)
    ag.start_recording()
    try:
        c1 = ag.relu(bn(conv(x, "c1w", "c1b", 1, 1), "c1"))
        s1 = ag.relu(bn(conv(c1, "s1a_w", "s1a_b", 1, 1), "s1a"))
        s1 = bn(conv(s1, "s1b_w", "s1b_b", 1, 1), "s1b")
        h = ag.relu(ag.add(s1, c1))
        s2 = ag.relu(bn(conv(h, "s2a_w", "s2a_b", 2, 1), "s2a"))
        s2 = bn(conv(s2, "s2b_w", "s2b_b", 1, 1), "s2b")
        sc2 = bn(conv(h, "s2sc_w", "s2sc_b", 2, 0), "s2sc")
        h = ag.relu(ag.add(s2, sc2))
        s3 = ag.relu(bn(conv(h, "s3a_w", "s3a_b", 2, 1), "s3a"))
        s3 = bn(conv(s3, "s3b_w", "s3b_b", 1, 1), "s3b")
        sc3 = bn(conv(h, "s3sc_w", "s3sc_b", 2, 0), "s3sc")
        h = ag.relu(ag.add(s3, sc3))
        g = ag.avgpool2d(h, 7, 7)
        flat = ag.reshape(g, [B, -1])
        logits = ag.linear(flat, make_tensor(p["fcw"], True),
                           make_tensor(p["fcb"], True))
    finally:
        ag.stop_recording()
    logits.backward(dY_np)


# ============================================================================
# PyTorch 同构孪生（结构逐层一致：conv/BN/ReLU/残差/GAP/fc）
# ============================================================================
class ResNet8MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=False))
        self.s1a = nn.Sequential(nn.Conv2d(16, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=False))
        self.s1b = nn.Sequential(nn.Conv2d(16, 16, 3, 1, 1), nn.BatchNorm2d(16))
        self.s2a = nn.Sequential(nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=False))
        self.s2b = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32))
        self.s2sc = nn.Sequential(nn.Conv2d(16, 32, 1, 2, 0), nn.BatchNorm2d(32))
        self.s3a = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=False))
        self.s3b = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64))
        self.s3sc = nn.Sequential(nn.Conv2d(32, 64, 1, 2, 0), nn.BatchNorm2d(64))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        c1 = self.c1(x)
        h = torch.relu(self.s1b(self.s1a(c1)) + c1)
        h = torch.relu(self.s2b(self.s2a(h)) + self.s2sc(h))
        h = torch.relu(self.s3b(self.s3a(h)) + self.s3sc(h))
        g = torch.nn.functional.adaptive_avg_pool2d(h, 1)
        return self.fc(torch.flatten(g, 1))


def main():
    print("=== 8 层经典 CNN（ResNet-8）MNIST 训练速度对比 ===")
    print(f"数据: {DATA_ROOT}\\MNIST\\raw（download=False）\n")
    X, y = load_mnist(subset=512)

    torch_model = ResNet8MNIST()
    torch_model.eval()  # 计时期间 BN 走 eval 语义两侧对齐？——否：SGN 侧 batchnorm2d 走
    # train 统计；torch 侧保持 train() 使 BN 语义一致（更新 running stats）。
    torch_model.train()

    print(f"{'B':>4} {'SGN fwd+bwd (ms)':>18} {'PyTorch fwd+bwd (ms)':>22} {'SGN/PyTorch':>12}")
    results = {}
    p = init_resnet8_mnist()   # 权重数值初始化一次，计时区外复用（公平口径）
    for B in (4, 8, 16):
        x_np = X[:B]
        dY_np = (np.random.default_rng(0).standard_normal((B, 10)) * 0.01).astype(np.float32)
        x_t = torch.tensor(x_np)
        dY_t = torch.tensor(dY_np)

        # SGN 计时
        for _ in range(3):
            sgn_resnet8_forward_backward(x_np, p, dY_np)
        t_sgn = []
        for _ in range(10):
            t0 = time.perf_counter()
            sgn_resnet8_forward_backward(x_np, p, dY_np)
            t_sgn.append(time.perf_counter() - t0)

        # PyTorch 计时
        for _ in range(3):
            torch_model.zero_grad()
            torch_model(x_t).backward(dY_t)
        t_pt = []
        for _ in range(10):
            torch_model.zero_grad()
            t0 = time.perf_counter()
            torch_model(x_t).backward(dY_t)
            t_pt.append(time.perf_counter() - t0)

        sgn_ms = float(np.median(t_sgn)) * 1000
        pt_ms = float(np.median(t_pt)) * 1000
        results[B] = (sgn_ms, pt_ms)
        print(f"{B:>4} {sgn_ms:>18.2f} {pt_ms:>22.2f} {sgn_ms / pt_ms:>11.1f}x")

    print("\n结论：8 层经典架构（ResNet-8/MNIST）基线已刷新；"
          "mkern R1-R3 不在 float 训练主路径，本表用于确认基线未回归"
          "（归因声明见脚本头注与性能白皮书 §3）。")


if __name__ == "__main__":
    main()
