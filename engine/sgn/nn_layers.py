# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""sgn.nn_layers — 标准 Module 子类

提供类 PyTorch 的标准层，使用 sgn.nn.Module 基类构建。
每个类在 __init__ 中自动创建和注册参数，forward 调用 C++ autograd 算子。

用法：
    import sgn
    nn = sgn.nn

    # 单层
    fc = nn.Linear(784, 256)
    conv = nn.Conv2d(3, 32, 3, stride=1, padding=1)
    bn = nn.BatchNorm2d(32)

    # 组合
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
"""

from __future__ import annotations

import numpy as np
import math as _math

# 用相对导入代替顶层 `import sgn`，避免在 `import engine.sgn`
# 身份下（sys.path 不含 engine/）破坏加载。
from . import nn
from . import autograd as ag


class Linear(nn.Module):
    """全连接层: Y = X @ W^T + b

    Args:
        in_features: 输入维度
        out_features: 输出维度
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        w = nn.Parameter([out_features, in_features])
        if in_features <= 0:
            raise ValueError(f"in_features 必须为正，得到 {in_features}")
        bound = _math.sqrt(6.0 / in_features) * _math.sqrt(2.0)
        nn.uniform_(w.tensor(), -bound, bound)
        self.register_parameter("weight", w)

        b = nn.Parameter([out_features])
        nn.fill_(b.tensor(), 0.0)
        self.register_parameter("bias", b)

    def forward(self, inputs: list) -> ag.Tensor:
        x = inputs[0]
        return ag.linear(x, self.weight.tensor(), self.bias.tensor())


class Conv2d(nn.Module):
    """2D 卷积层: Y = Conv2d(X, W) + b

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步幅（默认 1）
        padding: 填充（默认 0）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        k = kernel_size
        w = nn.Parameter([out_channels, in_channels, k, k])
        fan_in = in_channels * k * k
        if fan_in <= 0:
            raise ValueError(
                f"fan_in 必须为正（in_channels={in_channels}, k={k}）")
        bound = _math.sqrt(6.0 / fan_in) * _math.sqrt(2.0)
        nn.uniform_(w.tensor(), -bound, bound)
        self.register_parameter("weight", w)

        b = nn.Parameter([out_channels])
        nn.fill_(b.tensor(), 0.0)
        self.register_parameter("bias", b)

    def forward(self, inputs: list) -> ag.Tensor:
        x = inputs[0]
        return ag.conv2d(
            x, self.weight.tensor(), self.bias.tensor(),
            self.stride, self.padding,
        )


class ReLU(nn.Module):
    """ReLU 激活函数: Y = max(0, X)

    无参数，仅封装 autograd.relu 算子。
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: list) -> ag.Tensor:
        return ag.relu(inputs[0])


class MaxPool2d(nn.Module):
    """2D 最大池化

    Args:
        kernel_size: 池化窗口大小
        stride: 步幅（默认等于 kernel_size）
    """

    def __init__(self, kernel_size: int, stride: int = -1):
        super().__init__()
        self.kernel_size = kernel_size
        self._stride = stride

    def forward(self, inputs: list) -> ag.Tensor:
        return ag.maxpool2d(inputs[0], self.kernel_size, self._stride)


class BatchNorm2d(nn.Module):
    """2D BatchNorm（训练模式）

    Args:
        num_features: 特征通道数
        momentum: 滑动平均动量（默认 0.1）
        eps: 数值稳定性（默认 1e-5）
    """

    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        gamma = nn.Parameter([num_features])
        nn.fill_(gamma.tensor(), 1.0)
        self.register_parameter("gamma", gamma)

        beta = nn.Parameter([num_features])
        nn.fill_(beta.tensor(), 0.0)
        self.register_parameter("beta", beta)

        rm = nn.Buffer([num_features])
        nn.fill_(rm.tensor(), 0.0)
        self.register_buffer("running_mean", rm)

        rv = nn.Buffer([num_features])
        nn.fill_(rv.tensor(), 1.0)
        self.register_buffer("running_var", rv)

    def forward(self, inputs: list) -> ag.Tensor:
        x = inputs[0]
        return ag.batchnorm2d(
            x,
            self.gamma.tensor(), self.beta.tensor(),
            self.running_mean.tensor(), self.running_var.tensor(),
            self.momentum, self.eps,
        )


class Sequential(nn.Module):
    """顺序容器，按顺序执行各层。

    Args:
        *layers: 可变数量的 Module 子类实例

    Example:
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
    """

    def __init__(self, *layers):
        super().__init__()
        # 安全审计 2026-08-16 P2/B3：_layers（Python list）与 C++ 侧
        # register_module 的 children_ 双重存储——构造后请勿直接改 _layers
        # （不会同步 children_，state_dict/named_parameters 将不一致）；
        # 统一入口约束：仅在 __init__ 时构建一次
        self._layers = list(layers)
        for i, layer in enumerate(self._layers):
            self.register_module(str(i), layer)

    def forward(self, inputs: list) -> ag.Tensor:
        x = inputs[0]
        for layer in self._layers:
            x = layer.forward([x])
        return x


# 导出列表
__all__ = [
    "Linear",
    "Conv2d",
    "ReLU",
    "MaxPool2d",
    "BatchNorm2d",
    "Sequential",
]


class Dropout(nn.Module):
    """Inverted Dropout：训练期随机置零并反缩放，推理期恒等。

    语义（与 PyTorch 一致）：训练期每个元素以概率 p 置零，存留元素乘
    1/(1-p)（期望不变）；eval 期恒等。梯度经 mul 算子自动回传——
    存活位置 dX = dY/(1-p)，置零位置 dX = 0。

    落位说明（模块化）：依赖 mul 算子（autograd mul，基础设施批次 3）。

    参数：
        p: 置零概率 ∈ [0, 1)
        rng: 可选 np.random.Generator（可复现；默认每次 forward 新建——
             需要可复现时传入固定 seed 的 Generator）
    """

    def __init__(self, p: float = 0.5, rng: np.random.Generator | None = None):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout p must be in [0, 1), got {p}")
        self.p = float(p)
        self._rng = rng

    def forward(self, inputs: list) -> ag.Tensor:
        x = inputs[0]
        if not self.training or self.p == 0.0:
            return x
        rng = self._rng if self._rng is not None else np.random.default_rng()
        keep = 1.0 - self.p
        mask = (rng.random(x.to_numpy().shape) >= self.p) / keep
        mask_t = ag.Tensor(np.ascontiguousarray(mask, dtype=np.float32))
        return ag.mul(x, mask_t)


class LayerNorm(nn.Module):
    """LayerNorm：沿末维 C 归一化（逐样本），gamma/beta 可学习。

    Y = (X - mean) / sqrt(var + eps) * gamma + beta（X 为 (B, C)）
    依赖 autograd layernorm 算子（ops_norm.cpp，归一化族独立文件）。

    参数：
        C: 归一化维度大小
        eps: 方差稳定项（默认 1e-5）
    """

    def __init__(self, C: int, eps: float = 1e-5):
        super().__init__()
        if C <= 0:
            raise ValueError(f"C 必须为正，得到 {C}")
        self.C = int(C)
        self.eps = float(eps)

        g = nn.Parameter([C])
        nn.fill_(g.tensor(), 1.0)
        self.register_parameter("gamma", g)

        b = nn.Parameter([C])
        nn.fill_(b.tensor(), 0.0)
        self.register_parameter("beta", b)

    def forward(self, inputs: list) -> ag.Tensor:
        x = inputs[0]
        return ag.layernorm(x, self.gamma.tensor(), self.beta.tensor(), self.eps)
