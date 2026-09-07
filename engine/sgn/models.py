# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""sgn.models — 稳定模式（nn.Module）预定义模型

提供常见的 MLP 和 CNN 模型封装，开箱即用。
这些模型是 nn.Module 子类，参数自动管理，支持 forward/backward。

用法：
    import sgn
    ag = sgn.autograd

    # MLP 示例（MNIST: 784 → 128 → 64 → 10）
    model = sgn.models.MLP(input_size=784, hidden1=128, hidden2=64, num_classes=10)
    x = ag.Tensor.from_numpy(x_np)
    ag.start_recording()
    y = model.forward([x])
    ag.stop_recording()
    y.backward(dY_np)

    # CNN4 示例（CIFAR-10: 32×32×3 → 10）
    model = sgn.models.CNN4(in_channels=3, img_size=32, num_classes=10)
"""

from __future__ import annotations

import math as _math

# 用相对导入代替顶层 `import sgn`，使本模块在 `import engine.sgn`
# 身份下（sys.path 不含 engine/）也能正常解析，避免破坏 models 子模块加载。
from . import nn
from . import autograd as ag


class MLP(nn.Module):
    """标准 3 层 MLP（稳定模式）。

    架构：input_size → hidden1 → hidden2 → num_classes
    每层之间用 ReLU 激活（最后一层无激活）。

    参考：legacy/traditional/baseline/mlp_mnist.py, class MLP

    Args:
        input_size: 输入维度（MNIST = 784, CIFAR-10 flattened = 3072）
        hidden1: 第 1 个隐藏层大小（默认 128）
        hidden2: 第 2 个隐藏层大小（默认 64）
        num_classes: 输出类别数（默认 10）
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden1: int = 128,
        hidden2: int = 64,
        num_classes: int = 10,
    ):
        super().__init__()

        # fc1: input_size → hidden1
        self._reg_param("fc1_w", [hidden1, input_size])
        self._reg_param("fc1_b", [hidden1])

        # fc2: hidden1 → hidden2
        self._reg_param("fc2_w", [hidden2, hidden1])
        self._reg_param("fc2_b", [hidden2])

        # fc3: hidden2 → num_classes
        self._reg_param("fc3_w", [num_classes, hidden2])
        self._reg_param("fc3_b", [num_classes])

    def _reg_param(self, name: str, shape: list[int]) -> None:
        """创建并注册参数，用 Kaiming 均匀初始化。"""
        p = nn.Parameter(shape)
        # Kaiming uniform 初始化（gain=sqrt(2) 适配 ReLU）。
        # 安全审计 2026-08-16 P1：fan_in <= 0 显式报错（原实现除零崩溃）
        fan_in = shape[1] if len(shape) >= 2 else shape[0]
        if fan_in <= 0:
            raise ValueError(f"参数 {name} 的 fan_in 必须为正，shape={shape}")
        bound = _math.sqrt(6.0 / fan_in) * _math.sqrt(2.0)
        nn.uniform_(p.tensor(), -bound, bound)
        self.register_parameter(name, p)

    def forward(self, inputs: list) -> ag.Tensor:
        """前向传播。

        Args:
            inputs: [x]，x.shape = (B, input_size)

        Returns:
            y.shape = (B, num_classes)
        """
        x = inputs[0]

        # Layer 1: Linear(hidden1) → ReLU
        x = ag.linear(x, self.fc1_w.tensor(), self.fc1_b.tensor())
        x = ag.relu(x)

        # Layer 2: Linear(hidden2) → ReLU
        x = ag.linear(x, self.fc2_w.tensor(), self.fc2_b.tensor())
        x = ag.relu(x)

        # Layer 3: Linear(num_classes) — 无激活，接 CrossEntropyLoss
        x = ag.linear(x, self.fc3_w.tensor(), self.fc3_b.tensor())

        return x


class CNN4(nn.Module):
    """4 层卷积神经网络（稳定模式，CIFAR-10 规模）。

    架构：
        Conv2d(in_channels, 32, k=3, p=1) → ReLU → MaxPool(2)
        Conv2d(32, 64, k=3, p=1) → ReLU → MaxPool(2)
        Flatten
        Linear(64 * (H/4)², 256) → ReLU
        Linear(256, num_classes)

    参考：engine/sgn/tests/architecture/test_dual_mode.py, class ConvNet

    Args:
        in_channels: 输入通道数（CIFAR-10 = 3, MNIST = 1）
        img_size: 输入图像尺寸（CIFAR-10 = 32, MNIST = 28）
        num_classes: 输出类别数（默认 10）
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 32,
        num_classes: int = 10,
    ):
        super().__init__()

        # 两次 MaxPool(2) 后尺寸变成 H/4
        h_after_pool = img_size // 4
        fc_in = 64 * h_after_pool * h_after_pool

        # Conv1: in_channels → 32, k=3, p=1
        self._reg_param("conv1_w", [32, in_channels, 3, 3])
        self._reg_param("conv1_b", [32])

        # Conv2: 32 → 64, k=3, p=1
        self._reg_param("conv2_w", [64, 32, 3, 3])
        self._reg_param("conv2_b", [64])

        # FC1: 64 * (H/4)² → 256
        self._reg_param("fc1_w", [256, fc_in])
        self._reg_param("fc1_b", [256])

        # FC2: 256 → num_classes
        self._reg_param("fc2_w", [num_classes, 256])
        self._reg_param("fc2_b", [num_classes])

    def _reg_param(self, name: str, shape: list[int]) -> None:
        """创建并注册参数，用 Kaiming 均匀初始化。"""
        p = nn.Parameter(shape)
        # Kaiming uniform 初始化（gain=sqrt(2) 适配 ReLU）。
        # 安全审计 2026-08-16 P1：fan_in <= 0 显式报错（原实现除零崩溃）
        fan_in = shape[1] if len(shape) >= 2 else shape[0]
        if fan_in <= 0:
            raise ValueError(f"参数 {name} 的 fan_in 必须为正，shape={shape}")
        bound = _math.sqrt(6.0 / fan_in) * _math.sqrt(2.0)
        nn.uniform_(p.tensor(), -bound, bound)
        self.register_parameter(name, p)

    def forward(self, inputs: list) -> ag.Tensor:
        """前向传播。

        Args:
            inputs: [x]，x.shape = (B, in_channels, img_size, img_size)

        Returns:
            y.shape = (B, num_classes)
        """
        x = inputs[0]

        # Layer 1: Conv2d(32, k=3, p=1) → ReLU → MaxPool(2)
        x = ag.conv2d(x, self.conv1_w.tensor(), self.conv1_b.tensor(), 1, 1)
        x = ag.relu(x)
        x = ag.maxpool2d(x, 2, 2)

        # Layer 2: Conv2d(64, k=3, p=1) → ReLU → MaxPool(2)
        x = ag.conv2d(x, self.conv2_w.tensor(), self.conv2_b.tensor(), 1, 1)
        x = ag.relu(x)
        x = ag.maxpool2d(x, 2, 2)

        # Flatten
        x = ag.reshape(x, [x.shape[0], -1])

        # Layer 3: Linear(256) → ReLU
        x = ag.linear(x, self.fc1_w.tensor(), self.fc1_b.tensor())
        x = ag.relu(x)

        # Layer 4: Linear(num_classes) — 无激活
        x = ag.linear(x, self.fc2_w.tensor(), self.fc2_b.tensor())

        return x