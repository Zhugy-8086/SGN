# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""sgn.optim — 优化器

提供极简的优化器实现，让 SGN 可以独立运行，无需依赖 PyTorch 或其他浮点框架。

用法：
    import sgn
    optim = sgn.optim

    model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    optimizer = optim.SGD(model, lr=0.01)

    for step in range(100):
        with ag.record_scope(clear=True):
            y_pred = model([x])
        # 计算损失梯度：MSE 的 dL/dy_pred = (y_pred - y_label) / N
        loss_grad = (y_pred.to_numpy() - y_label) / batch_size
        y_pred.backward(loss_grad)
        optimizer.step()    # 参数更新
        optimizer.zero_grad()  # 清零梯度
"""

from __future__ import annotations

import numpy as np


class SGD:
    """随机梯度下降优化器（极简版）。

    通过 state_dict + load_state_dict 实现参数更新，
    无需修改 Module 内部状态，兼容所有 sgn.nn.Module 子类。

    Args:
        model: sgn.nn.Module 实例
        lr: 学习率（默认 0.01）

    Usage:
        optimizer = SGD(model, lr=0.01)
        optimizer.step()       # 更新所有参数
        optimizer.zero_grad()  # 清零梯度
    """

    def __init__(self, model, lr: float = 0.01):
        self.model = model
        self.lr = lr

    def step(self) -> None:
        """执行一步参数更新：param = param - lr * grad"""
        state = self.model.state_dict()
        for name, p in self.model.named_parameters():
            g = p.grad
            if g is not None:
                # 安全审计 2026-08-16 O1：g 可能为 float64（外部 numpy 运算），
                # state 会被提升为 float64，load_state_dict 的 dtype cast 抛
                # TypeError 训练崩溃——显式回 float32（引擎统一精度）
                state[name] = (state[name] - self.lr * g).astype(np.float32)
        self.model.load_state_dict(state)

    def zero_grad(self) -> None:
        """清零所有参数的梯度"""
        self.model.zero_grad()


__all__ = ["SGD"]