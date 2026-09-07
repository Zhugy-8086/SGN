"""带梯度日志的训练示例 — 演示 GradLogger 的三种动态调整方式

本脚本用 sgn.nn 构建一个简单 MLP，使用 sgn.loss.CrossEntropyLoss 训练，
在训练循环中演示：

  方式 1：运行时动态开关 enabled（热身期开启，稳定后关闭）
  方式 2：采样频率 log_every=N（长训练中每 N 步输出一次）
  方式 3：运行时切换采样频率（调试模式 → 生产模式）

运行方式：
    cd SGN
    python examples/train_with_logging.py
"""

import sys
import os
import time
import numpy as np

# ---- 路径设置 ----
_sgn_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "engine"))
sys.path.insert(0, _sgn_root)

import sgn

ag = sgn.autograd
nn = sgn.nn

np.random.seed(42)


def make_model():
    """3 层 MLP: 784 → 128 → 64 → 10"""
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


def make_data(steps, batch_size):
    """合成数据模拟 MNIST"""
    rng = np.random.RandomState(42)
    x = rng.randn(steps, batch_size, 784).astype(np.float32)
    y = rng.randint(0, 10, (steps, batch_size)).astype(np.int64)
    return x, y


# ============================================================================
# 方式 1：运行时动态开关 enabled
# ============================================================================
def demo_dynamic_toggle():
    """热身期（前 5 步）开启日志观察梯度，之后关闭以全速训练。"""
    print("=" * 70)
    print("方式 1：运行时动态开关 enabled")
    print("=" * 70)

    model = make_model()
    optimizer = sgn.optim.SGD(model, lr=0.01)

    # verbose=True 开启日志，log_every=1 每步都输出
    criterion = sgn.loss.CrossEntropyLoss(verbose=True, log_every=1)

    steps = 20
    batch_size = 32
    x_all, y_all = make_data(steps, batch_size)

    warmup = 5  # 前 5 步看日志

    for step in range(steps):
        x_np = x_all[step]
        y_np = y_all[step]

        # 前向
        with ag.record_scope(clear=True):
            x = ag.Tensor.from_numpy(x_np.copy())
            logits = model.forward([x])

        out_np = logits.to_numpy()

        # 用 sgn.loss 计算损失和梯度（内部触发 GradLogger）
        loss, dY = criterion(out_np, y_np)

        # 反向
        logits.backward(dY.astype(np.float32))
        optimizer.step()
        optimizer.zero_grad()

        # 热身期结束后关闭日志
        if step == warmup - 1:
            criterion._logger.enabled = False
            print(f"  ... 热身期结束，关闭日志，全速训练剩余 {steps - warmup} 步 ...\n")

    print(f"  最终 loss: {loss:.4f}\n")


# ============================================================================
# 方式 2：采样频率 log_every=N
# ============================================================================
def demo_sampling_frequency():
    """100 步训练，每 20 步输出一次日志，避免日志刷屏。"""
    print("=" * 70)
    print("方式 2：采样频率 log_every=N（每 20 步输出一次）")
    print("=" * 70)

    model = make_model()
    optimizer = sgn.optim.SGD(model, lr=0.01)

    # log_every=20：每 20 次调用输出一次
    criterion = sgn.loss.CrossEntropyLoss(verbose=True, log_every=20)

    steps = 100
    batch_size = 32
    x_all, y_all = make_data(steps, batch_size)

    for step in range(steps):
        x_np = x_all[step]
        y_np = y_all[step]

        with ag.record_scope(clear=True):
            x = ag.Tensor.from_numpy(x_np.copy())
            logits = model.forward([x])

        out_np = logits.to_numpy()
        loss, dY = criterion(out_np, y_np)

        logits.backward(dY.astype(np.float32))
        optimizer.step()
        optimizer.zero_grad()

    print(f"\n  100 步训练完成，仅输出 5 条日志（step 20/40/60/80/100）")
    print(f"  最终 loss: {loss:.4f}\n")


# ============================================================================
# 方式 3：运行时切换采样频率（调试 → 生产）
# ============================================================================
def demo_switch_frequency():
    """前 10 步逐步输出（调试模式），之后切换为每 50 步输出（生产模式）。"""
    print("=" * 70)
    print("方式 3：运行时切换采样频率（调试 → 生产）")
    print("=" * 70)

    model = make_model()
    optimizer = sgn.optim.SGD(model, lr=0.01)

    # 初始：调试模式，每步输出
    criterion = sgn.loss.CrossEntropyLoss(verbose=True, log_every=1)

    steps = 110
    batch_size = 32
    x_all, y_all = make_data(steps, batch_size)

    debug_steps = 10  # 前 10 步逐步输出

    for step in range(steps):
        x_np = x_all[step]
        y_np = y_all[step]

        with ag.record_scope(clear=True):
            x = ag.Tensor.from_numpy(x_np.copy())
            logits = model.forward([x])

        out_np = logits.to_numpy()
        loss, dY = criterion(out_np, y_np)

        logits.backward(dY.astype(np.float32))
        optimizer.step()
        optimizer.zero_grad()

        # 调试结束后切换到生产模式
        if step == debug_steps - 1:
            criterion._logger.log_every = 50
            criterion._logger._call_count = 0  # 重置计数器
            print(f"  ... 调试期结束，切换 log_every=50，进入生产模式 ...\n")

    print(f"  最终 loss: {loss:.4f}\n")


# ============================================================================
# 附加：优化器侧独立复用 GradLogger
# ============================================================================
def demo_optimizer_logging():
    """在优化器侧用独立 GradLogger 记录参数梯度，与 loss 日志互补。"""
    print("=" * 70)
    print("附加：优化器侧独立复用 GradLogger")
    print("=" * 70)

    model = make_model()
    optimizer = sgn.optim.SGD(model, lr=0.01)

    criterion = sgn.loss.CrossEntropyLoss(verbose=False)  # loss 侧静默
    opt_logger = sgn.logger.GradLogger(tag="OptimLog", enabled=True, log_every=5)

    steps = 25
    batch_size = 32
    x_all, y_all = make_data(steps, batch_size)

    for step in range(steps):
        x_np = x_all[step]
        y_np = y_all[step]

        with ag.record_scope(clear=True):
            x = ag.Tensor.from_numpy(x_np.copy())
            logits = model.forward([x])

        out_np = logits.to_numpy()
        loss, dY = criterion(out_np, y_np)

        logits.backward(dY.astype(np.float32))

        # 优化器侧：记录关键层的权重梯度
        for name, p in model.named_parameters():
            if p.grad is not None:
                opt_logger.log_tensor(name, p.grad, step=step)

        optimizer.step()
        optimizer.zero_grad()

    print(f"\n  25 步训练完成，优化器侧每 5 步输出一次参数梯度统计\n")


# ============================================================================
# 主入口
# ============================================================================
def main():
    print("=" * 70)
    print("SGN 梯度日志训练示例 — GradLogger 三种动态调整方式")
    print(f"SGN 版本: {sgn.version()}")
    print("=" * 70)
    print()

    demo_dynamic_toggle()
    demo_sampling_frequency()
    demo_switch_frequency()
    demo_optimizer_logging()

    print("=" * 70)
    print("总结")
    print("=" * 70)
    print("""
    GradLogger 三种动态调整方式：

    1. 运行时开关 enabled
       criterion._logger.enabled = False  # 立即关闭
       criterion._logger.enabled = True   # 立即开启
       适用：热身期观察，稳定后全速训练

    2. 采样频率 log_every=N
       criterion = CrossEntropyLoss(verbose=True, log_every=20)
       适用：长训练循环，每 N 步采样一次

    3. 运行时切换频率
       criterion._logger.log_every = 50
       criterion._logger._call_count = 0  # 重置计数器
       适用：调试模式 → 生产模式无缝切换

    附加：优化器侧独立复用
       opt_logger = sgn.logger.GradLogger(tag="OptimLog", enabled=True, log_every=5)
       opt_logger.log_tensor(name, grad)
       适用：与 loss 日志互补，监控参数更新健康度
    """)


if __name__ == "__main__":
    main()
