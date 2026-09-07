"""多策略交叉测试：FLOAT32 / GEF / SR 配合 sgn.loss 跑 MNIST

验证 loss 模块在不同反向传播策略下都能正常工作。

运行方式：
    cd SGN
    python examples/test_multi_strategy_mnist.py
"""

import sys
import os
import time
import numpy as np

_sgn_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "engine"))
sys.path.insert(0, _sgn_root)

import sgn

ag = sgn.autograd
nn = sgn.nn


def build_model():
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


def run_training(strategy, model, criterion, x_all, y_all, *, steps=100, lr=0.01):
    """用指定策略训练，返回 (losses, elapsed, grad_ok)"""
    # 设置策略
    ag.set_backward_strategy(strategy)
    if strategy in (ag.BackwardStrategy.GEF, ag.BackwardStrategy.SR):
        ag.set_quant_config(bits=16, clip_sigma=4.0)

    optimizer = sgn.optim.SGD(model, lr=lr)
    losses = []
    has_nan = False

    t0 = time.perf_counter()
    for step in range(steps):
        x_np = x_all[step]
        y_np = y_all[step]

        with ag.record_scope(clear=True):
            x = ag.Tensor.from_numpy(x_np.copy())
            logits = model.forward([x])

        out_np = logits.to_numpy()
        loss, dY = criterion(out_np, y_np)

        if np.isnan(loss) or np.isinf(loss):
            has_nan = True
            break

        logits.backward(dY.astype(np.float32))
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss)

    elapsed = time.perf_counter() - t0

    # 梯度流通检查
    grad_ok = True
    with ag.record_scope():
        x_test = ag.Tensor.from_numpy(np.random.randn(4, 784).astype(np.float32))
        y = model.forward([x_test])
    dY_test = np.random.randn(4, 10).astype(np.float32)
    y.backward(dY_test)
    for name, p in model.named_parameters():
        g = p.grad
        if g is None or float(np.linalg.norm(g)) == 0:
            grad_ok = False
            break

    return losses, elapsed, grad_ok, has_nan


def main():
    print("=" * 60)
    print("多策略交叉测试: FLOAT32 / GEF / SR × sgn.loss")
    print(f"SGN 版本: {sgn.version()}")
    print("=" * 60)

    # 共享配置
    steps = 100
    batch_size = 32
    lr = 0.01

    # 合成数据（固定种子，保证可比性）
    data_rng = np.random.RandomState(42)
    x_all = data_rng.randn(steps, batch_size, 784).astype(np.float32)
    y_all = data_rng.randint(0, 10, (steps, batch_size)).astype(np.int64)

    strategies = [
        ("FLOAT32", ag.BackwardStrategy.FLOAT32, False),
        ("GEF",     ag.BackwardStrategy.GEF,     True),
        ("SR",      ag.BackwardStrategy.SR,      True),
    ]

    results = {}

    for name, strategy, needs_quant in strategies:
        print(f"\n{'─' * 60}")
        print(f"测试策略: {name}")
        print(f"{'─' * 60}")

        # 每个策略用独立模型（从相同初始状态开始）
        np.random.seed(42)
        model = build_model()
        criterion = sgn.loss.CrossEntropyLoss()

        losses, elapsed, grad_ok, has_nan = run_training(
            strategy, model, criterion, x_all, y_all,
            steps=steps, lr=lr
        )

        results[name] = {
            "losses": losses,
            "elapsed": elapsed,
            "grad_ok": grad_ok,
            "has_nan": has_nan,
            "final_loss": losses[-1] if losses else float('nan'),
            "avg_loss_last10": float(np.mean(losses[-10:])) if len(losses) >= 10 else float('nan'),
        }

        status = "OK" if (not has_nan and grad_ok) else "FAIL"
        print(f"  状态: {status}")
        print(f"  耗时: {elapsed:.2f}s")
        print(f"  步数: {len(losses)}/{steps}")
        if losses:
            print(f"  初始 loss: {losses[0]:.4f}")
            print(f"  最终 loss: {losses[-1]:.4f}")
            if len(losses) >= 10:
                print(f"  平均 loss (最后 10 步): {np.mean(losses[-10:]):.4f}")
        if has_nan:
            print(f"  [FAIL] 出现 NaN/Inf!")
        if not grad_ok:
            print(f"  [FAIL] 梯度流通异常!")

    # 汇总对比
    print(f"\n{'=' * 60}")
    print("汇总对比")
    print(f"{'=' * 60}")
    print(f"{'策略':<12} {'状态':<8} {'初始loss':<12} {'最终loss':<12} {'耗时':<10} {'梯度流通'}")
    print(f"{'─' * 60}")

    all_ok = True
    for name in ["FLOAT32", "GEF", "SR"]:
        r = results[name]
        ok = not r["has_nan"] and r["grad_ok"]
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        init_loss = f"{r['losses'][0]:.4f}" if r["losses"] else "N/A"
        final_loss = f"{r['final_loss']:.4f}" if not np.isnan(r['final_loss']) else "N/A"
        grad_str = "OK" if r["grad_ok"] else "FAIL"
        print(f"{name:<12} {status:<8} {init_loss:<12} {final_loss:<12} {r['elapsed']:.2f}s     {grad_str}")

    # 交叉验证：各策略 loss 下降趋势是否合理
    print(f"\n交叉验证:")
    baseline_final = results["FLOAT32"]["final_loss"]
    for name in ["GEF", "SR"]:
        r = results[name]
        if not np.isnan(r["final_loss"]) and not np.isnan(baseline_final):
            ratio = r["final_loss"] / baseline_final
            # GEF/SR 可能略高于或接近 FLOAT32，只要不是数量级差异就算正常
            if ratio < 0.01 or ratio > 100:
                print(f"  [WARN] {name}/FLOAT32 loss ratio = {ratio:.2f} (异常)")
                all_ok = False
            else:
                print(f"  [OK] {name}/FLOAT32 loss ratio = {ratio:.2f}")

    print(f"\n{'=' * 60}")
    print(f"总体结果: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    print(f"{'=' * 60}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())