"""多 epoch 真实 MNIST 训练 — 多策略对比

在真实 MNIST 数据上运行 5 epoch 训练，对比 FLOAT32 / GEF / SR 三种
反向传播策略的收敛曲线和最终准确率。

运行方式：
    cd SGN
    python examples/test_multi_epoch_mnist.py
"""

import sys
import os
import time
import gzip
import struct
import numpy as np

_sgn_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "engine"))
sys.path.insert(0, _sgn_root)

import sgn

ag = sgn.autograd
nn = sgn.nn


# ============================================================================
# MNIST 数据加载（纯 numpy，无 PyTorch 依赖）
# ============================================================================

def _load_mnist_images(path):
    with gzip.open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    return data.astype(np.float32) / 255.0


def _load_mnist_labels(path):
    with gzip.open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def load_mnist(data_dir):
    """加载 MNIST，返回 (x_train, y_train, x_test, y_test)"""
    x_train = _load_mnist_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    y_train = _load_mnist_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    x_test  = _load_mnist_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    y_test  = _load_mnist_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    return x_train, y_train, x_test, y_test


# ============================================================================
# 模型
# ============================================================================

def build_model():
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def accuracy(logits, y_true):
    return float(np.mean(np.argmax(logits, axis=1) == y_true))


# ============================================================================
# 训练
# ============================================================================

def train_one_epoch(model, criterion, optimizer, x_train, y_train, batch_size, *,
                    spot_check_every=0, baseline=None):
    """训练一个 epoch，返回 (avg_loss, accuracy, grad_warnings)"""
    n = len(x_train)
    indices = np.random.permutation(n)
    total_loss = 0.0
    total_correct = 0
    grad_warnings = 0

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        x_np = x_train[batch_idx]
        y_np = y_train[batch_idx]

        with ag.record_scope(clear=True):
            x = ag.Tensor.from_numpy(x_np.copy())
            logits = model.forward([x])

        out_np = logits.to_numpy()
        loss, dY = criterion(out_np, y_np)

        logits.backward(dY.astype(np.float32))
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss * len(batch_idx)
        total_correct += int(np.sum(np.argmax(out_np, axis=1) == y_np))

        # 周期性梯度抽检
        if spot_check_every > 0 and baseline and (start // batch_size) % spot_check_every == 0:
            result = _spot_check(model, criterion, x_np, y_np)
            if result and baseline:
                if result[0] > baseline[0] * 5:
                    grad_warnings += 1

    return total_loss / n, total_correct / n, grad_warnings


def evaluate(model, x_test, y_test, batch_size=256):
    """评估准确率"""
    n = len(x_test)
    all_preds = []
    for start in range(0, n, batch_size):
        x_np = x_test[start:start + batch_size]
        with ag.record_scope():
            x = ag.Tensor.from_numpy(x_np.copy())
            logits = model.forward([x])
        all_preds.append(np.argmax(logits.to_numpy(), axis=1))
    preds = np.concatenate(all_preds)
    return float(np.mean(preds == y_test[:len(preds)]))


def _spot_check(model, criterion, x_np, y_np, eps=1e-2, n_samples=10):
    """快速梯度抽检"""
    from sgn.loss import LossDiagnoser
    diagnoser = LossDiagnoser(model, criterion, x_np, y_np, sgn_module=sgn)
    report = diagnoser.diagnose(modes=['gradient'], eps=eps, n_samples=n_samples)
    for c in report.checks:
        if c.name == 'numerical_gradient' and 'median_diff=' in c.message:
            parts = c.message.split(', ')
            md = float([p for p in parts if 'median_diff=' in p][0].split('=')[1])
            mx = float([p for p in parts if 'max_diff=' in p][0].split('=')[1].split()[0])
            mp = [p for p in parts if 'at ' in p][0].split('at ')[1].split(' ')[0]
            return md, mx, mp
    return None


# ============================================================================
# 主流程
# ============================================================================

def main():
    print("=" * 60)
    print("多 epoch 真实 MNIST 训练 — 多策略对比")
    print(f"SGN 版本: {sgn.version()}")
    print("=" * 60)

    # ---- 数据 ----
    # 安全审计 2026-08-16 A1-1/C2-1：traditional/ 已迁入 legacy/（原路径断裂）
    # 2026-08-16 legacy 独立：MNIST 数据迁至 engine/sgn/tests/refs/data/MNIST/raw
    data_dir = os.path.join(
        os.path.dirname(__file__), "..", "engine", "sgn", "tests", "refs", "data", "MNIST", "raw"
    )
    print(f"\n加载 MNIST: {data_dir}")
    x_train, y_train, x_test, y_test = load_mnist(data_dir)
    print(f"  训练集: {x_train.shape}, 标签: {y_train.shape}")
    print(f"  测试集: {x_test.shape}, 标签: {y_test.shape}")

    # ---- 配置 ----
    epochs = 5
    batch_size = 64
    lr = 0.01
    spot_check_every = 50  # 每 50 个 batch 抽检一次

    strategies = [
        ("FLOAT32", ag.BackwardStrategy.FLOAT32, False),
        ("GEF",     ag.BackwardStrategy.GEF,     True),
        ("SR",      ag.BackwardStrategy.SR,      True),
    ]

    all_results = {}

    for name, strategy, needs_quant in strategies:
        print(f"\n{'─' * 60}")
        print(f"策略: {name}")
        print(f"{'─' * 60}")

        # 独立模型
        np.random.seed(42)
        model = build_model()
        model.train()
        criterion = sgn.loss.CrossEntropyLoss()

        ag.set_backward_strategy(strategy)
        if needs_quant:
            ag.set_quant_config(bits=16, clip_sigma=4.0)

        optimizer = sgn.optim.SGD(model, lr=lr)

        # 基线梯度
        sample_x = x_train[:4]
        sample_y = y_train[:4]
        baseline = _spot_check(model, criterion, sample_x, sample_y)
        if baseline:
            print(f"  基线梯度: median_diff={baseline[0]:.2e}")

        history = []
        t0 = time.perf_counter()

        for epoch in range(epochs):
            avg_loss, train_acc, grad_warns = train_one_epoch(
                model, criterion, optimizer, x_train, y_train, batch_size,
                spot_check_every=spot_check_every, baseline=baseline
            )
            test_acc = evaluate(model, x_test, y_test)
            history.append((epoch + 1, avg_loss, train_acc, test_acc, grad_warns))

            warn_str = f"  grad_warns={grad_warns}" if grad_warns > 0 else ""
            print(f"  epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, "
                  f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}{warn_str}")

        elapsed = time.perf_counter() - t0

        # 梯度流通检查
        with ag.record_scope():
            y = model.forward([ag.Tensor.from_numpy(x_test[:4].copy())])
        dY_test = np.random.randn(4, 10).astype(np.float32)
        y.backward(dY_test)
        grad_ok = all(
            p.grad is not None and float(np.linalg.norm(p.grad)) > 0
            for _, p in model.named_parameters()
        )

        all_results[name] = {
            "history": history,
            "elapsed": elapsed,
            "grad_ok": grad_ok,
            "final_test_acc": history[-1][3],
        }

        print(f"  耗时: {elapsed:.1f}s, 梯度流通: {'OK' if grad_ok else 'FAIL'}")

    # ---- 汇总 ----
    print(f"\n{'=' * 60}")
    print("汇总对比")
    print(f"{'=' * 60}")
    print(f"{'策略':<12} {'最终loss':<12} {'训练acc':<12} {'测试acc':<12} {'耗时':<10} {'梯度流通'}")
    print(f"{'─' * 60}")

    all_ok = True
    for name in ["FLOAT32", "GEF", "SR"]:
        r = all_results[name]
        h = r["history"]
        last = h[-1]
        ok = r["grad_ok"]
        if not ok:
            all_ok = False
        print(f"{name:<12} {last[1]:<12.4f} {last[2]:<12.4f} {last[3]:<12.4f} "
              f"{r['elapsed']:.1f}s     {'OK' if ok else 'FAIL'}")

    # 准确率曲线
    print(f"\n准确率曲线:")
    print(f"  {'Epoch':<8} {'FLOAT32':<12} {'GEF':<12} {'SR':<12}")
    for epoch in range(epochs):
        vals = []
        for name in ["FLOAT32", "GEF", "SR"]:
            vals.append(f"{all_results[name]['history'][epoch][3]:.4f}")
        print(f"  {epoch + 1:<8} {vals[0]:<12} {vals[1]:<12} {vals[2]:<12}")

    print(f"\n{'=' * 60}")
    print(f"总体结果: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    print(f"{'=' * 60}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())