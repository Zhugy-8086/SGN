"""
MLP + MNIST 浮点 baseline

标准 PyTorch 实现，不使用任何 SGN API。
作为后续阶段 1.1~1.4 整数版本的对照基准。

架构：784 → 128 → 64 → 10
训练：SGD + momentum，5 epoch，学习率 0.01
预期准确率：~97%（5 epoch，CPU）

用法：
    python baseline/mlp_mnist.py                    # 默认 5 epoch
    python baseline/mlp_mnist.py --epochs 10        # 跑 10 epoch
    python baseline/mlp_mnist.py --quick            # 快速验证模式（1 epoch，小批量数据）
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# 将 baseline 目录加入 path 以便导入 common
BASELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASELINE_DIR))

from common.data_loader import load_dataset, get_dataset_info
from common.metrics import MetricsRecorder, Timer, evaluate_model


class MLP(nn.Module):
    """标准 MLP：784 → 128 → 64 → 10

    用 ReLU 激活，最后一层无激活（接 CrossEntropyLoss）。
    """

    def __init__(self, input_size: int = 784, hidden1: int = 128, hidden2: int = 64, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.01,
    momentum: float = 0.9,
    seed: int = 42,
    output_dir: str = "baseline/results",
    quick: bool = False,
    sample_activation_grad: bool = True,
):
    """训练 baseline MLP。

    Args:
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        momentum: SGD momentum
        seed: 随机种子
        output_dir: 结果输出目录
        quick: 快速模式（1 epoch，小批次，仅验证流程通畅）
        sample_activation_grad: 是否采样激活/梯度分布
    """
    # 固定随机种子
    torch.manual_seed(seed)

    # 快速模式覆盖
    if quick:
        epochs = 1
        batch_size = 32
        print("[QUICK MODE] 1 epoch, batch_size=32")

    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Threads: {torch.get_num_threads()}")

    # 加载数据
    print("\n=== Loading MNIST ===")
    with Timer() as t:
        train_loader, val_loader, test_loader = load_dataset(
            "mnist", batch_size=batch_size, val_ratio=0.1, seed=seed
        )
    print(f"Data loaded in {t.elapsed:.1f}s")
    info = get_dataset_info("mnist")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Input size: {info['input_size']}, Num classes: {info['num_classes']}")

    # 模型
    model = MLP(
        input_size=info["input_size"],
        hidden1=128,
        hidden2=64,
        num_classes=info["num_classes"],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== Model ===")
    print(f"  MLP: {info['input_size']} -> 128 -> 64 -> {info['num_classes']}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Optimizer: SGD(lr={lr}, momentum={momentum})")
    print(f"  Loss: CrossEntropyLoss")

    # 指标记录器
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    recorder = MetricsRecorder(
        experiment_name="baseline_mlp_mnist",
        output_dir=str(output_path),
    )
    recorder.set_meta(
        model="MLP_784_128_64_10",
        dataset="MNIST",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
        seed=seed,
        optimizer="SGD",
        quick_mode=quick,
        n_params=n_params,
        device=str(device),
        torch_version=torch.__version__,
    )

    # 训练循环
    print(f"\n=== Training ({epochs} epochs) ===")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            recorder.record_batch(loss.item())
            _, predicted = output.max(1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

            # 打印进度
            if (batch_idx + 1) % 200 == 0 or batch_idx == 0:
                print(
                    f"  Epoch {epoch+1}/{epochs} "
                    f"[{batch_idx+1}/{len(train_loader)} "
                    f"({100.*(batch_idx+1)/len(train_loader):.0f}%)] "
                    f"loss={loss.item():.4f}"
                )

        epoch_time = time.time() - epoch_start
        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total

        # 验证集评估
        val_acc, val_loss = evaluate_model(model, val_loader, criterion, device)

        # 记录当前 lr（本 baseline 用固定 lr，记录供后续对比）
        current_lr = optimizer.param_groups[0]["lr"]

        recorder.record_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            epoch_time=epoch_time,
            lr=current_lr,
        )

        # 采样激活/梯度分布（每 2 个 epoch 采一次，或最后一 epoch）
        if sample_activation_grad and ((epoch + 1) % 2 == 0 or epoch == epochs - 1):
            recorder.record_activation_stats(epoch + 1, model)
            recorder.record_gradient_stats(epoch + 1, model)

        # 内存
        mem = recorder.measure_memory()

        print(
            f"  Epoch {epoch+1} done: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"time={epoch_time:.1f}s mem={mem:.0f}MB"
        )

    # 测试集评估
    print(f"\n=== Final Test ===")
    test_acc, test_loss = evaluate_model(model, test_loader, criterion, device)
    recorder.record_final_test(test_acc, test_loss)
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Test loss: {test_loss:.4f}")

    # 保存指标
    metrics_path = recorder.save_json()
    print(f"\n=== Metrics saved to: {metrics_path} ===")

    # 保存模型
    model_path = output_path / "baseline_mlp_mnist.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_size": info["input_size"],
                "hidden1": 128,
                "hidden2": 64,
                "num_classes": info["num_classes"],
            },
            "test_acc": test_acc,
            "test_loss": test_loss,
            "epochs": epochs,
        },
        model_path,
    )
    print(f"=== Model saved to: {model_path} ===")

    # 打印总结
    summary = recorder.to_dict()["summary"]
    print(f"\n=== Summary ===")
    print(f"  Best val acc: {summary['best_val_acc']:.4f} (epoch {summary['best_val_epoch']})")
    print(f"  Test acc: {summary['final_test_acc']:.4f}")
    print(f"  Total train time: {summary['total_train_time_s']:.1f}s")
    print(f"  Avg epoch time: {summary['avg_epoch_time_s']:.1f}s")
    print(f"  Peak memory: {summary['peak_memory_mb']:.0f}MB")

    return test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP + MNIST baseline")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output-dir", type=str, default="baseline/results")
    parser.add_argument("--quick", action="store_true", help="快速验证模式")
    parser.add_argument(
        "--no-sampling", action="store_true", help="不采样激活/梯度分布（加速）"
    )
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        output_dir=args.output_dir,
        quick=args.quick,
        sample_activation_grad=not args.no_sampling,
    )
