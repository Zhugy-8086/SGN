"""CNN + CIFAR-10 浮点 baseline

标准 PyTorch 实现，不使用任何 SGN API。
作为后续阶段 2.1~2.4 整数版本的对照基准。

架构（CIFAR-10）：
    Input: (3, 32, 32)
    ├── Conv2d(3, 32, k=3, p=1) → BN(32) + ReLU → MaxPool2d(2)  → (32, 16, 16)
    ├── Conv2d(32, 64, k=3, p=1) → BN(64) + ReLU → MaxPool2d(2) → (64, 8, 8)
    ├── Flatten → (4096,)
    ├── Linear(4096, 128) + ReLU
    └── Linear(128, 10)

参数量: ~530K
预期 test_acc: ~75% (CIFAR-10, 5 epoch, CPU)

用法：
    python baseline/cnn_cifar10.py                 # 默认 5 epoch
    python baseline/cnn_cifar10.py --epochs 10     # 跑 10 epoch
    python baseline/cnn_cifar10.py --quick         # 快速验证模式（1 epoch，小批量数据）
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


class CNNCifar10(nn.Module):
    """CNN for CIFAR-10: 两个 Conv 块 + 两个 FC 层

    Conv 块: Conv2d → BatchNorm2d → ReLU → MaxPool2d
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 卷积块 1: (3, 32, 32) → (32, 16, 16)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # 卷积块 2: (32, 16, 16) → (64, 8, 8)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 全连接层: (64*8*8=4096) → 128 → 10
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # x: (B, 3, 32, 32)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # (B, 32, 16, 16)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # (B, 64, 8, 8)
        x = x.view(x.size(0), -1)  # (B, 4096)
        x = self.relu(self.fc1(x))  # (B, 128)
        x = self.fc2(x)  # (B, 10)
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
    """训练 baseline CNN (CIFAR-10)。

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
    torch.manual_seed(seed)

    if quick:
        epochs = 1
        batch_size = 32
        print("[QUICK MODE] 1 epoch, batch_size=32")

    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Threads: {torch.get_num_threads()}")

    # 加载数据（CNN 用 flatten=False，保留图像形状）
    print("\n=== Loading CIFAR-10 ===")
    with Timer() as t:
        train_loader, val_loader, test_loader = load_dataset(
            "cifar10", batch_size=batch_size, flatten=False, val_ratio=0.1, seed=seed
        )
    print(f"Data loaded in {t.elapsed:.1f}s")
    info = get_dataset_info("cifar10")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Num classes: {info['num_classes']}")

    # 模型
    model = CNNCifar10(num_classes=info["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== Model ===")
    print(f"  CNN: Conv(3→32) → Conv(32→64) → FC(4096→128) → FC(128→10)")
    print(f"  Parameters: {n_params:,}")
    print(f"  Optimizer: SGD(lr={lr}, momentum={momentum})")
    print(f"  Loss: CrossEntropyLoss")

    # 指标记录器
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    recorder = MetricsRecorder(
        experiment_name="baseline_cnn_cifar10",
        output_dir=str(output_path),
    )
    recorder.set_meta(
        model="CNN_CIFAR10",
        dataset="CIFAR-10",
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

            if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
                print(
                    f"  Epoch {epoch+1}/{epochs} "
                    f"[{batch_idx+1}/{len(train_loader)} "
                    f"({100.*(batch_idx+1)/len(train_loader):.0f}%)] "
                    f"loss={loss.item():.4f}"
                )

        epoch_time = time.time() - epoch_start
        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total

        val_acc, val_loss = evaluate_model(model, val_loader, criterion, device)
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

        if sample_activation_grad and ((epoch + 1) % 2 == 0 or epoch == epochs - 1):
            recorder.record_activation_stats(epoch + 1, model)
            recorder.record_gradient_stats(epoch + 1, model)

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
    model_path = output_path / "baseline_cnn_cifar10.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
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
    parser = argparse.ArgumentParser(description="CNN + CIFAR-10 baseline")
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
