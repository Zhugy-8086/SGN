"""6 层 CNN+BN 浮点 baseline (CIFAR-10)

Stage 2.9 / B2-simplified 的载体模型。

设计目标：
  - 6 个参数层（3 Conv + 3 FC），每层配 BN
  - 无残差连接，梯度流清晰，易隔离 Level_f/Level_b 效果
  - 层深度梯度明显：Layer 6（靠近 loss）梯度方差大，Layer 1（远离 loss）梯度方差小
  - 适合验证 Level_b 非均匀 bits 分配（exp19 H_B2 已用合成数据确认）
  - CIFAR-10 ~30s/epoch（CPU）

架构（CIFAR-10, input (3, 32, 32)）：
    Layer 1: Conv2d(3, 32, k=3, p=1) → BN(32) → ReLU → MaxPool(2)   → (32, 16, 16)
    Layer 2: Conv2d(32, 64, k=3, p=1) → BN(64) → ReLU → MaxPool(2)  → (64, 8, 8)
    Layer 3: Conv2d(64, 128, k=3, p=1) → BN(128) → ReLU → MaxPool(2) → (128, 4, 4)
    Layer 4: Linear(2048, 256) → BN1d(256) → ReLU
    Layer 5: Linear(256, 128) → BN1d(128) → ReLU
    Layer 6: Linear(128, 10)

参数量: ~653K
预期 test_acc: ~78-80% (CIFAR-10, 10 epoch, CPU)

层深度与梯度方差关系（exp10 结论）：
    Layer 6 (靠近 loss): 梯度方差大 → Level_b 应分配更多 bits (16-20)
    Layer 1 (远离 loss): 梯度方差小 → Level_b 可用较少 bits (8-12)
    这正是 Level_b 非均匀 bits 分配的验证场景

用法：
    python baseline/cnn6_cifar10.py --check       # 检查模型能否使用（不训练）
    python baseline/cnn6_cifar10.py --epochs 10   # 训练 10 epoch
    python baseline/cnn6_cifar10.py --quick       # 快速验证模式
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


class CNN6Cifar10(nn.Module):
    """6 层 CNN+BN for CIFAR-10: 3 Conv 块 + 3 FC 层

    每个卷积块: Conv2d → BatchNorm2d → ReLU → MaxPool2d
    每个全连接层: Linear → BatchNorm1d → ReLU (最后一层除外)

    层命名规则（用于后续 HC 集成时的层标识）：
        conv1/bn1, conv2/bn2, conv3/bn3, fc1/bn4, fc2/bn5, fc3

    层深度索引（从浅到深，用于 Level_b 调度）：
        depth[0] = conv1 (最浅，梯度方差最小)
        depth[1] = conv2
        depth[2] = conv3
        depth[3] = fc1
        depth[4] = fc2
        depth[5] = fc3 (最深，梯度方差最大)
    """

    # 层名称列表（用于 HC 集成时的层标识，顺序从浅到深）
    LAYER_NAMES = ["conv1", "conv2", "conv3", "fc1", "fc2", "fc3"]

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 卷积块 1: (3, 32, 32) → (32, 16, 16)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # 卷积块 2: (32, 16, 16) → (64, 8, 8)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 卷积块 3: (64, 8, 8) → (128, 4, 4)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 全连接层: (128*4*4=2048) → 256 → 128 → 10
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn4 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # x: (B, 3, 32, 32)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # (B, 32, 16, 16)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # (B, 64, 8, 8)
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # (B, 128, 4, 4)
        x = x.view(x.size(0), -1)  # (B, 2048)
        x = self.relu(self.bn4(self.fc1(x)))  # (B, 256)
        x = self.relu(self.bn5(self.fc2(x)))  # (B, 128)
        x = self.fc3(x)  # (B, 10)
        return x

    def get_layer_depths(self):
        """返回各层的深度索引（0=最浅, 5=最深）

        用于 Level_b 调度时识别层的深度位置。
        exp10 结论：深层（靠近 loss）梯度方差大，需要更多 bits。
        """
        return {name: i for i, name in enumerate(self.LAYER_NAMES)}


def check_model():
    """检查模型能否使用（不训练）

    验证内容：
    1. 模型实例化无错误
    2. Forward pass shape 正确
    3. Backward pass 梯度流通
    4. 参数量统计
    5. 单 batch 数据加载验证
    6. BN 统计更新验证
    """
    print("=" * 60)
    print("6 层 CNN+BN 模型检查")
    print("=" * 60)

    device = torch.device("cpu")
    model = CNN6Cifar10(num_classes=10).to(device)

    # 1. 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[1] 参数量统计:")
    print(f"    总参数量:   {total_params:,} ({total_params / 1e3:.1f}K)")
    print(f"    可训练参数: {trainable_params:,}")

    # 逐层参数量
    print(f"    逐层明细:")
    layer_params = [
        ("conv1", model.conv1, model.bn1),
        ("conv2", model.conv2, model.bn2),
        ("conv3", model.conv3, model.bn3),
        ("fc1", model.fc1, model.bn4),
        ("fc2", model.fc2, model.bn5),
        ("fc3", model.fc3, None),
    ]
    for name, main_layer, bn_layer in layer_params:
        p_main = sum(p.numel() for p in main_layer.parameters())
        p_bn = sum(p.numel() for p in bn_layer.parameters()) if bn_layer else 0
        print(f"      {name}: {p_main:,} + BN {p_bn:,} = {p_main + p_bn:,}")

    # 2. Forward shape 检查
    print(f"\n[2] Forward shape 检查:")
    model.eval()
    test_shapes = [
        (1, 3, 32, 32, "单样本"),
        (4, 3, 32, 32, "小 batch"),
        (64, 3, 32, 32, "标准 batch"),
    ]
    for batch_size, c, h, w, desc in test_shapes:
        x = torch.randn(batch_size, c, h, w)
        with torch.no_grad():
            y = model(x)
        expected = (batch_size, 10)
        status = "✓" if y.shape == expected else "✗"
        print(f"    {desc} ({batch_size},3,32,32) → {tuple(y.shape)} {status} (期望 {expected})")

    # 3. Backward 梯度流检查
    print(f"\n[3] Backward 梯度流检查:")
    model.train()
    x = torch.randn(8, 3, 32, 32)
    target = torch.randint(0, 10, (8,))
    criterion = nn.CrossEntropyLoss()

    y = model(x)
    loss = criterion(y, target)
    loss.backward()

    all_have_grad = True
    print(f"    Loss = {loss.item():.4f}")
    print(f"    各层梯度:")
    for name, main_layer, bn_layer in layer_params:
        for pname, p in main_layer.named_parameters():
            grad_ok = p.grad is not None and p.grad.abs().sum().item() > 0
            if not grad_ok:
                all_have_grad = False
            status = "✓" if grad_ok else "✗"
            print(f"      {name}.{pname}: grad_norm={p.grad.norm().item():.6f} {status}")
        if bn_layer is not None:
            for pname, p in bn_layer.named_parameters():
                grad_ok = p.grad is not None and p.grad.abs().sum().item() > 0
                if not grad_ok:
                    all_have_grad = False
                status = "✓" if grad_ok else "✗"
                if p.grad is not None:
                    print(f"      {name}.bn.{pname}: grad_norm={p.grad.norm().item():.6f} {status}")
                else:
                    print(f"      {name}.bn.{pname}: grad=None {status}")

    print(f"\n    梯度流总状态: {'✓ 全部通畅' if all_have_grad else '✗ 有梯度缺失'}")

    # 4. BN 统计更新验证
    print(f"\n[4] BN 统计更新验证:")
    model.train()
    # 记录初始 running_mean
    bn_layers = [model.bn1, model.bn2, model.bn3, model.bn4, model.bn5]
    initial_means = [bn.running_mean.clone() for bn in bn_layers]
    # 跑几个 batch
    for _ in range(5):
        x = torch.randn(32, 3, 32, 32)
        model(x)
    # 检查 running_mean 是否变化
    all_updated = True
    for i, (bn, init_m) in enumerate(zip(bn_layers, initial_means)):
        changed = not torch.equal(bn.running_mean, init_m)
        if not changed:
            all_updated = False
        status = "✓" if changed else "✗"
        print(f"    bn{i+1}.running_mean 已更新 {status}")
    print(f"\n    BN 统计更新总状态: {'✓ 全部更新' if all_updated else '✗ 有未更新的 BN'}")

    # 5. 数据加载验证
    print(f"\n[5] 数据加载验证 (CIFAR-10):")
    try:
        train_loader, val_loader, test_loader = load_dataset(
            "cifar10", batch_size=4, flatten=False, val_ratio=0.1
        )
        info = get_dataset_info("cifar10")
        print(f"    类别数: {info['num_classes']}")
        print(f"    输入尺寸: {info['input_size']}")
        print(f"    训练 batches: {len(train_loader)}")
        print(f"    验证 batches: {len(val_loader)}")
        print(f"    测试 batches: {len(test_loader)}")

        # 取一个 batch 验证
        x_batch, y_batch = next(iter(train_loader))
        print(f"    训练 batch shape: {tuple(x_batch.shape)}, dtype={x_batch.dtype}")
        print(f"    标签 shape: {tuple(y_batch.shape)}, dtype={y_batch.dtype}")

        # 用真实数据做一次 forward
        model.eval()
        with torch.no_grad():
            y = model(x_batch)
        print(f"    真实数据 forward: {tuple(x_batch.shape)} → {tuple(y.shape)} ✓")
        print(f"    数据加载状态: ✓ 正常")
    except Exception as e:
        print(f"    数据加载状态: ✗ 错误 - {e}")

    # 6. 层深度信息
    print(f"\n[6] 层深度信息 (用于 Level_b 调度):")
    depths = model.get_layer_depths()
    for name, depth in depths.items():
        gradient_var_expectation = "高 (靠近 loss)" if depth >= 4 else \
                                   "中" if depth >= 2 else "低 (远离 loss)"
        print(f"    depth[{name}] = {depth}  梯度方差预期: {gradient_var_expectation}")

    print("\n" + "=" * 60)
    print("检查完成。模型可用于后续 HC 集成和训练。")
    print("=" * 60)
    return True


def train(
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 0.01,
    momentum: float = 0.9,
    seed: int = 42,
    output_dir: str = "baseline/results",
    quick: bool = False,
):
    """训练 6 层 CNN+BN (CIFAR-10)"""
    torch.manual_seed(seed)

    print(f"训练 6 层 CNN+BN (CIFAR-10)")
    print(f"  epochs={epochs}, batch_size={batch_size}, lr={lr}, momentum={momentum}")

    device = torch.device("cpu")
    model = CNN6Cifar10().to(device)

    train_loader, val_loader, test_loader = load_dataset(
        "cifar10", batch_size=batch_size, flatten=False, val_ratio=0.1, seed=seed
    )
    info = get_dataset_info("cifar10")
    print(f"  类别数: {info['num_classes']}, 输入尺寸: {info['input_size']}")
    print(f"  训练 batches: {len(train_loader)}, 测试 batches: {len(test_loader)}")

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            _, predicted = out.max(1)
            correct += predicted.eq(y).sum().item()
            total += x.size(0)

        scheduler.step()
        train_acc = correct / total
        avg_loss = total_loss / total

        # 测试
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                test_loss += criterion(out, y).item() * x.size(0)
                _, predicted = out.max(1)
                test_correct += predicted.eq(y).sum().item()
                test_total += x.size(0)
        test_acc = test_correct / test_total
        test_loss /= test_total

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch}/{epochs} ({elapsed:.1f}s): "
            f"train_loss={avg_loss:.4f}, train_acc={train_acc:.4f}, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
        )

    print(f"\n训练完成。最终 test_acc: {test_acc:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="6 层 CNN+BN (CIFAR-10)")
    parser.add_argument("--check", action="store_true", help="只检查模型能否使用，不训练")
    parser.add_argument("--epochs", type=int, default=10, help="训练 epoch 数")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--quick", action="store_true", help="快速验证模式")
    args = parser.parse_args()

    if args.check:
        check_model()
        return

    train(
        epochs=1 if args.quick else args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
