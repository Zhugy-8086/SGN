"""
统一数据加载入口
所有 baseline 和 stage 代码都通过此模块加载数据，保证一致性。

支持数据集：MNIST、Fashion-MNIST、CIFAR-10、CIFAR-100
"""
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class FlattenTransform:
    """可 pickle 的展平变换（替代 transforms.Lambda）。

    transforms.Lambda 使用闭包，在 Windows spawn 模式下（num_workers>0）
    无法 pickle 导致 worker 启动失败。此类为顶层类，可被 pickle。
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1)


# 数据集默认配置
_DATASET_CONFIG = {
    "mnist": {
        "cls": datasets.MNIST,
        "mean": (0.1307,),
        "std": (0.3081,),
        "num_classes": 10,
        "input_size": 28 * 28,
    },
    "fashion_mnist": {
        "cls": datasets.FashionMNIST,
        "mean": (0.2860,),
        "std": (0.3530,),
        "num_classes": 10,
        "input_size": 28 * 28,
    },
    "cifar10": {
        "cls": datasets.CIFAR10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "num_classes": 10,
        "input_size": 32 * 32 * 3,
    },
    "cifar100": {
        "cls": datasets.CIFAR100,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "num_classes": 100,
        "input_size": 32 * 32 * 3,
    },
}


def get_transform(dataset_name: str, flatten: bool = True) -> transforms.Compose:
    """获取指定数据集的标准预处理。

    Args:
        dataset_name: 数据集名称（mnist/fashion_mnist/cifar10/cifar100）
        flatten: True 则展平为一维张量（MLP 用），False 则保留图像形状（CNN/Transformer 用）

    Returns:
        torchvision transforms.Compose
    """
    cfg = _DATASET_CONFIG[dataset_name]
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ]
    if flatten:
        transform_list.append(FlattenTransform())
    return transforms.Compose(transform_list)


def load_dataset(
    dataset_name: str,
    root: str = None,
    batch_size: int = 64,
    flatten: bool = True,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """加载指定数据集，返回 (train_loader, val_loader, test_loader)。

    Args:
        dataset_name: 数据集名称
        root: 数据集根目录，默认 traditional/data
        batch_size: 批次大小
        flatten: 是否展平（MLP 用 True，CNN/Transformer 用 False）
        val_ratio: 从训练集中划分验证集的比例
        num_workers: DataLoader 工作进程数
        seed: 随机种子（用于可复现的 train/val 划分）

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if dataset_name not in _DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset: {dataset_name}, "
            f"supported: {list(_DATASET_CONFIG.keys())}"
        )

    if root is None:
        # 默认存到 traditional/data 下
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
        )
    os.makedirs(root, exist_ok=True)

    cfg = _DATASET_CONFIG[dataset_name]
    transform = get_transform(dataset_name, flatten=flatten)

    # 下载并加载训练集
    train_set = cfg["cls"](
        root=root, train=True, download=True, transform=transform
    )
    test_set = cfg["cls"](
        root=root, train=False, download=True, transform=transform
    )

    # 划分训练集和验证集
    if val_ratio > 0:
        n_total = len(train_set)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val
        generator = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(
            train_set, [n_train, n_val], generator=generator
        )
    else:
        val_set = test_set  # 不划分时用 test 作 val（不推荐）

    # num_workers>0 时启用 persistent_workers 避免 epoch 间重建 worker；
    # prefetch_factor 仅在 num_workers>0 时有效，否则 PyTorch 报警告
    _persistent = num_workers > 0
    _prefetch = 4 if num_workers > 0 else None

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        persistent_workers=_persistent, prefetch_factor=_prefetch,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        persistent_workers=_persistent, prefetch_factor=_prefetch,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        persistent_workers=_persistent, prefetch_factor=_prefetch,
    )

    return train_loader, val_loader, test_loader


def get_dataset_info(dataset_name: str) -> dict:
    """获取数据集元信息：num_classes、input_size 等。"""
    if dataset_name not in _DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    cfg = _DATASET_CONFIG[dataset_name]
    return {
        "num_classes": cfg["num_classes"],
        "input_size": cfg["input_size"],
        "mean": cfg["mean"],
        "std": cfg["std"],
    }


if __name__ == "__main__":
    # 自测：尝试加载 MNIST 的一个批次
    print("Testing MNIST loading...")
    train_loader, val_loader, test_loader = load_dataset(
        "mnist", batch_size=4, val_ratio=0.1
    )
    for x, y in train_loader:
        print(f"  batch x shape: {x.shape}, x dtype: {x.dtype}")
        print(f"  batch y shape: {y.shape}, y dtype: {y.dtype}")
        print(f"  x range: [{x.min().item():.4f}, {x.max().item():.4f}]")
        print(f"  y values: {y.tolist()}")
        break
    info = get_dataset_info("mnist")
    print(f"  num_classes: {info['num_classes']}")
    print(f"  input_size: {info['input_size']}")
    print("MNIST OK")
