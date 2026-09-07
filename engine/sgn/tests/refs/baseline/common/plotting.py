"""
训练动态绘图工具

生成对比图表：
- loss 曲线对比（baseline vs MSInt/HC/SGN 版）
- 准确率曲线对比
- 激活值分布对比（直方图）
- 梯度分布对比（直方图 + 重尾特征可视化）
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # 非交互后端，适合远程服务器
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(
    metrics_dict: Dict[str, dict],
    output_path: str,
    title: str = "Training Loss Curves",
):
    """绘制多个实验的 loss 曲线对比图。

    Args:
        metrics_dict: {实验名: metrics 字典}
        output_path: 输出图片路径
        title: 图表标题
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, metrics in metrics_dict.items():
        ts = metrics.get("time_series", {})
        train_losses = ts.get("train_losses", [])
        val_losses = ts.get("val_losses", [])
        epochs = range(1, len(train_losses) + 1)
        if train_losses:
            axes[0].plot(epochs, train_losses, label=f"{name} (train)", linewidth=1.5)
        if val_losses:
            axes[0].plot(
                epochs, val_losses, label=f"{name} (val)", linewidth=1.5, linestyle="--"
            )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    # 准确率
    for name, metrics in metrics_dict.items():
        ts = metrics.get("time_series", {})
        train_accs = ts.get("train_accs", [])
        val_accs = ts.get("val_accs", [])
        epochs = range(1, len(train_accs) + 1)
        if train_accs:
            axes[1].plot(epochs, train_accs, label=f"{name} (train)", linewidth=1.5)
        if val_accs:
            axes[1].plot(
                epochs, val_accs, label=f"{name} (val)", linewidth=1.5, linestyle="--"
            )

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_activation_distribution(
    metrics_dict: Dict[str, dict],
    output_path: str,
    layer_name: Optional[str] = None,
    epoch_idx: int = -1,
):
    """绘制激活值分布对比（不同实验的同一层在指定 epoch 的统计值对比）。

    Args:
        metrics_dict: {实验名: metrics 字典}
        output_path: 输出图片路径
        layer_name: 指定层名，None 则用第一个权重层
        epoch_idx: 从 activation_stats 中取第几个 epoch 的数据，-1 为最后一个
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.15
    stats_names = ["mean", "std", "abs_mean"]
    x = np.arange(len(stats_names))

    for i, (name, metrics) in enumerate(metrics_dict.items()):
        act_stats = metrics.get("activation_stats", [])
        if not act_stats:
            continue
        stats = act_stats[epoch_idx]["layers"]
        if layer_name is None:
            layer_name = list(stats.keys())[0]
        if layer_name not in stats:
            continue
        layer_stats = stats[layer_name]
        values = [layer_stats[s] for s in stats_names]
        ax.bar(
            x + i * bar_width,
            values,
            bar_width,
            label=name,
        )

    ax.set_xticks(x + bar_width * (len(metrics_dict) - 1) / 2)
    ax.set_xticklabels(stats_names)
    ax.set_ylabel("Value")
    ax.set_title(
        f"Activation Stats - {layer_name or 'first layer'} (epoch_idx={epoch_idx})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_gradient_distribution(
    metrics_dict: Dict[str, dict],
    output_path: str,
    layer_name: Optional[str] = None,
    epoch_idx: int = -1,
):
    """绘制梯度分布对比（重点看重尾特征 p99/median）。

    Args:
        metrics_dict: {实验名: metrics 字典}
        output_path: 输出图片路径
        layer_name: 指定层名
        epoch_idx: 从 gradient_stats 中取第几个 epoch 的数据
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bar_width = 0.15
    abs_stats = ["abs_mean", "std"]
    x_abs = np.arange(len(abs_stats))

    # 左图：绝对值均值和标准差
    for i, (name, metrics) in enumerate(metrics_dict.items()):
        grad_stats = metrics.get("gradient_stats", [])
        if not grad_stats:
            continue
        stats = grad_stats[epoch_idx]["layers"]
        if layer_name is None:
            layer_name = list(stats.keys())[0]
        if layer_name not in stats:
            continue
        layer_stats = stats[layer_name]
        values = [layer_stats[s] for s in abs_stats]
        axes[0].bar(x_abs + i * bar_width, values, bar_width, label=name)

    axes[0].set_xticks(x_abs + bar_width * (len(metrics_dict) - 1) / 2)
    axes[0].set_xticklabels(abs_stats)
    axes[0].set_ylabel("Value")
    axes[0].set_title(
        f"Gradient Magnitude - {layer_name or 'first layer'}"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # 右图：重尾特征 p99/median（>1 表示重尾分布）
    for i, (name, metrics) in enumerate(metrics_dict.items()):
        grad_stats = metrics.get("gradient_stats", [])
        if not grad_stats:
            continue
        stats = grad_stats[epoch_idx]["layers"]
        if layer_name is None:
            layer_name = list(stats.keys())[0]
        if layer_name not in stats:
            continue
        layer_stats = stats[layer_name]
        p99_ratio = layer_stats.get("p99_over_median", 0)
        axes[1].bar(i * bar_width, p99_ratio, bar_width, label=name)

    axes[1].set_xticks([])
    axes[1].set_ylabel("p99 / median ratio")
    axes[1].set_title(
        "Heavy-tail Indicator (p99/median, >1 = heavy-tailed)"
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].axhline(y=1, color="r", linestyle="--", alpha=0.5, label="uniform baseline")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def load_metrics_json(path: str) -> dict:
    """加载 metrics JSON 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    print("plotting module OK")
    print("Use plot_loss_curves(metrics_dict, output_path) to compare experiments")
