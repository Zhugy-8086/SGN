"""
训练指标记录工具

记录训练过程中的 4 项核心指标：
- 准确率（train/val/test）
- 速度（每 epoch 耗时、每秒样本数）
- 内存（峰值内存占用）
- 训练动态（loss 曲线、激活/梯度分布）

所有指标以 JSON 格式输出，便于跨阶段对比。
"""
import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch


class MetricsRecorder:
    """训练过程指标记录器。"""

    def __init__(self, experiment_name: str, output_dir: Optional[str] = None):
        """初始化指标记录器。

        Args:
            experiment_name: 实验名称（如 "baseline_mlp_mnist"）
            output_dir: 结果输出目录，默认当前目录
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 时序记录
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []
        self.epoch_times: List[float] = []
        self.lr_history: List[float] = []

        # 每 batch 的 loss（用于细粒度分析）
        self.batch_losses: List[float] = []

        # 峰值内存（MB）
        self.peak_memory_mb: float = 0.0

        # 最终评估结果
        self.final_test_acc: float = 0.0
        self.final_test_loss: float = 0.0

        # 激活/梯度分布采样（每 N epoch 采一次）
        self.activation_stats: List[dict] = []
        self.gradient_stats: List[dict] = []

        # 元信息
        self.meta: dict = {}

    def set_meta(self, **kwargs):
        """设置实验元信息（模型配置、训练超参等）。"""
        self.meta.update(kwargs)

    def record_batch(self, loss: float):
        """记录单个 batch 的 loss。"""
        self.batch_losses.append(float(loss))

    def record_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        epoch_time: float,
        lr: float,
    ):
        """记录一个 epoch 结束时的指标。"""
        self.train_losses.append(float(train_loss))
        self.val_losses.append(float(val_loss))
        self.train_accs.append(float(train_acc))
        self.val_accs.append(float(val_acc))
        self.epoch_times.append(float(epoch_time))
        self.lr_history.append(float(lr))

    def record_final_test(self, test_acc: float, test_loss: float):
        """记录最终测试集结果。"""
        self.final_test_acc = float(test_acc)
        self.final_test_loss = float(test_loss)

    def record_activation_stats(self, epoch: int, model: torch.nn.Module):
        """采样模型各层激活值统计（均值、方差、最大值、最小值）。

        在指定 epoch 调用，用于分析训练动态。
        """
        stats = {"epoch": epoch, "layers": {}}
        for name, param in model.named_parameters():
            if "weight" in name:
                stats["layers"][name] = {
                    "mean": float(param.data.mean()),
                    "std": float(param.data.std()),
                    "min": float(param.data.min()),
                    "max": float(param.data.max()),
                    "abs_mean": float(param.data.abs().mean()),
                }
        self.activation_stats.append(stats)

    def record_gradient_stats(self, epoch: int, model: torch.nn.Module):
        """采样模型各层梯度统计（均值、方差、最大值、最小值）。

        在指定 epoch 调用，用于分析梯度分布（与 Xi 2023 论文中的
        heavy-tailed gradient distribution 对照）。
        """
        stats = {"epoch": epoch, "layers": {}}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                stats["layers"][name] = {
                    "mean": float(grad.mean()),
                    "std": float(grad.std()),
                    "min": float(grad.min()),
                    "max": float(grad.max()),
                    "abs_mean": float(grad.abs().mean()),
                    # 重尾特征：计算 99 百分位与中位数之比
                    "p99_over_median": float(
                        (torch.quantile(grad.flatten().abs().float(), 0.99) + 1e-10)
                        / (torch.quantile(grad.flatten().abs().float(), 0.5) + 1e-10)
                    ),
                }
        self.gradient_stats.append(stats)

    def measure_memory(self):
        """测量当前内存占用（MB）。CPU 版用 RSS。"""
        try:
            import psutil

            process = psutil.Process()
            mem_mb = process.memory_info().rss / 1024 / 1024
            if mem_mb > self.peak_memory_mb:
                self.peak_memory_mb = mem_mb
            return mem_mb
        except ImportError:
            return 0.0

    def to_dict(self) -> dict:
        """汇总所有指标为字典。"""
        return {
            "experiment_name": self.experiment_name,
            "meta": self.meta,
            "summary": {
                "final_test_acc": self.final_test_acc,
                "final_test_loss": self.final_test_loss,
                "best_val_acc": max(self.val_accs) if self.val_accs else 0.0,
                "best_val_epoch": (
                    self.val_accs.index(max(self.val_accs)) + 1
                    if self.val_accs
                    else 0
                ),
                "total_train_time_s": sum(self.epoch_times),
                "avg_epoch_time_s": (
                    sum(self.epoch_times) / len(self.epoch_times)
                    if self.epoch_times
                    else 0.0
                ),
                "peak_memory_mb": self.peak_memory_mb,
            },
            "time_series": {
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "train_accs": self.train_accs,
                "val_accs": self.val_accs,
                "epoch_times": self.epoch_times,
                "lr_history": self.lr_history,
                "batch_losses": self.batch_losses[-200:],  # 只保留最后 200 个
            },
            "activation_stats": self.activation_stats,
            "gradient_stats": self.gradient_stats,
        }

    def save_json(self, filename: Optional[str] = None) -> Path:
        """保存所有指标到 JSON 文件。"""
        if filename is None:
            filename = f"metrics_{self.experiment_name}.json"
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path


class Timer:
    """简单计时器。"""

    def __init__(self):
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple:
    """评估模型在数据集上的准确率和平均 loss。

    Returns:
        (accuracy, avg_loss)
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
            n_batches += 1
            _, predicted = output.max(1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    return correct / total, total_loss / max(n_batches, 1)


if __name__ == "__main__":
    # 自测
    rec = MetricsRecorder("test_experiment", output_dir="./test_metrics")
    rec.set_meta(model="MLP", dataset="MNIST", epochs=5)
    for epoch in range(5):
        rec.record_epoch(
            epoch=epoch,
            train_loss=1.0 - epoch * 0.15,
            val_loss=1.1 - epoch * 0.13,
            train_acc=0.3 + epoch * 0.12,
            val_acc=0.28 + epoch * 0.11,
            epoch_time=5.0 + epoch * 0.1,
            lr=0.001,
        )
    rec.record_final_test(0.95, 0.15)
    path = rec.save_json()
    print(f"Saved to: {path}")
    print(f"Summary: {rec.to_dict()['summary']}")
