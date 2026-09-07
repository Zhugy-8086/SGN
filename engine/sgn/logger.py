# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""sgn.logger — 梯度日志与诊断工具

提供统一的日志与诊断功能：

  - 运行时日志：梯度/张量统计，支持开关控制和采样频率
  - 开发期诊断：数值梯度验证、前向/反向健康检查、SGN 特定检查

核心类：
  - GradLogger: 统一的日志 + 诊断入口

用法：
    from sgn.logger import GradLogger

    # 运行时日志
    logger = GradLogger(tag="LossLog", enabled=True, log_every=10)
    logger.log_stats("MSELoss", loss=0.5, grad=grad_array)
    logger.log_tensor("conv1_w_grad", grad_array)

    # 开发期诊断（训练前验证梯度和网络健康）
    report = logger.diagnose(model, criterion, sample_x, sample_y)
    if report.is_ready_for_training():
        print("可以开始训练！")
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Union, List, Tuple


def _array_stats(arr: np.ndarray) -> dict:
    """计算数组的关键统计量。

    处理 NaN/Inf：先统计数量，再在有限元素上计算统计值。

    Returns:
        包含 l2, min, max, mean, std, zero_ratio, nan_cnt, inf_cnt, size 的 dict
    """
    g = np.asarray(arr)
    nan_cnt = int(np.sum(np.isnan(g)))
    inf_cnt = int(np.sum(np.isinf(g)))

    if nan_cnt + inf_cnt > 0:
        g_clean = g[np.isfinite(g)]
    else:
        g_clean = g

    if g_clean.size > 0:
        return {
            # 安全审计 2026-08-16 Q2：l2 用 g_clean（与 min/max 口径一致）——
            # 原实现用含 NaN/Inf 的 g，NaN 污染结果、±Inf 虚增范数
            "l2": float(np.linalg.norm(g_clean)),
            "min": float(np.min(g_clean)),
            "max": float(np.max(g_clean)),
            "mean": float(np.mean(g_clean)),
            "std": float(np.std(g_clean)),
            "zero_ratio": float(np.sum(np.abs(g_clean) < 1e-12) / g.size),
            "nan_cnt": nan_cnt,
            "inf_cnt": inf_cnt,
            "shape": g.shape,
        }
    else:
        return {
            "l2": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0,
            "zero_ratio": 0.0, "nan_cnt": nan_cnt, "inf_cnt": inf_cnt,
            "shape": g.shape,
        }


class GradLogger:
    """梯度与张量统计日志器。

    可嵌入到损失函数、优化器、训练循环中，按需输出统计信息。
    通过 enabled 开关控制，关闭时零开销（仅一次 bool 判断）。
    通过 log_every 控制采样频率，避免高频训练循环中日志过多。

    动态调整策略：
      1. 运行时直接赋值 enabled（如训练前 100 步开启，之后关闭）
      2. 设置 log_every=N，每 N 步输出一次（适合长训练循环）
      3. 两者可组合：enabled=True + log_every=100 → 每 100 步输出

    Attributes:
        tag: 日志前缀标签，如 "LossLog"、"OptimLog"
        enabled: 是否启用日志输出
        log_every: 采样间隔，每 N 次调用输出一次（0 或 1 = 每次都输出）
    """

    def __init__(self, tag: str = "GradLog", enabled: bool = False,
                 log_every: int = 1):
        self.tag = tag
        self.enabled = enabled
        self.log_every = max(1, log_every)
        self._call_count = 0

    def _should_log(self) -> bool:
        """判断当前是否应该输出日志（采样门控）。"""
        if not self.enabled:
            return False
        self._call_count += 1
        return self._call_count % self.log_every == 0

    def log_stats(self, name: str, loss: Optional[float] = None,
                  grad: Optional[np.ndarray] = None, **extra):
        """输出单项统计日志。

        Args:
            name: 条目名称（如损失函数类名 "MSELoss"）
            loss: 损失值（可选）
            grad: 梯度数组（可选）
            **extra: 额外键值对，追加到日志末尾
        """
        if not self._should_log():
            return

        parts = [f"[{self.tag}] {name}"]

        if loss is not None:
            parts.append(f"loss={loss:.6e}")

        if grad is not None:
            s = _array_stats(grad)
            parts.append(
                f"grad shape={s['shape']} | "
                f"L2={s['l2']:.4e} | "
                f"min={s['min']:.4e} max={s['max']:.4e} | "
                f"mean={s['mean']:.4e} std={s['std']:.4e} | "
                f"zero%={s['zero_ratio']*100:.1f} | "
                f"NaN={s['nan_cnt']} Inf={s['inf_cnt']}"
            )

        for k, v in extra.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4e}")
            else:
                parts.append(f"{k}={v}")

        print(" | ".join(parts))

    def log_sub(self, parent: str, child: str, weight: Optional[float] = None,
                grad: Optional[np.ndarray] = None, **extra):
        """输出子项统计日志（如 WeightedSumLoss 的逐项分解）。

        Args:
            parent: 父级名称（如 "WeightedSumLoss"）
            child: 子项名称（如 "mse"）
            weight: 子项权重（可选）
            grad: 子项梯度数组（可选）
            **extra: 额外键值对
        """
        if not self._should_log():
            return

        header = f"[{self.tag}] {parent}/{child}"
        if weight is not None:
            header += f" (w={weight})"

        parts = [header]

        if grad is not None:
            s = _array_stats(grad)
            parts.append(
                f"sub-grad L2={s['l2']:.4e} | "
                f"min={s['min']:.4e} max={s['max']:.4e} | "
                f"mean={s['mean']:.4e} std={s['std']:.4e}"
            )

        for k, v in extra.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4e}")
            else:
                parts.append(f"{k}={v}")

        print(" | ".join(parts))

    def log_tensor(self, name: str, tensor: np.ndarray, **extra):
        """输出任意张量的统计日志（不依赖 loss 概念）。

        适用于优化器、层输出等场景。

        Args:
            name: 张量名称（如 "conv1_w_grad"）
            tensor: 张量数组
            **extra: 额外键值对
        """
        if not self._should_log():
            return

        s = _array_stats(tensor)
        parts = [
            f"[{self.tag}] {name}",
            f"shape={s['shape']}",
            f"L2={s['l2']:.4e}",
            f"min={s['min']:.4e} max={s['max']:.4e}",
            f"mean={s['mean']:.4e} std={s['std']:.4e}",
            f"zero%={s['zero_ratio']*100:.1f}",
            f"NaN={s['nan_cnt']} Inf={s['inf_cnt']}",
        ]

        for k, v in extra.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4e}")
            else:
                parts.append(f"{k}={v}")

        print(" | ".join(parts))


    # ------------------------------------------------------------------
    # 诊断功能 — 集成 LossDiagnoser，统一日志 + 诊断入口
    # ------------------------------------------------------------------

    def diagnose(self, model, criterion, sample_input: np.ndarray,
                 sample_target: np.ndarray, *,
                 modes: Optional[List[str]] = None,
                 eps: float = 1e-2, n_samples: int = 50,
                 sgn_module=None):
        """一键诊断：运行所有检查，通过 logger 输出结果，返回诊断报告。

        内部调用 sgn.loss.LossDiagnoser，延迟导入避免循环依赖。

        Args:
            model: sgn.nn.Module 实例
            criterion: 损失函数实例（BaseLoss 子类）
            sample_input: 样本输入 (numpy array)
            sample_target: 样本标签 (numpy array)
            modes: 检查模式列表，可选 'static','forward','backward','gradient','sgn'
            eps: 数值梯度扰动大小（默认 1e-2，适配 float32）
            n_samples: 数值梯度抽样参数数量
            sgn_module: sgn 模块引用（可选，自动推断）

        Returns:
            DiagnoseReport 实例（含 summary() 和 is_ready_for_training()）
        """
        # 延迟导入，避免 logger.py ↔ loss.py 循环依赖
        from .loss import LossDiagnoser

        if sgn_module is None:
            # 安全审计 2026-08-16 Q1：原硬编码 `import engine.sgn`——build/
            # 直载 .pyd 模式（sys.modules["sgn"]）下会二次加载引擎包，
            # 两个 sgn 实例并存导致 isinstance 检查失效。改 sys.modules 查找，
            # 优先已加载实例，均未加载时回退包模式导入。
            import sys as _sys
            sgn_module = _sys.modules.get("sgn") or _sys.modules.get("engine.sgn")
            if sgn_module is None:
                import engine.sgn as _sgn
                sgn_module = _sgn

        diagnoser = LossDiagnoser(
            model, criterion, sample_input, sample_target,
            sgn_module=sgn_module,
        )
        report = diagnoser.diagnose(modes=modes, eps=eps, n_samples=n_samples)

        # 通过 logger 输出诊断摘要
        if self.enabled:
            print()
            print(report.summary())
            print()

        return report

    def check_gradient(self, model, criterion, sample_input: np.ndarray,
                       sample_target: np.ndarray, *,
                       eps: float = 1e-2, n_samples: int = 50,
                       sgn_module=None):
        """快速数值梯度检查（仅 gradient 模式）。

        比 diagnose() 快，适合训练中周期性抽样检查。

        Args:
            model: sgn.nn.Module 实例
            criterion: 损失函数实例
            sample_input: 样本输入
            sample_target: 样本标签
            eps: 数值梯度扰动大小
            n_samples: 抽样参数数量
            sgn_module: sgn 模块引用

        Returns:
            CheckResult 实例
        """
        report = self.diagnose(
            model, criterion, sample_input, sample_target,
            modes=['gradient'], eps=eps, n_samples=n_samples,
            sgn_module=sgn_module,
        )
        grad_results = [c for c in report.checks if c.name == 'numerical_gradient']
        return grad_results[0] if grad_results else None

    def check_forward(self, model, criterion, sample_input: np.ndarray,
                      sample_target: np.ndarray, *,
                      sgn_module=None):
        """快速前向检查（NaN/Inf、输出范围、loss 值）。

        Args:
            model: sgn.nn.Module 实例
            criterion: 损失函数实例
            sample_input: 样本输入
            sample_target: 样本标签
            sgn_module: sgn 模块引用

        Returns:
            DiagnoseReport 实例
        """
        return self.diagnose(
            model, criterion, sample_input, sample_target,
            modes=['static', 'forward'], sgn_module=sgn_module,
        )


__all__ = ["GradLogger", "_array_stats"]
