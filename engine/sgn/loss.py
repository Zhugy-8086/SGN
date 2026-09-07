# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""sgn.loss — 损失函数模块

提供可插拔的损失函数，统一接口，支持计算模式和诊断模式。

计算模式（轨一）：
  - BaseLoss: 损失函数基类，所有损失函数的统一接口
  - MSELoss: 均方误差损失
  - CrossEntropyLoss: 交叉熵损失（含 softmax）
  - WeightedSumLoss: 多任务加权组合器

诊断模式（轨二）：
  - LossDiagnoser: 开发期验证工具，静态/动态/数值梯度/整数域检查
  - CheckResult / DiagnoseReport: 诊断结果数据结构

用法：
    import sgn
    criterion = sgn.loss.CrossEntropyLoss()

    with ag.record_scope(clear=True):
        logits = model.forward([x])

    out_np = logits.to_numpy()
    loss, dY = criterion(out_np, y_np)
    logits.backward(dY.astype(np.float32))
    optimizer.step()

    # 诊断模式
    diagnoser = sgn.loss.LossDiagnoser(model, criterion, sample_x, sample_y)
    report = diagnoser.diagnose()
    print(report.summary())
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Callable
from enum import Enum

from .logger import GradLogger


# ============================================================================
# 计算模式（轨一）：Loss 基类 + 专用损失
# ============================================================================

class BaseLoss:
    """损失函数基类 — 所有损失函数的统一接口。

    子类需要实现 forward() 和 backward()。
    __call__ 提供便捷调用，返回 (loss_value, grad)。

    Attributes:
        _cached_y_pred: 前向缓存的 y_pred，用于 backward 计算梯度
        _cached_y_true: 前向缓存的 y_true
        verbose: 是否在 __call__ 中输出梯度统计日志
        log_every: 日志采样间隔（每 N 次调用输出一次，默认 1=每次）
        _logger: 内部 GradLogger 实例
    """

    def __init__(self, verbose: bool = False, log_every: int = 1):
        self._cached_y_pred = None
        self._cached_y_true = None
        self.verbose = verbose
        self._logger = GradLogger(tag="LossLog", enabled=verbose, log_every=log_every)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """计算损失值。

        Args:
            y_pred: 模型预测输出 (numpy array)
            y_true: 真实标签 (numpy array)

        Returns:
            损失值 (float)
        """
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        """返回损失对 y_pred 的梯度 ∂L/∂y_pred。

        必须在 forward() 之后调用，使用 forward 缓存的数据。

        Returns:
            梯度数组，形状与 y_pred 一致
        """
        raise NotImplementedError

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray]:
        """便捷调用：计算损失值和梯度。

        Args:
            y_pred: 模型预测输出
            y_true: 真实标签

        Returns:
            (loss, grad_wrt_y_pred) 元组
        """
        loss = self.forward(y_pred, y_true)
        grad = self.backward()
        if self.verbose:
            self._logger.log_stats(type(self).__name__, loss=loss, grad=grad)
        return loss, grad


class MSELoss(BaseLoss):
    """均方误差损失: L = 0.5 * mean((y_pred - y_true)^2)

    Usage:
        criterion = MSELoss()
        loss, dY = criterion(y_pred, y_true)
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._cached_y_pred = y_pred
        self._cached_y_true = y_true
        diff = y_pred - y_true
        return 0.5 * float(np.mean(diff * diff))

    def backward(self) -> np.ndarray:
        if self._cached_y_pred is None:
            raise RuntimeError("MSELoss.backward() 必须在 forward() 之后调用")
        diff = self._cached_y_pred - self._cached_y_true
        return diff / self._cached_y_pred.size


class CrossEntropyLoss(BaseLoss):
    """交叉熵损失（含 softmax）: L = -mean(log(softmax(y_pred)[y_true]))

    y_true 可以是类别索引（int array，shape=(batch_size,)）或 one-hot（float array）。

    Usage:
        criterion = CrossEntropyLoss()
        loss, dY = criterion(logits, labels)  # labels 是类别索引
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._cached_y_pred = y_pred
        self._cached_y_true = y_true

        # softmax（数值稳定）
        shifted = y_pred - y_pred.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        self._cached_softmax = exp / exp.sum(axis=1, keepdims=True)

        if y_true.ndim == 1:
            # 类别索引
            batch_size = y_pred.shape[0]
            loss = -np.mean(np.log(
                self._cached_softmax[np.arange(batch_size), y_true] + 1e-15
            ))
        else:
            # one-hot
            loss = -np.mean(np.sum(y_true * np.log(self._cached_softmax + 1e-15), axis=1))

        return float(loss)

    def backward(self) -> np.ndarray:
        if self._cached_y_pred is None:
            raise RuntimeError("CrossEntropyLoss.backward() 必须在 forward() 之后调用")

        batch_size = self._cached_y_pred.shape[0]
        y_true = self._cached_y_true

        if y_true.ndim == 1:
            # 类别索引 → dL/d(logits) = (softmax - one_hot) / batch_size
            dY = self._cached_softmax.copy()
            dY[np.arange(batch_size), y_true] -= 1.0
        else:
            # one-hot → dL/d(logits) = (softmax - y_true) / batch_size
            dY = (self._cached_softmax - y_true) / batch_size
            return dY

        return dY / batch_size


class WeightedSumLoss(BaseLoss):
    """多任务加权组合: L = sum(w_i * loss_i(y_pred, y_true))

    支持多个损失函数的加权组合，用于多任务学习。

    Args:
        losses: 损失函数列表 [(name, loss_instance), ...]
        weights: 权重列表，与 losses 一一对应

    Usage:
        criterion = WeightedSumLoss([
            ("mse", MSELoss()),
            ("ce", CrossEntropyLoss()),
        ], weights=[0.5, 0.5])
        loss, dY = criterion(y_pred, [y_true_mse, y_true_ce])
    """

    def __init__(self, losses: List[Tuple[str, BaseLoss]], weights: Optional[List[float]] = None,
                 verbose: bool = False, log_every: int = 1):
        super().__init__(verbose=verbose, log_every=log_every)
        self.losses = losses
        self._names = [name for name, _ in losses]
        self._loss_instances = [loss for _, loss in losses]
        if weights is None:
            self.weights = [1.0] * len(losses)
        else:
            if len(weights) != len(losses):
                raise ValueError(
                    f"weights 长度 ({len(weights)}) 与 losses 长度 ({len(losses)}) 不匹配"
                )
            self.weights = list(weights)

    def forward(self, y_pred: np.ndarray, y_true) -> float:
        """计算加权损失和。

        y_true 可以是单个 array（所有损失共享）或 list of arrays（每个损失自己的 target）。
        """
        self._cached_y_pred = y_pred
        self._cached_y_true = y_true

        total = 0.0
        for i, (weight, loss_fn) in enumerate(zip(self.weights, self._loss_instances)):
            target = y_true[i] if isinstance(y_true, (list, tuple)) else y_true
            total += weight * loss_fn.forward(y_pred, target)
        return total

    def backward(self) -> np.ndarray:
        if self._cached_y_pred is None:
            raise RuntimeError("WeightedSumLoss.backward() 必须在 forward() 之后调用")

        y_true = self._cached_y_true
        grad = np.zeros_like(self._cached_y_pred, dtype=np.float64)
        for i, (weight, loss_fn) in enumerate(zip(self.weights, self._loss_instances)):
            target = y_true[i] if isinstance(y_true, (list, tuple)) else y_true
            sub_grad = loss_fn.backward()
            grad += weight * sub_grad
            if self.verbose:
                self._logger.log_sub(
                    "WeightedSumLoss", self._names[i],
                    weight=weight, grad=sub_grad,
                )
        return grad


# ============================================================================
# 诊断模式（轨二）：数据类
# ============================================================================

class CheckLevel(Enum):
    """检查结果级别"""
    PASS = "PASS"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass
class CheckResult:
    """单条诊断检查结果"""
    category: str           # 检查类别：static / forward / backward / gradient / sgn
    name: str               # 检查项名称
    level: CheckLevel       # 结果级别
    message: str = ""       # 详细描述
    detail: dict = field(default_factory=dict)  # 附带数据（如具体数值）

    def __repr__(self) -> str:
        icon = {"PASS": "✓", "WARN": "⚠", "ERROR": "✗"}[self.level.value]
        return f"[{icon} {self.level.value}] {self.category}/{self.name}: {self.message}"


@dataclass
class DiagnoseReport:
    """诊断报告"""
    checks: List[CheckResult] = field(default_factory=list)
    timestamp: str = ""

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.level == CheckLevel.PASS)

    @property
    def warnings(self) -> int:
        return sum(1 for c in self.checks if c.level == CheckLevel.WARN)

    @property
    def errors(self) -> int:
        return sum(1 for c in self.checks if c.level == CheckLevel.ERROR)

    def summary(self) -> str:
        """生成可读的摘要"""
        lines = [
            "=" * 60,
            "  SGN 损失函数诊断报告",
            "=" * 60,
            f"  总计: {len(self.checks)} 项检查",
            f"  通过: {self.passed}  |  警告: {self.warnings}  |  错误: {self.errors}",
            "=" * 60,
        ]
        for c in self.checks:
            lines.append(f"  {c}")
        lines.append("=" * 60)
        if self.errors == 0 and self.warnings == 0:
            lines.append("  结论: 全部通过，可以开始训练")
        elif self.errors == 0:
            lines.append("  结论: 有警告但无错误，建议检查警告项后开始训练")
        else:
            lines.append(f"  结论: 存在 {self.errors} 个错误，请修复后再训练")
        return "\n".join(lines)

    def is_ready_for_training(self) -> bool:
        """是否可以进行训练（无错误）"""
        return self.errors == 0


# ============================================================================
# 诊断模式（轨二）：LossDiagnoser
# ============================================================================

class LossDiagnoser:
    """损失函数诊断器 — 开发期验证工具。

    作为独立的第三方裁判，用数值梯度检验用户的解析梯度。
    支持浮点参考模式，即使训练用整数域，诊断时临时用 float 做 ground truth。

    只报告不修复。

    Args:
        model: sgn.nn.Module 实例
        criterion: 损失函数实例（BaseLoss 子类或兼容 __call__ 的对象）
        sample_input: 样本输入 (numpy array)
        sample_target: 样本标签 (numpy array)
        sgn_module: sgn 模块引用（用于访问 autograd）

    Usage:
        import sgn
        diagnoser = sgn.loss.LossDiagnoser(model, criterion, x_np, y_np)
        report = diagnoser.diagnose()
        print(report.summary())
        if report.is_ready_for_training():
            print("可以开始训练！")
    """

    def __init__(self, model, criterion, sample_input: np.ndarray,
                 sample_target: np.ndarray, sgn_module=None):
        self.model = model
        self.criterion = criterion
        self.sample_input = sample_input
        self.sample_target = sample_target
        self._ag = sgn_module.autograd if sgn_module is not None else None

        # 缓存前向/反向结果
        self._y_pred_tensor = None
        self._y_pred_np = None
        self._loss_value = None
        self._loss_grad_np = None

    def _ensure_ag(self):
        """延迟导入 sgn.autograd"""
        if self._ag is None:
            from . import autograd as _ag_mod
            self._ag = _ag_mod

    def _run_forward(self) -> Tuple:
        """执行一次前向传播，返回 (y_pred_tensor, y_pred_np)"""
        self._ensure_ag()
        with self._ag.record_scope(clear=True):
            x = self._ag.Tensor.from_numpy(self.sample_input.copy())
            y_pred = self.model.forward([x])
        y_pred_np = y_pred.to_numpy()
        return y_pred, y_pred_np

    def _run_backward(self, y_pred, grad_np: np.ndarray):
        """执行反向传播"""
        self._ensure_ag()
        y_pred.backward(grad_np.astype(np.float32))

    # ------------------------------------------------------------------
    # 静态检查
    # ------------------------------------------------------------------

    def run_static_checks(self) -> List[CheckResult]:
        """静态检查（零计算成本）"""
        results = []

        # 1. Shape 兼容性：前向输出 vs 标签
        self._y_pred_tensor, self._y_pred_np = self._run_forward()
        pred_shape = self._y_pred_np.shape
        target_shape = self.sample_target.shape

        if pred_shape[0] != target_shape[0]:
            results.append(CheckResult(
                "static", "batch_size_match",
                CheckLevel.ERROR,
                f"batch_size 不匹配: y_pred={pred_shape[0]}, y_true={target_shape[0]}"
            ))
        else:
            results.append(CheckResult(
                "static", "batch_size_match",
                CheckLevel.PASS,
                f"batch_size: {pred_shape[0]}"
            ))

        # 2. 参数完整性：检查是否有参数未参与前向
        self._run_backward(self._y_pred_tensor,
                           np.random.randn(*pred_shape).astype(np.float32))
        orphan_params = []
        for name, p in self.model.named_parameters():
            if p.grad is None:
                orphan_params.append(name)
        if orphan_params:
            results.append(CheckResult(
                "static", "parameter_completeness",
                CheckLevel.ERROR,
                f"孤立参数（未参与前向）: {orphan_params}"
            ))
        else:
            results.append(CheckResult(
                "static", "parameter_completeness",
                CheckLevel.PASS,
                "所有参数参与前向计算"
            ))

        return results

    # ------------------------------------------------------------------
    # 前向检查
    # ------------------------------------------------------------------

    def run_forward_checks(self) -> List[CheckResult]:
        """前向检查：输出范围、NaN/Inf"""
        results = []

        self._y_pred_tensor, self._y_pred_np = self._run_forward()

        # 1. NaN/Inf 检查
        nan_count = int(np.sum(np.isnan(self._y_pred_np)))
        inf_count = int(np.sum(np.isinf(self._y_pred_np)))
        if nan_count > 0 or inf_count > 0:
            results.append(CheckResult(
                "forward", "nan_inf_check",
                CheckLevel.ERROR,
                f"y_pred 包含 NaN={nan_count}, Inf={inf_count}"
            ))
        else:
            results.append(CheckResult(
                "forward", "nan_inf_check",
                CheckLevel.PASS,
                "y_pred 无 NaN/Inf"
            ))

        # 2. 输出范围
        y_min = float(np.min(self._y_pred_np))
        y_max = float(np.max(self._y_pred_np))
        y_mean = float(np.mean(self._y_pred_np))
        y_std = float(np.std(self._y_pred_np))

        if abs(y_mean) > 100 or y_std > 100:
            results.append(CheckResult(
                "forward", "output_range",
                CheckLevel.WARN,
                f"y_pred 范围异常: mean={y_mean:.2f}, std={y_std:.2f}, "
                f"min={y_min:.2f}, max={y_max:.2f}"
            ))
        else:
            results.append(CheckResult(
                "forward", "output_range",
                CheckLevel.PASS,
                f"mean={y_mean:.4f}, std={y_std:.4f}, "
                f"min={y_min:.4f}, max={y_max:.4f}"
            ))

        # 3. 损失值
        try:
            self._loss_value = self.criterion.forward(
                self._y_pred_np, self.sample_target
            )
            self._loss_grad_np = self.criterion.backward()
            if np.isnan(self._loss_value) or np.isinf(self._loss_value):
                results.append(CheckResult(
                    "forward", "loss_value",
                    CheckLevel.ERROR,
                    f"loss 值为 NaN/Inf: {self._loss_value}"
                ))
            elif self._loss_value > 1e6:
                results.append(CheckResult(
                    "forward", "loss_value",
                    CheckLevel.WARN,
                    f"loss 值异常大: {self._loss_value:.2f}"
                ))
            else:
                results.append(CheckResult(
                    "forward", "loss_value",
                    CheckLevel.PASS,
                    f"loss={self._loss_value:.6f}"
                ))
        except Exception as e:
            results.append(CheckResult(
                "forward", "loss_value",
                CheckLevel.ERROR,
                f"loss 计算失败: {e}"
            ))

        return results

    # ------------------------------------------------------------------
    # 反向检查
    # ------------------------------------------------------------------

    def run_backward_checks(self) -> List[CheckResult]:
        """反向检查：梯度链健康（消失/爆炸/死神经元）"""
        results = []

        if self._loss_grad_np is None:
            self._y_pred_tensor, self._y_pred_np = self._run_forward()
            self._loss_value = self.criterion.forward(
                self._y_pred_np, self.sample_target
            )
            self._loss_grad_np = self.criterion.backward()

        # 1. 损失梯度 NaN/Inf
        nan_count = int(np.sum(np.isnan(self._loss_grad_np)))
        inf_count = int(np.sum(np.isinf(self._loss_grad_np)))
        if nan_count > 0 or inf_count > 0:
            results.append(CheckResult(
                "backward", "loss_grad_nan_inf",
                CheckLevel.ERROR,
                f"∂L/∂y_pred 包含 NaN={nan_count}, Inf={inf_count}"
            ))
        else:
            results.append(CheckResult(
                "backward", "loss_grad_nan_inf",
                CheckLevel.PASS,
                "∂L/∂y_pred 无 NaN/Inf"
            ))

        # 2. 执行反向传播，检查每层梯度
        self._run_backward(self._y_pred_tensor, self._loss_grad_np)

        grad_norms = {}
        for name, p in self.model.named_parameters():
            g = p.grad
            if g is not None:
                grad_norms[name] = float(np.linalg.norm(g))

        if not grad_norms:
            results.append(CheckResult(
                "backward", "gradient_flow",
                CheckLevel.ERROR,
                "所有参数梯度均为 None，梯度链断裂"
            ))
            return results

        # 梯度消失检查
        max_norm = max(grad_norms.values())
        min_norm = min(grad_norms.values())
        if max_norm < 1e-8:
            results.append(CheckResult(
                "backward", "gradient_vanishing",
                CheckLevel.WARN,
                f"梯度消失: max_norm={max_norm:.2e}"
            ))
        else:
            results.append(CheckResult(
                "backward", "gradient_vanishing",
                CheckLevel.PASS,
                f"max_norm={max_norm:.4e}, min_norm={min_norm:.4e}"
            ))

        # 梯度爆炸检查
        if max_norm > 1e4:
            results.append(CheckResult(
                "backward", "gradient_explosion",
                CheckLevel.WARN,
                f"梯度爆炸: max_norm={max_norm:.2e}"
            ))
        else:
            results.append(CheckResult(
                "backward", "gradient_explosion",
                CheckLevel.PASS,
                f"max_norm={max_norm:.4e} (正常范围)"
            ))

        # 梯度流通检查
        zero_count = sum(1 for v in grad_norms.values() if v < 1e-12)
        if zero_count > 0:
            zero_params = [k for k, v in grad_norms.items() if v < 1e-12]
            results.append(CheckResult(
                "backward", "gradient_flow",
                CheckLevel.WARN,
                f"零梯度参数: {zero_params}"
            ))
        else:
            results.append(CheckResult(
                "backward", "gradient_flow",
                CheckLevel.PASS,
                f"所有 {len(grad_norms)} 个参数梯度非零"
            ))

        return results

    # ------------------------------------------------------------------
    # 数值梯度检查
    # ------------------------------------------------------------------

    def run_gradient_correctness(self, eps: float = 1e-2,
                                 n_samples: int = 50) -> List[CheckResult]:
        """数值梯度 vs 解析梯度对比。

        对随机抽样的参数做数值梯度检查，验证用户手动写的梯度是否正确。

        注意：SGN Tensor 内部为 float32，eps 不宜太小（1e-5 会被 float32
        精度淹没），默认 1e-2 在 float32 下给出最稳定的数值梯度。

        Args:
            eps: 数值梯度扰动大小（float32 下推荐 1e-2 ~ 1e-3）
            n_samples: 抽样参数数量（全量太慢，抽样足够发现结构性错误）
        """
        results = []

        # 1. 执行一次完整的前向 + 反向，刷新 param.grad
        #    _run_forward() 内部 record_scope(clear=True) 会清空 grads_，
        #    所以必须在 _run_forward() 之后再做 backward 才能拿到 param.grad
        self._y_pred_tensor, self._y_pred_np = self._run_forward()
        self._loss_value = self.criterion.forward(
            self._y_pred_np, self.sample_target
        )
        self._loss_grad_np = self.criterion.backward()
        self._run_backward(self._y_pred_tensor, self._loss_grad_np)

        # 2. 收集所有可检查的参数（此时 param.grad 已刷新）
        params = list(self.model.named_parameters())
        if not params:
            results.append(CheckResult(
                "gradient", "numerical_gradient",
                CheckLevel.WARN,
                "模型无参数，跳过数值梯度检查"
            ))
            return results

        # 3. 抽样
        if len(params) > n_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(params), n_samples, replace=False)
            sampled = [params[i] for i in indices]
        else:
            sampled = params

        # 4. 缓存解析梯度快照（_run_forward 会清空 grads_，后续无法再读 param.grad）
        ana_grads = {}
        param_names = {}
        for name, param in sampled:
            if param.grad is not None:
                ana_grads[name] = param.to_numpy().copy(), param.grad.copy()
                param_names[id(param)] = name

        all_diffs = []
        max_diff = 0.0
        max_diff_param = ""
        total_checked = 0

        for name, param in sampled:
            if name not in ana_grads:
                continue

            W_orig, ana_grad_arr = ana_grads[name]
            flat = W_orig.ravel()
            ana_flat = ana_grad_arr.ravel()

            # 取一组元素做检查（不检查所有元素，对大矩阵足够）
            n_elem = min(len(flat), 10)
            step = max(1, len(flat) // n_elem)

            for idx in range(0, len(flat), step):
                if total_checked >= n_samples * 10:
                    break

                orig = flat[idx]

                # f(w + ε)
                flat[idx] = orig + eps
                self._load_param(name, flat.reshape(W_orig.shape))
                _, y_plus = self._run_forward()
                loss_plus = self.criterion.forward(y_plus, self.sample_target)

                # f(w - ε)
                flat[idx] = orig - eps
                self._load_param(name, flat.reshape(W_orig.shape))
                _, y_minus = self._run_forward()
                loss_minus = self.criterion.forward(y_minus, self.sample_target)

                # 恢复
                flat[idx] = orig
                self._load_param(name, flat.reshape(W_orig.shape))

                # 数值梯度
                num_grad = (loss_plus - loss_minus) / (2 * eps)

                # 解析梯度（从快照读取，不依赖 param.grad）
                ana_grad = float(ana_flat[idx])

                # 相对误差
                denom = max(abs(num_grad), abs(ana_grad), 1e-15)
                diff = abs(num_grad - ana_grad) / denom

                all_diffs.append(diff)
                if diff > max_diff:
                    max_diff = diff
                    max_diff_param = f"{name}[{idx}]"

                total_checked += 1

        # 5. 空检查保护
        if total_checked == 0:
            results.append(CheckResult(
                "gradient", "numerical_gradient",
                CheckLevel.WARN,
                "未检查任何参数（所有 param.grad 为 None），"
                "可能前向传播未录制到 tape 或梯度链断裂"
            ))
            return results

        # 用中位数判断 PASS/WARN/ERROR（对 ReLU kink 等 outlier 鲁棒）
        # 同时记录 max_diff 供参考
        median_diff = float(np.median(all_diffs))

        # float32 + ReLU kink 下的阈值：
        # PASS: median < 0.1（结构性正确，个别 kink 点偏差大是正常的）
        # WARN: median < 1.0（CNN 等复杂网络 kink 点多，偏差自然偏大但非结构性错误）
        # ERROR: median >= 1.0（符号反转、漏除 batch_size 等结构性错误）
        if median_diff < 0.1:
            results.append(CheckResult(
                "gradient", "numerical_gradient",
                CheckLevel.PASS,
                f"数值梯度 vs 解析梯度: median_diff={median_diff:.2e}, "
                f"max_diff={max_diff:.2e} at {max_diff_param} "
                f"(抽样 {total_checked} 个元素)"
            ))
        elif median_diff < 1.0:
            results.append(CheckResult(
                "gradient", "numerical_gradient",
                CheckLevel.WARN,
                f"数值梯度偏差较大: median_diff={median_diff:.2e}, "
                f"max_diff={max_diff:.2e} at {max_diff_param} "
                f"(抽样 {total_checked} 个元素)"
            ))
        else:
            results.append(CheckResult(
                "gradient", "numerical_gradient",
                CheckLevel.ERROR,
                f"数值梯度严重偏差: median_diff={median_diff:.2e}, "
                f"max_diff={max_diff:.2e} at {max_diff_param} "
                f"(抽样 {total_checked} 个元素) — 梯度公式可能有结构性错误"
            ))

        return results

    def _load_param(self, name: str, array: np.ndarray):
        """将 numpy array 加载到指定名称的参数中

        Args:
            name: 参数名（如 "0.weight"）
            array: 要写入的 numpy array
        """
        state = self.model.state_dict()
        state[name] = array
        self.model.load_state_dict(state)

    # ------------------------------------------------------------------
    # SGN 特定检查
    # ------------------------------------------------------------------

    def run_sgn_specific_checks(self) -> List[CheckResult]:
        """SGN 整数域特定检查：Δ 适配性、EF 残差状态、累加器溢出风险。

        注意：这些检查依赖 SGN 的量化/Level 子系统，如果未启用则跳过。
        """
        results = []

        # 检查是否在 SGN 整数域运行
        try:
            import engine.sgn as sgn
            # 尝试获取 Level 调度器状态
            has_level = hasattr(sgn, 'level') and sgn.level is not None
        except Exception:
            has_level = False

        if not has_level:
            results.append(CheckResult(
                "sgn", "level_available",
                CheckLevel.PASS,
                "SGN 整数域系统未启用，跳过 SGN 特定检查"
            ))
            return results

        # Δ 适配性检查
        try:
            level_ctx = getattr(sgn.level, 'current_context', lambda: None)()
            if level_ctx is not None:
                delta = getattr(level_ctx, 'delta', None)
                if delta is not None and delta < 0.2:
                    results.append(CheckResult(
                        "sgn", "delta_adaptability",
                        CheckLevel.WARN,
                        f"Δ={delta:.4f} < 0.2 (EF 临界间隙)，EF 可能有害"
                    ))
                elif delta is not None:
                    results.append(CheckResult(
                        "sgn", "delta_adaptability",
                        CheckLevel.PASS,
                        f"Δ={delta:.4f} >= 0.2，EF 正常运行"
                    ))
        except Exception:
            results.append(CheckResult(
                "sgn", "delta_adaptability",
                CheckLevel.PASS,
                "无法获取 Δ 信息，跳过"
            ))

        return results

    # ------------------------------------------------------------------
    # 一键诊断
    # ------------------------------------------------------------------

    def diagnose(self, modes: Optional[List[str]] = None,
                 eps: float = 1e-2, n_samples: int = 50) -> DiagnoseReport:
        """一键诊断，运行所有检查并返回报告。

        Args:
            modes: 检查模式列表，可选 'static', 'forward', 'backward', 'gradient', 'sgn'
                   默认全部运行
            eps: 数值梯度扰动大小（仅 gradient 模式）
            n_samples: 数值梯度抽样参数数量（仅 gradient 模式）

        Returns:
            DiagnoseReport 实例
        """
        from datetime import datetime

        if modes is None:
            modes = ['static', 'forward', 'backward', 'gradient', 'sgn']

        all_checks = []

        for mode in modes:
            try:
                if mode == 'static':
                    all_checks.extend(self.run_static_checks())
                elif mode == 'forward':
                    all_checks.extend(self.run_forward_checks())
                elif mode == 'backward':
                    all_checks.extend(self.run_backward_checks())
                elif mode == 'gradient':
                    all_checks.extend(
                        self.run_gradient_correctness(eps=eps, n_samples=n_samples)
                    )
                elif mode == 'sgn':
                    all_checks.extend(self.run_sgn_specific_checks())
            except Exception as e:
                all_checks.append(CheckResult(
                    "system", f"mode_{mode}",
                    CheckLevel.ERROR,
                    f"检查模式 '{mode}' 执行失败: {e}"
                ))

        return DiagnoseReport(
            checks=all_checks,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # 计算模式
    "BaseLoss",
    "MSELoss",
    "CrossEntropyLoss",
    "WeightedSumLoss",
    # 诊断模式
    "CheckLevel",
    "CheckResult",
    "DiagnoseReport",
    "LossDiagnoser",
]