# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""测试 sgn.loss 模块 — 损失函数正确性

覆盖：
  - MSELoss: forward/backward 正确性、边界条件
  - CrossEntropyLoss: forward/backward（类别索引 + one-hot）、梯度性质
  - WeightedSumLoss: 加权组合正确性
  - 数值梯度验证：解析梯度 vs 有限差分
  - 错误处理：forward 前调用 backward 抛异常

运行：
    cd engine/sgn
    python tests/test_loss.py
"""

import sys
import os
import numpy as np

# 确保项目根（含 engine/ 包）在 sys.path，支持 `python tests/test_loss.py` 直接运行
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# 用相对导入 engine.sgn 代替顶层 `import sgn`，避免在 pytest 全量收集时
# 与已加载的 engine.sgn 包产生 sys.modules 冲突（顶层 `import sgn` 会
# 解析到别名 _sgn_native 或产生 INTERNALERROR）。
import engine.sgn as sgn
from engine.sgn.loss import (
    BaseLoss, MSELoss, CrossEntropyLoss, WeightedSumLoss,
    CheckLevel, CheckResult, DiagnoseReport, LossDiagnoser,
)
print(f"[OK] sgn.loss 导入成功")

np.random.seed(42)

# ============================================================
# 测试框架
# ============================================================

_passed = 0
_failed = 0


def _check(name, condition, detail=""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  [PASS] {name}")
    else:
        _failed += 1
        print(f"  [FAIL] {name}: {detail}")


# ============================================================
# MSELoss 测试
# ============================================================

def test_mse_forward():
    """MSELoss 前向：与 numpy 手动计算对比"""
    print("\n--- test_mse_forward ---")
    criterion = MSELoss()
    y_pred = np.random.randn(8, 5).astype(np.float64)
    y_true = np.random.randn(8, 5).astype(np.float64)

    loss = criterion.forward(y_pred, y_true)

    # 手动计算: 0.5 * mean((y_pred - y_true)^2)
    diff = y_pred - y_true
    expected = 0.5 * np.mean(diff * diff)

    _check("forward 值匹配", abs(loss - expected) < 1e-12,
           f"got {loss}, expected {expected}")
    _check("forward 返回 float", isinstance(loss, float),
           f"got {type(loss)}")


def test_mse_backward():
    """MSELoss 反向：梯度 shape 和数值正确性"""
    print("\n--- test_mse_backward ---")
    criterion = MSELoss()
    y_pred = np.random.randn(8, 5).astype(np.float64)
    y_true = np.random.randn(8, 5).astype(np.float64)

    criterion.forward(y_pred, y_true)
    grad = criterion.backward()

    # 手动计算: (y_pred - y_true) / y_pred.size
    expected = (y_pred - y_true) / y_pred.size

    _check("backward shape", grad.shape == y_pred.shape,
           f"got {grad.shape}, expected {y_pred.shape}")
    _check("backward 数值", np.max(np.abs(grad - expected)) < 1e-12,
           f"max_diff={np.max(np.abs(grad - expected)):.2e}")


def test_mse_zero_loss():
    """MSELoss 边界：y_pred == y_true 时 loss=0, grad=0"""
    print("\n--- test_mse_zero_loss ---")
    criterion = MSELoss()
    y = np.random.randn(4, 3).astype(np.float64)

    loss, grad = criterion(y, y)

    _check("零损失", abs(loss) < 1e-15, f"loss={loss}")
    _check("零梯度", np.max(np.abs(grad)) < 1e-15,
           f"max|grad|={np.max(np.abs(grad)):.2e}")


# ============================================================
# CrossEntropyLoss 测试
# ============================================================

def _manual_ce_loss(y_pred, y_true):
    """手动计算 CE loss，返回 (loss, softmax)"""
    shifted = y_pred - y_pred.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    softmax = exp / exp.sum(axis=1, keepdims=True)

    if y_true.ndim == 1:
        batch_size = y_pred.shape[0]
        loss = -np.mean(np.log(softmax[np.arange(batch_size), y_true] + 1e-15))
    else:
        loss = -np.mean(np.sum(y_true * np.log(softmax + 1e-15), axis=1))

    return float(loss), softmax


def test_ce_forward_class_index():
    """CrossEntropyLoss 前向（类别索引标签）"""
    print("\n--- test_ce_forward_class_index ---")
    criterion = CrossEntropyLoss()
    y_pred = np.random.randn(8, 10).astype(np.float64)
    y_true = np.random.randint(0, 10, size=8)

    loss = criterion.forward(y_pred, y_true)
    expected, _ = _manual_ce_loss(y_pred, y_true)

    _check("forward 值匹配", abs(loss - expected) < 1e-12,
           f"got {loss}, expected {expected}")


def test_ce_forward_onehot():
    """CrossEntropyLoss 前向（one-hot 标签）"""
    print("\n--- test_ce_forward_onehot ---")
    criterion = CrossEntropyLoss()
    y_pred = np.random.randn(8, 10).astype(np.float64)
    y_true_idx = np.random.randint(0, 10, size=8)
    y_true = np.eye(10)[y_true_idx].astype(np.float64)

    loss = criterion.forward(y_pred, y_true)
    expected, _ = _manual_ce_loss(y_pred, y_true)

    _check("forward 值匹配", abs(loss - expected) < 1e-12,
           f"got {loss}, expected {expected}")

    # 两种标签格式结果一致
    loss_idx = criterion.forward(y_pred, y_true_idx)
    _check("类别索引 vs one-hot 一致", abs(loss - loss_idx) < 1e-12,
           f"idx={loss_idx}, onehot={loss}")


def test_ce_backward_class_index():
    """CrossEntropyLoss 反向（类别索引）"""
    print("\n--- test_ce_backward_class_index ---")
    criterion = CrossEntropyLoss()
    y_pred = np.random.randn(8, 10).astype(np.float64)
    y_true = np.random.randint(0, 10, size=8)

    criterion.forward(y_pred, y_true)
    grad = criterion.backward()

    # 手动计算: (softmax - one_hot) / batch_size
    _, softmax = _manual_ce_loss(y_pred, y_true)
    one_hot = np.zeros_like(y_pred)
    one_hot[np.arange(8), y_true] = 1.0
    expected = (softmax - one_hot) / 8

    _check("backward shape", grad.shape == y_pred.shape,
           f"got {grad.shape}, expected {y_pred.shape}")
    _check("backward 数值", np.max(np.abs(grad - expected)) < 1e-12,
           f"max_diff={np.max(np.abs(grad - expected)):.2e}")


def test_ce_backward_onehot():
    """CrossEntropyLoss 反向（one-hot）"""
    print("\n--- test_ce_backward_onehot ---")
    criterion = CrossEntropyLoss()
    y_pred = np.random.randn(8, 10).astype(np.float64)
    y_true_idx = np.random.randint(0, 10, size=8)
    y_true = np.eye(10)[y_true_idx].astype(np.float64)

    criterion.forward(y_pred, y_true)
    grad = criterion.backward()

    # 手动计算: (softmax - y_true) / batch_size
    _, softmax = _manual_ce_loss(y_pred, y_true)
    expected = (softmax - y_true) / 8

    _check("backward 数值", np.max(np.abs(grad - expected)) < 1e-12,
           f"max_diff={np.max(np.abs(grad - expected)):.2e}")

    # 两种标签格式梯度一致
    criterion.forward(y_pred, y_true_idx)
    grad_idx = criterion.backward()
    _check("类别索引 vs one-hot 梯度一致",
           np.max(np.abs(grad - grad_idx)) < 1e-12,
           f"max_diff={np.max(np.abs(grad - grad_idx)):.2e}")


def test_ce_gradient_sum_zero():
    """CrossEntropyLoss 梯度性质：每行梯度之和为 0"""
    print("\n--- test_ce_gradient_sum_zero ---")
    criterion = CrossEntropyLoss()
    y_pred = np.random.randn(8, 10).astype(np.float64)
    y_true = np.random.randint(0, 10, size=8)

    _, grad = criterion(y_pred, y_true)
    row_sums = grad.sum(axis=1)

    _check("行和为零", np.max(np.abs(row_sums)) < 1e-12,
           f"max|row_sum|={np.max(np.abs(row_sums)):.2e}")


def test_ce_perfect_prediction():
    """CrossEntropyLoss 边界：完美预测时 loss≈0"""
    print("\n--- test_ce_perfect_prediction ---")
    criterion = CrossEntropyLoss()
    y_true = np.array([0, 1, 2, 3])
    # 正确类 logit 远大于其他类
    y_pred = np.full((4, 4), -100.0)
    for i, t in enumerate(y_true):
        y_pred[i, t] = 100.0

    loss, grad = criterion(y_pred, y_true)

    _check("完美预测 loss≈0", loss < 1e-40,
           f"loss={loss}")
    # softmax ≈ one_hot，梯度接近 0
    _check("完美预测梯度≈0", np.max(np.abs(grad)) < 1e-30,
           f"max|grad|={np.max(np.abs(grad)):.2e}")


# ============================================================
# WeightedSumLoss 测试
# ============================================================

def test_weighted_sum():
    """WeightedSumLoss 加权组合正确性"""
    print("\n--- test_weighted_sum ---")
    mse = MSELoss()
    ce = CrossEntropyLoss()
    criterion = WeightedSumLoss(
        [("mse", mse), ("ce", ce)],
        weights=[0.3, 0.7],
    )

    y_pred = np.random.randn(8, 10).astype(np.float64)
    y_true_mse = np.random.randn(8, 10).astype(np.float64)
    y_true_ce = np.random.randint(0, 10, size=8)

    loss, grad = criterion(y_pred, [y_true_mse, y_true_ce])

    # 手动计算
    loss_mse = mse.forward(y_pred, y_true_mse)
    grad_mse = mse.backward()
    loss_ce = ce.forward(y_pred, y_true_ce)
    grad_ce = ce.backward()
    expected_loss = 0.3 * loss_mse + 0.7 * loss_ce
    expected_grad = 0.3 * grad_mse + 0.7 * grad_ce

    _check("加权 loss 匹配", abs(loss - expected_loss) < 1e-12,
           f"got {loss}, expected {expected_loss}")
    _check("加权 grad 匹配", np.max(np.abs(grad - expected_grad)) < 1e-12,
           f"max_diff={np.max(np.abs(grad - expected_grad)):.2e}")

    # 默认权重=1.0
    criterion_default = WeightedSumLoss([("mse", MSELoss()), ("ce", CrossEntropyLoss())])
    _check("默认权重=1.0", all(w == 1.0 for w in criterion_default.weights),
           f"weights={criterion_default.weights}")

    # 权重长度不匹配抛异常
    try:
        WeightedSumLoss([("mse", MSELoss())], weights=[1.0, 2.0])
        _check("权重不匹配报错", False, "未抛异常")
    except ValueError:
        _check("权重不匹配报错", True)


# ============================================================
# 错误处理
# ============================================================

def test_backward_before_forward():
    """错误处理：forward 前调用 backward 抛异常"""
    print("\n--- test_backward_before_forward ---")
    mse = MSELoss()
    try:
        mse.backward()
        _check("MSE 抛异常", False, "未抛异常")
    except RuntimeError:
        _check("MSE 抛异常", True)

    ce = CrossEntropyLoss()
    try:
        ce.backward()
        _check("CE 抛异常", False, "未抛异常")
    except RuntimeError:
        _check("CE 抛异常", True)

    wsl = WeightedSumLoss([("mse", MSELoss())])
    try:
        wsl.backward()
        _check("WeightedSum 抛异常", False, "未抛异常")
    except RuntimeError:
        _check("WeightedSum 抛异常", True)


# ============================================================
# 数值梯度验证
# ============================================================

def test_numerical_gradient():
    """数值梯度验证：所有损失函数的解析梯度 vs 有限差分"""
    print("\n--- test_numerical_gradient ---")

    eps = 1e-6
    n_samples = 20  # 抽样元素数

    def numerical_check(name, criterion, y_pred, y_true):
        """对 y_pred 的随机抽样元素做数值梯度检查"""
        # 解析梯度
        loss = criterion.forward(y_pred, y_true)
        ana_grad = criterion.backward()

        # 数值梯度
        flat = y_pred.ravel().copy()
        rng = np.random.default_rng(42)
        indices = rng.choice(len(flat), min(n_samples, len(flat)), replace=False)

        max_diff = 0.0
        for idx in indices:
            orig = flat[idx]

            flat[idx] = orig + eps
            loss_plus = criterion.forward(flat.reshape(y_pred.shape), y_true)

            flat[idx] = orig - eps
            loss_minus = criterion.forward(flat.reshape(y_pred.shape), y_true)

            flat[idx] = orig  # 恢复

            num_grad = (loss_plus - loss_minus) / (2 * eps)
            ana_val = float(ana_grad.ravel()[idx])
            denom = max(abs(num_grad), abs(ana_val), 1e-15)
            diff = abs(num_grad - ana_val) / denom
            if diff > max_diff:
                max_diff = diff

        _check(f"{name} 数值梯度 (max_diff={max_diff:.2e})", max_diff < 1e-5,
               f"max_diff={max_diff:.2e}")

    # MSE
    y_pred = np.random.randn(8, 5).astype(np.float64)
    y_true = np.random.randn(8, 5).astype(np.float64)
    numerical_check("MSELoss", MSELoss(), y_pred, y_true)

    # CE (类别索引)
    y_pred = np.random.randn(8, 10).astype(np.float64)
    y_true = np.random.randint(0, 10, size=8)
    numerical_check("CELoss(class_index)", CrossEntropyLoss(), y_pred, y_true)

    # CE (one-hot)
    y_true_oh = np.eye(10)[y_true].astype(np.float64)
    numerical_check("CELoss(onehot)", CrossEntropyLoss(), y_pred, y_true_oh)

    # WeightedSum
    y_true_mse = np.random.randn(8, 10).astype(np.float64)
    wsl = WeightedSumLoss(
        [("mse", MSELoss()), ("ce", CrossEntropyLoss())],
        weights=[0.5, 0.5],
    )
    numerical_check("WeightedSumLoss", wsl, y_pred, [y_true_mse, y_true])


# ============================================================
# LossDiagnoser 测试 — 诊断器检测能力
# ============================================================

def _build_simple_model():
    """构建简单的 2 层 MLP 用于诊断器测试"""
    return sgn.nn.Sequential(
        sgn.nn.Linear(4, 8),
        sgn.nn.ReLU(),
        sgn.nn.Linear(8, 2),
    )


class _WrongMSELoss_NoSizeDiv(MSELoss):
    """错误 1：梯度未除以 y_pred.size（常见 bug）"""
    def backward(self) -> np.ndarray:
        if self._cached_y_pred is None:
            raise RuntimeError("forward() must be called before backward()")
        return self._cached_y_pred - self._cached_y_true  # 缺少 / y_pred.size


class _WrongMSELoss_SignFlipped(MSELoss):
    """错误 2：梯度符号反转"""
    def backward(self) -> np.ndarray:
        grad = super().backward()
        return -grad


class _CorrectMSELoss(MSELoss):
    """正确实现（对照组）"""
    pass


def test_diagnoser_static_checks():
    """静态检查：shape 匹配、参数完整性"""
    print("\n--- test_diagnoser_static_checks ---")
    model = _build_simple_model()
    criterion = MSELoss()
    x_np = np.random.randn(3, 4).astype(np.float32)
    y_np = np.random.randn(3, 2).astype(np.float32)

    diagnoser = LossDiagnoser(model, criterion, x_np, y_np)
    results = diagnoser.run_static_checks()

    # 应该有 batch_size_match 和 parameter_completeness 两项
    names = {r.name for r in results}
    _check("包含 batch_size_match", "batch_size_match" in names)
    _check("包含 parameter_completeness", "parameter_completeness" in names)

    batch_check = next(r for r in results if r.name == "batch_size_match")
    _check("batch_size 匹配通过", batch_check.level == CheckLevel.PASS,
           batch_check.message)

    param_check = next(r for r in results if r.name == "parameter_completeness")
    _check("参数完整性通过", param_check.level == CheckLevel.PASS,
           param_check.message)


def test_diagnoser_forward_checks():
    """前向检查：NaN/Inf、输出范围、loss 值"""
    print("\n--- test_diagnoser_forward_checks ---")
    model = _build_simple_model()
    criterion = MSELoss()
    x_np = np.random.randn(3, 4).astype(np.float32)
    y_np = np.random.randn(3, 2).astype(np.float32)

    diagnoser = LossDiagnoser(model, criterion, x_np, y_np)
    results = diagnoser.run_forward_checks()

    names = {r.name for r in results}
    _check("包含 nan_inf_check", "nan_inf_check" in names)
    _check("包含 output_range", "output_range" in names)
    _check("包含 loss_value", "loss_value" in names)

    for r in results:
        _check(f"forward/{r.name} 非 ERROR", r.level != CheckLevel.ERROR,
               f"{r.level.value}: {r.message}")


def test_diagnoser_backward_checks():
    """反向检查：梯度消失/爆炸/流通"""
    print("\n--- test_diagnoser_backward_checks ---")
    model = _build_simple_model()
    criterion = MSELoss()
    x_np = np.random.randn(3, 4).astype(np.float32)
    y_np = np.random.randn(3, 2).astype(np.float32)

    diagnoser = LossDiagnoser(model, criterion, x_np, y_np)
    results = diagnoser.run_backward_checks()

    names = {r.name for r in results}
    _check("包含 loss_grad_nan_inf", "loss_grad_nan_inf" in names)
    _check("包含 gradient_vanishing", "gradient_vanishing" in names)
    _check("包含 gradient_explosion", "gradient_explosion" in names)
    _check("包含 gradient_flow", "gradient_flow" in names)

    for r in results:
        _check(f"backward/{r.name} 非 ERROR", r.level != CheckLevel.ERROR,
               f"{r.level.value}: {r.message}")


def test_diagnoser_detects_wrong_gradient():
    """核心：注入错误梯度，验证诊断器能检测到"""
    print("\n--- test_diagnoser_detects_wrong_gradient ---")

    x_np = np.random.randn(3, 4).astype(np.float32)
    y_np = np.random.randn(3, 2).astype(np.float32)

    # ---- 错误 1：梯度未除以 y_pred.size ----
    model1 = _build_simple_model()
    wrong_criterion1 = _WrongMSELoss_NoSizeDiv()
    diagnoser1 = LossDiagnoser(model1, wrong_criterion1, x_np, y_np)
    results1 = diagnoser1.run_gradient_correctness(eps=1e-2, n_samples=20)
    grad_check1 = results1[0]

    # 未除以 size 的梯度偏差应该很大（factor = y_pred.size = 6）
    _check("错误1(未除size) 被检测为 ERROR/WARN",
           grad_check1.level in (CheckLevel.ERROR, CheckLevel.WARN),
           f"level={grad_check1.level.value}, {grad_check1.message}")

    # ---- 错误 2：梯度符号反转 ----
    model2 = _build_simple_model()
    wrong_criterion2 = _WrongMSELoss_SignFlipped()
    diagnoser2 = LossDiagnoser(model2, wrong_criterion2, x_np, y_np)
    results2 = diagnoser2.run_gradient_correctness(eps=1e-2, n_samples=20)
    grad_check2 = results2[0]

    # 符号反转的 relative diff ≈ 2.0
    _check("错误2(符号反转) 被检测为 ERROR",
           grad_check2.level == CheckLevel.ERROR,
           f"level={grad_check2.level.value}, {grad_check2.message}")


def test_diagnoser_correct_gradient():
    """正确梯度应通过诊断"""
    print("\n--- test_diagnoser_correct_gradient ---")

    model = _build_simple_model()
    criterion = _CorrectMSELoss()
    x_np = np.random.randn(3, 4).astype(np.float32)
    y_np = np.random.randn(3, 2).astype(np.float32)

    diagnoser = LossDiagnoser(model, criterion, x_np, y_np)
    results = diagnoser.run_gradient_correctness(eps=1e-2, n_samples=20)
    grad_check = results[0]

    _check("正确梯度 诊断通过", grad_check.level == CheckLevel.PASS,
           f"level={grad_check.level.value}, {grad_check.message}")


def test_diagnoser_report():
    """报告格式：summary() 输出和 is_ready_for_training()"""
    print("\n--- test_diagnoser_report ---")

    model = _build_simple_model()
    criterion = MSELoss()
    x_np = np.random.randn(3, 4).astype(np.float32)
    y_np = np.random.randn(3, 2).astype(np.float32)

    # 仅静态 + 前向（快速模式）
    diagnoser = LossDiagnoser(model, criterion, x_np, y_np)
    report = diagnoser.diagnose(modes=['static', 'forward'])

    _check("report 非空", len(report.checks) > 0)
    _check("passed >= 0", report.passed >= 0)
    _check("errors == 0", report.errors == 0,
           f"errors={report.errors}")
    _check("is_ready_for_training()", report.is_ready_for_training())

    summary = report.summary()
    _check("summary 包含标题", "SGN 损失函数诊断报告" in summary)
    _check("summary 包含结论", "结论" in summary)

    # 全量诊断
    full_report = diagnoser.diagnose()  # 默认全部模式
    _check("全量诊断 无 ERROR", full_report.errors == 0,
           f"errors={full_report.errors}, checks={len(full_report.checks)}")


def test_diagnoser_gradlogger_integration():
    """GradLogger 集成：diagnose/check_gradient/check_forward 方法"""
    print("\n--- test_diagnoser_gradlogger_integration ---")
    from engine.sgn.logger import GradLogger

    model = _build_simple_model()
    criterion = MSELoss()
    x_np = np.random.randn(3, 4).astype(np.float32)
    y_np = np.random.randn(3, 2).astype(np.float32)

    # 关闭日志输出，只测试返回值
    logger = GradLogger(tag="TestLog", enabled=False)

    # check_gradient
    grad_result = logger.check_gradient(model, criterion, x_np, y_np,
                                        eps=1e-2, n_samples=10)
    _check("check_gradient 返回 CheckResult", grad_result is not None)
    _check("check_gradient name=numerical_gradient",
           grad_result.name == "numerical_gradient")
    _check("check_gradient 正确梯度 PASS",
           grad_result.level == CheckLevel.PASS,
           f"level={grad_result.level.value}, {grad_result.message}")

    # check_forward
    fwd_report = logger.check_forward(model, criterion, x_np, y_np)
    _check("check_forward 返回 DiagnoseReport", fwd_report is not None)
    _check("check_forward 无错误", fwd_report.errors == 0)

    # diagnose（全模式）
    report = logger.diagnose(model, criterion, x_np, y_np,
                             eps=1e-2, n_samples=10)
    _check("diagnose 返回 DiagnoseReport", report is not None)
    _check("diagnose 无错误", report.errors == 0,
           f"errors={report.errors}")
    _check("diagnose 包含检查项", len(report.checks) > 0)


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  sgn.loss 模块测试 — 损失函数正确性")
    print("=" * 60)

    test_mse_forward()
    test_mse_backward()
    test_mse_zero_loss()
    test_ce_forward_class_index()
    test_ce_forward_onehot()
    test_ce_backward_class_index()
    test_ce_backward_onehot()
    test_ce_gradient_sum_zero()
    test_ce_perfect_prediction()
    test_weighted_sum()
    test_backward_before_forward()
    test_numerical_gradient()

    # ---- LossDiagnoser 测试 ----
    print("\n" + "=" * 60)
    print("  LossDiagnoser 诊断器检测能力测试")
    print("=" * 60)
    test_diagnoser_static_checks()
    test_diagnoser_forward_checks()
    test_diagnoser_backward_checks()
    test_diagnoser_detects_wrong_gradient()
    test_diagnoser_correct_gradient()
    test_diagnoser_report()
    test_diagnoser_gradlogger_integration()

    print("\n" + "=" * 60)
    total = _passed + _failed
    print(f"  结果: {_passed}/{total} passed", end="")
    if _failed == 0:
        print(" — All tests passed!")
    else:
        print(f" — {_failed} tests failed.")
    print("=" * 60)
    sys.exit(0 if _failed == 0 else 1)
