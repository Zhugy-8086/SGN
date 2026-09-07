"""诊断器抓 Bug 演示：注入梯度错误 → 诊断器捕获 → 修复 → 重验证

展示 LossDiagnoser 在实际工作流中的价值：
  1. 故意注入一个常见 bug（CE loss 忘记除以 batch_size）
  2. 用诊断器在训练前捕获它
  3. 修复后重新诊断，确认通过

运行方式：
    cd SGN
    python examples/test_diagnoser_catch_bug.py
"""

import sys
import os
import numpy as np

_sgn_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "engine"))
sys.path.insert(0, _sgn_root)

import sgn

ag = sgn.autograd
nn = sgn.nn


# ============================================================
# Buggy 损失函数：CrossEntropyLoss 忘记除以 batch_size
# 这是实际开发中最常见的梯度错误之一
# ============================================================
class BuggyCrossEntropyLoss:
    """有 bug 的交叉熵损失 — 忘记除以 batch_size。

    与 sgn.loss.CrossEntropyLoss 的唯一区别：
      正确: dY = (softmax - one_hot) / batch_size
      错误: dY = (softmax - one_hot)           ← 梯度放大 batch_size 倍
    """

    def __init__(self):
        self._cached_y_pred = None
        self._cached_y_true = None
        self._cached_softmax = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._cached_y_pred = y_pred
        self._cached_y_true = y_true

        shifted = y_pred - y_pred.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        self._cached_softmax = exp / exp.sum(axis=1, keepdims=True)

        batch_size = y_pred.shape[0]
        if y_true.ndim == 1:
            loss = -np.mean(np.log(
                self._cached_softmax[np.arange(batch_size), y_true] + 1e-15
            ))
        else:
            loss = -np.mean(np.sum(y_true * np.log(self._cached_softmax + 1e-15), axis=1))
        return float(loss)

    def backward(self) -> np.ndarray:
        batch_size = self._cached_y_pred.shape[0]
        y_true = self._cached_y_true

        if y_true.ndim == 1:
            dY = self._cached_softmax.copy()
            dY[np.arange(batch_size), y_true] -= 1.0
            # BUG: 忘记除以 batch_size
            # 正确: return dY / batch_size
            return dY
        else:
            dY = self._cached_softmax - y_true
            return dY


def main():
    print("=" * 70)
    print("诊断器抓 Bug 演示")
    print("=" * 70)

    # ---- 1. 构建 MLP 模型 ----
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

    sample_x = np.random.randn(4, 784).astype(np.float32)
    sample_y = np.random.randint(0, 10, size=4)

    # ================================================================
    # 场景 1: 使用 Buggy CrossEntropyLoss（忘记除以 batch_size）
    # ================================================================
    print("\n" + "=" * 70)
    print("场景 1: Buggy CE Loss — 忘记除以 batch_size")
    print("=" * 70)
    print("  Bug 描述: dY = softmax - one_hot  (缺少 /batch_size)")
    print("  预期: 诊断器应检测到 ERROR，因为梯度放大 batch_size 倍")
    print()

    from sgn.loss import LossDiagnoser

    buggy_criterion = BuggyCrossEntropyLoss()
    diagnoser = LossDiagnoser(model, buggy_criterion, sample_x, sample_y, sgn_module=sgn)
    buggy_report = diagnoser.diagnose(
        modes=['static', 'forward', 'backward', 'gradient'],
        eps=1e-2, n_samples=20
    )
    print(buggy_report.summary())

    # 检查是否抓到 bug（WARN 或 ERROR 都算捕获）
    grad_checks = [c for c in buggy_report.checks if c.name == 'numerical_gradient']
    bug_caught = any(
        c.level.name in ('ERROR', 'WARN') and 'median_diff=' in c.message
        for c in grad_checks
    )

    if bug_caught:
        print("\n  >>> 诊断器成功捕获了梯度错误！")
        for c in grad_checks:
            if 'median_diff=' in c.message:
                import re
                m = re.search(r'median_diff=([\d.e+-]+)', c.message)
                if m:
                    val = float(m.group(1))
                    print(f"  >>> median_diff = {val:.2e} (正常应 < 0.001)")
    else:
        print("\n  >>> 诊断器未捕获到错误 — 需要检查诊断逻辑")

    # ================================================================
    # 场景 2: 修复 — 使用正确的 CrossEntropyLoss
    # ================================================================
    print("\n" + "=" * 70)
    print("场景 2: 修复后 — 正确 CE Loss")
    print("=" * 70)
    print("  修复: 使用 sgn.loss.CrossEntropyLoss()")
    print("  预期: 诊断器应全部 PASS")
    print()

    correct_criterion = sgn.loss.CrossEntropyLoss()
    diagnoser2 = LossDiagnoser(model, correct_criterion, sample_x, sample_y, sgn_module=sgn)
    correct_report = diagnoser2.diagnose(
        modes=['static', 'forward', 'backward', 'gradient'],
        eps=1e-2, n_samples=20
    )
    print(correct_report.summary())

    if correct_report.is_ready_for_training():
        print("\n  >>> 修复后诊断通过，可以开始训练！")
    else:
        print("\n  >>> 修复后仍有问题")

    # ================================================================
    # 对比总结
    # ================================================================
    print("\n" + "=" * 70)
    print("对比总结")
    print("=" * 70)

    buggy_grad = [c for c in buggy_report.checks if c.name == 'numerical_gradient']
    correct_grad = [c for c in correct_report.checks if c.name == 'numerical_gradient']

    b_warn_or_err = buggy_report.errors + buggy_report.warnings
    c_warn_or_err = correct_report.errors + correct_report.warnings
    print(f"  Buggy CE Loss:  {buggy_report.errors} 错误, {buggy_report.warnings} 警告, {buggy_report.passed} 通过")
    print(f"  Correct CE Loss: {correct_report.errors} 错误, {correct_report.warnings} 警告, {correct_report.passed} 通过")

    if buggy_grad and correct_grad:
        import re
        b_msg = buggy_grad[0].message
        c_msg = correct_grad[0].message
        b_m = re.search(r'median_diff=([\d.e+-]+)', b_msg)
        c_m = re.search(r'median_diff=([\d.e+-]+)', c_msg)
        if b_m and c_m:
            b_val = float(b_m.group(1))
            c_val = float(c_m.group(1))
            print(f"\n  数值梯度偏差:")
            print(f"    Buggy:  median_diff = {b_val:.2e}")
            print(f"    Correct: median_diff = {c_val:.2e}")
            print(f"    差异:   {b_val / max(c_val, 1e-10):.0f}×")

    if b_warn_or_err > 0 and c_warn_or_err == 0:
        print(f"\n  >>> 结论: 诊断器成功捕获了梯度 bug，修复后全部通过。")
        print(f"  >>> 这证明了 LossDiagnoser 在实际工作流中的价值。")
        return 0
    else:
        print(f"\n  >>> 诊断器未按预期工作，请检查。")
        return 1


if __name__ == "__main__":
    sys.exit(main())