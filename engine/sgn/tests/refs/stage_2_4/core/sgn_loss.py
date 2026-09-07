"""CrossEntropy Loss with Softmax（numpy 实现，无 autograd）

阶段 2.4 Step 2: MNIST 快速验证

设计:
  - 与 Softmax 融合（数值稳定版）
  - 前向: loss = -log(softmax(logits)[target]) 的 batch 均值
  - 反向: grad = (softmax(logits) - one_hot(target)) / batch_size

参考: stage_1_4_independent/core/loss.py（Python list 版本，本文件是 numpy 版本）
"""
from __future__ import annotations

import numpy as np


class CrossEntropyLoss:
    """CrossEntropy Loss with Softmax (numpy)

    前向:
      1. softmax(logits) = exp(logits - max) / sum(exp(logits - max))
      2. loss = -mean(log(softmax[target]))

    反向:
      grad = (softmax - one_hot(target)) / batch_size
    """

    def __init__(self):
        self._softmax: np.ndarray = None
        self._target: np.ndarray = None

    def forward(self, logits: np.ndarray, target: np.ndarray) -> float:
        """前向: 计算 cross-entropy loss

        Args:
            logits: (B, C) 未归一化 logits
            target: (B,) int 类别索引

        Returns:
            loss: 标量
        """
        # 数值稳定的 softmax
        logits_max = logits.max(axis=1, keepdims=True)
        exp = np.exp(logits - logits_max)
        softmax = exp / exp.sum(axis=1, keepdims=True)
        self._softmax = softmax
        self._target = target

        B = logits.shape[0]
        # loss = -log(softmax[range(B), target]).mean()
        loss = -np.log(softmax[np.arange(B), target] + 1e-12).mean()
        return float(loss)

    def backward(self) -> np.ndarray:
        """反向: 返回 logits 的梯度

        grad = (softmax - one_hot(target)) / batch_size
        """
        B = self._softmax.shape[0]
        grad = self._softmax.copy()
        grad[np.arange(B), self._target] -= 1.0
        grad /= B
        return grad.astype(np.float32)

    def __repr__(self) -> str:
        return "CrossEntropyLoss()"


# ============================================================
# 自检
# ============================================================

def _self_check() -> None:
    print("=" * 60)
    print("sgn_loss.py self-check")
    print("=" * 60)

    loss_fn = CrossEntropyLoss()
    logits = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    target = np.array([2, 0])

    loss = loss_fn.forward(logits, target)
    print(f"  logits={logits.tolist()}, target={target.tolist()}")
    print(f"  loss={loss:.6f}")

    # 手动验证
    import math
    exps0 = [math.exp(v) for v in [1.0, 2.0, 3.0]]
    s0 = sum(exps0)
    p0 = [e / s0 for e in exps0]
    expected_loss = (-math.log(p0[2]) + (-math.log(1/3))) / 2
    assert abs(loss - expected_loss) < 1e-5, f"loss 计算错误: {loss} vs {expected_loss}"
    print(f"  ✓ loss 计算正确（期望 {expected_loss:.6f}）")

    grad = loss_fn.backward()
    expected_grad = []
    for i in range(2):
        row = []
        for j in range(3):
            g = p0[j] if i == 0 else 1/3
            if j == target[i]:
                g -= 1.0
            row.append(g / 2)
        expected_grad.append(row)
    max_err = max(
        abs(grad[i, j] - expected_grad[i][j])
        for i in range(2) for j in range(3)
    )
    assert max_err < 1e-6, f"梯度计算错误: max_err={max_err}"
    print(f"  ✓ 梯度计算正确（max_err={max_err:.2e}）")

    print("\n" + "=" * 60)
    print("sgn_loss.py self-check 全部通过")
    print("=" * 60)


if __name__ == "__main__":
    _self_check()
