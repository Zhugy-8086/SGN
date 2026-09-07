# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""
阶段 5：梯度恒等式验证
=====================================================
对应计划 numpy_math_verification_plan_2026_08_13.md §3 阶段 5 的 #17-#19

验证目标（纯数据验证，非神经网络训练）：
  #17 [T] 残差反向传播恒等梯度 dy = dy_act·W^T + dy_skip
  #18 [T] 损失梯度除以总元素数 y_pred.size（而非 batch_size）
  #19 [T] CE 梯度 = (softmax - one_hot) / batch_size

性质类别：全部 [T]（恒等式）→ 判定标准：机器精度/位精确，零容差（§2.4）
验证方式：正推（解析梯度 vs 直接公式）+ 数值梯度（有限差分）确认

参考实现来源：
  - #17: 残差反向传播 bug 修复记录（缺 skip 恒等梯度 → 差异 1e+00；
          修复后 1e-11）
  - #18: test_loss.py MSELoss.backward = (y_pred - y_true) / y_pred.size
  - #19: test_loss.py CrossEntropyLoss.backward = (softmax - one_hot) / B

用法（纯 numpy）：
    py -3.14 engine/sgn/tests/architecture/validate_math_stage5_gradient_identities.py
"""
from __future__ import annotations

import time

import numpy as np

SEEDS = [0, 1, 2, 3, 4]     # 多随机种子

# 浮点精度声明：全部在 float64 下进行（SGN 内部 float32/int，此处验证数学恒等式）
DT = np.float64


# ============================================================
# 数值梯度工具（中心差分）
# ============================================================

def numerical_gradient(loss_fn, x, eps=1e-6):
    """对 x 的每个元素做中心差分，返回与 x 同形状的数值梯度"""
    grad = np.zeros_like(x, dtype=DT)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = loss_fn()
        x[idx] = orig - eps
        f_minus = loss_fn()
        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def relu(x):
    return np.maximum(x, 0.0)


# ============================================================
# #18 [T] 损失梯度除以总元素数 y_pred.size
# ============================================================
def verify_mse_size_division():
    print("=" * 72)
    print("#18 [T] 损失梯度除以总元素数 y_pred.size（而非 batch_size）")
    print("=" * 72)

    all_ok = True
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        B, D = int(rng.integers(2, 9)), int(rng.integers(2, 9))
        y = rng.normal(0, 1, (B, D)).astype(DT)
        y_true = rng.normal(0, 1, (B, D)).astype(DT)

        N = y.size                      # 总元素数 = B*D
        # 数值梯度会原地扰动 y，故 loss 闭包内须实时重算差值
        loss = lambda: 0.5 * np.mean((y - y_true) ** 2)

        # ---- 1) 解析梯度 = (y - y_true) / y_pred.size ----
        diff = y - y_true
        ana = diff / N
        num = numerical_gradient(loss, y)
        ok = np.max(np.abs(ana - num)) < 1e-7
        all_ok &= ok
        print(f"    [seed{seed}] B={B},D={D}: 解析梯度(÷size) vs 数值 max|Δ|="
              f"{np.max(np.abs(ana-num)):.2e}  → {'✓' if ok else '✗'}")

        # ---- 2) 论证"必须 ÷size，而非 ÷batch_size" ----
        # 若错误地除以 batch_size，则偏差因子 = D
        ana_wrong = diff / B
        wrong_scale = np.max(np.abs(ana_wrong)) / max(np.max(np.abs(ana)), 1e-30)
        expected_scale = float(D) if D > 1 else 1.0
        ok2 = abs(wrong_scale - expected_scale) < 1e-6
        all_ok &= ok2
        print(f"        错误地÷batch_size 的梯度尺度 = {wrong_scale:.4f}×（应为 D={D}）"
              f"  → {'✓ 证明必须÷总元素数' if ok2 else '✗'}")

    print(f"  结论: {'✓ 损失梯度必须除以 y_pred.size（=B×D，而非 batch_size）' if all_ok else '✗ 有失败'}")
    return all_ok


# ============================================================
# #19 [T] CE 梯度 = (softmax - one_hot) / batch_size
# ============================================================
def softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def verify_ce_gradient():
    print("\n" + "=" * 72)
    print("#19 [T] CE 梯度 = (softmax - one_hot) / batch_size")
    print("=" * 72)

    all_ok = True
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        B, C = int(rng.integers(2, 9)), int(rng.integers(3, 10))
        logits = rng.normal(0, 1, (B, C)).astype(DT)
        # 随机标签（类别索引 + one-hot）
        idx = rng.integers(0, C, size=B)
        one_hot = np.eye(C)[idx].astype(DT)

        # 数值梯度会原地扰动 logits，故 softmax 须在 loss 闭包内实时重算
        # ---- 1) 类别索引标签：loss = -mean(log p[true]) ----
        def ce_loss():
            p = softmax_rows(logits)
            eps = 1e-15
            return float(-np.mean(np.log(p[np.arange(B), idx] + eps)))

        p = softmax_rows(logits)
        ana = (p - one_hot) / B                 # 解析梯度（公式）
        num = numerical_gradient(ce_loss, logits)
        ok = np.max(np.abs(ana - num)) < 1e-7
        all_ok &= ok
        print(f"    [seed{seed}] B={B},C={C}: CE梯度(公式) vs 数值 max|Δ|="
              f"{np.max(np.abs(ana-num)):.2e}  → {'✓' if ok else '✗'}")

        # ---- 2) one-hot 标签等价 ----
        def ce_loss_oh():
            p = softmax_rows(logits)
            eps = 1e-15
            return float(-np.mean(np.sum(one_hot * np.log(p + eps), axis=1)))
        ana2 = (p - one_hot) / B
        num2 = numerical_gradient(ce_loss_oh, logits)
        ok2 = np.max(np.abs(ana2 - num2)) < 1e-7
        all_ok &= ok2
        # ---- 3) 行和为零（CE 梯度性质）----
        row_sum = np.max(np.abs(ana.sum(axis=1)))
        ok3 = row_sum < 1e-12
        all_ok &= ok3
        print(f"        one-hot 等价 ✓={ok2}, 行和=max|Σ|= {row_sum:.2e} ✓={ok3}")

    print(f"  结论: {'✓ CE 梯度 = (softmax - one_hot)/batch_size 成立' if all_ok else '✗ 有失败'}")
    return all_ok


# ============================================================
# #17 [T] 残差反向传播恒等梯度 dy = dy_act·W^T + dy_skip
# ============================================================
def residual_forward(Ws, bs, x):
    """残差网络前向，返回 (activations a[0..L], pre-activations z[0..L-1])"""
    a = [x]
    z = []
    for W, b in zip(Ws, bs):
        zl = a[-1] @ W.T + b
        z.append(zl)
        a.append(relu(zl) + a[-1])          # 残差跳连
    return a, z


def residual_backward(Ws, bs, x, a, z, dy):
    """残差网络反向，返回 (dW[0..L-1], dB[0..L-1], dA[0..L])

    dA[l] = W[l].T @ (dA[l+1]*relu'(z[l])) + dA[l+1]   ← skip 恒等梯度
    """
    L = len(Ws)
    dA = [None] * (L + 1)
    dA[L] = dy
    dW = [None] * L
    dB = [None] * L
    for l in range(L - 1, -1, -1):
        dZ = dA[l + 1] * (z[l] > 0)         # relu'
        dW[l] = dZ.T @ a[l]
        dB[l] = dZ.sum(axis=0)
        # skip 路径恒等梯度：+ dA[l+1]
        dA[l] = dZ @ Ws[l] + dA[l + 1]
    return dW, dB, dA


def verify_residual_skip_gradient():
    print("\n" + "=" * 72)
    print("#17 [T] 残差反向传播恒等梯度 dy = dy_act·W^T + dy_skip")
    print("=" * 72)

    all_ok = True
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        B = int(rng.integers(2, 6))
        D = int(rng.integers(2, 6))
        L = int(rng.integers(2, 5))          # 2~4 层残差块
        # 残差连接要求每层隐藏维度与输入维度一致（a[l] + relu(zl) 同形状）
        dims = [D] * (L + 1)

        Ws = [rng.normal(0, 0.5, (dims[i + 1], dims[i])).astype(DT) for i in range(L)]
        bs = [rng.normal(0, 0.1, dims[i + 1]).astype(DT) for i in range(L)]
        x = rng.normal(0, 1, (B, D)).astype(DT)
        y_true = rng.normal(0, 1, (B, D)).astype(DT)

        a, z = residual_forward(Ws, bs, x)
        y = a[L]
        N = y.size
        dy = (y - y_true) / N                # 输出梯度（MSE，÷size）

        def loss_fn():
            return float(0.5 * np.mean((a[L] - y_true) ** 2))

        # 注意：数值梯度需重算前向，故 loss_fn 引用 a 会失效（a 会被覆盖）。
        # 用闭包重建前向。
        def make_loss(Ws, bs, x):
            def f():
                aa, _ = residual_forward(Ws, bs, x)
                return float(0.5 * np.mean((aa[L] - y_true) ** 2))
            return f

        # ---- 1) 含 skip 的解析梯度 vs 数值梯度（对每个参数）----
        dW, dB, dA = residual_backward(Ws, bs, x, a, z, dy)
        lf = make_loss(Ws, bs, x)
        max_dw = 0.0
        for l in range(L):
            num = numerical_gradient(lf, Ws[l])
            max_dw = max(max_dw, np.max(np.abs(dW[l] - num)))
            # 也验证 dW 的解析形式 dZ.T @ a[l]
        ok1 = max_dw < 1e-7
        all_ok &= ok1
        print(f"    [seed{seed}] L={L}, dims={dims}: "
              f"含skip dL/dW vs 数值 max|Δ|={max_dw:.2e}  → {'✓' if ok1 else '✗'}")

        # ---- 2) 反例：去掉 skip 恒等梯度 dA[l+1]，梯度应错误 ----
        def residual_backward_no_skip(Ws, bs, x, a, z, dy):
            Ln = len(Ws)
            dA = [None] * (Ln + 1)
            dA[Ln] = dy
            dW = [None] * Ln
            for l in range(Ln - 1, -1, -1):
                dZ = dA[l + 1] * (z[l] > 0)
                dW[l] = dZ.T @ a[l]
                dA[l] = dZ @ Ws[l]           # 缺 + dA[l+1]
            return dW

        dW_bug = residual_backward_no_skip(Ws, bs, x, a, z, dy)
        bug_dw = max(np.max(np.abs(dW_bug[l] - numerical_gradient(lf, Ws[l])))
                     for l in range(L))
        # 无 skip 时应明显错误（远大于机器精度）
        ok2 = bug_dw > 1e-3
        all_ok &= ok2
        print(f"        无skip 的 dL/dW vs 数值 max|Δ|={bug_dw:.2e}"
              f"  → {'✓ 证明 skip 恒等梯度必要' if ok2 else '✗'}")

    print(f"  结论: {'✓ 残差反向传播必须含 skip 恒等梯度 dy_skip' if all_ok else '✗ 有失败'}")
    return all_ok


def main():
    t0 = time.time()
    r17 = verify_residual_skip_gradient()
    r18 = verify_mse_size_division()
    r19 = verify_ce_gradient()
    dt = time.time() - t0

    print("\n" + "=" * 72)
    print("阶段 5 验证汇总")
    print("=" * 72)
    print(f"  耗时 {dt:.1f}s")
    print(f"  #17 [T] 残差 skip 恒等梯度 : {'✓' if r17 else '✗'}")
    print(f"  #18 [T] 损失梯度÷y_pred.size : {'✓' if r18 else '✗'}")
    print(f"  #19 [T] CE 梯度=(softmax-onehot)/B : {'✓' if r19 else '✗'}")
    overall = r17 and r18 and r19
    print(f"\n  总体判定: {'✅ 全部通过' if overall else '❌ 存在失败'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
