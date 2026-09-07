"""MNIST MLP 训练示例 — 端到端验证版

使用 sgn.nn 标准层 (Linear, ReLU, Sequential)、sgn.loss 内置损失函数和
record_scope 上下文管理器构建一个 3 层 MLP 并用合成数据训练。

端到端验证流程：
  1. 训练前：LossDiagnoser 完整诊断（静态/前向/反向/数值梯度）
  2. 训练中：每 N 步周期性数值梯度抽检，监控梯度健康度
  3. 训练后：梯度流通 + 最终数值梯度验证

运行方式：
    cd SGN
    python examples/mnist_mlp.py

如果你有真实的 MNIST 数据，参考 engine/sgn/tests/refs/baseline/common/data_loader.py
（2026-08-16 legacy 独立后 data_loader 迁至 tests/refs/baseline/common/）
"""

import sys
import os
import time
import numpy as np

# ---- 路径设置 ----
_sgn_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "engine"))
sys.path.insert(0, _sgn_root)

import sgn

ag = sgn.autograd
nn = sgn.nn

np.random.seed(42)


def _run_gradient_spot_check(model, criterion, x_np, y_np, *,
                             eps=1e-2, n_samples=10):
    """快速数值梯度抽检：在当前 batch 上验证解析梯度正确性。

    返回 (median_diff, max_diff, max_param_name) 或 None（检查失败时）。
    """
    from sgn.loss import LossDiagnoser
    diagnoser = LossDiagnoser(model, criterion, x_np, y_np, sgn_module=sgn)
    report = diagnoser.diagnose(modes=['gradient'], eps=eps, n_samples=n_samples)
    for c in report.checks:
        if c.name == 'numerical_gradient':
            # 从 summary 行提取数值
            msg = c.message
            if 'median_diff=' in msg:
                parts = msg.split(', ')
                median_part = [p for p in parts if 'median_diff=' in p][0]
                max_part = [p for p in parts if 'max_diff=' in p][0]
                median_diff = float(median_part.split('=')[1])
                max_diff = float(max_part.split('=')[1].split()[0])
                name_part = [p for p in parts if 'at ' in p][0]
                max_param = name_part.split('at ')[1].split(' ')[0]
                return median_diff, max_diff, max_param
    return None


def main():
    print("=" * 60)
    print("MNIST MLP 训练示例 — 端到端验证")
    print(f"SGN 版本: {sgn.version()}")
    print("=" * 60)

    # ---- 1. 定义网络 ----
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    print(f"\n模型结构:\n{model}")

    # ---- 2. 超参数 ----
    steps = 100
    batch_size = 32
    lr = 0.01
    check_interval = 25  # 每 N 步做一次梯度抽检

    print(f"\n训练配置: steps={steps}, batch_size={batch_size}, lr={lr}")
    print(f"梯度抽检: 每 {check_interval} 步 (n_samples=10)")

    # ---- 3. 损失函数 + 优化器 ----
    criterion = sgn.loss.CrossEntropyLoss()
    optimizer = sgn.optim.SGD(model, lr=lr)

    # ---- 4. 训练前：完整诊断 ----
    print("\n" + "=" * 60)
    print("阶段 1/3: 训练前完整诊断")
    print("=" * 60)
    from sgn.loss import LossDiagnoser
    sample_x = np.random.randn(4, 784).astype(np.float32)
    sample_y = np.random.randint(0, 10, size=4)
    diagnoser = LossDiagnoser(model, criterion, sample_x, sample_y, sgn_module=sgn)
    report = diagnoser.diagnose(modes=['static', 'forward', 'backward', 'gradient'],
                                eps=1e-2, n_samples=20)
    print(report.summary())
    if not report.is_ready_for_training():
        print("\n[FAIL] 训练前诊断未通过，请修复梯度问题后再训练")
        return 1

    # 记录基线梯度指标
    grad_history = []
    baseline = _run_gradient_spot_check(model, criterion, sample_x, sample_y,
                                        eps=1e-2, n_samples=10)
    if baseline:
        grad_history.append(("init", baseline[0], baseline[1], baseline[2]))
        print(f"  基线数值梯度: median_diff={baseline[0]:.2e}, max_diff={baseline[1]:.2e}")

    # ---- 5. 合成数据 ----
    data_rng = np.random.RandomState(42)
    x_all = data_rng.randn(steps, batch_size, 784).astype(np.float32)
    y_all = data_rng.randint(0, 10, (steps, batch_size)).astype(np.int64)

    # ---- 6. 训练循环 + 周期性抽检 ----
    print("\n" + "=" * 60)
    print("阶段 2/3: 训练 + 周期性梯度抽检")
    print("=" * 60)

    losses = []
    t0 = time.perf_counter()

    for step in range(steps):
        x_np = x_all[step]
        y_np = y_all[step]

        # 前向传播
        with ag.record_scope(clear=True):
            x = ag.Tensor.from_numpy(x_np.copy())
            logits = model.forward([x])

        # 损失 + 梯度
        out_np = logits.to_numpy()
        loss, dY = criterion(out_np, y_np)

        # 反向传播
        logits.backward(dY.astype(np.float32))

        # 权重更新
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss)

        # 周期性梯度抽检
        if (step + 1) % check_interval == 0:
            result = _run_gradient_spot_check(model, criterion, x_np, y_np,
                                              eps=1e-2, n_samples=10)
            if result:
                median_diff, max_diff, max_param = result
                grad_history.append((step + 1, median_diff, max_diff, max_param))
                # 退化检测：如果 median_diff 超过基线的 5 倍，发出警告
                if baseline and median_diff > baseline[0] * 5:
                    print(f"  step {step + 1:3d}/{steps}: loss={loss:.4f}  "
                          f"[WARN] 梯度退化: median_diff={median_diff:.2e} (基线 {baseline[0]:.2e}×{median_diff/baseline[0]:.1f})")
                else:
                    print(f"  step {step + 1:3d}/{steps}: loss={loss:.4f}  "
                          f"[grad] median_diff={median_diff:.2e}")
            else:
                print(f"  step {step + 1:3d}/{steps}: loss={loss:.4f}  "
                      f"[grad] 抽检失败")
        elif (step + 1) % 10 == 0:
            print(f"  step {step + 1:3d}/{steps}: loss={loss:.4f}")

    elapsed = time.perf_counter() - t0
    avg_loss = float(np.mean(losses[-10:]))
    print(f"\n训练完成: {elapsed:.2f}s")
    print(f"最终 loss: {losses[-1]:.4f}")
    print(f"平均 loss (最后 10 步): {avg_loss:.4f}")

    # ---- 7. 训练后：最终验证 ----
    print("\n" + "=" * 60)
    print("阶段 3/3: 训练后最终验证")
    print("=" * 60)

    model.eval()
    x_test = np.random.randn(4, 784).astype(np.float32)

    with ag.record_scope():
        y = model.forward([ag.Tensor.from_numpy(x_test)])
    out = y.to_numpy()
    print(f"  推理输出 shape: {out.shape}  (期望 (4, 10))")

    # 梯度流通检查
    dY_test = np.random.randn(4, 10).astype(np.float32)
    y.backward(dY_test)
    all_ok = True
    for name, p in model.named_parameters():
        g = p.grad
        gnorm = float(np.linalg.norm(g)) if g is not None else 0
        ok = gnorm > 0
        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {name}: grad_norm={gnorm:.6f}")

    # 最终数值梯度验证
    print()
    final_result = _run_gradient_spot_check(model, criterion, x_test,
                                            np.random.randint(0, 10, size=4),
                                            eps=1e-2, n_samples=20)
    if final_result:
        median_diff, max_diff, max_param = final_result
        print(f"  最终数值梯度: median_diff={median_diff:.2e}, max_diff={max_diff:.2e}")

    # 梯度健康度摘要
    print(f"\n梯度健康度追踪 ({len(grad_history)} 个采样点):")
    print(f"  {'阶段':<8} {'median_diff':<14} {'max_diff':<14} {'参数'}")
    for stage, md, mx, name in grad_history:
        flag = " <-- 退化" if (baseline and md > baseline[0] * 5) else ""
        print(f"  step {str(stage):<5} {md:<14.2e} {mx:<14.2e} {name}{flag}")

    print(f"\n验证结果: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())