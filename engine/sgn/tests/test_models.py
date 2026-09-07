# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""验证 MLP 和 CNN4 模型在稳定模式下可以正确前向+反向。

运行：
    cd engine/sgn/build && python ../tests/test_models.py
"""

import sys
import os
import numpy as np

# 确保项目根（含 engine/ 包）在 sys.path，支持 `python tests/test_models.py` 直接运行
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _PROJ_ROOT)

# 用相对导入 engine.sgn 代替顶层 `import sgn`，避免 pytest 全量收集时
# `sgn` 被解析为 .pyd（非包）导致 `sgn.models` 等子模块不可访问。
import engine.sgn as sgn
print(f"[OK] sgn 模块导入成功，version={sgn.version()}")

from engine.sgn.models import MLP, CNN4

ag = sgn.autograd

np.random.seed(42)


def test_mlp():
    """验证 MLP 前向+反向可正常执行，梯度流通。"""
    print("\n" + "=" * 60)
    print("测试 MLP（MNIST: 784 → 128 → 64 → 10）")
    print("=" * 60)

    model = MLP(input_size=784, hidden1=128, hidden2=64, num_classes=10)

    B = 4
    x_np = np.random.randn(B, 784).astype(np.float32)
    dY_np = np.random.randn(B, 10).astype(np.float32)
    x = ag.Tensor.from_numpy(x_np.copy())

    # 前向
    ag.clear()
    ag.start_recording()
    y = model.forward([x])
    ag.stop_recording()

    out_np = y.to_numpy()
    print(f"  前向输出 shape: {out_np.shape}  (期望 (B, 10))")
    assert out_np.shape == (B, 10), f"shape 错误: {out_np.shape}"

    # 反向
    y.backward(dY_np.copy())

    # 检查梯度流通
    print("\n  --- 梯度流通检查 ---")
    all_ok = True
    for name, p in model.named_parameters():
        g = p.grad
        if g is None:
            print(f"  [FAIL] {name}: grad=None")
            all_ok = False
        else:
            gnorm = float(np.linalg.norm(g))
            ok = "OK" if gnorm > 0 else "FAIL"
            print(f"  [{ok}] {name}: grad_norm={gnorm:.6f}, shape={g.shape}")
            if gnorm == 0:
                all_ok = False

    print(f"\n  MLP 测试: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def test_cnn4():
    """验证 CNN4 前向+反向可正常执行，梯度流通。"""
    print("\n" + "=" * 60)
    print("测试 CNN4（CIFAR-10: 3×32×32 → 10）")
    print("=" * 60)

    model = CNN4(in_channels=3, img_size=32, num_classes=10)

    B = 4
    x_np = np.random.randn(B, 3, 32, 32).astype(np.float32)
    dY_np = np.random.randn(B, 10).astype(np.float32)
    x = ag.Tensor.from_numpy(x_np.copy())

    # 前向
    ag.clear()
    ag.start_recording()
    y = model.forward([x])
    ag.stop_recording()

    out_np = y.to_numpy()
    print(f"  前向输出 shape: {out_np.shape}  (期望 (B, 10))")
    assert out_np.shape == (B, 10), f"shape 错误: {out_np.shape}"

    # 反向
    y.backward(dY_np.copy())

    # 检查梯度流通
    print("\n  --- 梯度流通检查 ---")
    all_ok = True
    for name, p in model.named_parameters():
        g = p.grad
        if g is None:
            print(f"  [FAIL] {name}: grad=None")
            all_ok = False
        else:
            gnorm = float(np.linalg.norm(g))
            ok = "OK" if gnorm > 0 else "FAIL"
            print(f"  [{ok}] {name}: grad_norm={gnorm:.6f}, shape={g.shape}")
            if gnorm == 0:
                all_ok = False

    print(f"\n  CNN4 测试: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def test_mlp_flexible_shapes():
    """验证 MLP 可用不同配置。"""
    print("\n" + "=" * 60)
    print("测试 MLP 不同配置")
    print("=" * 60)

    all_ok = True
    for input_size, hidden1, hidden2, num_classes in [
        (784, 128, 64, 10),    # MNIST 默认
        (3072, 512, 256, 100), # CIFAR-100 大 MLP
        (64, 32, 16, 5),       # 小数据集
    ]:
        model = MLP(input_size, hidden1, hidden2, num_classes)
        B = 2
        x_np = np.random.randn(B, input_size).astype(np.float32)
        dY_np = np.random.randn(B, num_classes).astype(np.float32)
        x = ag.Tensor.from_numpy(x_np.copy())

        ag.clear()
        ag.start_recording()
        y = model.forward([x])
        ag.stop_recording()
        y.backward(dY_np.copy())

        out_np = y.to_numpy()
        ok = out_np.shape == (B, num_classes)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] MLP({input_size}→{hidden1}→{hidden2}→{num_classes}): out={out_np.shape}")
        if not ok:
            all_ok = False

    print(f"\n  灵活配置测试: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def test_cnn4_flexible_shapes():
    """验证 CNN4 可用不同配置。"""
    print("\n" + "=" * 60)
    print("测试 CNN4 不同配置")
    print("=" * 60)

    all_ok = True
    for in_channels, img_size, num_classes in [
        (3, 32, 10),   # CIFAR-10
        (1, 28, 10),   # MNIST
        (3, 16, 5),    # 小图像
    ]:
        model = CNN4(in_channels, img_size, num_classes)
        B = 2
        x_np = np.random.randn(B, in_channels, img_size, img_size).astype(np.float32)
        dY_np = np.random.randn(B, num_classes).astype(np.float32)
        x = ag.Tensor.from_numpy(x_np.copy())

        ag.clear()
        ag.start_recording()
        y = model.forward([x])
        ag.stop_recording()
        y.backward(dY_np.copy())

        out_np = y.to_numpy()
        ok = out_np.shape == (B, num_classes)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] CNN4({in_channels}ch, {img_size}px, {num_classes}cls): out={out_np.shape}")
        if not ok:
            all_ok = False

    print(f"\n  灵活配置测试: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    tests = [
        ("MLP 基本功能", test_mlp),
        ("CNN4 基本功能", test_cnn4),
        ("MLP 灵活配置", test_mlp_flexible_shapes),
        ("CNN4 灵活配置", test_cnn4_flexible_shapes),
    ]

    results = []
    print("=" * 60)
    print("稳定模式模型验证")
    print("=" * 60)

    for name, fn in tests:
        try:
            ok = fn()
        except Exception as e:
            import traceback
            print(f"\n  [EXCEPTION] {name}: {e}")
            traceback.print_exc()
            ok = False
        results.append((name, ok))

    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_pass = False

    print(f"\n总体: {'全部通过' if all_pass else '存在失败'}")
    sys.exit(0 if all_pass else 1)