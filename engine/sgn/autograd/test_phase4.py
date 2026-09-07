# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""test_phase4.py - Python 集成验证（Phase 4）

Stage 3.2 Phase 4 Task A4.4：验证 Python 端使用 sgn.autograd 模块。

验证内容：
  1. sgn.autograd 模块可导入
  2. Tensor 创建（from_numpy / 构造函数）+ numpy 互转
  3. matmul autograd：a @ b @ c 的 backward 与 PyTorch 对比
  4. 前向算子（linear/relu/conv2d/maxpool）数值正确
  5. requires_grad=False 的 input 无梯度

运行：cd engine/sgn/build && python ../autograd/test_phase4.py
"""

import sys
import os
import numpy as np

# 添加 build 目录到 path（sgn.cp314-win_amd64.pyd 所在位置）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

try:
    import sgn
    print(f"[OK] sgn 模块导入成功，version={sgn.version()}")
except ImportError as e:
    # 安全审计 2026-08-16 A3-1：模块级 sys.exit(1) 杀死 pytest 收集进程
    try:
        import pytest
        pytest.skip(f"sgn .pyd 不可用: {e}", allow_module_level=True)
    except ImportError:
        print(f"[FAIL] 无法导入 sgn: {e}")
        sys.exit(1)

# 检查 autograd 子模块
if not hasattr(sgn, 'autograd'):
    print("[FAIL] sgn.autograd 子模块不存在")
    sys.exit(1)
print(f"[OK] sgn.autograd 子模块存在: {sgn.autograd}")


# ============================================================================
# 测试 1: Tensor 创建 + numpy 互转
# ============================================================================
def test_tensor_numpy():
    print("\n[test_tensor_numpy] Tensor 与 numpy 互转")
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    # from_numpy
    t = sgn.autograd.Tensor.from_numpy(arr)
    assert list(t.shape) == [2, 3], f"shape 错误: {t.shape}"
    assert t.ndim == 2
    assert t.numel == 6

    # to_numpy
    arr2 = t.to_numpy()
    np.testing.assert_array_equal(arr2, arr)
    print("  from_numpy / to_numpy OK")

    # 构造函数
    t2 = sgn.autograd.Tensor([3, 4])
    assert list(t2.shape) == [3, 4]
    assert t2.numel == 12
    print("  Tensor(shape) 构造 OK")

    # requires_grad 属性
    t3 = sgn.autograd.Tensor.from_numpy(arr)
    assert t3.requires_grad == False
    t3.requires_grad = True
    assert t3.requires_grad == True
    print("  requires_grad 属性 OK")
    print("  PASS")


# ============================================================================
# 测试 2: matmul autograd 与 PyTorch 对比
# ============================================================================
def test_matmul_autograd():
    print("\n[test_matmul_autograd] a @ b @ c backward vs PyTorch")
    sgn.autograd.clear()

    # 与 Phase 2 C++ 测试相同的数据
    a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    b_np = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1]], dtype=np.float32)
    c_np = np.array([
        [1, 2, 3, 4, 5],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [2, 1, 2, 1, 2]
    ], dtype=np.float32)

    a = sgn.autograd.Tensor.from_numpy(a_np); a.requires_grad = True
    b = sgn.autograd.Tensor.from_numpy(b_np); b.requires_grad = True
    c = sgn.autograd.Tensor.from_numpy(c_np); c.requires_grad = True

    # 前向（录制）
    sgn.autograd.start_recording()
    d = sgn.autograd.matmul(a, b)
    e = sgn.autograd.matmul(d, c)
    sgn.autograd.stop_recording()

    # 检查前向数值
    expected_e = a_np @ b_np @ c_np
    np.testing.assert_allclose(e.to_numpy(), expected_e, atol=1e-5)
    print("  forward 数值正确")

    # backward (grad_e = ones)
    grad_e = np.ones((2, 5), dtype=np.float32)
    e.backward(grad_e)

    # PyTorch 期望值（Phase 2 已验证）
    expected_grad_a = np.array([[18, 10, 28], [18, 10, 28]], dtype=np.float32)
    expected_grad_b = np.array([
        [75, 10, 15, 40],
        [105, 14, 21, 56],
        [135, 18, 27, 72]
    ], dtype=np.float32)
    expected_grad_c = np.array([
        [14, 14, 14, 14, 14],
        [16, 16, 16, 16, 16],
        [14, 14, 14, 14, 14],
        [16, 16, 16, 16, 16]
    ], dtype=np.float32)

    np.testing.assert_allclose(a.grad, expected_grad_a, atol=1e-4)
    np.testing.assert_allclose(b.grad, expected_grad_b, atol=1e-4)
    np.testing.assert_allclose(c.grad, expected_grad_c, atol=1e-4)
    print("  grad_a/grad_b/grad_c 与 PyTorch 一致")
    print("  PASS")


# ============================================================================
# 测试 3: requires_grad=False 的 input 无梯度
# ============================================================================
def test_no_grad_input():
    print("\n[test_no_grad_input] a.requires_grad=False")
    sgn.autograd.clear()

    a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    b_np = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1]], dtype=np.float32)

    a = sgn.autograd.Tensor.from_numpy(a_np); a.requires_grad = False
    b = sgn.autograd.Tensor.from_numpy(b_np); b.requires_grad = True

    sgn.autograd.start_recording()
    c = sgn.autograd.matmul(a, b)
    sgn.autograd.stop_recording()

    assert c.requires_grad == True, "c.requires_grad 应为 True（因为 b 需要）"

    grad_c = np.ones((2, 4), dtype=np.float32)
    c.backward(grad_c)

    assert a.grad is None, "a.requires_grad=False 时 grad 应为 None"
    assert b.grad is not None, "b.requires_grad=True 时 grad 不应为 None"
    print("  a.grad=None (正确), b.grad 存在 (正确)")
    print("  PASS")


# ============================================================================
# 测试 4: 前向算子数值正确（linear/relu/conv2d/maxpool）
# ============================================================================
def test_forward_ops():
    print("\n[test_forward_ops] 前向算子数值验证")
    sgn.autograd.clear()

    # linear
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    w_np = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1], [2, 1, 2]], dtype=np.float32)
    b_np = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    x = sgn.autograd.Tensor.from_numpy(x_np)
    w = sgn.autograd.Tensor.from_numpy(w_np)
    b = sgn.autograd.Tensor.from_numpy(b_np)
    y = sgn.autograd.linear_forward(x, w, b)
    expected_y = x_np @ w_np.T + b_np
    np.testing.assert_allclose(y.to_numpy(), expected_y, atol=1e-5)
    print("  linear_forward OK")

    # relu
    x_np2 = np.array([[-1, 2, -3, 4, 0, -5, 6, -7]], dtype=np.float32)
    x2 = sgn.autograd.Tensor.from_numpy(x_np2)
    y2 = sgn.autograd.relu_forward(x2)
    expected_y2 = np.maximum(x_np2, 0)
    np.testing.assert_allclose(y2.to_numpy(), expected_y2, atol=1e-5)
    print("  relu_forward OK")

    # conv2d
    x_np3 = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32)
    w_np3 = np.array([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]], dtype=np.float32)
    b_np3 = np.array([0.0], dtype=np.float32)
    x3 = sgn.autograd.Tensor.from_numpy(x_np3)
    w3 = sgn.autograd.Tensor.from_numpy(w_np3)
    b3 = sgn.autograd.Tensor.from_numpy(b_np3)
    y3 = sgn.autograd.conv2d_forward(x3, w3, b3, stride=1, padding=1)
    # Phase 3 已验证的期望值
    expected_y3 = np.array([[[[6, 8, 3], [12, 15, 8], [7, 12, 14]]]], dtype=np.float32)
    np.testing.assert_allclose(y3.to_numpy(), expected_y3, atol=1e-4)
    print("  conv2d_forward OK")

    # maxpool
    x_np4 = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8],
                        [9, 10, 11, 12], [13, 14, 15, 16]]]], dtype=np.float32)
    x4 = sgn.autograd.Tensor.from_numpy(x_np4)
    y4 = sgn.autograd.maxpool2d_forward(x4, kernel=2, stride=2)
    expected_y4 = np.array([[[[6, 8], [14, 16]]]], dtype=np.float32)
    np.testing.assert_allclose(y4.to_numpy(), expected_y4, atol=1e-5)
    print("  maxpool2d_forward OK")
    print("  PASS")


# ============================================================================
# main
# ============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("Phase 4: Python 集成验证")
    print("=" * 50)

    tests = [
        test_tensor_numpy,
        test_matmul_autograd,
        test_no_grad_input,
        test_forward_ops,
    ]

    failures = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  FAIL: {e}")
            failures += 1

    print("\n" + "=" * 50)
    if failures == 0:
        print(f"全部测试通过 ({len(tests)}/{len(tests)})")
    else:
        print(f"{failures} 个测试失败")
    print("=" * 50)
    sys.exit(failures)
