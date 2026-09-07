# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""test_phase5.py - 6 层 CNN+BN 前向+反向验证（Phase 5 Task A5.2b）

Stage 3.2 Phase 5 Task A5.2b：用 C++ Autograd tape 实现 6 层 CNN 前向+反向，
与 PyTorch baseline (CNN6Cifar10) 对比前向输出和梯度正确性。

架构（与 legacy/traditional/baseline/cnn6_cifar10.py 完全一致）：
    Layer 1: Conv2d(3, 32, k=3, p=1) → BN2d(32) → ReLU → MaxPool(2)   → (32, 16, 16)
    Layer 2: Conv2d(32, 64, k=3, p=1) → BN2d(64) → ReLU → MaxPool(2)  → (64, 8, 8)
    Layer 3: Conv2d(64, 128, k=3, p=1) → BN2d(128) → ReLU → MaxPool(2) → (128, 4, 4)
    Layer 4: Linear(2048, 256) → BN1d(256) → ReLU
    Layer 5: Linear(256, 128) → BN1d(128) → ReLU
    Layer 6: Linear(128, 10)

验证内容：
  1. C++ Autograd 前向输出 vs PyTorch 前向输出（数值匹配）
  2. C++ Autograd 反向梯度 vs PyTorch 反向梯度（各参数 grad 匹配）
  3. 梯度流通（所有参数都有非零梯度）

运行：cd engine/sgn/build && python ../autograd/test_phase5.py
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn

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

# 将 baseline 目录加入 path 以便导入 CNN6Cifar10
# 安全审计 2026-08-16 A3-2：traditional/ 已迁入 legacy/（原 ../../../traditional
# 路径断裂，本脚本曾完全无法运行）
# 2026-08-16 legacy 独立：baseline 参考实现迁至 tests/refs/baseline/（legacy 侧不再引用）
_BASELINE_DIR = os.path.join(os.path.dirname(__file__), '..', 'tests', 'refs', 'baseline')
if not os.path.isdir(_BASELINE_DIR):
    try:
        import pytest
        pytest.skip(f"PyTorch baseline 不存在: {_BASELINE_DIR}", allow_module_level=True)
    except ImportError:
        print(f"[FAIL] baseline 目录不存在: {_BASELINE_DIR}")
        sys.exit(1)
sys.path.insert(0, _BASELINE_DIR)
from cnn6_cifar10 import CNN6Cifar10  # noqa: E402


# ============================================================================
# 工具函数
# ============================================================================
def make_tensor(np_arr, requires_grad=False):
    """从 numpy array 创建 sgn Tensor"""
    t = sgn.autograd.Tensor.from_numpy(np.ascontiguousarray(np_arr, dtype=np.float32))
    t.requires_grad = requires_grad
    return t


def compare(name, torch_val, cpp_val, atol=1e-3, rtol=1e-3):
    """对比 PyTorch 和 C++ 的数值，返回 max_diff"""
    if isinstance(torch_val, torch.Tensor):
        torch_np = torch_val.detach().numpy()
    else:
        torch_np = np.array(torch_val, dtype=np.float32)
    cpp_np = np.array(cpp_val, dtype=np.float32) if not isinstance(cpp_val, np.ndarray) else cpp_val

    if torch_np.shape != cpp_np.shape:
        print(f"  [FAIL] {name}: shape mismatch torch={torch_np.shape} vs cpp={cpp_np.shape}")
        return float('inf')

    max_diff = float(np.max(np.abs(torch_np - cpp_np))) if torch_np.size > 0 else 0.0
    rel_diff = max_diff / (float(np.max(np.abs(torch_np))) + 1e-12)
    ok = np.allclose(torch_np, cpp_np, atol=atol, rtol=rtol)
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {name}: max_diff={max_diff:.6e}, rel_diff={rel_diff:.6e} (atol={atol})")
    return max_diff


# ============================================================================
# C++ Autograd 6 层 CNN 前向+反向
# ============================================================================
def cpp_cnn6_forward_backward(x_np, weights, bn_params, dY_np):
    """用 sgn.autograd 算子构建 6 层 CNN，前向+反向

    Args:
        x_np: (B, 3, 32, 32) 输入
        weights: dict，包含 conv1-3/fc1-3 的 weight 和 bias（numpy float32）
        bn_params: dict，包含 bn1-5 的 gamma/beta/running_mean/running_var（numpy float32）
        dY_np: (B, 10) loss 对 output 的梯度

    Returns:
        (output_np, grads_dict)
        output_np: (B, 10) 前向输出
        grads_dict: {name: grad_numpy} 各参数的梯度
    """
    sgn.autograd.clear()

    B = x_np.shape[0]
    ag = sgn.autograd

    # --- 创建权重 Tensor（requires_grad=True）---
    w_conv1 = make_tensor(weights['conv1_w'], requires_grad=True)
    b_conv1 = make_tensor(weights['conv1_b'], requires_grad=True)
    w_conv2 = make_tensor(weights['conv2_w'], requires_grad=True)
    b_conv2 = make_tensor(weights['conv2_b'], requires_grad=True)
    w_conv3 = make_tensor(weights['conv3_w'], requires_grad=True)
    b_conv3 = make_tensor(weights['conv3_b'], requires_grad=True)
    w_fc1 = make_tensor(weights['fc1_w'], requires_grad=True)
    b_fc1 = make_tensor(weights['fc1_b'], requires_grad=True)
    w_fc2 = make_tensor(weights['fc2_w'], requires_grad=True)
    b_fc2 = make_tensor(weights['fc2_b'], requires_grad=True)
    w_fc3 = make_tensor(weights['fc3_w'], requires_grad=True)
    b_fc3 = make_tensor(weights['fc3_b'], requires_grad=True)

    # --- BN 参数（gamma/beta requires_grad=True，running stats requires_grad=False）---
    bn_gammas = {}
    bn_betas = {}
    bn_rm = {}
    bn_rv = {}
    for i in range(1, 6):
        bn_gammas[i] = make_tensor(bn_params[f'bn{i}_gamma'], requires_grad=True)
        bn_betas[i] = make_tensor(bn_params[f'bn{i}_beta'], requires_grad=True)
        bn_rm[i] = make_tensor(bn_params[f'bn{i}_running_mean'], requires_grad=False)
        bn_rv[i] = make_tensor(bn_params[f'bn{i}_running_var'], requires_grad=False)

    # --- 输入 ---
    x = make_tensor(x_np, requires_grad=False)

    momentum = 0.1
    eps = 1e-5

    # --- 前向（录制到 tape）---
    ag.start_recording()
    try:
        # Layer 1: Conv2d(3, 32, k=3, p=1) → BN2d(32) → ReLU → MaxPool(2)
        y = ag.conv2d(x, w_conv1, b_conv1, stride=1, padding=1)
        y = ag.batchnorm2d(y, bn_gammas[1], bn_betas[1], bn_rm[1], bn_rv[1], momentum, eps)
        y = ag.relu(y)
        y = ag.maxpool2d(y, kernel=2, stride=2)  # (B, 32, 16, 16)

        # Layer 2: Conv2d(32, 64, k=3, p=1) → BN2d(64) → ReLU → MaxPool(2)
        y = ag.conv2d(y, w_conv2, b_conv2, stride=1, padding=1)
        y = ag.batchnorm2d(y, bn_gammas[2], bn_betas[2], bn_rm[2], bn_rv[2], momentum, eps)
        y = ag.relu(y)
        y = ag.maxpool2d(y, kernel=2, stride=2)  # (B, 64, 8, 8)

        # Layer 3: Conv2d(64, 128, k=3, p=1) → BN2d(128) → ReLU → MaxPool(2)
        y = ag.conv2d(y, w_conv3, b_conv3, stride=1, padding=1)
        y = ag.batchnorm2d(y, bn_gammas[3], bn_betas[3], bn_rm[3], bn_rv[3], momentum, eps)
        y = ag.relu(y)
        y = ag.maxpool2d(y, kernel=2, stride=2)  # (B, 128, 4, 4)

        # Flatten: (B, 128, 4, 4) → (B, 2048)
        y = ag.reshape(y, [B, -1])

        # Layer 4: Linear(2048, 256) → BN1d(256) → ReLU
        y = ag.linear(y, w_fc1, b_fc1)
        y = ag.bn_train(y, bn_gammas[4], bn_betas[4], bn_rm[4], bn_rv[4], momentum, eps, dim=0)
        y = ag.relu(y)

        # Layer 5: Linear(256, 128) → BN1d(128) → ReLU
        y = ag.linear(y, w_fc2, b_fc2)
        y = ag.bn_train(y, bn_gammas[5], bn_betas[5], bn_rm[5], bn_rv[5], momentum, eps, dim=0)
        y = ag.relu(y)

        # Layer 6: Linear(128, 10)
        y = ag.linear(y, w_fc3, b_fc3)
    finally:
        ag.stop_recording()

    # --- 前向输出 ---
    output_np = y.to_numpy()

    # --- 反向 ---
    y.backward(dY_np)

    # --- 收集梯度 ---
    grads = {}
    grads['conv1_w'] = w_conv1.grad
    grads['conv1_b'] = b_conv1.grad
    grads['conv2_w'] = w_conv2.grad
    grads['conv2_b'] = b_conv2.grad
    grads['conv3_w'] = w_conv3.grad
    grads['conv3_b'] = b_conv3.grad
    grads['fc1_w'] = w_fc1.grad
    grads['fc1_b'] = b_fc1.grad
    grads['fc2_w'] = w_fc2.grad
    grads['fc2_b'] = b_fc2.grad
    grads['fc3_w'] = w_fc3.grad
    grads['fc3_b'] = b_fc3.grad
    for i in range(1, 6):
        grads[f'bn{i}_gamma'] = bn_gammas[i].grad
        grads[f'bn{i}_beta'] = bn_betas[i].grad

    return output_np, grads


# ============================================================================
# 主测试
# ============================================================================
def test_cnn6_forward_backward():
    """6 层 CNN 前向+反向：C++ Autograd vs PyTorch"""
    print("\n[test_cnn6_forward_backward] 6 层 CNN 前向+反向 vs PyTorch")

    torch.manual_seed(42)
    np.random.seed(42)

    B = 4  # 小 batch 用于验证

    # --- 1. 创建 PyTorch 模型 ---
    model = CNN6Cifar10(num_classes=10)
    model.train()  # BN 用 batch 统计
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    # 生成随机输入
    x_np = np.random.randn(B, 3, 32, 32).astype(np.float32)
    x_torch = torch.from_numpy(x_np)

    # --- 2. 提取 PyTorch 权重 ---
    weights = {
        'conv1_w': model.conv1.weight.detach().numpy().copy(),
        'conv1_b': model.conv1.bias.detach().numpy().copy(),
        'conv2_w': model.conv2.weight.detach().numpy().copy(),
        'conv2_b': model.conv2.bias.detach().numpy().copy(),
        'conv3_w': model.conv3.weight.detach().numpy().copy(),
        'conv3_b': model.conv3.bias.detach().numpy().copy(),
        'fc1_w': model.fc1.weight.detach().numpy().copy(),
        'fc1_b': model.fc1.bias.detach().numpy().copy(),
        'fc2_w': model.fc2.weight.detach().numpy().copy(),
        'fc2_b': model.fc2.bias.detach().numpy().copy(),
        'fc3_w': model.fc3.weight.detach().numpy().copy(),
        'fc3_b': model.fc3.bias.detach().numpy().copy(),
    }

    # BN 参数（保存初始 running stats，因为 PyTorch forward 会更新它们）
    bn_layers = [model.bn1, model.bn2, model.bn3, model.bn4, model.bn5]
    bn_params = {}
    for i, bn in enumerate(bn_layers, 1):
        bn_params[f'bn{i}_gamma'] = bn.weight.detach().numpy().copy()
        bn_params[f'bn{i}_beta'] = bn.bias.detach().numpy().copy()
        bn_params[f'bn{i}_running_mean'] = bn.running_mean.detach().numpy().copy()
        bn_params[f'bn{i}_running_var'] = bn.running_var.detach().numpy().copy()

    # --- 3. PyTorch 前向+反向 ---
    y_torch = model(x_torch)  # (B, 10)
    dY_np = np.random.randn(B, 10).astype(np.float32)
    dY_torch = torch.from_numpy(dY_np)
    model.zero_grad()
    y_torch.backward(dY_torch)

    # --- 4. C++ Autograd 前向+反向 ---
    y_cpp, grads_cpp = cpp_cnn6_forward_backward(x_np, weights, bn_params, dY_np)

    # --- 5. 对比前向输出 ---
    print("\n  --- 前向输出对比 ---")
    fwd_max_diff = compare("output (B,10)", y_torch, y_cpp, atol=1e-3, rtol=1e-3)

    # --- 6. 对比梯度 ---
    print("\n  --- 梯度对比 ---")
    grad_max_diffs = {}

    # Conv/FC 权重和 bias
    torch_grads = {
        'conv1_w': model.conv1.weight.grad,
        'conv1_b': model.conv1.bias.grad,
        'conv2_w': model.conv2.weight.grad,
        'conv2_b': model.conv2.bias.grad,
        'conv3_w': model.conv3.weight.grad,
        'conv3_b': model.conv3.bias.grad,
        'fc1_w': model.fc1.weight.grad,
        'fc1_b': model.fc1.bias.grad,
        'fc2_w': model.fc2.weight.grad,
        'fc2_b': model.fc2.bias.grad,
        'fc3_w': model.fc3.weight.grad,
        'fc3_b': model.fc3.bias.grad,
    }
    for i, bn in enumerate(bn_layers, 1):
        torch_grads[f'bn{i}_gamma'] = bn.weight.grad
        torch_grads[f'bn{i}_beta'] = bn.bias.grad

    all_ok = True
    for name in torch_grads:
        if grads_cpp.get(name) is None:
            print(f"  [FAIL] {name}: C++ grad is None")
            all_ok = False
            continue
        # conv 层梯度累加较多，用稍大容差
        atol = 1e-2 if name.startswith('conv') else 1e-3
        d = compare(name, torch_grads[name], grads_cpp[name], atol=atol, rtol=1e-2)
        grad_max_diffs[name] = d
        if d > atol:
            all_ok = False

    # --- 7. 梯度流通检查 ---
    print("\n  --- 梯度流通检查 ---")
    for name in torch_grads:
        g = grads_cpp.get(name)
        if g is None:
            print(f"  [FAIL] {name}: grad=None (梯度未流通)")
            all_ok = False
        else:
            gnorm = float(np.linalg.norm(g))
            status = "OK" if gnorm > 0 else "FAIL"
            print(f"  [{status}] {name}: grad_norm={gnorm:.6f}")
            if gnorm == 0:
                all_ok = False

    print(f"\n  前向 max_diff: {fwd_max_diff:.6e}")
    print(f"  梯度 max_diff: {max(grad_max_diffs.values()) if grad_max_diffs else 0:.6e}")
    print(f"  总状态: {'PASS' if all_ok and fwd_max_diff < 1e-3 else 'FAIL'}")
    return all_ok and fwd_max_diff < 1e-3


# ============================================================================
# main
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 5 Task A5.2b: 6 层 CNN 前向+反向验证")
    print("=" * 60)

    success = test_cnn6_forward_backward()

    print("\n" + "=" * 60)
    if success:
        print("PASS: C++ Autograd 6 层 CNN 前向+反向与 PyTorch 一致")
    else:
        print("FAIL: 存在不一致，请检查上面的输出")
    print("=" * 60)
    sys.exit(0 if success else 1)
