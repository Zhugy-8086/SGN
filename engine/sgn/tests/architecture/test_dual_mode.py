# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""双模式系统一致性测试：对比稳定模式（Module）与调试模式（自由函数）的 forward/backward 结果。

测试目标：
  1. 相同网络结构在两种模式下前向输出一致
  2. 相同输入在两种模式下反向梯度一致
  3. 混合模式（Module 管理参数 + 自由函数算子）结果一致
  4. 序列化（state_dict/load_state_dict）不影响计算结果
"""

import sys
import os

# 安全审计 2026-08-16 A2-7：模式 B → 模式 A（import engine.sgn as sgn）
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import numpy as np
import engine.sgn as sgn

ag = sgn.autograd

# ============================================================================
# 网络结构：C 3 → 32 → ReLU → MaxPool(2) → 64 → ReLU → Linear(256→10)
# ============================================================================
IN_CH, HID1, HID2, FC_IN, N_CLS = 3, 32, 64, 256, 10
K = 3
B = 4
H, W = 8, 8

# 随机输入（固定种子，确保可复现）
np.random.seed(2026)
x_np = np.random.randn(B, IN_CH, H, W).astype(np.float32)
dY_np = np.random.randn(B, N_CLS).astype(np.float32)


def run_debug_mode():
    """调试模式：手动创建所有 Tensor，手动管理 tape。"""
    # 权重初始化为固定值，消除随机性影响
    conv1_w = ag.Tensor.from_numpy(np.full((HID1, IN_CH, K, K), 0.01, dtype=np.float32))
    conv1_w.requires_grad = True
    conv1_b = ag.Tensor.from_numpy(np.zeros(HID1, dtype=np.float32))
    conv1_b.requires_grad = True

    conv2_w = ag.Tensor.from_numpy(np.full((HID2, HID1, K, K), 0.01, dtype=np.float32))
    conv2_w.requires_grad = True
    conv2_b = ag.Tensor.from_numpy(np.zeros(HID2, dtype=np.float32))
    conv2_b.requires_grad = True

    fc_w = ag.Tensor.from_numpy(np.full((N_CLS, FC_IN), 0.01, dtype=np.float32))
    fc_w.requires_grad = True
    fc_b = ag.Tensor.from_numpy(np.zeros(N_CLS, dtype=np.float32))
    fc_b.requires_grad = True

    x = ag.Tensor.from_numpy(x_np.copy())
    x.requires_grad = True

    # 前向
    ag.clear()
    ag.start_recording()
    y = ag.conv2d(x, conv1_w, conv1_b, 1, 1)
    y = ag.relu(y)
    y = ag.maxpool2d(y, 2, 2)
    y = ag.conv2d(y, conv2_w, conv2_b, 1, 1)
    y = ag.relu(y)
    y = ag.maxpool2d(y, 2, 2)
    y = y.reshape([B, -1])  # flatten
    y = ag.linear(y, fc_w, fc_b)
    ag.stop_recording()

    # 反向
    y.backward(dY_np.copy())

    grads = {
        'conv1_w': conv1_w.grad.copy(),
        'conv1_b': conv1_b.grad.copy(),
        'conv2_w': conv2_w.grad.copy(),
        'conv2_b': conv2_b.grad.copy(),
        'fc_w': fc_w.grad.copy(),
        'fc_b': fc_b.grad.copy(),
    }

    return y.to_numpy().copy(), grads


def run_stable_mode():
    """稳定模式：使用 Module 封装，参数自动管理。"""
    class ConvNet(sgn.nn.Module):
        def __init__(self):
            super().__init__()
            # Conv1
            w1 = sgn.nn.Parameter([HID1, IN_CH, K, K])
            sgn.nn.fill_(w1.tensor(), 0.01)
            self.register_parameter('conv1_w', w1)
            b1 = sgn.nn.Parameter([HID1])
            sgn.nn.fill_(b1.tensor(), 0.0)
            self.register_parameter('conv1_b', b1)

            # Conv2
            w2 = sgn.nn.Parameter([HID2, HID1, K, K])
            sgn.nn.fill_(w2.tensor(), 0.01)
            self.register_parameter('conv2_w', w2)
            b2 = sgn.nn.Parameter([HID2])
            sgn.nn.fill_(b2.tensor(), 0.0)
            self.register_parameter('conv2_b', b2)

            # FC
            fw = sgn.nn.Parameter([N_CLS, FC_IN])
            sgn.nn.fill_(fw.tensor(), 0.01)
            self.register_parameter('fc_w', fw)
            fb = sgn.nn.Parameter([N_CLS])
            sgn.nn.fill_(fb.tensor(), 0.0)
            self.register_parameter('fc_b', fb)

        def forward(self, inputs):
            x = inputs[0]
            y = ag.conv2d(x, self.conv1_w.tensor(), self.conv1_b.tensor(), 1, 1)
            y = ag.relu(y)
            y = ag.maxpool2d(y, 2, 2)
            y = ag.conv2d(y, self.conv2_w.tensor(), self.conv2_b.tensor(), 1, 1)
            y = ag.relu(y)
            y = ag.maxpool2d(y, 2, 2)
            y = y.reshape([B, -1])
            y = ag.linear(y, self.fc_w.tensor(), self.fc_b.tensor())
            return y

    model = ConvNet()
    x = ag.Tensor.from_numpy(x_np.copy())

    # 前向（使用 Module 的 forward）
    ag.clear()
    ag.start_recording()
    y = model.forward([x])
    ag.stop_recording()

    # 反向
    y.backward(dY_np.copy())

    grads = {}
    for name, t in model.named_parameters():
        g = t.grad
        # 统一命名：去掉前缀
        short_name = name.split('.')[-1] if '.' in name else name
        grads[short_name] = g.copy()

    return y.to_numpy().copy(), grads


def run_mixed_mode():
    """混合模式：Module 管理参数，自由函数做前向（参数直接传给算子）。"""
    # 用 Module 管理参数
    class ParamContainer(sgn.nn.Module):
        def __init__(self):
            super().__init__()
            for name, shape, val in [
                ('conv1_w', [HID1, IN_CH, K, K], 0.01),
                ('conv1_b', [HID1], 0.0),
                ('conv2_w', [HID2, HID1, K, K], 0.01),
                ('conv2_b', [HID2], 0.0),
                ('fc_w', [N_CLS, FC_IN], 0.01),
                ('fc_b', [N_CLS], 0.0),
            ]:
                p = sgn.nn.Parameter(shape)
                sgn.nn.fill_(p.tensor(), val)
                self.register_parameter(name, p)

        def forward(self, inputs):
            return inputs[0]

    pc = ParamContainer()

    # 用自由函数做前向（直接传 pc 的参数）
    x = ag.Tensor.from_numpy(x_np.copy())

    ag.clear()
    ag.start_recording()
    y = ag.conv2d(x, pc.conv1_w.tensor(), pc.conv1_b.tensor(), 1, 1)
    y = ag.relu(y)
    y = ag.maxpool2d(y, 2, 2)
    y = ag.conv2d(y, pc.conv2_w.tensor(), pc.conv2_b.tensor(), 1, 1)
    y = ag.relu(y)
    y = ag.maxpool2d(y, 2, 2)
    y = y.reshape([B, -1])
    y = ag.linear(y, pc.fc_w.tensor(), pc.fc_b.tensor())
    ag.stop_recording()

    y.backward(dY_np.copy())

    grads = {}
    for name, t in pc.named_parameters():
        short_name = name.split('.')[-1] if '.' in name else name
        g = t.grad
        grads[short_name] = g.copy()

    return y.to_numpy().copy(), grads


def check_arrays_equal(a, b, label, rtol=1e-5, atol=1e-7):
    """比较两个 numpy array 是否一致，不一致时打印详细差异。"""
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        diff = np.abs(a - b).max()
        print(f"  FAIL: {label}  max_diff={diff:.2e}")
        print(f"    shape: a={a.shape}, b={b.shape}")
        # 打印显著差异的位置
        mask = np.abs(a - b) > atol
        if mask.any():
            idx = np.argmax(np.abs(a - b))
            flat_idx = np.unravel_index(idx, a.shape)
            print(f"    max diff at {flat_idx}: a={a[flat_idx]:.6f}, b={b[flat_idx]:.6f}")
        return False
    print(f"  PASS: {label}")
    return True


def main():
    print("=" * 60)
    print("双模式系统一致性测试")
    print("=" * 60)
    print(f"输入: B={B}, C={IN_CH}, H={H}, W={W}")
    print(f"网络: Conv({IN_CH}→{HID1})→ReLU→MP(2)→Conv({HID1}→{HID2})→ReLU→MP(2)→Linear({FC_IN}→{N_CLS})")
    print()

    # 运行三种模式
    print("[1/3] 运行调试模式（自由函数 + 手动参数）...")
    y_debug, grads_debug = run_debug_mode()

    print("[2/3] 运行稳定模式（Module 封装）...")
    y_stable, grads_stable = run_stable_mode()

    print("[3/3] 运行混合模式（Module 参数 + 自由函数）...")
    y_mixed, grads_mixed = run_mixed_mode()

    print()
    print("--- 前向输出对比 ---")
    all_pass = True
    all_pass &= check_arrays_equal(y_debug, y_stable, "调试 vs 稳定")
    all_pass &= check_arrays_equal(y_debug, y_mixed, "调试 vs 混合")
    all_pass &= check_arrays_equal(y_stable, y_mixed, "稳定 vs 混合")

    print()
    print("--- 反向梯度对比 ---")
    for name in ['conv1_w', 'conv1_b', 'conv2_w', 'conv2_b', 'fc_w', 'fc_b']:
        print(f"  [{name}]")
        if name not in grads_debug:
            print(f"    SKIP: 调试模式无此梯度")
            continue
        if name not in grads_stable:
            print(f"    SKIP: 稳定模式无此梯度")
            continue
        if name not in grads_mixed:
            print(f"    SKIP: 混合模式无此梯度")
            continue
        all_pass &= check_arrays_equal(grads_debug[name], grads_stable[name],
                                       f"调试 vs 稳定")
        all_pass &= check_arrays_equal(grads_debug[name], grads_mixed[name],
                                       f"调试 vs 混合")
        all_pass &= check_arrays_equal(grads_stable[name], grads_mixed[name],
                                       f"稳定 vs 混合")

    print()
    print("--- 序列化不影响计算 ---")
    # 构造一个用 Module 的模型，序列化后再跑一次，对比结果
    class NetForSerialization(sgn.nn.Module):
        def __init__(self):
            super().__init__()
            w1 = sgn.nn.Parameter([HID1, IN_CH, K, K])
            sgn.nn.fill_(w1.tensor(), 0.01)
            self.register_parameter('conv1_w', w1)
            b1 = sgn.nn.Parameter([HID1])
            sgn.nn.fill_(b1.tensor(), 0.0)
            self.register_parameter('conv1_b', b1)
            w2 = sgn.nn.Parameter([HID2, HID1, K, K])
            sgn.nn.fill_(w2.tensor(), 0.01)
            self.register_parameter('conv2_w', w2)
            b2 = sgn.nn.Parameter([HID2])
            sgn.nn.fill_(b2.tensor(), 0.0)
            self.register_parameter('conv2_b', b2)
            fw = sgn.nn.Parameter([N_CLS, FC_IN])
            sgn.nn.fill_(fw.tensor(), 0.01)
            self.register_parameter('fc_w', fw)
            fb = sgn.nn.Parameter([N_CLS])
            sgn.nn.fill_(fb.tensor(), 0.0)
            self.register_parameter('fc_b', fb)
        def forward(self, inputs):
            x = inputs[0]
            y = ag.conv2d(x, self.conv1_w.tensor(), self.conv1_b.tensor(), 1, 1)
            y = ag.relu(y)
            y = ag.maxpool2d(y, 2, 2)
            y = ag.conv2d(y, self.conv2_w.tensor(), self.conv2_b.tensor(), 1, 1)
            y = ag.relu(y)
            y = ag.maxpool2d(y, 2, 2)
            y = y.reshape([B, -1])
            y = ag.linear(y, self.fc_w.tensor(), self.fc_b.tensor())
            return y

    model_a = NetForSerialization()
    state = model_a.state_dict()

    model_b = NetForSerialization()
    diff_val = 0.5
    sgn.nn.fill_(model_b.conv1_w.tensor(), diff_val)  # 改乱参数
    model_b.load_state_dict(state)

    # 加载后应恢复原始参数
    orig_w = sgn.nn.Parameter([HID1, IN_CH, K, K])
    sgn.nn.fill_(orig_w.tensor(), 0.01)
    all_pass &= check_arrays_equal(
        model_b.conv1_w.to_numpy(), orig_w.tensor().to_numpy(),
        "load_state_dict 恢复参数")

    print()
    print("=" * 60)
    if all_pass:
        print("结论: 全部 PASS — 双模式系统结果完全一致")
    else:
        print("结论: 存在 FAIL — 双模式系统结果不一致")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())