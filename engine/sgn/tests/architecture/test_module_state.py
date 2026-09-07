# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""Module register_module 名称处理 + state_dict 序列化逻辑单元测试。

覆盖点（对应 module.cpp / module_bindings.cpp）：
  A. register_module
     A1 正常注册：children() 返回子模块且保留 Python 子类类型
     A2 重名注册：抛 ValueError（C++ invalid_argument），且无半注册状态
        （安全审计 2026-08-16 M-2/M-5：C++ 先抛，Python 侧不追加）
     A3 嵌套路径名：named_parameters 产生 "child.param" 点分路径
     A4 train/eval 递归传播到子模块
     A5 register_parameter(name, None) 取消注册 + Python 属性删除
  B. state_dict / load_state_dict
     B1 键完整性：参数 + buffer，键为点分路径名
     B2 state_dict 返回拷贝：修改返回数组不影响模型
     B3 round-trip：改乱 → load → 逐键恢复原值
     B4 load 后前向输出与原模型 bit 一致
     B5 shape 校验：ndim 不匹配抛 RuntimeError
     B6 shape 校验：同 ndim 但维度不同抛 RuntimeError
     B7 缺失键跳过、多余键忽略（静默）
     B8 forcecast：float64 数组加载后数值正确（审计 M-6）

运行：
    python tests/architecture/test_module_state.py
"""

import sys
import os

# 同 test_dual_mode.py：项目根入 sys.path，避免 import engine.sgn 身份破坏
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import numpy as np
import engine.sgn as sgn

ag = sgn.autograd

_pass = 0
_fail = 0


def check(cond, label, detail=""):
    """单条断言，记录 PASS/FAIL。"""
    global _pass, _fail
    if cond:
        _pass += 1
        print(f"  PASS: {label}")
    else:
        _fail += 1
        print(f"  FAIL: {label}  {detail}")
    return cond


# ============================================================================
# 测试用模块
# ============================================================================

class MyLinear(sgn.nn.Module):
    """带自定义类名的线性层（验证 children() 保留 Python 子类类型）。"""

    def __init__(self, in_f, out_f):
        super().__init__()
        w = sgn.nn.Parameter([out_f, in_f])
        nn_fill(w, 0.01)
        self.register_parameter("weight", w)
        b = sgn.nn.Parameter([out_f])
        nn_fill(b, 0.0)
        self.register_parameter("bias", b)

    def forward(self, inputs):
        x = inputs[0]
        return ag.linear(x, self.weight.tensor(), self.bias.tensor())


class WithBuffer(sgn.nn.Module):
    """含 buffer 的模块（buffer 不进 parameters()，但进 state_dict）。"""

    def __init__(self, in_f, out_f):
        super().__init__()
        self.fc = MyLinear(in_f, out_f)
        self.register_module("fc", self.fc)
        rm = ag.Tensor.from_numpy(np.zeros(out_f, dtype=np.float32))
        self.register_buffer("running_mean", sgn.nn.Buffer(rm))

    def forward(self, inputs):
        return self.fc(inputs)


def nn_fill(param, val):
    sgn.nn.fill_(param.tensor(), val)


def build_parent():
    """父模块：1 个具名子模块 + 1 个 Sequential（str(i) 命名）+ 直接 buffer。"""
    parent = WithBuffer(4, 8)
    parent.seq = sgn.nn.Sequential(
        MyLinear(8, 8),
        sgn.nn.ReLU(),
        MyLinear(8, 3),
    )
    parent.register_module("seq", parent.seq)
    return parent


# ============================================================================
# A. register_module
# ============================================================================

def test_register_module():
    print("\n--- A. register_module ---")

    # A1 正常注册 + 子类类型保留
    parent = build_parent()
    kids = parent.children()
    check(len(kids) == 2, "A1 children() 数量 == 2",
          f"got {len(kids)}")
    check(isinstance(kids[0], MyLinear), "A1 children() 保留 Python 子类类型",
          f"got {type(kids[0]).__name__}")
    check(isinstance(parent.seq, sgn.nn.Sequential) and len(parent.seq._layers) == 3,
          "A1 Sequential 内 3 层")

    # A2 重名注册：C++ 抛 invalid_argument → ValueError；无半注册状态
    dup_child = MyLinear(8, 8)
    raised = False
    try:
        parent.register_module("fc", dup_child)
    except (ValueError, RuntimeError) as e:
        raised = True
        check("duplicate child name 'fc'" in str(e), "A2 异常信息含 duplicate child name",
              str(e))
    check(raised, "A2 重名注册抛异常")
    check(len(parent.children()) == 2, "A2 Python 侧 children() 数量不变（无半注册）",
          f"got {len(parent.children())}")
    np_names = [n for n, _ in parent.named_parameters()]
    check(len(np_names) == len(set(np_names)),
          "A2 named_parameters 无重复路径", str(np_names))

    # A3 嵌套路径名
    keys = {n for n, _ in parent.named_parameters()}
    expect = {"fc.weight", "fc.bias",                  # WithBuffer.fc
              "seq.0.weight", "seq.0.bias",            # Sequential[0] MyLinear
              "seq.2.weight", "seq.2.bias"}            # Sequential[2] MyLinear
    check(keys == expect, "A3 嵌套点分路径名完整",
          f"missing={expect - keys}, extra={keys - expect}")

    # A4 train/eval 递归
    parent.eval()
    check(parent.training is False, "A4 父模块 eval")
    check(parent.fc.training is False and parent.seq.training is False,
          "A4 eval 递归到子模块")
    check(parent.seq._layers[0].training is False, "A4 eval 递归到孙子模块")
    parent.train()
    check(parent.seq._layers[2].training is True, "A4 train 递归恢复")

    # A5 register_parameter(name, None) 取消注册 + 属性删除
    parent.fc.register_parameter("weight", None)
    check(not hasattr(parent.fc, "weight"), "A5 取消注册后 Python 属性删除")
    fc_keys = {n for n, _ in parent.fc.named_parameters()}
    check("weight" not in fc_keys and "bias" in fc_keys,
          "A5 named_parameters 移除该参数", str(fc_keys))


# ============================================================================
# B. state_dict / load_state_dict
# ============================================================================

def test_state_dict():
    print("\n--- B. state_dict / load_state_dict ---")

    model = build_parent()

    # B1 键完整性：参数 + buffer，点分路径
    state = model.state_dict()
    expect_keys = {"fc.weight", "fc.bias", "running_mean",
                   "seq.0.weight", "seq.0.bias",
                   "seq.2.weight", "seq.2.bias"}
    check(set(state.keys()) == expect_keys, "B1 state_dict 键完整（参数+buffer）",
          f"missing={expect_keys - set(state.keys())}, extra={set(state.keys()) - expect_keys}")
    check(all(isinstance(v, np.ndarray) for v in state.values()),
          "B1 值均为 numpy.ndarray")
    check(state["running_mean"].shape == (8,), "B1 buffer 形状正确")
    check(state["seq.0.weight"].shape == (8, 8), "B1 嵌套参数形状正确")

    # B2 返回拷贝：修改 state 不影响模型
    state["fc.weight"][0, 0] = 999.0
    check(float(model.fc.weight.tensor().to_numpy()[0, 0]) == float(np.float32(0.01)),
          "B2 修改返回数组不影响模型参数")
    state2 = model.state_dict()

    # B3 round-trip：改乱 → load → 恢复
    ref = {k: v.copy() for k, v in state2.items()}
    nn_fill(model.fc.weight, 0.5)                              # 改乱直接参数
    sgn.nn.fill_(model.seq._layers[2].weight.tensor(), 0.7)    # 改乱嵌套参数
    sgn.nn.fill_(model.running_mean.tensor(), 0.3)             # 改乱 buffer
    model.load_state_dict(state2)
    ok = all(np.array_equal(model.state_dict()[k], ref[k]) for k in expect_keys)
    check(ok, "B3 round-trip 逐键恢复原值")

    # B4 load 后前向输出与原模型一致（bit 级）
    x_np = np.random.RandomState(7).randn(2, 4).astype(np.float32)

    def fwd(m):
        ag.clear()
        ag.start_recording()
        y = m.forward([ag.Tensor.from_numpy(x_np.copy())])
        ag.stop_recording()
        return y.to_numpy().copy()

    model_clean = build_parent()
    y0 = fwd(model_clean)
    # 改乱 model 后恢复
    nn_fill(model.fc.weight, 0.9)
    model.load_state_dict(state2)
    y1 = fwd(model)
    check(np.array_equal(y0, y1), "B4 load 后前向输出 bit 级一致",
          f"max_diff={np.abs(y0 - y1).max():.3e}" if y0.shape == y1.shape else "shape")

    # B5/B6 shape 校验
    bad_ndim = dict(state2)
    bad_ndim["fc.weight"] = np.zeros((8, 4, 1), dtype=np.float32)  # 3D vs 2D
    raised = False
    try:
        model.load_state_dict(bad_ndim)
    except RuntimeError as e:
        raised = "ndim mismatch" in str(e)
    check(raised, "B5 ndim 不匹配抛 RuntimeError('ndim mismatch')")

    bad_shape = dict(state2)
    bad_shape["fc.weight"] = np.zeros((8, 7), dtype=np.float32)    # 2D 但 dim1 不同
    raised = False
    try:
        model.load_state_dict(bad_shape)
    except RuntimeError as e:
        raised = "shape mismatch" in str(e)
    check(raised, "B6 维度大小不匹配抛 RuntimeError('shape mismatch')")

    # B7 缺失键跳过、多余键忽略
    missing = {k: v for k, v in state2.items() if k != "fc.bias"}  # 缺 fc.bias
    model.load_state_dict(missing)                                 # 不抛
    check(np.array_equal(model.state_dict()["fc.bias"], ref["fc.bias"]),
          "B7 缺失键被跳过（原值保留）")
    extra = dict(state2)
    extra["ghost.key"] = np.zeros(3, dtype=np.float32)          # 多余键
    model.load_state_dict(extra)                                # 不抛
    check(np.array_equal(model.state_dict()["fc.weight"], ref["fc.weight"]),
          "B7 多余键被忽略，其余正常加载")

    # B8 forcecast：float64 → float32 数值正确（M-6）
    f64 = {k: v.astype(np.float64) for k, v in state2.items()}
    model.load_state_dict(f64)
    check(np.array_equal(model.state_dict()["fc.weight"], ref["fc.weight"]),
          "B8 float64 数组 forcecast 加载 bit 级正确")


def main():
    print("=" * 60)
    print("Module register_module + state_dict 序列化单元测试")
    print("=" * 60)

    test_register_module()
    test_state_dict()

    print()
    print("=" * 60)
    print(f"结论: PASS={_pass}, FAIL={_fail}")
    print("=" * 60)
    return 0 if _fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
