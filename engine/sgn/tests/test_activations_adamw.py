# test_activations_adamw.py - 基础设施扩展验收：sigmoid/tanh 激活算子 + AdamW
#
# 预先登记验收（基础设施扩展第一批 2026-09-07）：
#   A1 sigmoid/tanh 前向值 = numpy 参考逐位一致（float32 容差 1e-6）
#   A2 梯度 = 解析式（sigmoid: c·y·(1−y)；tanh: c·(1−y²)），经 tape backward
#      容差标定：f32 tape 算术舍入噪声实测 ≤3e-5 绝对 / 1.4e-3 相对
#      （f64 解析参考 vs f32 tape 链），故 atol=1e-4 + rtol=1e-3
#   A3 AdamW 解耦衰减：零梯度 + wd>0 → 参数收缩且 m/v 保持零（衰减不污染统计）
#   A4 AdamW 单步 = 手算参考（收缩 + Adam 主式）
#
# 运行：SGN 根目录 pytest engine/sgn/tests/test_activations_adamw.py

import numpy as np
import pytest

import engine.sgn as sgn_pkg

ag = sgn_pkg.autograd


def _tensor(arr, requires_grad=True):
    t = ag.Tensor(np.ascontiguousarray(arr, dtype=np.float32))
    t.requires_grad = requires_grad
    return t


# ---------------------------------------------------------------- A1 前向
def test_a1_sigmoid_forward_values():
    x = np.array([-4.0, -1.0, 0.0, 1.0, 4.0], dtype=np.float32)
    y = ag.sigmoid(_tensor(x, requires_grad=False))
    ref = 1.0 / (1.0 + np.exp(-x.astype(np.float64)))
    assert np.allclose(y.to_numpy(), ref, atol=1e-6)


def test_a1_tanh_forward_values():
    x = np.array([-4.0, -1.0, 0.0, 1.0, 4.0], dtype=np.float32)
    y = ag.tanh(_tensor(x, requires_grad=False))
    assert np.allclose(y.to_numpy(), np.tanh(x.astype(np.float64)), atol=1e-6)


# ---------------------------------------------------------------- A2 梯度
@pytest.mark.parametrize("op", ["sigmoid", "tanh"])
def test_a2_gradient_through_tape(op):
    """解析梯度经 tape backward：sigmoid c·y·(1−y)；tanh c·(1−y²)。"""
    rng = np.random.default_rng(hash(op) % 1000)
    x_np = rng.normal(0, 1.5, size=(3, 7)).astype(np.float32)
    x = _tensor(x_np)
    with ag.record_scope(clear=True):
        y = getattr(ag, op)(x)
        c = rng.normal(0, 1, size=x_np.shape).astype(np.float32)
        y.backward(c)
    got = x.grad
    y_np = y.to_numpy().astype(np.float64)
    ref = (c.astype(np.float64) * (y_np * (1 - y_np)
                                   if op == "sigmoid"
                                   else 1 - y_np ** 2)).astype(np.float32)
    assert got.shape == x_np.shape
    if not np.allclose(got, ref, atol=1e-4, rtol=1e-3):
        print(f"[DIAG {op}] got={got!r} ref={ref!r}", flush=True)
    assert np.allclose(got, ref, atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------- A3/A4 AdamW
def test_a3_adamw_decoupled_decay():
    """零梯度 + wd>0：参数按 (1−lr·wd) 收缩，m/v 保持零（衰减不进统计）。"""
    from engine.sgn.optimizer import AdamW
    p = {"w": np.ones(4, dtype=np.float32)}
    opt = AdamW(p, lr=0.1, weight_decay=0.5)
    opt.step({"w": np.zeros(4, dtype=np.float32)})       # 零梯度
    np.testing.assert_allclose(p["w"], 1.0 * (1.0 - 0.1 * 0.5), rtol=0, atol=1e-7)
    assert np.allclose(opt._state["w"]["m"], 0.0)        # 衰减未污染动量
    assert np.allclose(opt._state["w"]["v"], 0.0)


def test_a4_adamw_single_step_reference():
    """单步 = 手算参考：收缩 + Adam 主式（含偏差修正）。"""
    from engine.sgn.optimizer import AdamW
    lr, b1, b2, eps, wd = 0.01, 0.9, 0.999, 1e-8, 0.1
    p = {"w": np.array([1.0, -2.0], dtype=np.float32)}
    g = np.array([0.5, -1.0], dtype=np.float32)
    w0 = p["w"].astype(np.float64).copy()                # 先存初值（step 原地改）
    opt = AdamW(p, lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
    opt.step({"w": g})
    w = w0 * (1.0 - lr * wd)                             # 解耦收缩
    m = (1 - b1) * g
    v = (1 - b2) * g.astype(np.float64) ** 2
    m_hat = m / (1 - b1)
    v_hat = v / (1 - b2)
    ref = w - lr * m_hat / (np.sqrt(v_hat) + eps)
    np.testing.assert_allclose(p["w"], ref, rtol=1e-5)


def test_a4_adamw_decay_not_in_momentum():
    """与 Adam（衰减混入梯度）对照：AdamW 的 v 统计不含 wd 项。"""
    from engine.sgn.optimizer import Adam, AdamW
    g = np.array([0.3], dtype=np.float32)
    pa = {"w": np.ones(1, dtype=np.float32)}
    pw = {"w": np.ones(1, dtype=np.float32)}
    oa = Adam(pa, lr=0.01, weight_decay=0.5)
    ow = AdamW(pw, lr=0.01, weight_decay=0.5)
    oa.step({"w": g})
    ow.step({"w": g})
    # Adam 的 g' = g + wd·p = 0.8；AdamW 的 g' = 0.3 → v 统计不同
    assert not np.allclose(oa._state["w"]["v"], ow._state["w"]["v"])
    # 但 AdamW 的 v 恰等于无衰减 Adam 的 v
    pz = {"w": np.ones(1, dtype=np.float32)}
    oz = Adam(pz, lr=0.01, weight_decay=0.0)
    oz.step({"w": g})
    assert np.allclose(ow._state["w"]["v"], oz._state["w"]["v"])
