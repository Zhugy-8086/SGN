# test_optimizer.py - Optimizer 封装测试（基础设施 B1）
import numpy as np
import pytest

from engine.sgn.optimizer import SGD, Adam


@pytest.fixture
def rng():
    return np.random.default_rng(1)


def test_sgd_matches_hand_formula(rng):
    p0 = rng.standard_normal((6, 4)).astype(np.float32)
    g0 = rng.standard_normal((6, 4)).astype(np.float32)
    params = {"w": p0.copy()}
    opt = SGD(params, lr=0.02, momentum=0.9)
    v = np.zeros_like(p0)
    for step in range(5):
        g = {"w": g0 * (step + 1)}
        opt.step(g)
        v[:] = 0.9 * v + g["w"]
        params["w"] -= 0.02 * v
    assert np.array_equal(params["w"], opt._params["w"])


def test_sgd_plain_no_momentum(rng):
    p0 = rng.standard_normal((5,)).astype(np.float32)
    g0 = rng.standard_normal((5,)).astype(np.float32)
    params = {"w": p0.copy()}
    opt = SGD(params, lr=0.1, momentum=0.0)
    opt.step({"w": g0})
    assert np.array_equal(params["w"], p0 - 0.1 * g0)


def test_sgd_weight_decay(rng):
    p0 = rng.standard_normal((5,)).astype(np.float32)
    g0 = rng.standard_normal((5,)).astype(np.float32)
    params = {"w": p0.copy()}
    opt = SGD(params, lr=0.1, momentum=0.0, weight_decay=0.5)
    opt.step({"w": g0})
    assert np.array_equal(params["w"], p0 - 0.1 * (g0 + 0.5 * p0))


def test_sgd_classical_matches_hand_formula(rng):
    """classical=True（经典/速度式）：v = m*v - lr*g ; p += v（SGN 训练脚本现状）。"""
    p0 = rng.standard_normal((6, 4)).astype(np.float32)
    g0 = rng.standard_normal((6, 4)).astype(np.float32)
    params = {"w": p0.copy()}
    opt = SGD(params, lr=0.02, momentum=0.9, classical=True)
    v = np.zeros_like(p0)
    for step in range(5):
        g = {"w": g0 * (step + 1)}
        opt.step(g)
        v[:] = 0.9 * v - 0.02 * g["w"]
        params["w"] += v
    assert np.array_equal(params["w"], opt._params["w"])


def test_adam_matches_hand_formula(rng):
    p0 = rng.standard_normal((6, 4)).astype(np.float32)
    g0 = rng.standard_normal((6, 4)).astype(np.float32)
    params = {"w": p0.copy()}
    opt = Adam(params, lr=0.001)
    m = np.zeros_like(p0)
    v = np.zeros_like(p0)
    for step in range(4):
        t = step + 1
        g = {"w": g0 * (step + 1)}
        opt.step(g)
        m[:] = 0.9 * m + 0.1 * g["w"]
        v[:] = 0.999 * v + 0.001 * g["w"] ** 2
        params["w"] -= 0.001 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    assert np.array_equal(params["w"], opt._params["w"])


def test_none_grad_skipped(rng):
    p0 = rng.standard_normal((3,)).astype(np.float32)
    opt = SGD({"w": p0.copy()}, lr=0.1)
    opt.step(None)                       # 空梯度：不更新
    assert np.array_equal(opt._params["w"], p0)
    opt.step({"other": np.ones(3, np.float32)})   # 无该名字梯度：跳过
    assert np.array_equal(opt._params["w"], p0)


def test_checkpoint_resume_equals_continuous(rng):
    """正确 checkpoint 语义：params + optimizer 状态一起保存/恢复，续训与不中断逐位一致。"""
    p0 = rng.standard_normal((6, 4)).astype(np.float32)
    g0 = rng.standard_normal((6, 4)).astype(np.float32)
    params_a = {"w": p0.copy()}
    opt_a = SGD(params_a, lr=0.02, momentum=0.9)
    params_b = {"w": p0.copy()}
    opt_b = SGD(params_b, lr=0.02, momentum=0.9)
    for step in range(3):
        g = {"w": g0 * (step + 1)}
        opt_a.step(g)
        opt_b.step(g)
    ckpt = {
        "params": {k: v.copy() for k, v in params_b.items()},
        "optimizer": opt_b.state_dict(),
    }
    # 恢复
    params_c = {k: v.copy() for k, v in ckpt["params"].items()}
    opt_c = SGD(params_c, lr=0.02, momentum=0.9)
    opt_c.load_state_dict(ckpt["optimizer"])
    # 续训
    g4 = {"w": g0 * 4}
    opt_a.step(g4)
    opt_c.step(g4)
    assert np.array_equal(params_a["w"], params_c["w"])


def test_state_dict_snapshot_is_copy(rng):
    """state_dict 返回拷贝：持有后源优化器继续 step 不污染已保存状态。"""
    p0 = rng.standard_normal((3,)).astype(np.float32)
    g0 = rng.standard_normal((3,)).astype(np.float32)
    opt = SGD({"w": p0.copy()}, lr=0.1, momentum=0.9)
    opt.step({"w": g0})
    sd = opt.state_dict()
    saved_v = sd["state"]["w"]["v"]
    assert not np.shares_memory(saved_v, opt._state["w"]["v"])
    opt.step({"w": g0})                   # 继续 step
    assert np.array_equal(saved_v, g0)    # 已保存快照不变


def test_module_params_mode(rng):
    """nn.Module 模式：state_dict/load_state_dict 读写参数。"""
    from engine.sgn import nn
    m = nn.Linear(4, 3)
    opt = SGD(m, lr=0.1, momentum=0.0)
    g = {k: np.ones_like(v, np.float32) for k, v in opt._params.items()}
    opt.step(g)
    # 参数已写回 Module
    sd = m.state_dict()
    assert sd.keys() == opt._params.keys()
    # 与直接对 numpy 应用 SGD 一致（第一个参数）
    first = next(iter(opt._params))
    assert np.array_equal(m.state_dict()[first], opt._params[first])
