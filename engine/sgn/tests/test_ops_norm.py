# test_ops_norm.py - 归一化族算子验收（批次 4）：LayerNorm（2D 沿末维）
#
# 预先登记验收（基础设施缺口分析批次 4，2026-09-07）：
#   N1 前向 = numpy 参考逐元素（f32 容差 1e-6）
#   N2 dX 梯度经 tape = 解析式（f32 噪声容差，同 A2/B2 标定口径）
#   N3 dgamma/dbeta 经 tape = 解析式（dγ = Σ_b dY·x̂，dβ = Σ_b dY）
#   N4 eps 语义：var + eps（非 sqrt(var)+eps）
#
# 运行：SGN 根目录 pytest engine/sgn/tests/test_ops_norm.py

import numpy as np
import pytest

import engine.sgn as sgn_pkg

ag = sgn_pkg.autograd
nn = sgn_pkg.nn


def _reference(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    y = x_hat * gamma + beta
    return y, x_hat, (1.0 / np.sqrt(var + eps))


def _ln_module(C):
    m = nn.LayerNorm(C)
    # 固定 gamma/beta（非全 1）以检验 gamma/beta 梯度路径
    nn.fill_(m.gamma.tensor(), 1.2)   # gamma != 1（覆盖 γ 路径）
    nn.fill_(m.beta.tensor(), 0.5)    # beta != 0（覆盖 β 路径）
    return m


def test_n1_forward_matches_numpy():
    rng = np.random.default_rng(101)
    B, C = 6, 16
    x = rng.normal(0, 1.5, size=(B, C)).astype(np.float32)
    m = _ln_module(C)
    g = m.gamma.tensor().to_numpy().astype(np.float64)
    beta = m.beta.tensor().to_numpy().astype(np.float64)
    y_ref, _, _ = _reference(x.astype(np.float64), g, beta)
    x_t = ag.Tensor(np.ascontiguousarray(x))
    x_t.requires_grad = True
    with ag.record_scope(clear=True):
        y = m.forward([x_t])
    np.testing.assert_allclose(y.to_numpy(), y_ref.astype(np.float32),
                               rtol=0, atol=1e-6)


@pytest.mark.parametrize("op", ["sigmoid"])  # 占位防误改参数化结构
def _noop(op):
    assert True


def test_n2_dx_gradient_through_tape():
    rng = np.random.default_rng(103)
    B, C = 5, 12
    x_np = rng.normal(0, 1.5, size=(B, C)).astype(np.float32)
    m = _ln_module(C)
    g64 = m.gamma.tensor().to_numpy().astype(np.float64)
    beta64 = m.beta.tensor().to_numpy().astype(np.float64)
    x = ag.Tensor(np.ascontiguousarray(x_np))
    x.requires_grad = True
    with ag.record_scope(clear=True):
        y = m.forward([x])
        c = rng.normal(0, 1, size=(B, C)).astype(np.float32)
        y.backward(c)
    got = x.grad
    # f64 解析参考（A2/B2 同款噪声容差标定）
    y_ref64, x_hat64, rstd64 = _reference(x_np.astype(np.float64), g64, beta64)
    dyg = c * g64
    m1 = dyg.mean(axis=1, keepdims=True)
    m2 = (dyg * x_hat64).mean(axis=1, keepdims=True)
    ref = (rstd64 * (dyg - m1 - x_hat64 * m2)).astype(np.float32)
    np.testing.assert_allclose(got, ref, atol=1e-4, rtol=1e-3)


def test_n3_gamma_beta_gradients():
    rng = np.random.default_rng(107)
    B, C = 5, 12
    x_np = rng.normal(0, 1.5, size=(B, C)).astype(np.float32)
    m = _ln_module(C)
    g64 = m.gamma.tensor().to_numpy().astype(np.float64)
    x_t = ag.Tensor(np.ascontiguousarray(x_np))
    x_t.requires_grad = True
    with ag.record_scope(clear=True):
        y = m.forward([x_t])
        c = rng.normal(0, 1, size=(B, C)).astype(np.float32)
        y.backward(c)
    x_hat = (x_np - x_np.mean(axis=1, keepdims=True)) / np.sqrt(
        x_np.var(axis=1, keepdims=True) + m.eps)
    dgamma_ref = (c * x_hat).sum(axis=0)
    dbeta_ref = c.sum(axis=0)
    np.testing.assert_allclose(m.gamma.grad, dgamma_ref, atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(m.beta.grad, dbeta_ref, atol=1e-4, rtol=1e-3)


def test_n4_eps_semantics():
    """var + eps（eps 加在方差上，非 sqrt 内）——numel 差异可检出。"""
    rng = np.random.default_rng(109)
    B, C = 4, 8
    x = rng.normal(0, 2.0, size=(B, C)).astype(np.float32)
    gamma = np.ones(C, dtype=np.float32)
    beta = np.zeros(C, dtype=np.float32)
    mu = x.mean(axis=1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
    eps = 1e-5
    y_var_plus = (x - mu) / np.sqrt(var + eps)
    y_sqrt_var = (x - mu) / (np.sqrt(var) + eps)
    x_t = ag.Tensor(np.ascontiguousarray(x))
    x_t.requires_grad = False
    ctx_probe = None
    y = ag.layernorm(x_t, ag.Tensor(np.ascontiguousarray(gamma)),
                     ag.Tensor(np.ascontiguousarray(beta)), eps)
    np.testing.assert_allclose(y.to_numpy(), y_var_plus.astype(np.float32),
                               rtol=0, atol=1e-6)
