# test_ops_activation.py - 激活族独立文件（ops_activation.cpp）验收：gelu/silu
#
# 预先登记验收（基础设施缺口分析批次 2，2026-09-07）：
#   B1 gelu/silu 前向 = numpy 参考（erf 精确式，容差 f32 1e-6）
#   B2 gelu/silu 梯度经 tape backward = 解析式
#      gelu: Φ(x) + x·φ(x)（Φ=0.5(1+erf)，φ=exp(−x²/2)/√(2π)）
#      silu: sig(x)·(1 + x·(1−sig(x)))
#   容差按 f32 tape 舍入噪声标定（同 test_activations_adamw A2 口径）
#
# 运行：SGN 根目录 pytest engine/sgn/tests/test_ops_activation.py

import math

import numpy as np
import pytest

import engine.sgn as sgn_pkg

ag = sgn_pkg.autograd

_erf = np.vectorize(math.erf)


def _tensor(arr):
    t = ag.Tensor(np.ascontiguousarray(arr, dtype=np.float32))
    t.requires_grad = True
    return t


def test_b1_gelu_silu_forward():
    x = np.linspace(-3, 3, 25, dtype=np.float32)
    gelu_ref = 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))
    silu_ref = x / (1.0 + np.exp(-x.astype(np.float64)))
    y_gelu = ag.gelu(_tensor(x, ))
    y_gelu.requires_grad = False
    y_silu = ag.silu(_tensor(x))
    y_silu.requires_grad = False
    np.testing.assert_allclose(y_gelu.to_numpy(), gelu_ref, atol=1e-6)
    np.testing.assert_allclose(y_silu.to_numpy(), silu_ref, atol=1e-6)


@pytest.mark.parametrize("op", ["gelu", "silu"])
def test_b2_gradient_through_tape(op):
    rng = np.random.default_rng(97)
    x_np = rng.normal(0, 1.5, size=(3, 7)).astype(np.float32)
    x = _tensor(x_np)
    with ag.record_scope(clear=True):
        y = getattr(ag, op)(x)
        c = rng.normal(0, 1, size=x_np.shape).astype(np.float32)
        y.backward(c)
    got = x.grad
    assert got.shape == x_np.shape
    if op == "gelu":
        phi = 0.5 * (1.0 + _erf(x_np / np.sqrt(2.0)))
        pdf = np.exp(-0.5 * x_np.astype(np.float64) ** 2) / np.sqrt(2 * np.pi)
        ref = (c * (phi + x_np * pdf)).astype(np.float32)
    else:
        sig = 1.0 / (1.0 + np.exp(-x_np.astype(np.float64)))
        ref = (c * (sig * (1.0 + x_np * (1.0 - sig)))).astype(np.float32)
    if not np.allclose(got, ref, atol=1e-4, rtol=1e-3):
        print(f"[DIAG {op}] got={got!r} ref={ref!r}", flush=True)
    assert np.allclose(got, ref, atol=1e-4, rtol=1e-3)
