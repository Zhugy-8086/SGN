# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""int8 对（h,l）叶梯度存储 A″ 三层访问验证

对应 msint_int8_pair_grad_carrier_2026_08_31.md §八.4 / StrategyContext::pair_grad_store：
  L0 主业默认：开关关 = 现状 float32（逐位不变，开关 off 基准）
  L1 主业+省显存：开关开 + SR/A1 + bits=16 → 单路径叶 pair 存储（2B/元素），
     grad() 透明解码（惰性缓存）——**同种子下与 L0 叶 grad 逐位相等**（pair 化
     不改训练动态：同 SRNG 同舍入序列 + fine≡Q16-SR，C++ P2 + 数学层 V7 的实现侧兑现）
  L2 研究：grad_pair(id) → PairGradView（h/l 肢、q 往返、fine/coarse 双读、dot8 消费）
  回退矩阵：FLOAT32 / GEF / 多路径叶（权重共享）自动回退 float（grad_pair → None）
  clear() 清 pair 存储

运行: python engine/sgn/tests/architecture/test_pair_grad_store.py
"""

import os
import sys
from pathlib import Path

import numpy as np

# 项目惯用法：直载 build/ 下 .pyd（sgn.cp314-win_amd64.pyd），与 validate_* 一致
_ENGINE_BUILD = Path(__file__).resolve().parents[2] / "build"
sys.path.insert(0, str(_ENGINE_BUILD))

import sgn

ag = sgn.autograd

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {name}")


def make_case(seed=7):
    """单层线性：Y = X @ W，W 叶梯度 (4,3)。"""
    rng = np.random.RandomState(seed)
    X = rng.randn(2, 4).astype(np.float32)
    W = (rng.randn(4, 3) * 0.5).astype(np.float32)
    gout = rng.randn(2, 3).astype(np.float32)
    return X, W, gout


def run_backward(X, W, gout, strategy, pair_store, shared=False):
    """跑一次 backward，返回 (W_grad_numpy, W_pair_view_or_None)。"""
    ag.clear()
    ag.set_backward_strategy(getattr(ag.BackwardStrategy, strategy))
    ag.set_sr_seed(42)
    ag.set_pair_grad_store(pair_store)
    ag.set_quant_config(bits=16)
    ag.start_recording()
    x_t = ag.Tensor.from_numpy(X)
    w_t = ag.Tensor.from_numpy(W)
    w_t.requires_grad = True
    if shared:
        h = ag.matmul(x_t, w_t)          # W 第一次作为 input（此 output 无梯度播种）
        x2_t = ag.Tensor.from_numpy(X)   # 同 tape 第二条路径共享 W
        y = ag.matmul(x2_t, w_t)         # W 第二次 → count=2 → 回退 float
    else:
        y = ag.matmul(x_t, w_t)
    ag.stop_recording()
    y.backward(gout)
    grad = None if w_t.grad is None else np.asarray(w_t.grad)
    view = ag.grad_pair(w_t.id)
    ag.set_pair_grad_store(False)
    return grad, view


# ---- T1: L0/L1 bit-exact 对照（SR / A1，同种子开关 off/on）----
def test_l0_l1_bitexact():
    # F4 扩展（2026-09-02 审查）：多 seed × 多分布。
    # 分布设计（作用于 dW = Xᵀ·dY 的实际幅值）：
    #   normal   — 常规训练分布（远零邻域）
    #   tiny     — gout ~ 1e-14 → dW max|g| < 1e-12（触发 sr_quantize_grad
    #              early-return 保持原值 vs pair 版编码 q=0 的【已声明语义分叉】：
    #              此分布下 L0/L1 不要求逐位相等，仅验证 pair 侧 scale=1.0 / q=0）
    #   wide     — 跨数量级（类真实梯度）
    #   clipped  — 大幅值触发 clip_bound 路径
    for strategy in ("SR", "A1"):
        for dist_name, wgen, gscale in (
            ("normal",  lambda r: r.randn(4, 3).astype(np.float32) * 0.5, 1.0),
            ("tiny",    lambda r: r.randn(4, 3).astype(np.float32) * 0.5, 1e-14),
            ("wide",    lambda r: (r.randn(4, 3)
                                   * np.float_power(10.0, r.randint(-6, 2, (4, 3)))).astype(np.float32), 1.0),
            ("clipped", lambda r: (r.randn(4, 3) * 50).astype(np.float32), 1.0),
        ):
            for seed in (7, 42, 2026):
                rng = np.random.RandomState(seed)
                X = rng.randn(2, 4).astype(np.float32)
                W = wgen(np.random.RandomState(seed + 100))
                gout = (rng.randn(2, 3) * gscale).astype(np.float32)

                grad_off, view_off = run_backward(X, W, gout, strategy, False)
                check(f"[{strategy}/{dist_name}/s{seed}] L0 无 pair 视图",
                      view_off is None)
                grad_on, view_on = run_backward(X, W, gout, strategy, True)
                check(f"[{strategy}/{dist_name}/s{seed}] L1 pair 视图存在",
                      view_on is not None)

                if dist_name == "tiny":
                    # 已声明语义分叉（见 pair_grad_carrier.h 头注）：
                    # float 版保持原值（~1e-14 量级）；pair 版 max_abs<1e-12 →
                    # scale=1.0、q 全 0。此处验证分叉行为符合声明而非逐位相等。
                    check(f"[{strategy}/{dist_name}/s{seed}] tiny: pair scale==1.0",
                          view_on.scale == 1.0)
                    check(f"[{strategy}/{dist_name}/s{seed}] tiny: q 全 0",
                          int(np.abs(np.asarray(view_on.q())).max()) == 0)
                else:
                    ok = (grad_off is not None and grad_on is not None
                          and grad_off.shape == grad_on.shape
                          and np.array_equal(grad_off, grad_on))
                    check(f"[{strategy}/{dist_name}/s{seed}] L0/L1 叶 grad 逐位相等",
                          ok)


# ---- T2: L1 grad() 透明解码 == 视图 fine ----
def test_l1_transparent_decode():
    X, W, gout = make_case()
    grad_on, view = run_backward(X, W, gout, "SR", True)
    check("L1 grad 非空", grad_on is not None)
    fine = np.asarray(view.fine())
    # grad() 走惰性解码缓存；fine 是同一解码公式——逐位相等
    check("grad() == PairGradView.fine()（同一解码路径）",
          grad_on.shape == fine.reshape(grad_on.shape).shape
          and np.array_equal(grad_on, fine.reshape(grad_on.shape)))
    # scale 一致性：max|grad| / 32767 ≈ view.scale（SR 网格化后 max|q|=32767）
    q = np.asarray(view.q())
    check("max|q| == 32767（per-tensor scale 定义）", int(np.abs(q).max()) == 32767)


# ---- T3: L2 视图细节（肢/dtype/q 往返/dot8 消费）----
def test_view_details():
    X, W, gout = make_case(11)
    _, view = run_backward(X, W, gout, "SR", True)
    n = X.shape[1] * W.shape[1]
    h = np.asarray(view.h())
    l = np.asarray(view.l())
    q = np.asarray(view.q())
    check("n == 元素数", view.n == n)
    check("h/l dtype uint8", h.dtype == np.uint8 and l.dtype == np.uint8)
    check("h 值域 [0,255]", 0 <= h.min() and h.max() <= 255)
    # q 往返：256·h + l − 32768 == q（编码恒等式）
    q_ref = 256 * h.astype(np.int32) + l.astype(np.int32) - 32768
    check("q 往返恒等式（256h+l−32768）", np.array_equal(q.astype(np.int32), q_ref))
    check("q 值域 [-32767, 32767]", -32767 <= q.min() and q.max() <= 32767)
    # fine = q·scale
    fine = np.asarray(view.fine())
    check("fine == q·scale（float32 逐位）",
          np.array_equal(fine, (q.astype(np.float32)) * np.float32(view.scale)))
    # coarse = 256·h_s·scale（h_s = h − 128）
    coarse = np.asarray(view.coarse())
    coarse_ref = (256 * (h.astype(np.int32) - 128)).astype(np.float32) * np.float32(view.scale)
    check("coarse == 256·h_s·scale", np.array_equal(coarse, coarse_ref))
    # dot8 消费 vs numpy int64 参照（fine / coarse）
    rng = np.random.RandomState(3)
    w8 = rng.randint(-128, 128, size=n).astype(np.int8)
    df = view.dot_fine(w8)
    dc = view.dot_coarse(w8)
    ref_f = int(np.sum(q.astype(np.int64) * w8.astype(np.int64)))
    h_s = h.astype(np.int32) - 128
    ref_c = int(np.sum((256 * h_s).astype(np.int64) * w8.astype(np.int64)))
    check("dot_fine == int64 参照（2×dot8 恒等式）", int(df) == ref_f)
    check("dot_coarse == int64 参照（1×dot8 恒等式）", int(dc) == ref_c)


# ---- T4: 回退矩阵（FLOAT32 / GEF / 多路径叶）----
def test_fallback_matrix():
    X, W, gout = make_case()
    # FLOAT32：不量化，pair 不适用
    _, v1 = run_backward(X, W, gout, "FLOAT32", True)
    check("FLOAT32 回退（grad_pair None）", v1 is None)
    # GEF：确定性舍入路径，第一期不接 pair
    _, v2 = run_backward(X, W, gout, "GEF", True)
    check("GEF 回退（grad_pair None）", v2 is None)
    # 多路径叶（权重共享）：自动回退 float
    grad_shared, v3 = run_backward(X, W, gout, "SR", True, shared=True)
    check("多路径叶回退（grad_pair None）", v3 is None)
    check("多路径叶 grad 正常产出（float 路径）", grad_shared is not None)


# ---- T5: clear 清 pair 存储 + 开关状态 ----
def test_clear_and_state():
    X, W, gout = make_case()
    ag.clear()
    ag.set_pair_grad_store(True)
    check("开关状态查询", ag.pair_grad_store() is True or ag.pair_grad_store() == True)
    run_backward(X, W, gout, "SR", True)
    ag.clear()
    # clear 后：同 id 体系已清，重建 tensor 无法对应旧 pair——用新 backward 验证干净
    grad2, view2 = run_backward(X, W, gout, "SR", True)
    check("clear 后重新 backward 正常", grad2 is not None and view2 is not None)
    check("重新 backward 后 pair 尺寸正确", view2.n == W.size)
    ag.set_pair_grad_store(False)


def main():
    print("=" * 70)
    print("int8 对（h,l）叶梯度存储 A″ 三层访问验证")
    print("=" * 70)
    test_l0_l1_bitexact()
    test_l1_transparent_decode()
    test_view_details()
    test_fallback_matrix()
    test_clear_and_state()
    print(f"结论: PASS={PASS}, FAIL={FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
