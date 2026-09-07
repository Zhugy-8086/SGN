# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""pysgn_col2im 验证脚本

验证内容：
  1. 正确性：C 扩展 col2im vs numpy _col2im，max_diff < 1e-6
  2. 性能：C 扩展 vs numpy，加速比测量
  3. 边界情况：padding=0/1, stride=1/2, 不同 kh/kw
  4. ResNet-18 代表性 Conv 配置实测

运行：
    cd engine/sgn/tests/hc_ext/
    py -3.14 test_col2im_c.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# ============================================================
# 环境设置
# ============================================================
_cpu_count = os.cpu_count() or 4
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, _cpu_count - 2)))

HERE = Path(__file__).resolve().parent
# 安全审计 2026-08-16 A2-7：模式 B → 模式 A（import engine.sgn as sgn）
_PROJECT_ROOT = HERE.parents[3]  # hc_ext -> tests -> sgn -> engine -> SGN 根
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import engine.sgn as sgn  # noqa: E402
pysgn_col2im = sgn.col2im_c  # 原独立 pysgn_col2im 已合并到 sgn.col2im_c 子模块


# ============================================================
# numpy 参考实现（与 sgn_layers.py _col2im 一致）
# ============================================================

def numpy_col2im(x_col: np.ndarray, x_shape: tuple,
                 kh: int, kw: int, stride: int, padding: int) -> np.ndarray:
    """与 sgn_layers._col2im 完全一致的 numpy 实现（参考实现）"""
    B, C, H, W = x_shape
    H_out = (H + 2 * padding - kh) // stride + 1
    W_out = (W + 2 * padding - kw) // stride + 1
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding

    x_padded = np.zeros((B, C, H_padded, W_padded), dtype=x_col.dtype)
    x_col_reshaped = x_col.reshape(B, C, kh, kw, H_out, W_out)

    for i in range(kh):
        for j in range(kw):
            x_padded[:, :, i:i + stride * H_out:stride,
                     j:j + stride * W_out:stride] += \
                x_col_reshaped[:, :, i, j, :, :]

    if padding > 0:
        return x_padded[:, :, padding:padding + H, padding:padding + W]
    return x_padded


def c_col2im(x_col: np.ndarray, x_shape: tuple,
             kh: int, kw: int, stride: int, padding: int) -> np.ndarray:
    """C 扩展 col2im 包装（接口与 numpy_col2im 一致）

    内部完成 reshape + 调用 C 扩展 + 裁剪 padding。
    """
    B, C, H, W = x_shape
    H_out = (H + 2 * padding - kh) // stride + 1
    W_out = (W + 2 * padding - kw) // stride + 1
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding

    # x_col: (B, K, L) → (B, C, kh, kw, H_out, W_out) C-contiguous
    # 注意：x_col 可能非连续（来自 transpose），需 ascontiguousarray
    x_col_6d = np.ascontiguousarray(x_col).reshape(
        B, C, kh, kw, H_out, W_out).astype(np.float32, copy=False)
    if not x_col_6d.flags["C_CONTIGUOUS"]:
        x_col_6d = np.ascontiguousarray(x_col_6d)

    x_padded = np.zeros((B, C, H_padded, W_padded), dtype=np.float32)

    pysgn_col2im.col2im_add(
        x_col_6d, x_padded,
        B, C, kh, kw, H_out, W_out, stride,
        H_padded, W_padded,
    )

    if padding > 0:
        return x_padded[:, :, padding:padding + H, padding:padding + W]
    return x_padded


# ============================================================
# 工具函数
# ============================================================

def _timeit(fn, n_iter: int) -> float:
    """计时辅助：返回 n_iter 次调用的平均秒数（不含 warmup）"""
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    return (time.perf_counter() - t0) / n_iter


def _fmt_ms(t: float) -> str:
    if t < 1e-3:
        return f"{t * 1e6:8.2f} µs"
    return f"{t * 1000:8.2f} ms"


def _print_section(title: str):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


# ============================================================
# 1. 正确性验证
# ============================================================

def test_correctness():
    """正确性验证：C 扩展 vs numpy，多种配置"""
    _print_section("1. 正确性验证 (C 扩展 vs numpy _col2im)")

    # (label, B, C, H, W, kh, kw, stride, padding)
    configs = [
        # 边界情况
        ("pad=0, stride=1, k=1",    4, 8, 7, 7, 1, 1, 1, 0),
        ("pad=0, stride=1, k=3",    4, 8, 5, 5, 3, 3, 1, 0),
        ("pad=0, stride=2, k=3",    4, 8, 7, 7, 3, 3, 2, 0),
        ("pad=1, stride=1, k=3",    4, 8, 5, 5, 3, 3, 1, 1),
        ("pad=1, stride=2, k=3",    4, 8, 5, 5, 3, 3, 2, 1),
        ("pad=0, stride=1, k=1x1",  4, 8, 7, 7, 1, 1, 1, 0),
        # 非对称核
        ("pad=0, stride=1, k=1x3",  4, 8, 5, 7, 1, 3, 1, 0),
        ("pad=0, stride=1, k=3x1",  4, 8, 7, 5, 3, 1, 1, 0),
        # ResNet-18 代表性配置
        ("conv1: 3→64, 32×32, k3s1p1",   8,  3,  32, 32, 3, 3, 1, 1),
        ("layer1: 64ch, 32×32, k3s1p1",  8, 64,  32, 32, 3, 3, 1, 1),
        ("layer2: 64ch, 32×32, k3s2p1",  8, 64,  32, 32, 3, 3, 2, 1),
        ("layer2.sc: 64ch, k1s2p0",      8, 64,  32, 32, 1, 1, 2, 0),
        ("layer4: 512ch, 4×4, k3s1p1",   8, 512,  4,  4, 3, 3, 1, 1),
    ]

    all_pass = True
    print(f"\n  {'配置':<32s} {'形状(x_col)':<20s} {'max_diff':<12s} {'结果'}")
    print(f"  {'-' * 32} {'-' * 20} {'-' * 12} {'-' * 6}")

    for label, B, C, H, W, kh, kw, stride, padding in configs:
        H_out = (H + 2 * padding - kh) // stride + 1
        W_out = (W + 2 * padding - kw) // stride + 1
        K = C * kh * kw
        L = H_out * W_out

        # 随机 x_col: (B, K, L)
        rng = np.random.RandomState(42)
        x_col = (rng.randn(B, K, L).astype(np.float32) * 0.1)
        x_shape = (B, C, H, W)

        # numpy 参考结果
        x_ref = numpy_col2im(x_col, x_shape, kh, kw, stride, padding)
        # C 扩展结果
        x_c = c_col2im(x_col, x_shape, kh, kw, stride, padding)

        max_diff = float(np.abs(x_ref - x_c).max())
        ok = max_diff < 1e-6
        all_pass = all_pass and ok

        shape_str = f"({B},{K},{L})"
        print(f"  {label:<32s} {shape_str:<20s} {max_diff:<12.2e} {'✓ PASS' if ok else '✗ FAIL'}")

    print(f"\n  总结: {'全部通过 ✓' if all_pass else '存在失败 ✗'}")
    return all_pass


# ============================================================
# 2. 性能对比
# ============================================================

def test_performance():
    """性能对比：C 扩展 vs numpy"""
    _print_section("2. 性能对比 (C 扩展 vs numpy)")

    # (label, B, C, H, W, kh, kw, stride, padding, n_iter)
    configs = [
        ("conv1: 3→64, 32×32, s1",      64,   3,  32, 32, 3, 3, 1, 1, 30),
        ("layer1: 64ch, 32×32, s1",     64,  64,  32, 32, 3, 3, 1, 1, 20),
        ("layer2: 64ch, 32×32, s2",     64,  64,  32, 32, 3, 3, 2, 1, 20),
        ("layer2.sc: 64ch, k1s2",       64,  64,  32, 32, 1, 1, 2, 0, 30),
        ("layer3: 128ch, 16×16, s2",    64, 128,  16, 16, 3, 3, 2, 1, 30),
        ("layer4: 256ch, 8×8, s2",      64, 256,   8,  8, 3, 3, 2, 1, 30),
        ("layer4: 512ch, 4×4, s1",      64, 512,   4,  4, 3, 3, 1, 1, 30),
    ]

    print(f"\n  {'配置':<28s} {'numpy':<12s} {'C扩展':<12s} {'加速比':<10s}")
    print(f"  {'-' * 28} {'-' * 12} {'-' * 12} {'-' * 10}")

    np_total = 0.0
    c_total = 0.0
    for label, B, C, H, W, kh, kw, stride, padding, n_iter in configs:
        H_out = (H + 2 * padding - kh) // stride + 1
        W_out = (W + 2 * padding - kw) // stride + 1
        K = C * kh * kw
        L = H_out * W_out

        rng = np.random.RandomState(42)
        x_col = (rng.randn(B, K, L).astype(np.float32) * 0.1)
        x_shape = (B, C, H, W)

        # numpy 计时
        t_np = _timeit(lambda: numpy_col2im(x_col, x_shape, kh, kw, stride, padding), n_iter)
        # C 扩展计时
        t_c = _timeit(lambda: c_col2im(x_col, x_shape, kh, kw, stride, padding), n_iter)

        speedup = t_np / t_c if t_c > 0 else float("inf")
        np_total += t_np
        c_total += t_c

        print(f"  {label:<28s} {_fmt_ms(t_np):<12s} {_fmt_ms(t_c):<12s} {speedup:<10.2f}x")

    print(f"\n  合计:")
    print(f"    numpy:  {_fmt_ms(np_total)}")
    print(f"    C 扩展: {_fmt_ms(c_total)}")
    print(f"    整体加速比: {np_total / c_total:.2f}x")

    return {"np_total": np_total, "c_total": c_total}


# ============================================================
# 3. 边界情况
# ============================================================

def test_edge_cases():
    """边界情况：极端配置"""
    _print_section("3. 边界情况验证")

    cases = [
        # (label, B, C, H, W, kh, kw, stride, padding)
        ("1x1 卷积 pad=0",        2, 4, 8, 8, 1, 1, 1, 0),
        ("1x1 卷积 pad=0 stride=2", 2, 4, 8, 8, 1, 1, 2, 0),
        ("3x3 pad=1 同尺寸",      2, 4, 8, 8, 3, 3, 1, 1),
        ("3x3 pad=0 stride=1",    2, 4, 5, 5, 3, 3, 1, 0),
        ("非方阵 H≠W",            2, 4, 7, 11, 3, 3, 1, 1),
        ("非方阵 stride=2",       2, 4, 9, 13, 3, 3, 2, 1),
        ("大 kernel 5x5",         2, 4, 12, 12, 5, 5, 1, 2),
        ("大 kernel 5x5 stride=2", 2, 4, 14, 14, 5, 5, 2, 2),
        ("batch=1",               1, 8, 8, 8, 3, 3, 1, 1),
        ("大通道 C=256",          2, 256, 8, 8, 3, 3, 1, 1),
        ("H_out=W_out=1",         2, 4, 3, 3, 3, 3, 1, 0),
        ("stride=3",              2, 4, 9, 9, 3, 3, 3, 0),
    ]

    all_pass = True
    print(f"\n  {'配置':<28s} {'max_diff':<12s} {'结果'}")
    print(f"  {'-' * 28} {'-' * 12} {'-' * 6}")

    for label, B, C, H, W, kh, kw, stride, padding in cases:
        H_out = (H + 2 * padding - kh) // stride + 1
        W_out = (W + 2 * padding - kw) // stride + 1
        if H_out <= 0 or W_out <= 0:
            print(f"  {label:<28s} {'跳过(无效)':<12s} SKIP")
            continue
        K = C * kh * kw
        L = H_out * W_out

        rng = np.random.RandomState(123)
        x_col = (rng.randn(B, K, L).astype(np.float32) * 0.1)
        x_shape = (B, C, H, W)

        x_ref = numpy_col2im(x_col, x_shape, kh, kw, stride, padding)
        x_c = c_col2im(x_col, x_shape, kh, kw, stride, padding)

        max_diff = float(np.abs(x_ref - x_c).max())
        ok = max_diff < 1e-6
        all_pass = all_pass and ok
        print(f"  {label:<28s} {max_diff:<12.2e} {'✓ PASS' if ok else '✗ FAIL'}")

    print(f"\n  总结: {'全部通过 ✓' if all_pass else '存在失败 ✗'}")
    return all_pass


# ============================================================
# 4. 非连续输入测试
# ============================================================

def test_non_contiguous():
    """验证非连续 x_col 输入（来自 transpose，模拟 backward 实际场景）"""
    _print_section("4. 非连续输入验证 (模拟 backward 的 transpose 场景)")

    B, C, H, W = 4, 16, 8, 8
    kh = kw = 3
    stride = 1
    padding = 1
    H_out = (H + 2 * padding - kh) // stride + 1
    W_out = (W + 2 * padding - kw) // stride + 1
    K = C * kh * kw
    L = H_out * W_out

    rng = np.random.RandomState(7)
    # 模拟 backward 中的场景：
    # grad_x_col_2d (B*L, K) → reshape(B, L, K) → transpose(0,2,1) → (B, K, L) 非连续
    grad_x_col_2d = rng.randn(B * L, K).astype(np.float32) * 0.1
    grad_x_col = grad_x_col_2d.reshape(B, L, K).transpose(0, 2, 1)  # (B, K, L) 非连续

    x_shape = (B, C, H, W)
    x_ref = numpy_col2im(grad_x_col, x_shape, kh, kw, stride, padding)
    x_c = c_col2im(grad_x_col, x_shape, kh, kw, stride, padding)

    max_diff = float(np.abs(x_ref - x_c).max())
    ok = max_diff < 1e-6
    print(f"\n  非连续输入 max_diff = {max_diff:.2e}")
    print(f"  x_col C_CONTIGUOUS = {grad_x_col.flags['C_CONTIGUOUS']}")
    print(f"  结果: {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 80)
    print("  col2im C 扩展验证脚本 (sgn.col2im_c)")
    print("=" * 80)
    print(f"  Python:       {sys.version.split()[0]}")
    print(f"  NumPy:        {np.__version__}")
    print(f"  sgn.col2im_c:  {pysgn_col2im.__version__}")
    print(f"  CPU 核心:     {_cpu_count}")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', '未设置')}")

    # 1. 正确性
    ok1 = test_correctness()

    # 2. 性能
    test_performance()

    # 3. 边界情况
    ok3 = test_edge_cases()

    # 4. 非连续输入
    ok4 = test_non_contiguous()

    # 总结
    _print_section("总结")
    print(f"\n  正确性验证:     {'✓ PASS' if ok1 else '✗ FAIL'}")
    print(f"  边界情况验证:   {'✓ PASS' if ok3 else '✗ FAIL'}")
    print(f"  非连续输入验证: {'✓ PASS' if ok4 else '✗ FAIL'}")
    all_ok = ok1 and ok3 and ok4
    print(f"\n  总体: {'✓ 全部通过' if all_ok else '✗ 存在失败'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
