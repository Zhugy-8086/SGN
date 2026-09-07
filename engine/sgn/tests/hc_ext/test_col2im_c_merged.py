# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""sgn.col2im_c 合并验证脚本

验证 pysgn_col2im 合并到 sgn.col2im_c 子模块后功能正常：
  1. 模块结构：col2im_c 子模块可访问，含 col2im_add 和 __version__
  2. 函数签名：col2im_add 参数正确，类型校验正常
  3. 正确性：col2im_add vs numpy 参考实现，max_diff < 1e-6
  4. 边界条件：空梯度、跨步卷积、非连续输入

运行：
    cd engine/sgn/tests/hc_ext/
    py -3.14 test_col2im_c_merged.py
"""
from __future__ import annotations

import os
import sys
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

# ============================================================
# 测试框架
# ============================================================
_tests: list[tuple[str, callable]] = []


def test(name: str):
    def deco(fn):
        _tests.append((name, fn))
        return fn
    return deco


def run_tests() -> int:
    passed = 0
    failed = 0
    for name, fn in _tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
            failed += 1
    print()
    print(f"{'=' * 70}")
    print(f"col2im_c 合并验证: {passed} passed, {failed} failed")
    print(f"{'=' * 70}")
    return 0 if failed == 0 else 1


# ============================================================
# 1. 模块结构验证
# ============================================================

@test("col2im_c 子模块可访问")
def test_module_accessible():
    """验证 sgn.col2im_c 子模块存在且包含预期属性"""
    col2im_c = sgn.col2im_c
    assert col2im_c is not None, "sgn.col2im_c 不存在"
    assert hasattr(col2im_c, "col2im_add"), "缺少 col2im_add 函数"
    assert hasattr(col2im_c, "__version__"), "缺少 __version__"
    assert col2im_c.__version__ == "1.0.0-col2im", \
        f"__version__ 应为 '1.0.0-col2im', 实际为 '{col2im_c.__version__}'"
    print(f"    __version__: {col2im_c.__version__}")


@test("col2im_add 函数签名正确")
def test_function_signature():
    """验证 col2im_add 函数调用参数正确（pybind11 builtin 不支持 inspect.signature）"""
    # 通过实际调用验证参数映射正确
    B, C, H, W = 2, 4, 8, 8
    kh = kw = 3
    stride = 1
    padding = 1
    H_out = (H + 2 * padding - kh) // stride + 1
    W_out = (W + 2 * padding - kw) // stride + 1
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    x_col = np.zeros((B, C, kh, kw, H_out, W_out), dtype=np.float32)
    x_padded = np.zeros((B, C, H_padded, W_padded), dtype=np.float32)

    # 按位置参数调用（11 个参数）
    sgn.col2im_c.col2im_add(x_col, x_padded, B, C, kh, kw,
                             H_out, W_out, stride, H_padded, W_padded)
    # 按关键字参数调用
    sgn.col2im_c.col2im_add(
        x_col=x_col, x_padded=x_padded, B=B, C=C, kh=kh, kw=kw,
        H_out=H_out, W_out=W_out, stride=stride,
        H_padded=H_padded, W_padded=W_padded)
    print("    ✓ 位置参数和关键字参数调用均正常")


@test("sgn.col2im_add 和 sgn.col2im_c.col2im_add 共存")
def test_both_versions_exist():
    """验证 C++ 和 C 两个版本的 col2im_add 都可用"""
    assert hasattr(sgn, "col2im_add"), "sgn.col2im_add (C++ 版) 不存在"
    assert hasattr(sgn.col2im_c, "col2im_add"), "sgn.col2im_c.col2im_add (C 版) 不存在"
    # 两个函数应是不同的对象（不同实现）
    assert sgn.col2im_add is not sgn.col2im_c.col2im_add, \
        "C++ 和 C 版本应是不同的函数对象"
    print("    C++ 版: sgn.col2im_add")
    print("    C  版: sgn.col2im_c.col2im_add")


# ============================================================
# 2. 正确性验证
# ============================================================

def _numpy_col2im(x_col: np.ndarray, x_shape: tuple,
                  kh: int, kw: int, stride: int, padding: int) -> np.ndarray:
    """numpy 参考实现"""
    B, C, H, W = x_shape
    H_out = (H + 2 * padding - kh) // stride + 1
    W_out = (W + 2 * padding - kw) // stride + 1
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding

    x_padded = np.zeros((B, C, H_padded, W_padded), dtype=x_col.dtype)
    x_col_6d = x_col.reshape(B, C, kh, kw, H_out, W_out)

    for i in range(kh):
        for j in range(kw):
            x_padded[:, :, i:i + stride * H_out:stride,
                     j:j + stride * W_out:stride] += x_col_6d[:, :, i, j, :, :]
    if padding > 0:
        return x_padded[:, :, padding:padding + H, padding:padding + W]
    return x_padded


@test("col2im_add 正确性（典型配置）")
def test_correctness():
    """典型 CNN 配置验证"""
    configs = [
        # (label, B, C, H, W, kh, kw, stride, padding)
        ("pad=0, stride=1, k=3",  4, 8, 5, 5, 3, 3, 1, 0),
        ("pad=1, stride=1, k=3",  4, 8, 5, 5, 3, 3, 1, 1),
        ("pad=0, stride=2, k=3",  4, 8, 7, 7, 3, 3, 2, 0),
        ("pad=1, stride=2, k=3",  4, 8, 7, 7, 3, 3, 2, 1),
        ("conv1 3→64, s1p1",      8, 3, 32, 32, 3, 3, 1, 1),
        ("layer1 64ch, s1p1",     8, 64, 32, 32, 3, 3, 1, 1),
        ("layer2 64ch, s2p1",     8, 64, 32, 32, 3, 3, 2, 1),
        ("layer4 512ch, s1p1",    8, 512, 4, 4, 3, 3, 1, 1),
    ]

    all_pass = True
    for label, B, C, H, W, kh, kw, stride, padding in configs:
        H_out = (H + 2 * padding - kh) // stride + 1
        W_out = (W + 2 * padding - kw) // stride + 1
        H_padded = H + 2 * padding
        W_padded = W + 2 * padding

        rng = np.random.RandomState(42)
        x_col = rng.randn(B, C * kh * kw, H_out * W_out).astype(np.float32)

        # numpy 参考结果
        x_ref = _numpy_col2im(x_col, (B, C, H, W), kh, kw, stride, padding)

        # C 扩展结果
        x_col_6d = np.ascontiguousarray(x_col).reshape(
            B, C, kh, kw, H_out, W_out).astype(np.float32, copy=False)
        x_padded = np.zeros((B, C, H_padded, W_padded), dtype=np.float32)
        sgn.col2im_c.col2im_add(
            x_col_6d, x_padded,
            B, C, kh, kw, H_out, W_out, stride, H_padded, W_padded)

        if padding > 0:
            x_c = x_padded[:, :, padding:padding + H, padding:padding + W]
        else:
            x_c = x_padded

        max_diff = float(np.abs(x_ref - x_c).max())
        ok = max_diff < 1e-6
        if not ok:
            print(f"    {label}: max_diff={max_diff:.2e} ✗")
            all_pass = False
        else:
            print(f"    {label}: max_diff={max_diff:.2e} ✓")

    assert all_pass, "存在配置未通过"


@test("col2im_add 非连续输入处理")
def test_non_contiguous():
    """模拟 backward 中 transpose 后的非连续输入"""
    B, C, H, W = 4, 16, 8, 8
    kh = kw = 3
    stride = 1
    padding = 1
    H_out = (H + 2 * padding - kh) // stride + 1
    W_out = (W + 2 * padding - kw) // stride + 1
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    K = C * kh * kw
    L = H_out * W_out

    rng = np.random.RandomState(7)
    # 模拟 backward: (B*L, K) → reshape(B, L, K) → transpose(0,2,1) → (B, K, L) 非连续
    grad_x_col_2d = rng.randn(B * L, K).astype(np.float32) * 0.1
    grad_x_col = grad_x_col_2d.reshape(B, L, K).transpose(0, 2, 1)

    x_ref = _numpy_col2im(grad_x_col, (B, C, H, W), kh, kw, stride, padding)

    x_col_6d = np.ascontiguousarray(grad_x_col).reshape(
        B, C, kh, kw, H_out, W_out).astype(np.float32, copy=False)
    x_padded = np.zeros((B, C, H_padded, W_padded), dtype=np.float32)
    sgn.col2im_c.col2im_add(
        x_col_6d, x_padded,
        B, C, kh, kw, H_out, W_out, stride, H_padded, W_padded)
    x_c = x_padded[:, :, padding:padding + H, padding:padding + W]

    max_diff = float(np.abs(x_ref - x_c).max())
    assert max_diff < 1e-6, f"非连续输入 max_diff={max_diff:.2e} (应 < 1e-6)"
    print(f"    max_diff={max_diff:.2e}, C_CONTIGUOUS={grad_x_col.flags['C_CONTIGUOUS']}")


@test("col2im_add 空梯度（全零输入）")
def test_zero_input():
    """全零输入应产生全零输出"""
    B, C, H, W = 4, 64, 16, 16
    kh = kw = 3
    stride = 1
    padding = 1
    H_out = (H + 2 * padding - kh) // stride + 1
    W_out = (W + 2 * padding - kw) // stride + 1
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding

    x_col = np.zeros((B, C, kh, kw, H_out, W_out), dtype=np.float32)
    x_padded = np.zeros((B, C, H_padded, W_padded), dtype=np.float32)
    sgn.col2im_c.col2im_add(
        x_col, x_padded,
        B, C, kh, kw, H_out, W_out, stride, H_padded, W_padded)

    assert x_padded.max() == 0.0 and x_padded.min() == 0.0, "全零输入应输出全零"
    print("    ✓ 全零输入输出全零")


@test("col2im_add 形状校验（异常输入）")
def test_shape_validation():
    """非法形状应抛出异常"""
    B, C, H, W = 4, 8, 8, 8
    kh = kw = 3
    stride = 1
    padding = 1
    H_out = (H + 2 * padding - kh) // stride + 1
    W_out = (W + 2 * padding - kw) // stride + 1
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding

    col2im_add = sgn.col2im_c.col2im_add

    # 正常输入（不应抛异常）
    x_col = np.zeros((B, C, kh, kw, H_out, W_out), dtype=np.float32)
    x_padded = np.zeros((B, C, H_padded, W_padded), dtype=np.float32)
    col2im_add(x_col, x_padded, B, C, kh, kw, H_out, W_out, stride, H_padded, W_padded)

    # x_col 维度错误
    try:
        x_col_bad = np.zeros((B, C, kh, kw, H_out), dtype=np.float32)
        col2im_add(x_col_bad, x_padded, B, C, kh, kw, H_out, W_out, stride, H_padded, W_padded)
        assert False, "5D x_col 应抛出异常"
    except RuntimeError:
        pass

    # 维度为负
    try:
        col2im_add(x_col, x_padded, -1, C, kh, kw, H_out, W_out, stride, H_padded, W_padded)
        assert False, "负维度应抛出异常"
    except RuntimeError:
        pass

    print("    ✓ 异常输入校验正确")


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("sgn.col2im_c 合并验证脚本")
    print("=" * 70)
    print(f"  Python:        {sys.version.split()[0]}")
    print(f"  NumPy:         {np.__version__}")
    print(f"  sgn.col2im_c:  {sgn.col2im_c.__version__}")
    print(f"  sgn 版本:      {sgn.version()}")
    print(f"  CPU 核心:      {_cpu_count}")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', '未设置')}")
    print()
    return run_tests()


if __name__ == "__main__":
    sys.exit(main())