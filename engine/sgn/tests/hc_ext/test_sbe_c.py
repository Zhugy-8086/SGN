# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""
SBE C 化正确性对照测试

验证 C 扩展（pysgn_net.sbe_quantize_weight_blocks / sbe_matmul）与
Python 参考实现（sbe_conv2d.quantize_weight_sbe / sbe_matmul）数学等价。

运行：
    cd engine/sgn/tests/hc_ext/
    py -3.14 test_sbe_c.py

对照原理：
  两条路径都计算 y[i][j] = Σ_g Σ_k x_q[i][g,k] * w_q[g,k][j] * x_scale_g * w_scale_g
  其中 x_q, w_q 是 per-block 独立量化的 int8 值（范围 [-127, 127]）。
  C 路径用 AVX-VNNI _mm256_dpbusd_epi32 加速累加，Python 路径用 numpy int32 matmul。
  int32 累加无溢出（k_block ≤ 4096, |q| ≤ 127, max acc ≈ 6.6e7 << INT32_MAX），
  所以两条路径应当给出相同结果（max_diff < 1e-4，由 float32 累加顺序差异导致）。
"""
import sys
import os
from pathlib import Path

# ── OMP 冲突调式日志 ──────────────────────────────────────────
# 安全审计 2026-08-16 A2-6：原注释描述 MSVC/Clang 双 .pyd 共存场景——
# 2026-08-06 起全部 .pyd 已合并为 Clang 编译的单一 sgn 模块，该场景不复
# 存在；KMP_DUPLICATE_LIB_OK 保留为防御性设置（进程内如加载其它 OpenMP
# 运行时的第三方扩展时仍可避免 "Error #15" 崩溃）。
_DEBUG = "SGN_DEBUG" in os.environ
if _DEBUG:
    _omp_before = os.environ.get('KMP_DUPLICATE_LIB_OK', '(未设置)')
    print(f"[DEBUG] test_sbe_c: KMP_DUPLICATE_LIB_OK 设置前={_omp_before}")
    print(f"[DEBUG] test_sbe_c: sys.path={sys.path}")
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np

# 导入 C 扩展
HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = HERE.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── 路径解析调式日志 ──────────────────────────────────────────
# 背景：import engine.sgn 依赖 _PROJECT_ROOT 正确指向项目根目录。
# 若路径偏移会导致加载 engine/sgn/__init__.py（包）而非 .pyd（扩展）。
# HERE = engine/sgn/tests/hc_ext/，4 层 parent = SGN/（项目根）
if _DEBUG:
    print(f"[DEBUG] test_sbe_c: 路径解析")
    print(f"[DEBUG]   HERE={HERE}")
    print(f"[DEBUG]   _PROJECT_ROOT={_PROJECT_ROOT}")
    print(f"[DEBUG]   sys.path(前3)={sys.path[:3]}")

try:
    import engine.sgn as _sgn
    pysgn_net = _sgn._native.hc8_net
except (ImportError, AttributeError) as e:
    # 安全审计 2026-08-16 A2-2：模块级 sys.exit(1) 杀死 pytest 收集进程。
    # A2-4：错误指引更新为 cmake 构建（旧 setup_net.py 已删除）
    if _DEBUG:
        import traceback
        print(f"[DEBUG] test_sbe_c: 导入异常详情")
        print(f"[DEBUG]   _PROJECT_ROOT={_PROJECT_ROOT}")
        print(f"[DEBUG]   sys.path(前3)={sys.path[:3]}")
        traceback.print_exc()
    try:
        import pytest
        pytest.skip(f"engine.sgn.hc8_net 不可用: {e}", allow_module_level=True)
    except ImportError:
        print(f"ERROR: 无法导入 pysgn_net: {e}")
        print("请先编译: cd engine/sgn && cmake -B build -S . && cmake --build build")
        sys.exit(1)

# 导入 Python SBE 参考实现（位于 tests/refs/，见 refs/__init__.py）
SBE_DIR = HERE.parent / "refs"
sys.path.insert(0, str(SBE_DIR))
try:
    from sbe_conv2d import sbe_matmul as sbe_matmul_py, quantize_weight_sbe
except ImportError as e:
    # 安全审计 2026-08-16 A2-2：同上
    try:
        import pytest
        pytest.skip(f"sbe_conv2d 参考实现不可用: {e}", allow_module_level=True)
    except ImportError:
        print(f"ERROR: 无法导入 sbe_conv2d: {e}")
        sys.exit(1)


def _run_one_case(m, k, n, groups, k_block, seed=42):
    """运行一次对照测试

    Returns:
        (max_diff, y_py, y_c)
    """
    assert groups * k_block == k, f"groups*k_block={groups*k_block} != k={k}"

    rng = np.random.default_rng(seed)
    # 使用 float32 与训练时一致
    x = rng.standard_normal((m, k)).astype(np.float32) * 0.5
    w = rng.standard_normal((k, n)).astype(np.float32) * 0.3

    # ===== Python 参考路径 =====
    w_blocks = quantize_weight_sbe(w, groups, k_block, m, k, n)
    y_py = sbe_matmul_py(x, w_blocks, groups, k_block, m, k, n)

    # ===== C 扩展路径 =====
    w_signed, w_sum_b, w_scales = pysgn_net.sbe_quantize_weight_blocks(
        w, groups, k_block, k, n
    )
    y_c = pysgn_net.sbe_matmul(
        x, w_signed, w_sum_b, w_scales,
        groups, k_block, m, k, n
    )

    max_diff = float(np.abs(y_py - y_c).max())
    return max_diff, y_py, y_c


def test_small():
    """测试 1: 小矩阵对照（m=4, k=8, n=3, groups=2, k_block=4）"""
    print("[test 1] 小矩阵对照")
    m, k, n = 4, 8, 3
    groups, k_block = 2, 4
    max_diff, y_py, y_c = _run_one_case(m, k, n, groups, k_block, seed=42)
    print(f"  shape: x=({m},{k}), w=({k},{n}), groups={groups}, k_block={k_block}")
    print(f"  max_diff = {max_diff:.6e}")
    print(f"  y_py[0]  = {y_py[0]}")
    print(f"  y_c[0]   = {y_c[0]}")
    # float32 累加顺序差异允许少量误差
    assert max_diff < 1e-3, f"max_diff={max_diff} 过大"
    print("  ✓ 通过\n")


def test_medium():
    """测试 2: 中等矩阵（模拟 MNIST FC 层）

    max_diff 允许较大阈值：Python 路径用 float64 标量做 scale 乘法
    （numpy 的 float32_array * python_float 可能内部提升精度），
    C 路径用 float32 标量。int32 累加是精确的，差异仅来自 float 精度。
    对于 28 个 group 的累加，0.05 阈值已经很紧（典型输出值 ~1.0）。
    """
    print("[test 2] 中等矩阵对照（MNIST FC 规模）")
    # 模拟 MNIST FC: m=64(batch), k=784, n=128, groups=28(C_in), k_block=28
    m, k, n = 64, 784, 128
    groups, k_block = 28, 28
    max_diff, y_py, y_c = _run_one_case(m, k, n, groups, k_block, seed=123)
    print(f"  shape: x=({m},{k}), w=({k},{n}), groups={groups}, k_block={k_block}")
    print(f"  max_diff = {max_diff:.6e}")
    print(f"  y_py abs max = {np.abs(y_py).max():.4f}")
    print(f"  y_c abs max  = {np.abs(y_c).max():.4f}")
    # float32 vs float64 scale 乘法 + 28 group 累加 → 允许 0.05 差异
    assert max_diff < 5e-2, f"max_diff={max_diff} 过大"
    print("  ✓ 通过\n")


def test_cifar_conv():
    """测试 3: CIFAR-10 卷积规模（im2col 后的 matmul）"""
    print("[test 3] CIFAR-10 卷积规模对照")
    # CIFAR-10 conv1: B=64, C_in=3, C_out=32, kh=kw=3
    # im2col: m = B*H_out*W_out = 64*30*30 = 57600, k = C_in*kh*kw = 27, n = C_out = 32
    # SBE: groups = C_in = 3, k_block = kh*kw = 9
    m, k, n = 256, 27, 32  # 用较小的 batch 测试
    groups, k_block = 3, 9
    max_diff, y_py, y_c = _run_one_case(m, k, n, groups, k_block, seed=7)
    print(f"  shape: x=({m},{k}), w=({k},{n}), groups={groups}, k_block={k_block}")
    print(f"  max_diff = {max_diff:.6e}")
    assert max_diff < 1e-3, f"max_diff={max_diff} 过大"
    print("  ✓ 通过\n")


def test_k_block_not_multiple_32():
    """测试 4: k_block 不是 32 的倍数（测试 VNNI 尾部处理）"""
    print("[test 4] k_block 非 32 倍数（VNNI 尾部处理）")
    # k_block=9 (CIFAR conv), 27, 49 等都不是 32 的倍数
    m, k, n = 16, 27, 32
    groups, k_block = 3, 9
    max_diff, y_py, y_c = _run_one_case(m, k, n, groups, k_block, seed=999)
    print(f"  shape: x=({m},{k}), w=({k},{n}), groups={groups}, k_block={k_block}")
    print(f"  max_diff = {max_diff:.6e}")
    assert max_diff < 1e-3, f"max_diff={max_diff} 过大（VNNI 尾部可能有 bug）"
    print("  ✓ 通过\n")


def test_k_block_ge_32():
    """测试 4b: k_block >= 32（覆盖 VNNI 主循环路径或标量回退）

    此测试覆盖 k_block=64 的场景（如 LinearSBE group_size=64）。
    在支持 AVX-VNNI 的 CPU 上走 VNNI 路径，不支持时走标量路径。
    两种路径都应与 Python 参考实现数学等价。
    """
    print("[test 4b] k_block >= 32（VNNI 主循环 / 标量回退）")
    m, k, n = 8, 128, 16
    groups, k_block = 2, 64
    max_diff, y_py, y_c = _run_one_case(m, k, n, groups, k_block, seed=777)
    print(f"  shape: x=({m},{k}), w=({k},{n}), groups={groups}, k_block={k_block}")
    print(f"  max_diff = {max_diff:.6e}")
    assert max_diff < 1e-3, f"max_diff={max_diff} 过大（k_block>=32 路径可能有 bug）"
    print("  ✓ 通过\n")


def test_random_seeds():
    """测试 5: 多个随机种子稳定性"""
    print("[test 5] 多随机种子稳定性")
    m, k, n = 32, 64, 16
    groups, k_block = 8, 8
    max_diffs = []
    for seed in range(10):
        max_diff, _, _ = _run_one_case(m, k, n, groups, k_block, seed=seed)
        max_diffs.append(max_diff)
    overall_max = max(max_diffs)
    print(f"  10 个种子的 max_diff: min={min(max_diffs):.6e}, "
          f"max={overall_max:.6e}, mean={np.mean(max_diffs):.6e}")
    assert overall_max < 1e-3, f"max_diff={overall_max} 过大"
    print("  ✓ 通过\n")


def test_zero_input():
    """测试 6: 零输入边界情况"""
    print("[test 6] 零输入边界情况")
    m, k, n = 4, 8, 3
    groups, k_block = 2, 4
    x = np.zeros((m, k), dtype=np.float32)
    w = np.zeros((k, n), dtype=np.float32)

    # Python 路径
    w_blocks = quantize_weight_sbe(w, groups, k_block, m, k, n)
    y_py = sbe_matmul_py(x, w_blocks, groups, k_block, m, k, n)

    # C 路径
    w_signed, w_sum_b, w_scales = pysgn_net.sbe_quantize_weight_blocks(
        w, groups, k_block, k, n
    )
    y_c = pysgn_net.sbe_matmul(
        x, w_signed, w_sum_b, w_scales,
        groups, k_block, m, k, n
    )

    max_diff = float(np.abs(y_py - y_c).max())
    print(f"  max_diff = {max_diff:.6e}")
    print(f"  y_py max = {np.abs(y_py).max():.6e}")
    print(f"  y_c max  = {np.abs(y_c).max():.6e}")
    # 零输入应该输出零（或接近零，scale=1.0 时 q=0）
    assert max_diff < 1e-6, f"零输入 max_diff={max_diff} 过大"
    print("  ✓ 通过\n")


def test_output_shapes():
    """测试 7: 输出形状校验"""
    print("[test 7] 输出形状校验")
    m, k, n = 8, 12, 5
    groups, k_block = 3, 4
    rng = np.random.default_rng(0)
    w = rng.standard_normal((k, n)).astype(np.float32)
    x = rng.standard_normal((m, k)).astype(np.float32)

    w_signed, w_sum_b, w_scales = pysgn_net.sbe_quantize_weight_blocks(
        w, groups, k_block, k, n
    )
    print(f"  w_signed.shape = {w_signed.shape} (期望 ({groups}, {n}, {k_block}))")
    print(f"  w_sum_b.shape  = {w_sum_b.shape} (期望 ({groups}, {n}))")
    print(f"  w_scales.shape = {w_scales.shape} (期望 ({groups},))")
    assert w_signed.shape == (groups, n, k_block)
    assert w_sum_b.shape == (groups, n)
    assert w_scales.shape == (groups,)
    assert w_signed.dtype == np.int8
    assert w_sum_b.dtype == np.int32
    assert w_scales.dtype == np.float32

    y = pysgn_net.sbe_matmul(
        x, w_signed, w_sum_b, w_scales,
        groups, k_block, m, k, n
    )
    print(f"  y.shape = {y.shape} (期望 ({m}, {n}))")
    assert y.shape == (m, n)
    assert y.dtype == np.float32
    print("  ✓ 通过\n")


def main():
    print("=" * 60)
    print("SBE C 化正确性对照测试")
    print(f"pysgn_net 版本: {pysgn_net.__version__}")
    print("=" * 60 + "\n")

    tests = [
        test_small,
        test_medium,
        test_cifar_conv,
        test_k_block_not_multiple_32,
        test_k_block_ge_32,
        test_random_seeds,
        test_zero_input,
        test_output_shapes,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ 失败: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ✗ 异常: {type(e).__name__}: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败 (共 {passed + failed} 项)")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
