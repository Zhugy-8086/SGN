# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""
pysgn_net 自检测试

验证 HC8 神经网络运算扩展的正确性。
对应纯 Python 版本：legacy/traditional/stage_1_3_int_path/hc_matmul.py 的 _self_check

运行：
    python test_pysgn_net.py

测试内容：
    1. 量化 / 反量化往返
    2. 矩阵乘正确性（小矩阵）
    3. ReLU 正确性
    4. 与纯 Python 实现的等价性（如果可用）
"""
import sys
import os
import math
from pathlib import Path

# ── OMP 冲突调式日志 ──────────────────────────────────────────
# 安全审计 2026-08-16 A2-6：原注释描述 MSVC/Clang 双 .pyd 共存场景——
# 2026-08-06 起全部 .pyd 已合并为 Clang 编译的单一 sgn 模块，该场景不复
# 存在；KMP_DUPLICATE_LIB_OK 保留为防御性设置（进程内如加载其它 OpenMP
# 运行时的第三方扩展时仍可避免 "Error #15" 崩溃）。
_DEBUG = "SGN_DEBUG" in os.environ
if _DEBUG:
    _omp_before = os.environ.get('KMP_DUPLICATE_LIB_OK', '(未设置)')
    print(f"[DEBUG] test_pysgn_net: KMP_DUPLICATE_LIB_OK 设置前={_omp_before}")
    print(f"[DEBUG] test_pysgn_net: sys.path={sys.path}")
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# 添加项目根目录到 sys.path，以便导入 engine.sgn
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── 路径解析调式日志 ──────────────────────────────────────────
# 背景：import engine.sgn 依赖 _PROJECT_ROOT 正确指向项目根目录。
# 如果 _PROJECT_ROOT 偏移（如 engine/ 而非 SGN/），则 import engine.sgn
# 可能解析为 engine/sgn/__init__.py（包）而非 sgn.cp*.pyd（扩展），
# 导致递归加载失败或模块属性缺失。
if _DEBUG:
    print(f"[DEBUG] test_pysgn_net: 路径解析")
    print(f"[DEBUG]   _PROJECT_ROOT={_PROJECT_ROOT}")
    print(f"[DEBUG]   sys.path(前3)={sys.path[:3]}")

try:
    import numpy as np
except ImportError:
    # 安全审计 2026-08-16 A2-2：模块级 sys.exit(1) 杀死 pytest 收集进程
    try:
        import pytest
        pytest.skip("numpy 不可用", allow_module_level=True)
    except ImportError:
        print("ERROR: 需要 numpy 才能运行此测试")
        sys.exit(1)

try:
    import engine.sgn as _sgn
    pysgn_net = _sgn._native.hc8_net
except (ImportError, AttributeError) as e:
    # 安全审计 2026-08-16 A2-2：同上，pytest 下 module-level skip。
    # A2-4：错误指引更新为 cmake 构建（旧 setup_net.py 已删除）
    if _DEBUG:
        import traceback
        print(f"[DEBUG] test_pysgn_net: 导入异常详情")
        print(f"[DEBUG]   _PROJECT_ROOT={_PROJECT_ROOT}")
        print(f"[DEBUG]   sys.path(前3)={sys.path[:3]}")
        traceback.print_exc()
    try:
        import pytest
        pytest.skip(f"engine.sgn.hc8_net 不可用: {e}", allow_module_level=True)
    except ImportError:
        print(f"ERROR: 无法导入 pysgn_net，请先编译：{e}")
        print("  cd engine/sgn")
        print("  cmake -B build -S . && cmake --build build")
        sys.exit(1)


def test_quantize_roundtrip():
    """测试 1: 量化 / 反量化往返"""
    print("[test 1] 量化 / 反量化往返")
    schema = pysgn_net.default_schema()
    values = [0.5, -0.3, 1.2, -2.7, 0.0, 0.01]
    scale = pysgn_net.quant_compute_scale(values)
    print(f"  scale = {scale:.6e}")

    bytes_h = pysgn_net.quantize(values, scale, schema)
    print(f"  bytes 长度 = {len(bytes_h)} (期望 {len(values) * 6})")
    assert len(bytes_h) == len(values) * 6

    back = pysgn_net.dequantize(bytes_h, scale, schema)
    print(f"  原始: {[f'{v:.4f}' for v in values]}")
    print(f"  反量化: {[f'{v:.4f}' for v in back]}")

    # 8-bit 量化误差应 < scale
    for v_orig, v_back in zip(values, back):
        err = abs(v_orig - v_back)
        assert err < scale * 1.5, f"量化误差过大：{v_orig} → {v_back}, err={err}, scale={scale}"
    print("  ✓ 量化误差在可接受范围\n")


def test_matmul():
    """测试 2: 矩阵乘正确性"""
    print("[test 2] 矩阵乘正确性")
    schema = pysgn_net.default_schema()
    # A = [[1.0, 2.0], [3.0, 4.0]]
    # B = [[5.0, 6.0], [7.0, 8.0]]
    # C = A @ B = [[19, 22], [43, 50]]
    a = [1.0, 2.0, 3.0, 4.0]
    b = [5.0, 6.0, 7.0, 8.0]
    m, k, n = 2, 2, 2

    a_scale = pysgn_net.quant_compute_scale(a)
    b_scale = pysgn_net.quant_compute_scale(b)
    bytes_a = pysgn_net.quantize(a, a_scale, schema)
    bytes_b = pysgn_net.quantize(b, b_scale, schema)

    bytes_c, c_scale = pysgn_net.matmul(
        bytes_a, bytes_b, m, k, n, a_scale, b_scale, schema
    )
    c_back = pysgn_net.dequantize(bytes_c, c_scale, schema)
    c_matrix = [c_back[i*n:(i+1)*n] for i in range(m)]
    print(f"  A = {a[:2]} / {a[2:]}")
    print(f"  B = {b[:2]} / {b[2:]}")
    print(f"  C (期望 [[19, 22], [43, 50]]) = {c_matrix}")
    print(f"  c_scale = {c_scale:.6e}")

    expected = [[19.0, 22.0], [43.0, 50.0]]
    mse = sum((c_matrix[i][j] - expected[i][j])**2
              for i in range(m) for j in range(n)) / (m * n)
    print(f"  MSE = {mse:.6e}")
    assert mse < 5.0, f"矩阵乘 MSE 过大：{mse}"
    print("  ✓ 矩阵乘正确性在可接受范围\n")


def test_relu():
    """测试 3: ReLU 正确性"""
    print("[test 3] ReLU 正确性")
    schema = pysgn_net.default_schema()
    # [[1.0, -2.0], [-3.0, 4.0]] → [[1.0, 0.0], [0.0, 4.0]]
    x = [1.0, -2.0, -3.0, 4.0]
    m, n = 2, 2

    x_scale = pysgn_net.quant_compute_scale(x)
    bytes_x = pysgn_net.quantize(x, x_scale, schema)
    bytes_out = pysgn_net.relu(bytes_x, m, n, schema)
    out = pysgn_net.dequantize(bytes_out, x_scale, schema)
    out_matrix = [out[i*n:(i+1)*n] for i in range(m)]
    print(f"  X = {x[:2]} / {x[2:]}")
    print(f"  ReLU(X) (期望 [[>0, 0], [0, >0]]) = {out_matrix}")

    assert out_matrix[0][0] > 0.5, f"ReLU(1.0) 应为正，得到 {out_matrix[0][0]}"
    assert abs(out_matrix[0][1]) < 0.5, f"ReLU(-2.0) 应为 0，得到 {out_matrix[0][1]}"
    assert abs(out_matrix[1][0]) < 0.5, f"ReLU(-3.0) 应为 0，得到 {out_matrix[1][0]}"
    assert out_matrix[1][1] > 3.5, f"ReLU(4.0) 应为正，得到 {out_matrix[1][1]}"
    print("  ✓ ReLU 正确\n")


def test_equiv_python_impl():
    """测试 4: 与纯 Python 实现的等价性（如果可用）"""
    print("[test 4] 与纯 Python 实现的等价性")
    try:
        # 2026-08-16 legacy 独立：stage_1_3 参考实现迁至 tests/refs/stage_1_3/
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "refs", "stage_1_3")))
        from hc_adapter import HC8, HC8WeightSchema
        from hc_matmul import tensor_to_hc8_matrix, hc8_matrix_to_tensor, hc_matmul as py_matmul
        import torch
    except ImportError as e:
        print(f"  跳过：纯 Python 实现不可用（{e}）\n")
        return

    schema_c = pysgn_net.default_schema()
    schema_py = HC8WeightSchema()

    # 用同一个 float 矩阵测试
    torch.manual_seed(42)
    a_tensor = torch.randn(3, 4)
    b_tensor = torch.randn(4, 5)

    # C 实现路径
    a_list = a_tensor.flatten().tolist()
    b_list = b_tensor.flatten().tolist()
    a_scale_c = pysgn_net.quant_compute_scale(a_list)
    b_scale_c = pysgn_net.quant_compute_scale(b_list)
    bytes_a = pysgn_net.quantize(a_list, a_scale_c, schema_c)
    bytes_b = pysgn_net.quantize(b_list, b_scale_c, schema_c)
    bytes_c, c_scale_c = pysgn_net.matmul(
        bytes_a, bytes_b, 3, 4, 5, a_scale_c, b_scale_c, schema_c
    )
    c_back_c = pysgn_net.dequantize(bytes_c, c_scale_c, schema_c)
    c_matrix_c = torch.tensor([c_back_c[i*5:(i+1)*5] for i in range(3)])

    # 纯 Python 实现路径
    a_hc = tensor_to_hc8_matrix(a_tensor, schema_py)
    b_hc = tensor_to_hc8_matrix(b_tensor, schema_py)
    c_hc = py_matmul(a_hc, b_hc)
    c_matrix_py = hc8_matrix_to_tensor(c_hc, schema_py)

    # 两者应近似（量化方案相同，但可能有 round 差异）
    mse = ((c_matrix_c - c_matrix_py) ** 2).mean().item()
    print(f"  C 实现结果: {c_matrix_c.flatten().tolist()}")
    print(f"  Python 实现结果: {c_matrix_py.flatten().tolist()}")
    print(f"  两者 MSE = {mse:.6e}")
    # 允许较小误差（量化 round 边界差异）
    assert mse < 1.0, f"C 与 Python 实现差异过大：MSE={mse}"
    print("  ✓ C 实现与纯 Python 实现等价性在可接受范围\n")


def main():
    print("=" * 60)
    print("pysgn_net 自检测试")
    print("=" * 60)
    print(f"pysgn_net 版本: {pysgn_net.__version__}")
    print(f"HC8_BYTES = {pysgn_net.HC8_BYTES}")
    print(f"默认 schema: {pysgn_net.default_schema()}")
    print()

    test_quantize_roundtrip()
    test_matmul()
    test_relu()
    test_equiv_python_impl()

    print("=" * 60)
    print("所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
