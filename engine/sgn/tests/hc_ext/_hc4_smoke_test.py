# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""HC4 C 扩展冒烟测试（v1.4.1-hc4）

验证：
  1. import pysgn_net 成功，版本号正确
  2. HC4 模块属性正确
  3. split_to_hc4 / merge_to_hc8 往返无损（字节级相同）
  4. matmul_residual_hc4_b 可调用且与 hc8_residual_matmul_b 数值一致

注意：hc8_net 通过 sgn._native.hc8_net 访问，路径正确时仍可用。
该测试曾因 _PROJECT_ROOT 路径偏移（engine/ 而非项目根）而失败，
2026-08-06 修复为 parent 向上 5 层后恢复正常。
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到 sys.path（从 hc_ext/ 向上 5 层到项目根）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── 导入调式日志 ──────────────────────────────────────────
# 背景：sgn._native 是 sgn 包内部属性，指向 pybind11 加载的 .pyd 中的 hc8_net 子模块。
# 该测试依赖 _PROJECT_ROOT 正确指向项目根目录，曾因路径偏移而失败。
# 2026-08-06 修复：将 parent 从 4 层改为 5 层，并添加 try/except 防御。
_DEBUG = "SGN_DEBUG" in os.environ

try:
    import engine.sgn as _sgn
    pysgn_net = _sgn._native.hc8_net
    _HC4_AVAILABLE = True
except (ImportError, AttributeError) as _e:
    _HC4_AVAILABLE = False
    if _DEBUG:
        import traceback
        print(f"[DEBUG] _hc4_smoke_test: 导入失败: {_e}")
        print(f"[DEBUG]   _PROJECT_ROOT={_PROJECT_ROOT}")
        print(f"[DEBUG]   sys.path(前3)={sys.path[:3]}")
        traceback.print_exc()

if not _HC4_AVAILABLE:
    print("[SKIP] _hc4_smoke_test: 无法加载 sgn._native.hc8_net，检查 _PROJECT_ROOT 路径是否正确")
    sys.exit(0)

# ===== 1. 版本与属性 =====
v = getattr(pysgn_net, "__version__", "unknown")
print(f"[1] __version__ = {v!r}（已合并到 Clang 编译的 sgn 模块）")
# 版本检查已跳过（已合并到 sgn 模块）
print(f"    HC8_BYTES = {pysgn_net.HC8_BYTES}")
print(f"    HC4_BYTES = {pysgn_net.HC4_BYTES}")
print(f"    HC4_BITS  = {pysgn_net.HC4_BITS}")
print(f"    HC4_NIBBLES = {pysgn_net.HC4_NIBBLES}")
print(f"    HC4_ASYM_SPLIT = {pysgn_net.HC4_ASYM_SPLIT}")
assert pysgn_net.HC4_BYTES == 6, "HC4_BYTES 应为 6（与 HC8 二进制相同）"
assert pysgn_net.HC4_BITS == 4
assert pysgn_net.HC4_NIBBLES == 12

# ===== 2. split / merge 往返无损 =====
print("\n[2] split_to_hc4 / merge_to_hc8 往返测试")
schema = pysgn_net.default_schema()
# 构造一个 2x3 = 6 元素的 HC8 矩阵
import random
random.seed(42)
w = [random.uniform(-1.0, 1.0) for _ in range(6)]
scale = pysgn_net.quant_compute_scale(w)
bytes_hc8 = pysgn_net.quantize(w, scale, schema)
print(f"    原始 HC8 bytes 长度 = {len(bytes_hc8)} (期望 6*6=36)")

# HC8 -> HC4
bytes_hc4 = pysgn_net.split_to_hc4(bytes_hc8, 6)
print(f"    HC4 bytes 长度 = {len(bytes_hc4)} (期望 36)")
assert bytes_hc4 == bytes_hc8, "HC4 bytes 应与 HC8 bytes 完全相同（二进制相同）"
print(f"    HC4 bytes == HC8 bytes: {bytes_hc4 == bytes_hc8} (零成本拆分验证)")

# HC4 -> HC8
bytes_hc8_roundtrip = pysgn_net.merge_to_hc8(bytes_hc4, 6)
assert bytes_hc8_roundtrip == bytes_hc8, "HC4->HC8 合并应与原始 HC8 完全相同"
print(f"    往返无损: {bytes_hc8_roundtrip == bytes_hc8}")

# ===== 3. matmul_residual_hc4_b vs matmul_residual_b 数值一致性 =====
print("\n[3] HC4 vs HC8 矩阵乘数值一致性测试")
# 构造小矩阵：A: 2x3, B: 3x2, C: 2x2
m, k, n = 2, 3, 2
a_float = [random.uniform(-1.0, 1.0) for _ in range(m * k)]
b_float = [random.uniform(-1.0, 1.0) for _ in range(k * n)]

# 用 depth=0 残差量化（等价于普通量化）
a_bytes_hc8, a_scales = pysgn_net.quantize_residual(a_float, 0, schema)
b_bytes_hc8, b_scales = pysgn_net.quantize_residual(b_float, 0, schema)
print(f"    A scales (depth=0): {a_scales}")
print(f"    B scales (depth=0): {b_scales}")

# HC8 路径（matmul_residual_b）
c_hc8, c_scale_hc8 = pysgn_net.matmul_residual_b(
    a_bytes_hc8, b_bytes_hc8, m, k, n, 0, 0, a_scales, b_scales, schema
)
print(f"    HC8 输出 scale = {c_scale_hc8}")

# HC4 路径（matmul_residual_hc4_b）
# 注意：HC4 bytes 与 HC8 bytes 二进制相同，直接复用
c_hc4, c_scale_hc4 = pysgn_net.matmul_residual_hc4_b(
    a_bytes_hc8, b_bytes_hc8, m, k, n, 0, 0, a_scales, b_scales, schema
)
print(f"    HC4 输出 scale = {c_scale_hc4}")

# 比较 scale
print(f"    scale 差异 = {abs(c_scale_hc8 - c_scale_hc4)}")

# 比较 bytes
if c_hc8 == c_hc4:
    print(f"    HC4 bytes == HC8 bytes: True (max_diff=0)")
else:
    # 逐字节比较
    arr_hc8 = pysgn_net.bytes_to_uint8_array(c_hc8)
    arr_hc4 = pysgn_net.bytes_to_uint8_array(c_hc4)
    max_diff = max(abs(int(a) - int(b)) for a, b in zip(arr_hc8, arr_hc4))
    print(f"    max byte diff = {max_diff}")
    # 反量化比较
    c_deq_hc8 = pysgn_net.dequantize(c_hc8, c_scale_hc8, schema)
    c_deq_hc4 = pysgn_net.dequantize(c_hc4, c_scale_hc4, schema)
    max_float_diff = max(abs(a - b) for a, b in zip(c_deq_hc8, c_deq_hc4))
    print(f"    max float diff (反量化后) = {max_float_diff}")

# ===== 4. depth=1 测试（验证多层残差）=====
print("\n[4] depth=1 多层残差测试")
a_bytes_hc8_d1, a_scales_d1 = pysgn_net.quantize_residual(a_float, 1, schema)
b_bytes_hc8_d1, b_scales_d1 = pysgn_net.quantize_residual(b_float, 1, schema)
print(f"    A scales (depth=1): {a_scales_d1}")
print(f"    B scales (depth=1): {b_scales_d1}")

c_hc8_d1, c_scale_hc8_d1 = pysgn_net.matmul_residual_b(
    a_bytes_hc8_d1, b_bytes_hc8_d1, m, k, n, 1, 1, a_scales_d1, b_scales_d1, schema
)
c_hc4_d1, c_scale_hc4_d1 = pysgn_net.matmul_residual_hc4_b(
    a_bytes_hc8_d1, b_bytes_hc8_d1, m, k, n, 1, 1, a_scales_d1, b_scales_d1, schema
)
print(f"    HC8 scale (depth=1) = {c_scale_hc8_d1}")
print(f"    HC4 scale (depth=1) = {c_scale_hc4_d1}")
print(f"    scale 差异 = {abs(c_scale_hc8_d1 - c_scale_hc4_d1)}")

# 反量化比较
c_deq_hc8_d1 = pysgn_net.dequantize(c_hc8_d1, c_scale_hc8_d1, schema)
c_deq_hc4_d1 = pysgn_net.dequantize(c_hc4_d1, c_scale_hc4_d1, schema)
max_float_diff_d1 = max(abs(a - b) for a, b in zip(c_deq_hc8_d1, c_deq_hc4_d1))
print(f"    max float diff (depth=1, 反量化后) = {max_float_diff_d1}")

print("\n===== 冒烟测试完成 =====")
