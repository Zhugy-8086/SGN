# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""HC16MS - MSInt int16 容器多视角存储验证脚本（方案 B）

验证项：
  1. 基本导入 + AVX2 检测
  2. 多视角读取一致性（HC16/HC8/HC4 同一块内存的不同解读）
  3. HC16 量化精度（vs numpy）
  4. HC8 量化精度（vs numpy）
  5. HC16 视角 matmul 精度（vs numpy float matmul）
  6. HC8 视角 matmul 精度（vs numpy float matmul，2x k 维度）
  7. 零拷贝切换验证：同一块内存，HC16 写入 → HC8 读取 → HC4 读取
  8. 性能对比
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# 添加项目根目录到 sys.path，以便导入 engine.sgn
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import engine.sgn as _sgn
hc16ms = _sgn._native.hc16ms  # Clang 编译的 sgn 模块的 hc16ms 子模块（替代旧 MSVC pysgn_hc16ms.pyd）


def _rel_error(ref: np.ndarray, approx: np.ndarray) -> float:
    ref_norm = float(np.linalg.norm(ref))
    if ref_norm == 0:
        return 0.0
    return float(np.linalg.norm(ref - approx) / ref_norm)


def _max_error(ref: np.ndarray, approx: np.ndarray) -> float:
    return float(np.max(np.abs(ref - approx)))


def test_1_basic():
    """测试 1: 基本导入 + AVX2 检测"""
    print("\n[测试 1] 基本导入 + AVX2 检测")
    avx2 = hc16ms.detect_avx2()
    print(f"  AVX2 支持: {'是' if avx2 else '否'}")
    assert avx2 == 1, "AVX2 未检测到（HC16 路径需要 AVX2）"
    print("  ✓ 通过")
    return True


def test_2_multiview_read_consistency():
    """测试 2: 多视角读取一致性
    验证同一块 int16 内存，HC16/HC8/HC4 视角的值符合位宽重新解释规则
    """
    print("\n[测试 2] 多视角读取一致性")
    # 构造测试数据：int16 值 0x1234 = 4660（正数，在 int16 范围内）
    # 二进制: 0001 0010 0011 0100
    # 小端序内存: 字节0=0x34, 字节1=0x12
    # HC16 视角: 0x1234 = 4660
    # HC8 视角: low=0x34=52, high=0x12=18
    # HC4 视角: h0=1, h1=2, h2=3, h3=4
    test_int16 = np.array([0x1234], dtype=np.int16)

    views = hc16ms.inspect_views(test_int16)
    hc16_val = int(views["hc16"][0])
    hc8_low = int(views["hc8"][0])
    hc8_high = int(views["hc8"][1])
    hc4_vals = [int(views["hc4"][i]) for i in range(4)]

    print(f"  原始 int16: 0x{int(test_int16[0]) & 0xFFFF:04x} = {int(test_int16[0])}")
    print(f"  HC16 视角: {hc16_val}")
    print(f"  HC8 视角: low={hc8_low}, high={hc8_high}")
    print(f"  HC4 视角: {hc4_vals}")

    assert hc16_val == 0x1234, f"HC16 视角错误: {hc16_val} != {0x1234}"
    assert hc8_low == 0x34, f"HC8 low 错误: {hc8_low} != {0x34}"
    assert hc8_high == 0x12, f"HC8 high 错误: {hc8_high} != {0x12}"
    assert hc4_vals == [1, 2, 3, 4], f"HC4 错误: {hc4_vals} != [1,2,3,4]"

    # 验证负数（int16 有符号，-1 的二进制是 0xFFFF）
    # 0xFFFF: low=0xFF=-1(int8), high=0xFF=-1(int8)
    # HC4: h0=15, h1=15, h2=15, h3=15
    test_neg = np.array([-1], dtype=np.int16)
    views_neg = hc16ms.inspect_views(test_neg)
    neg_hc16 = int(views_neg["hc16"][0])
    neg_hc8_low = int(views_neg["hc8"][0])
    neg_hc8_high = int(views_neg["hc8"][1])
    print(f"\n  负数测试: int16 = -1 (0xFFFF)")
    print(f"  HC16 视角: {neg_hc16}")
    print(f"  HC8 视角: low={neg_hc8_low}, high={neg_hc8_high}")
    assert neg_hc16 == -1, f"HC16 负数错误: {neg_hc16} != -1"
    assert neg_hc8_low == -1, f"HC8 low 负数错误: {neg_hc8_low} != -1"
    assert neg_hc8_high == -1, f"HC8 high 负数错误: {neg_hc8_high} != -1"

    print("  ✓ 通过")
    return True


def test_3_hc16_quantize_precision():
    """测试 3: HC16 量化精度（vs numpy）"""
    print("\n[测试 3] HC16 量化精度")
    np.random.seed(42)
    x = np.random.randn(1000).astype(np.float32)

    # C 扩展量化
    x_q, x_scale = hc16ms.quantize_hc16(x)
    x_deq = hc16ms.dequantize_hc16(x_q, x_scale)

    # numpy 参考量化
    x_max = float(np.abs(x).max())
    ref_scale = x_max / 32767.0
    ref_q = np.round(x / ref_scale)
    ref_q = np.clip(ref_q, -32767, 32767)
    ref_deq = (ref_q * ref_scale).astype(np.float32)

    max_err = _max_error(ref_deq, x_deq)
    rel_err = _rel_error(ref_deq, x_deq)
    print(f"  scale: C={x_scale:.6e}, numpy={ref_scale:.6e}, diff={abs(x_scale-ref_scale):.2e}")
    print(f"  max_error: {max_err:.2e}")
    print(f"  rel_error: {rel_err:.2e}")

    assert abs(x_scale - ref_scale) < 1e-6, f"scale 差异过大: {abs(x_scale - ref_scale)}"
    assert max_err < 1e-3, f"max_error 过大: {max_err}"
    print("  ✓ 通过")
    return True


def test_4_hc8_quantize_precision():
    """测试 4: HC8 量化精度（vs numpy）"""
    print("\n[测试 4] HC8 量化精度（2 int8/hc16ms_t）")
    np.random.seed(42)
    # HC8 视角: 输入长度必须为偶数（2 float/hc16ms_t）
    x = np.random.randn(2000).astype(np.float32)

    # C 扩展量化
    x_q, x_scale = hc16ms.quantize_hc8(x)
    x_deq = hc16ms.dequantize_hc8(x_q, x_scale)

    # numpy 参考量化（HC8 标准）
    x_max = float(np.abs(x).max())
    ref_scale = x_max / 127.0
    ref_q = np.round(x / ref_scale)
    ref_q = np.clip(ref_q, -127, 127)
    ref_deq = (ref_q * ref_scale).astype(np.float32)

    max_err = _max_error(ref_deq, x_deq)
    rel_err = _rel_error(ref_deq, x_deq)
    print(f"  scale: C={x_scale:.6e}, numpy={ref_scale:.6e}")
    print(f"  deq shape: C={x_deq.shape}, numpy={ref_deq.shape}")
    print(f"  max_error: {max_err:.2e}")
    print(f"  rel_error: {rel_err:.2e}")

    assert abs(x_scale - ref_scale) < 1e-6, f"scale 差异过大"
    assert max_err < 1e-3, f"max_error 过大: {max_err}"
    print("  ✓ 通过")
    return True


def test_5_hc16_matmul_precision():
    """测试 5: HC16 视角 matmul 精度（vs numpy float matmul）"""
    print("\n[测试 5] HC16 视角 matmul 精度")
    np.random.seed(42)
    m, k, n = 64, 128, 32
    x = np.random.randn(m, k).astype(np.float32) * 0.1
    w = np.random.randn(k, n).astype(np.float32) * 0.1

    # C 扩展 HC16 matmul
    y_c = hc16ms.quantized_matmul_hc16(x.flatten(), w.flatten(), m, k, n)

    # numpy 参考
    y_ref = (x @ w).astype(np.float32)

    max_err = _max_error(y_ref, y_c)
    rel_err = _rel_error(y_ref, y_c)
    print(f"  矩阵: {m}×{k} @ {k}×{n}")
    print(f"  max_error: {max_err:.2e}")
    print(f"  rel_error: {rel_err:.2e}")

    assert rel_err < 1e-3, f"rel_error 过大: {rel_err}"
    print("  ✓ 通过")
    return True


def test_6_hc8_matmul_precision():
    """测试 6: HC8 视角 matmul 精度
    HC8 视角下每个 hc16ms_t 存 2 个 int8，矩阵 A (m×k) → (m×2k), B (k×n) → (2k×n)

    布局说明：
      - A (m×2k) 行优先：a_flat[i*2k + 2j] 和 a_flat[i*2k + 2j+1] 相邻
        → quantize_hc8 1D 打包直接正确：a_q[i][j] = pack(a[i][2j+1], a[i][2j])
      - B (2k×n) 行优先：b[2j][l] 和 b[2j+1][l] 在内存中不相邻（间隔 n）
        → 需要重排为 B_packed[2*(j*n+l)] = b[2j][l], B_packed[2*(j*n+l)+1] = b[2j+1][l]
    """
    print("\n[测试 6] HC8 视角 matmul 精度（2x k 维度）")
    np.random.seed(42)
    m, k, n = 64, 128, 32  # hc16ms_t 维度

    # 生成参考矩阵
    x_ref = np.random.randn(m, 2 * k).astype(np.float32) * 0.1   # (m, 2k)
    w_ref = np.random.randn(2 * k, n).astype(np.float32) * 0.1   # (2k, n)

    # A 不需要重排（行优先 m×2k 的 1D 序列已正确配对）
    x_flat = x_ref.flatten()

    # B 需要重排：W_packed[2*(j*n+l)] = W[2j][l], W_packed[2*(j*n+l)+1] = W[2j+1][l]
    # W_ref (2k, n) → reshape (k, 2, n) → transpose (k, n, 2) → flatten (2*k*n,)
    w_packed = w_ref.reshape(k, 2, n).transpose(0, 2, 1).reshape(-1).astype(np.float32)

    # C 扩展 HC8 matmul
    y_c = hc16ms.quantized_matmul_hc8(x_flat, w_packed, m, k, n)

    # numpy 参考: 标准 matmul
    y_ref = (x_ref @ w_ref).astype(np.float32)

    max_err = _max_error(y_ref, y_c)
    rel_err = _rel_error(y_ref, y_c)
    print(f"  矩阵: {m}×{2*k} @ {2*k}×{n} (HC8 视角, hc16ms_t k={k})")
    print(f"  max_error: {max_err:.2e}")
    print(f"  rel_error: {rel_err:.2e}")

    assert rel_err < 2e-2, f"rel_error 过大: {rel_err}"  # HC8 int8 量化精度限制（254 级）
    print("  ✓ 通过")
    return True


def test_7_zero_copy_switch():
    """测试 7: 零拷贝切换验证
    同一块内存：HC16 写入 → HC8 读取 → HC4 读取
    验证三种视角对同一内存的不同解读

    注意：本测试直接操作 int16 原始值，不经过量化（quantize 会推导 scale 改变值）
    """
    print("\n[测试 7] 零拷贝切换验证")

    # 直接构造 int16 值 0x0123 = 291
    # HC8 视角: low=0x23=35, high=0x01=1
    # HC4 视角: h0=0, h1=1, h2=2, h3=3
    test_val = np.array([0x0123], dtype=np.int16)

    # 通过 inspect_views 一次读取所有视角（零拷贝，同一内存）
    views = hc16ms.inspect_views(test_val)

    hc16_v = int(views["hc16"][0])
    hc8_low = int(views["hc8"][0])
    hc8_high = int(views["hc8"][1])
    hc4 = [int(views["hc4"][i]) for i in range(4)]

    print(f"  同一块 2 字节内存的多视角解读:")
    print(f"    HC16 视角 (1 值): {hc16_v}")
    print(f"    HC8  视角 (2 值): low={hc8_low}, high={hc8_high}")
    print(f"    HC4  视角 (4 值): {hc4}")

    # 验证位宽重新解释的正确性
    assert hc16_v == 0x0123, f"HC16 错误: {hc16_v} != {0x0123}"
    assert hc8_low == 0x23 and hc8_high == 0x01, f"HC8 错误"
    assert hc4 == [0, 1, 2, 3], f"HC4 错误: {hc4}"

    # 验证反向构造：从 HC4 值重建 int16
    # h0=0, h1=1, h2=2, h3=3 → 高字节=0x01, 低字节=0x23 → int16=0x0123
    reconstructed = (hc4[0] << 12) | (hc4[1] << 8) | (hc4[2] << 4) | hc4[3]
    print(f"\n  从 HC4 重建 int16: {reconstructed} (期望 {0x0123})")
    assert reconstructed == 0x0123, f"HC4 重建失败: {reconstructed} != {0x0123}"

    # 验证多元素场景：构造 4 个 int16 值（均在 int16 范围内），验证批量多视角读取
    # 0x0123=291, 0x4567=17767, 0x7ABC=31420, -1=0xFFFF
    batch = np.array([0x0123, 0x4567, 0x7ABC, -1], dtype=np.int16)
    batch_views = hc16ms.inspect_views(batch)
    print(f"\n  批量多视角读取 (4 元素):")
    print(f"    HC16: {list(batch_views['hc16'])}")
    print(f"    HC8 low:  {list(batch_views['hc8'][::2])}")
    print(f"    HC8 high: {list(batch_views['hc8'][1::2])}")

    # 验证第一个元素的 HC8 视角
    assert int(batch_views["hc8"][0]) == 0x23  # 第一个 low
    assert int(batch_views["hc8"][1]) == 0x01  # 第一个 high
    # 验证第四个元素（-1 = 0xFFFF）：low=-1, high=-1
    assert int(batch_views["hc8"][6]) == -1  # 第四个 low
    assert int(batch_views["hc8"][7]) == -1  # 第四个 high

    print("  ✓ 通过 - 同一块内存成功以 3 种位宽读取，零拷贝切换验证")
    return True


def test_8_performance():
    """测试 8: 性能对比"""
    print("\n[测试 8] 性能对比")
    np.random.seed(42)

    # HC16 matmul 性能
    m, k, n = 256, 512, 256
    x = np.random.randn(m, k).astype(np.float32) * 0.1
    w = np.random.randn(k, n).astype(np.float32) * 0.1

    # 预热
    _ = hc16ms.quantized_matmul_hc16(x.flatten(), w.flatten(), m, k, n)

    t0 = time.time()
    for _ in range(5):
        y_c = hc16ms.quantized_matmul_hc16(x.flatten(), w.flatten(), m, k, n)
    t_hc16 = (time.time() - t0) / 5

    # numpy float32 BLAS 参考
    t0 = time.time()
    for _ in range(5):
        y_ref = (x @ w).astype(np.float32)
    t_blas = (time.time() - t0) / 5

    rel_err = _rel_error(y_ref, y_c)
    print(f"  HC16 matmul ({m}×{k} @ {k}×{n}):")
    print(f"    HC16 (AVX2): {t_hc16*1000:.2f} ms")
    print(f"    numpy BLAS:  {t_blas*1000:.2f} ms")
    print(f"    加速比: {t_blas/t_hc16:.2f}x")
    print(f"    rel_error: {rel_err:.2e}")

    # HC8 matmul 性能（W 需要重排布局）
    m2, k2, n2 = 128, 256, 128
    x2 = np.random.randn(m2, 2 * k2).astype(np.float32) * 0.1
    w2 = np.random.randn(2 * k2, n2).astype(np.float32) * 0.1
    # W 重排为 packed 布局（与 test_6 一致）
    w2_packed = w2.reshape(k2, 2, n2).transpose(0, 2, 1).reshape(-1).astype(np.float32)

    t0 = time.time()
    for _ in range(5):
        y_c2 = hc16ms.quantized_matmul_hc8(x2.flatten(), w2_packed, m2, k2, n2)
    t_hc8 = (time.time() - t0) / 5

    t0 = time.time()
    for _ in range(5):
        y_ref2 = (x2 @ w2).astype(np.float32)
    t_blas2 = (time.time() - t0) / 5

    rel_err2 = _rel_error(y_ref2, y_c2)
    print(f"\n  HC8 matmul ({m2}×{2*k2} @ {2*k2}×{n2}):")
    print(f"    HC8 (标量):  {t_hc8*1000:.2f} ms")
    print(f"    numpy BLAS:  {t_blas2*1000:.2f} ms")
    print(f"    rel_error: {rel_err2:.2e}")

    print("  ✓ 通过")
    return True


def main():
    print("=" * 70)
    print("HC16MS - MSInt int16 容器多视角存储验证（方案 B）")
    print("=" * 70)

    tests = [
        test_1_basic,
        test_2_multiview_read_consistency,
        test_3_hc16_quantize_precision,
        test_4_hc8_quantize_precision,
        test_5_hc16_matmul_precision,
        test_6_hc8_matmul_precision,
        test_7_zero_copy_switch,
        test_8_performance,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"验证结果: {passed} 通过, {failed} 失败")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
