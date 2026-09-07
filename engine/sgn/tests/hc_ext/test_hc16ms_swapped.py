# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""HC16MS - MSInt 读取顺序切换验证脚本

验证 inspect_views_swapped / matmul_hc16_swapped 接口的正确性：
  1. 基本导入 + AVX2 检测
  2. 切换视角读取正确性（核心验证）
  3. bswap16 可逆性
  4. 切换视角 matmul 精度
  5. 正向 vs 切换视角结果不同
  6. 现有正向接口回归（无破坏）
  7. 性能对比（可选）
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


def _bswap16_numpy(arr: np.ndarray) -> np.ndarray:
    """numpy 实现的 bswap16：int16 字节交换（0x1234 → 0x3412）"""
    u = arr.astype(np.uint16)
    return (((u >> 8) | ((u << 8) & 0xFFFF)) & 0xFFFF).astype(np.int16)


def test_1_basic():
    """测试 1: 基本导入 + AVX2 检测"""
    print("\n[测试 1] 基本导入 + AVX2 检测")
    avx2 = hc16ms.detect_avx2()
    print(f"  AVX2 支持: {'是' if avx2 else '否'}")
    assert avx2 == 1, "AVX2 未检测到（HC16 路径需要 AVX2）"

    # 验证切换视角接口存在
    assert hasattr(hc16ms, 'inspect_views_swapped'), "缺少 inspect_views_swapped 接口"
    assert hasattr(hc16ms, 'matmul_hc16_swapped'), "缺少 matmul_hc16_swapped 接口"
    print("  inspect_views_swapped: 存在")
    print("  matmul_hc16_swapped: 存在")
    print("  ✓ 通过")
    return True


def test_2_inspect_views_swapped():
    """测试 2: 切换视角读取正确性（核心验证）
    对 raw=0x1234 的 int16（小端序内存：低字节0x34, 高字节0x12）验证三种视角的切换读取结果。
      - 正向: HC16=4660, HC8=[low=52, high=18], HC4=[1,2,3,4]
      - 切换: HC16=13330 (0x3412), HC8=[low=18, high=52], HC4=[4,3,2,1]
    """
    print("\n[测试 2] 切换视角读取正确性（核心验证）")

    # 构造测试数据：int16 值 0x1234 = 4660
    # 小端序内存: 字节0=0x34 (低), 字节1=0x12 (高)
    test_int16 = np.array([0x1234], dtype=np.int16)

    # 先验证正向读取（作为对照）
    views_fwd = hc16ms.inspect_views(test_int16)
    fwd_hc16 = int(views_fwd["hc16"][0])
    fwd_hc8 = [int(views_fwd["hc8"][0]), int(views_fwd["hc8"][1])]
    fwd_hc4 = [int(views_fwd["hc4"][i]) for i in range(4)]
    print(f"  原始 int16: 0x{int(test_int16[0]) & 0xFFFF:04x} = {int(test_int16[0])}")
    print(f"  正向 HC16: {fwd_hc16} (期望 4660)")
    print(f"  正向 HC8:  [low={fwd_hc8[0]}, high={fwd_hc8[1]}] (期望 [52, 18])")
    print(f"  正向 HC4:  {fwd_hc4} (期望 [1, 2, 3, 4])")

    # 切换视角读取
    views_sw = hc16ms.inspect_views_swapped(test_int16)
    sw_hc16 = int(views_sw["hc16"][0])
    sw_hc8 = [int(views_sw["hc8"][0]), int(views_sw["hc8"][1])]
    sw_hc4 = [int(views_sw["hc4"][i]) for i in range(4)]
    print(f"\n  切换 HC16: {sw_hc16} (期望 {0x3412} = 13330)")
    print(f"  切换 HC8:  [low={sw_hc8[0]}, high={sw_hc8[1]}] (期望 [18, 52])")
    print(f"  切换 HC4:  {sw_hc4} (期望 [4, 3, 2, 1])")

    # 验证切换视角值
    assert sw_hc16 == 0x3412, f"切换 HC16 错误: {sw_hc16} != {0x3412} (13330)"
    assert sw_hc8 == [18, 52], f"切换 HC8 错误: {sw_hc8} != [18, 52]"
    assert sw_hc4 == [4, 3, 2, 1], f"切换 HC4 错误: {sw_hc4} != [4, 3, 2, 1]"

    # 验证 HC8 的 high/low 对调关系（正向 [52,18] → 切换 [18,52]）
    assert sw_hc8[0] == fwd_hc8[1], f"HC8 low 对调错误: {sw_hc8[0]} != {fwd_hc8[1]}"
    assert sw_hc8[1] == fwd_hc8[0], f"HC8 high 对调错误: {sw_hc8[1]} != {fwd_hc8[0]}"

    # 验证 HC4 的 nibble 反转关系（正向 [1,2,3,4] → 切换 [4,3,2,1]）
    assert sw_hc4 == fwd_hc4[::-1], f"HC4 反转错误: {sw_hc4} != {fwd_hc4[::-1]}"

    # 批量验证：多个元素
    batch = np.array([0x1234, 0x5678, 0x7ABC, -1], dtype=np.int16)
    batch_sw = hc16ms.inspect_views_swapped(batch)
    # 0x1234 → bswap16 = 0x3412 = 13330 (int16 正数)
    # 0x5678 → bswap16 = 0x7856 = 30806 (int16 正数)
    # 0x7ABC → bswap16 = 0xBC7A = -17286 (int16 有符号: 0xBC7A = 48250 - 65536)
    # -1 (0xFFFF) → bswap16 = 0xFFFF = -1
    expected_hc16 = [0x3412, 0x7856, -17286, -1]
    actual_hc16 = [int(x) for x in batch_sw["hc16"]]
    print(f"\n  批量切换 HC16: {actual_hc16}")
    print(f"  期望:          {expected_hc16}")
    assert actual_hc16 == expected_hc16, f"批量切换 HC16 错误"

    print("  ✓ 通过 - 切换视角读取值全部正确")
    return True


def test_3_bswap16_reversibility():
    """测试 3: bswap16 可逆性
    bswap16(bswap16(V)) = V
    实现：取原数组 v，做 swapped = inspect_views_swapped(v)["hc16"]，
         再做 double_swapped = inspect_views_swapped(swapped)["hc16"]，验证 double_swapped == v
    """
    print("\n[测试 3] bswap16 可逆性")
    np.random.seed(123)
    # 生成随机 int16 数组（覆盖正数、负数、边界值）
    v = np.random.randint(-32768, 32768, size=1000).astype(np.int16)

    # 添加边界值
    v[0] = 0        # 全零
    v[1] = -1       # 0xFFFF
    v[2] = 32767    # 0x7FFF
    v[3] = -32768   # 0x8000
    v[4] = 0x1234   # 测试用例
    v[5] = 0x3412   # bswap16(0x1234)
    v[6] = 0x00FF   # 255
    v[7] = -256     # 0xFF00 as signed int16

    # 第一次 bswap16
    swapped = hc16ms.inspect_views_swapped(v)["hc16"]
    # 验证 swapped == bswap16(v)（与 numpy 参考一致）
    ref_swapped = _bswap16_numpy(v)
    assert np.array_equal(swapped, ref_swapped), "第一次 bswap16 与 numpy 参考不一致"

    # 第二次 bswap16（应该回到原值）
    double_swapped = hc16ms.inspect_views_swapped(swapped)["hc16"]
    assert np.array_equal(double_swapped, v), "bswap16(bswap16(V)) != V"

    print(f"  数组长度: {len(v)}")
    print(f"  原始[0:8]:     {list(v[:8])}")
    print(f"  1次bswap[0:8]: {list(swapped[:8])}")
    print(f"  2次bswap[0:8]: {list(double_swapped[:8])}")
    print(f"  bswap16(bswap16(V)) == V: {'是' if np.array_equal(double_swapped, v) else '否'}")
    print("  ✓ 通过 - bswap16 完全可逆")
    return True


def test_4_matmul_swapped_precision():
    """测试 4: 切换视角 matmul 精度
    C 扩展 matmul_hc16_swapped vs numpy 参考（bswap16 + 反量化 + float matmul）
    """
    print("\n[测试 4] 切换视角 matmul 精度")
    np.random.seed(42)
    m, k, n = 32, 64, 16
    x = np.random.randn(m, k).astype(np.float32) * 0.1
    w = np.random.randn(k, n).astype(np.float32) * 0.1

    # 量化到 int16（quantize_hc16 返回 (int16数组, scale)）
    x_q, x_scale = hc16ms.quantize_hc16(x.flatten())
    w_q, w_scale = hc16ms.quantize_hc16(w.flatten())
    print(f"  矩阵: A({m}×{k}), B({k}×{n})")
    print(f"  x_q shape: {x_q.shape}, w_q shape: {w_q.shape}")
    print(f"  x_scale={x_scale:.6e}, w_scale={w_scale:.6e}")

    # C 扩展切换视角 matmul
    y_c = hc16ms.matmul_hc16_swapped(x_q, w_q, m, k, n, x_scale, w_scale)

    # numpy 参考：先对 A、B 的 int16 值做 bswap16，再反量化做 float matmul
    x_swapped = _bswap16_numpy(x_q)
    w_swapped = _bswap16_numpy(w_q)
    x_deq = x_swapped.astype(np.float32) * x_scale
    w_deq = w_swapped.astype(np.float32) * w_scale
    y_ref = (x_deq.reshape(m, k) @ w_deq.reshape(k, n)).astype(np.float32)

    max_err = _max_error(y_ref, y_c)
    rel_err = _rel_error(y_ref, y_c)
    print(f"  y_c shape: {y_c.shape}, y_ref shape: {y_ref.shape}")
    print(f"  max_error: {max_err:.2e}")
    print(f"  rel_error: {rel_err:.2e}")

    assert rel_err < 1e-3, f"rel_error 过大: {rel_err} (阈值 1e-3)"
    print("  ✓ 通过")
    return True


def test_5_forward_vs_swapped_different():
    """测试 5: 正向 vs 切换视角结果不同
    对同一块内存，正向 matmul 和切换 matmul 结果应该不同
    """
    print("\n[测试 5] 正向 vs 切换视角结果不同")
    np.random.seed(42)
    m, k, n = 32, 64, 16
    x = np.random.randn(m, k).astype(np.float32) * 0.1
    w = np.random.randn(k, n).astype(np.float32) * 0.1

    x_q, x_scale = hc16ms.quantize_hc16(x.flatten())
    w_q, w_scale = hc16ms.quantize_hc16(w.flatten())

    y_forward = hc16ms.matmul_hc16(x_q, w_q, m, k, n, x_scale, w_scale)
    y_swapped = hc16ms.matmul_hc16_swapped(x_q, w_q, m, k, n, x_scale, w_scale)

    rel_diff = _rel_error(y_forward, y_swapped)
    print(f"  正向 vs 切换 rel_diff: {rel_diff:.6f}")
    print(f"  y_forward[0,0]: {y_forward[0,0]:.6f}")
    print(f"  y_swapped[0,0]: {y_swapped[0,0]:.6f}")

    assert not np.allclose(y_forward, y_swapped), \
        "正向 matmul 和切换 matmul 结果相同（应该不同）"
    print("  ✓ 通过 - 两种视角结果确实不同")
    return True


def test_6_forward_regression():
    """测试 6: 现有正向接口回归（无破坏）
    验证 inspect_views(0x1234) 仍然返回正向值: HC16=4660, HC8=[52,18], HC4=[1,2,3,4]
    """
    print("\n[测试 6] 现有正向接口回归（无破坏）")
    test_int16 = np.array([0x1234], dtype=np.int16)
    views = hc16ms.inspect_views(test_int16)

    hc16_val = int(views["hc16"][0])
    hc8_low = int(views["hc8"][0])
    hc8_high = int(views["hc8"][1])
    hc4_vals = [int(views["hc4"][i]) for i in range(4)]

    print(f"  inspect_views(0x1234):")
    print(f"    HC16: {hc16_val} (期望 4660)")
    print(f"    HC8:  [low={hc8_low}, high={hc8_high}] (期望 [52, 18])")
    print(f"    HC4:  {hc4_vals} (期望 [1, 2, 3, 4])")

    assert hc16_val == 4660, f"HC16 回归错误: {hc16_val} != 4660"
    assert hc8_low == 52, f"HC8 low 回归错误: {hc8_low} != 52"
    assert hc8_high == 18, f"HC8 high 回归错误: {hc8_high} != 18"
    assert hc4_vals == [1, 2, 3, 4], f"HC4 回归错误: {hc4_vals} != [1, 2, 3, 4]"

    # 同时验证 quantized_matmul_hc16 正向接口仍然可用
    np.random.seed(7)
    m, k, n = 16, 32, 8
    x = np.random.randn(m, k).astype(np.float32) * 0.1
    w = np.random.randn(k, n).astype(np.float32) * 0.1
    y_c = hc16ms.quantized_matmul_hc16(x.flatten(), w.flatten(), m, k, n)
    y_ref = (x @ w).astype(np.float32)
    rel_err = _rel_error(y_ref, y_c)
    print(f"\n  正向 quantized_matmul_hc16 回归: rel_err={rel_err:.2e}")
    assert rel_err < 1e-3, f"正向 matmul 回归失败: rel_err={rel_err}"

    print("  ✓ 通过 - 正向接口无破坏")
    return True


def test_7_performance():
    """测试 7: 性能对比（可选）
    - bswap16 吞吐：inspect_views_swapped vs inspect_views
    - 切换 matmul vs 正向 matmul
    """
    print("\n[测试 7] 性能对比（可选）")
    np.random.seed(42)

    # inspect_views 吞吐对比
    n_elem = 100000
    data = np.random.randint(-32768, 32768, size=n_elem).astype(np.int16)

    # 预热
    _ = hc16ms.inspect_views(data)
    _ = hc16ms.inspect_views_swapped(data)

    n_iter = 50
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = hc16ms.inspect_views(data)
    t_fwd = (time.perf_counter() - t0) / n_iter

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = hc16ms.inspect_views_swapped(data)
    t_sw = (time.perf_counter() - t0) / n_iter

    print(f"  inspect_views ({n_elem} 元素):")
    print(f"    正向:   {t_fwd*1000:.3f} ms")
    print(f"    切换:   {t_sw*1000:.3f} ms")
    print(f"    比值:   {t_sw/t_fwd:.2f}x" if t_fwd > 0 else "    比值:   N/A")

    # matmul 性能对比（增大矩阵尺寸避免计时为零）
    m, k, n = 128, 256, 128
    x = np.random.randn(m, k).astype(np.float32) * 0.1
    w = np.random.randn(k, n).astype(np.float32) * 0.1
    x_q, x_scale = hc16ms.quantize_hc16(x.flatten())
    w_q, w_scale = hc16ms.quantize_hc16(w.flatten())

    # 预热
    _ = hc16ms.matmul_hc16(x_q, w_q, m, k, n, x_scale, w_scale)
    _ = hc16ms.matmul_hc16_swapped(x_q, w_q, m, k, n, x_scale, w_scale)

    n_iter_mm = 20
    t0 = time.perf_counter()
    for _ in range(n_iter_mm):
        _ = hc16ms.matmul_hc16(x_q, w_q, m, k, n, x_scale, w_scale)
    t_fwd_mm = (time.perf_counter() - t0) / n_iter_mm

    t0 = time.perf_counter()
    for _ in range(n_iter_mm):
        _ = hc16ms.matmul_hc16_swapped(x_q, w_q, m, k, n, x_scale, w_scale)
    t_sw_mm = (time.perf_counter() - t0) / n_iter_mm

    print(f"\n  matmul_hc16 ({m}×{k} @ {k}×{n}):")
    print(f"    正向:   {t_fwd_mm*1000:.3f} ms")
    print(f"    切换:   {t_sw_mm*1000:.3f} ms")
    print(f"    比值:   {t_sw_mm/t_fwd_mm:.2f}x" if t_fwd_mm > 0 else "    比值:   N/A")

    print("  ✓ 通过")
    return True


def test_8_nswap8_via_hc4():
    """测试 8: nswap8 间接验证 + 边界值
    通过 HC4 切换视角读取间接验证 nswap8（nibble 交换）的正确性。
    pysgn_hc16ms 没有直接暴露 nswap8 函数，但 HC4 切换视角读取内部使用 nswap8。

    验证关系：HC4 切换视角 == 正向 HC4 的 nibble 序列反转
    边界值：nswap8(0x00)=0x00, nswap8(0xFF)=0xFF, nswap8(0x0F)=0xF0, nswap8(0xF0)=0x0F
    可逆性：bswap16(bswap16(raw)) = raw（两次切换回到原值）

    注意 int16 有符号性：
      0x0000 = 0
      0xFFFF = -1（int16 有符号）
      0x0F0F = 3855
      0xF0F0 = -3856（int16 有符号：0xF0F0 = 61680 - 65536）
    """
    print("\n[测试 8] nswap8 间接验证 + 边界值（通过 HC4 切换视角）")

    # 测试用例：(int16 原始值, 描述, 正向 HC4 期望, 切换 HC4 期望)
    test_cases = [
        (0x1234, "对照（test_2 用例）",             [1, 2, 3, 4],     [4, 3, 2, 1]),
        (0x0000, "边界值: nswap8(0x00)=0x00",        [0, 0, 0, 0],     [0, 0, 0, 0]),
        (0xFFFF, "边界值: nswap8(0xFF)=0xFF (-1)",    [15, 15, 15, 15], [15, 15, 15, 15]),
        (0x0F0F, "nswap8(0x0F)=0xF0 效果 (3855)",     [0, 15, 0, 15],   [15, 0, 15, 0]),
        (0xF0F0, "nswap8(0xF0)=0x0F 效果 (-3856)",    [15, 0, 15, 0],   [0, 15, 0, 15]),
    ]

    # 收集所有 raw 值用于可逆性测试
    # 注意：0xFFFF=65535、0xF0F0=61680 超出 int16 正数范围，需先以 uint16 构造再
    # 视图转换为 int16，以保留比特模式（0xFFFF→-1, 0xF0F0→-3856）
    all_raw = np.array([tc[0] for tc in test_cases], dtype=np.uint16).view(np.int16)

    for raw_val, desc, fwd_expected, sw_expected in test_cases:
        arr = np.array([raw_val], dtype=np.uint16).view(np.int16)

        # 正向 HC4
        views_fwd = hc16ms.inspect_views(arr)
        fwd_hc4 = [int(views_fwd["hc4"][i]) for i in range(4)]

        # 切换 HC4
        views_sw = hc16ms.inspect_views_swapped(arr)
        sw_hc4 = [int(views_sw["hc4"][i]) for i in range(4)]

        print(f"  raw=0x{int(arr[0]) & 0xFFFF:04x} ({desc})")
        print(f"    正向 HC4: {fwd_hc4} (期望 {fwd_expected})")
        print(f"    切换 HC4: {sw_hc4} (期望 {sw_expected})")

        assert fwd_hc4 == fwd_expected, \
            f"正向 HC4 错误: raw=0x{int(arr[0]) & 0xFFFF:04x}, got {fwd_hc4}, 期望 {fwd_expected}"
        assert sw_hc4 == sw_expected, \
            f"切换 HC4 错误: raw=0x{int(arr[0]) & 0xFFFF:04x}, got {sw_hc4}, 期望 {sw_expected}"

        # 验证反转关系：切换 HC4 == 正向 HC4 反转（nibble 序列反转 = nswap8 效果的体现）
        assert sw_hc4 == fwd_hc4[::-1], \
            f"反转关系错误: sw={sw_hc4} != fwd[::-1]={fwd_hc4[::-1]}"

    # 可逆性验证：对切换后的 HC16 值再读一次切换视角，应回到正向值
    # bswap16(bswap16(raw)) = raw
    print("\n  可逆性验证: 对切换后的 HC16 值再读一次切换视角，应回到正向值")
    swapped_hc16 = hc16ms.inspect_views_swapped(all_raw)["hc16"]
    double_swapped_hc16 = hc16ms.inspect_views_swapped(swapped_hc16)["hc16"]

    print(f"  原始 raw:        {list(all_raw)}")
    print(f"  1 次 bswap16:    {list(swapped_hc16)}")
    print(f"  2 次 bswap16:    {list(double_swapped_hc16)}")
    assert np.array_equal(double_swapped_hc16, all_raw), \
        "HC16 两次切换未回到原值（bswap16 不可逆）"

    print("  ✓ 通过 - nswap8 间接验证 + 边界值 + 可逆性全部正确")
    return True


def test_9_matmul_swapped_k256():
    """测试 9: K=256 大维度 matmul 精度
    与 test_4 类似，但 K=256，验证大 K 维度下 int64 跨块累加无溢出，精度不退化。
    """
    print("\n[测试 9] K=256 大维度切换视角 matmul 精度")
    np.random.seed(2026)
    m, k, n = 16, 256, 32  # K=256 测试大维度累加
    x = np.random.randn(m, k).astype(np.float32) * 0.1
    w = np.random.randn(k, n).astype(np.float32) * 0.1
    x_q, x_scale = hc16ms.quantize_hc16(x.flatten())
    w_q, w_scale = hc16ms.quantize_hc16(w.flatten())
    # C 扩展切换视角 matmul
    y_c = hc16ms.matmul_hc16_swapped(x_q, w_q, m, k, n, x_scale, w_scale)
    # numpy 参考
    x_swapped = _bswap16_numpy(x_q)
    w_swapped = _bswap16_numpy(w_q)
    x_deq = x_swapped.astype(np.float32) * x_scale
    w_deq = w_swapped.astype(np.float32) * w_scale
    y_ref = (x_deq.reshape(m, k) @ w_deq.reshape(k, n)).astype(np.float32)
    rel_err = _rel_error(y_ref, y_c)

    print(f"  矩阵: A({m}×{k}), B({k}×{n}) (K=256 大维度)")
    print(f"  y_c shape: {y_c.shape}, y_ref shape: {y_ref.shape}")
    print(f"  rel_error: {rel_err:.2e}")
    assert rel_err < 1e-3, f"K=256 rel_error 过大: {rel_err}"
    print("  ✓ 通过 - 大 K 维度 int64 累加无溢出，精度不退化")
    return True


def test_10_matmul_swapped_reversibility():
    """测试 10: 切换视角 matmul 可逆性
    验证 bswap16(bswap16(A)) @ bswap16(bswap16(B)) = A @ B（bit-exact vs 正向 matmul）。

    正确的可逆性验证方式：
      先对 A 做一次 bswap 得到 A_sw，再调用 matmul_hc16_swapped(A_sw, B_sw)。
      matmul_hc16_swapped 内部会对 A_sw 再做 bswap，得到 bswap(bswap(A)) = A。
      所以 matmul_hc16_swapped(A_sw, B_sw) = A @ B = matmul_hc16(A, B)。
      两者应 bit-exact 相等。
    """
    print("\n[测试 10] 切换视角 matmul 可逆性")
    np.random.seed(999)
    m, k, n = 16, 64, 8
    x = np.random.randn(m, k).astype(np.float32) * 0.1
    w = np.random.randn(k, n).astype(np.float32) * 0.1
    x_q, x_scale = hc16ms.quantize_hc16(x.flatten())
    w_q, w_scale = hc16ms.quantize_hc16(w.flatten())

    # 正向 matmul
    y_forward = hc16ms.matmul_hc16(x_q, w_q, m, k, n, x_scale, w_scale)

    # 对 x_q, w_q 做两次 bswap16（应该回到原值）
    x_double_swapped = _bswap16_numpy(_bswap16_numpy(x_q))
    w_double_swapped = _bswap16_numpy(_bswap16_numpy(w_q))
    # 验证两次 bswap16 回到原值
    assert np.array_equal(x_double_swapped, x_q), "两次 bswap16 未回到原值"
    assert np.array_equal(w_double_swapped, w_q), "两次 bswap16 未回到原值"

    # 对 A 做一次 bswap 得到 A_sw = bswap(A)，B 同理
    x_swapped = _bswap16_numpy(x_q)  # A_sw = bswap(A)
    w_swapped = _bswap16_numpy(w_q)  # B_sw = bswap(B)

    # matmul_hc16_swapped 内部会对 x_swapped 再做 bswap，得到 bswap(bswap(A)) = A
    # 所以 y_from_swapped_input = A @ B = y_forward（应 bit-exact 相等）
    y_from_swapped_input = hc16ms.matmul_hc16_swapped(
        x_swapped, w_swapped, m, k, n, x_scale, w_scale)

    max_err = _max_error(y_forward, y_from_swapped_input)
    rel_err = _rel_error(y_forward, y_from_swapped_input)
    print(f"  矩阵: A({m}×{k}), B({k}×{n})")
    print(f"  正向 matmul vs (bswap输入的)切换 matmul: max_err={max_err:.2e}, rel_err={rel_err:.2e}")
    assert np.array_equal(y_forward, y_from_swapped_input), \
        f"切换视角 matmul 可逆性失败: max_err={max_err:.2e}"
    print("  ✓ 通过 - bswap(bswap(A)) @ bswap(bswap(B)) = A @ B（bit-exact）")
    return True


def main():
    print("=" * 70)
    print("HC16MS - MSInt 读取顺序切换验证")
    print("（inspect_views_swapped / matmul_hc16_swapped）")
    print("=" * 70)

    tests = [
        test_1_basic,
        test_2_inspect_views_swapped,
        test_3_bswap16_reversibility,
        test_4_matmul_swapped_precision,
        test_5_forward_vs_swapped_different,
        test_6_forward_regression,
        test_7_performance,
        test_8_nswap8_via_hc4,
        test_9_matmul_swapped_k256,
        test_10_matmul_swapped_reversibility,
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
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"验证结果: {passed} 通过, {failed} 失败")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
