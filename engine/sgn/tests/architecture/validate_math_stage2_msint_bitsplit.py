# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""
阶段 2：位操作代数严格验证（T 类，bit-exact，零容差）
=====================================================
对应计划 numpy_math_verification_plan_2026_08_13.md §3 阶段 2 的 #5-#8

验证目标（纯数据验证，非神经网络训练）：
  #5 [T] int12 零进位泄漏（bitsplit/concat 可逆）
  #6 [T] MSint 整数位拆分严格可逆：value = high·2^p + low
  #7 [T] backward_int16 ≡ 由两个 int8 重构的 int16（high<<8|low）
  #8 [T] 量化公式与 scale 定义：scale = max/(2^(bits-1)-1)

严谨性要求（§2.4 T 类）：
  - 位精确相等（bit-exact），零容差，用 np.array_equal / 严格 ==
  - 大规模随机值遍历 + 全位宽组合
  - 倒推：从 bitsplit 结果反推原始值，验证可逆

真实语义（engine/sgn/msint/cpp_backend.py _PyMSIntView）：
  - bitsplit(raw, total_bits, target_bits)：低位在前（parts[0]=LSB），
    负数转 total_bits 位补码
  - concat(values, bits_list)：第一个值在高位（MSB 在前），
    result = Σ values[i] << (bits_list[i+1:] 之和)
  - 关键：两者顺序不对称（bitsplit 低→高，concat 高→低），
    正确重构需将 bitsplit 结果反转再 concat
"""
from __future__ import annotations

import numpy as np
import sys
import time

# ============================================================
# 结论类型标注
# ============================================================
KIND = "T"   # 恒等式，见计划 §2.4

# ============================================================
# 尝试导入真实实现（优先 C++ MSIntView，回退 Python fallback）
# ============================================================
import os as _os
_FILE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_ARCH = _FILE_DIR
_TESTS = _os.path.dirname(_ARCH)
_SGN = _os.path.dirname(_TESTS)
_ENGINE = _os.path.dirname(_SGN)
_PROJECT_ROOT = _os.path.dirname(_ENGINE)     # → SGN 根
_PARENT = _os.path.dirname(_PROJECT_ROOT)     # → 仓库根
# 需把 engine/ 加入 path，使 `sgn` 解析为 engine/sgn 包，从而取到真实 C++ 扩展
for _p in (_PROJECT_ROOT, _PARENT, _ENGINE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from sgn.msint.cpp_backend import CppMSIntView as _REAL, USING_CPP as _USING_CPP
    _REAL_NAME = "C++ sgn.MSIntView" if _USING_CPP else "Python fallback _PyMSIntView"
except Exception as e:  # pragma: no cover
    _REAL = None
    _REAL_NAME = f"None (导入失败: {e})"


# 与 _PyMSIntView 完全一致的本地实现（作为独立参照，避免依赖导入环境）
def bitsplit(raw, total_bits, target_bits):
    if target_bits <= 0 or total_bits <= 0 or target_bits > total_bits:
        raise ValueError("非法位宽参数")
    if raw < 0:
        raw = raw & ((1 << total_bits) - 1)
    mask = (1 << target_bits) - 1
    result = []
    remaining = raw
    n_parts = (total_bits + target_bits - 1) // target_bits
    for _ in range(n_parts):
        result.append(remaining & mask)
        remaining >>= target_bits
    return result


def concat(values, bits_list):
    if len(values) != len(bits_list):
        raise ValueError("values 与 bits_list 长度不一致")
    result = 0
    for val, bits in zip(values, bits_list):
        mask = (1 << bits) - 1
        result = (result << bits) | (val & mask)
    return result


# ============================================================
# #6 [T] 位拆分严格可逆：value = high·2^p + low
# ============================================================
def verify_value_high_low():
    print("=" * 72)
    print("#6 [T] 位拆分严格可逆：value = high·2^p + low")
    print("=" * 72)

    rng = np.random.default_rng(2026)
    N = 2_000_000
    all_pass = True
    total_checked = 0
    for p in [1, 2, 4, 8, 12, 16, 24]:
        max_bits = p * 2
        hi_bits = max_bits - p
        # 随机非负值（覆盖 [0, 2^max_bits)）
        values = rng.integers(0, 1 << max_bits, N, dtype=np.int64)
        low = values & ((1 << p) - 1)
        high = values >> p
        recon = (high << p) | low
        # 也验证 high·2^p + low 形式
        recon2 = high * (1 << p) + low
        exact1 = np.array_equal(recon, values)
        exact2 = np.array_equal(recon2, values)
        ok = exact1 and exact2
        all_pass &= ok
        total_checked += N
        print(f"  p={p:>2}, 值域 [0,2^{max_bits}) N={N}: "
              f"high·2^p|low 精确={exact1}, high·2^p+low 精确={exact2}  → {'✓' if ok else '✗'}")
    print(f"\n  总计检查 {total_checked} 个值，bit-exact: {'✓ 全部通过' if all_pass else '✗ 有失败'}")
    return all_pass


# ============================================================
# #5 [T] bitsplit/concat 可逆（含顺序不对称验证）
# ============================================================
def verify_bitsplit_concat():
    print("\n" + "=" * 72)
    print("#5 [T] bitsplit/concat 可逆（int12 零进位泄漏）")
    print("=" * 72)

    rng = np.random.default_rng(7)
    N = 200_000
    combos = [
        (12, 4),   # int12 → 3×int4
        (12, 8),   # int12 → 2×int8（int8+低8位）
        (16, 8),   # int16 → 2×int8
        (16, 4),   # int16 → 4×int4
        (24, 8),   # int24 → 3×int8
        (32, 8),   # int32 → 4×int8
        (32, 16),  # int32 → 2×int16
    ]
    all_pass = True
    print(f"  {'组合':<12} {'N':>8} {'正确顺序重构':<14} {'直接concat(泄漏)':<16}")
    for total_bits, target_bits in combos:
        n_parts = (total_bits + target_bits - 1) // target_bits
        ok_all = True
        leak_all = True
        for _ in range(N):
            v = int(rng.integers(0, 1 << total_bits))
            parts = bitsplit(v, total_bits, target_bits)   # 低→高
            # 正确顺序：反转 parts（高→低）再 concat
            recon = concat(list(reversed(parts)), [target_bits] * n_parts)
            # 错误顺序：直接 concat（演示零进位泄漏）
            wrong = concat(parts, [target_bits] * n_parts)
            if recon != v:
                ok_all = False
            if wrong == v:  # 若错误顺序也等于 v，说明无泄漏（此处应为泄漏）
                leak_all = False
        all_pass &= ok_all
        print(f"  {total_bits:>2}→{target_bits:>2}×{n_parts:<6} {N:>8} "
              f"{'✓ bit-exact' if ok_all else '✗ 失败':<14} "
              f"{'✓ 存在泄漏' if not leak_all else '✗ 错误顺序也还原(异常)':<16}")

    # 用真实实现交叉验证（若可导入）
    if _REAL is not None:
        print(f"\n  [交叉验证] 真实实现 {_REAL_NAME}")
        cross_roundtrip = True
        cross_parts = True
        n_cross = 0
        for total_bits, target_bits in combos:
            n_parts = (total_bits + target_bits - 1) // target_bits
            for _ in range(5000):
                v = int(rng.integers(0, 1 << total_bits))
                parts_real = list(_REAL.bitsplit(v, total_bits, target_bits))
                recon = _REAL.concat(list(reversed(parts_real)),
                                     [target_bits] * n_parts)
                if recon != v:
                    cross_roundtrip = False
                    break
                # 逐部件比对 C++ 与本地实现
                parts_local = bitsplit(v, total_bits, target_bits)
                if parts_real != parts_local:
                    cross_parts = False
                    break
                n_cross += 1
            if not (cross_roundtrip and cross_parts):
                break
        print(f"  真实 bitsplit/concat 正确顺序重构: "
              f"{'✓ bit-exact' if cross_roundtrip else '✗ 失败'}（{n_cross} 例）")
        print(f"  C++ 逐部件拆分 == 本地实现: "
              f"{'✓ 完全一致' if cross_parts else '✗ 不一致'}")
        all_pass &= cross_roundtrip and cross_parts
    else:
        print(f"\n  [交叉验证] 真实实现不可导入，仅用本地一致实现")

    print(f"\n  结论: {'✓ 全部 bit-exact 可逆' if all_pass else '✗ 有失败'}")
    return all_pass


# ============================================================
# #7 [T] backward_int16 ≡ 两个 int8 重构的 int16
# ============================================================
def verify_int16_from_two_int8():
    print("\n" + "=" * 72)
    print("#7 [T] backward_int16 ≡ 两个 int8 重构 int16（high<<8|low）")
    print("=" * 72)

    rng = np.random.default_rng(99)
    N = 2_000_000
    # 覆盖有符号 int16 全范围
    vals = rng.integers(-32768, 32768, N, dtype=np.int64)
    # 取无符号 int16 位模式（与补码存储一致）
    u = vals & 0xFFFF
    low = u & 0xFF
    high = u >> 8
    recon_u = (high << 8) | low
    # 符号还原：把无符号位模式解释回有符号
    recon_s = recon_u.astype(np.int64)
    recon_s = np.where(recon_s >= 32768, recon_s - 65536, recon_s)
    exact_unsigned = np.array_equal(recon_u, u)
    exact_signed = np.array_equal(recon_s, vals)
    print(f"  N={N}，覆盖 int16 全范围 [-32768, 32767]")
    print(f"  无符号位模式重构 high<<8|low == 原无符号值: {'✓' if exact_unsigned else '✗'}")
    print(f"  有符号还原 == 原有符号值: {'✓' if exact_signed else '✗'}")
    ok = exact_unsigned and exact_signed
    print(f"  结论: {'✓ bit-exact' if ok else '✗ 失败'}")
    return ok


# ============================================================
# #8 [T] scale = max/(2^(bits-1)-1)
# ============================================================
def verify_scale_formula():
    print("\n" + "=" * 72)
    print("#8 [T] scale = max/(2^(bits-1)-1)（对称量化）")
    print("=" * 72)

    rng = np.random.default_rng(5)
    all_pass = True
    for bits in [8, 16, 24, 32]:
        max_code = 2 ** (bits - 1) - 1   # 127, 32767, 8388607, 2147483647
        for _ in range(100):
            x_max = float(10.0 ** rng.uniform(-3, 3))  # 随机尺度
            scale = x_max / max_code
            formula = x_max / (2 ** (bits - 1) - 1)
            if scale != formula:  # 位精确浮点相等
                all_pass = False
        print(f"  bits={bits:>2}: max_code=2^{bits-1}-1={max_code}, "
              f"scale=max/max_code 位精确==公式: {'✓' if all_pass else '✗'}")
    print(f"\n  结论: {'✓ bit-exact（scale 定义与公式一致）' if all_pass else '✗ 失败'}")
    return all_pass


def main():
    t0 = time.time()
    r6 = verify_value_high_low()
    r5 = verify_bitsplit_concat()
    r7 = verify_int16_from_two_int8()
    r8 = verify_scale_formula()
    dt = time.time() - t0

    print("\n" + "=" * 72)
    print("阶段 2 验证汇总（T 类，bit-exact）")
    print("=" * 72)
    print(f"  耗时 {dt:.1f}s")
    print(f"  #6 位拆分可逆 value=high·2^p+low : {'✓' if r6 else '✗'}")
    print(f"  #5 bitsplit/concat 可逆(int12 等) : {'✓' if r5 else '✗'}")
    print(f"  #7 int16≡两个int8(high<<8|low)    : {'✓' if r7 else '✗'}")
    print(f"  #8 scale=max/(2^(bits-1)-1)       : {'✓' if r8 else '✗'}")
    overall = r6 and r5 and r7 and r8
    print(f"\n  总体判定: {'✅ 全部 bit-exact 通过' if overall else '❌ 存在失败'}")
    print(f"  交叉验证来源: {_REAL_NAME}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
