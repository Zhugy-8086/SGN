# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""MSint 多输出层信息分离验证（范式假设 H2：正交/低冗余）

对标 docs/msint_multisplit_paradigm/MSint多精度拆分计算范式.md §2.3 多输出模式
  多输出模式产出 [fine, cross, coarse] 多尺度表示（int32→int16，n=2）：
    fine   = Σ w_l·x_l           （细粒度：低×低位带）
    cross  = Σ (w_h·x_l + w_l·x_h) （交叉修正项：混合位带）
    coarse = Σ w_h·x_h           （粗粒度：高×高位带）

验证内容（性质 [T] 位带依赖 + [E] 尺度分离）：
  #24a [T] 位带依赖（信息分离/低冗余）——精确（bit-exact）：
      - 只保留低 16 位（w_low = w&0xFFFF）：fine 不变，cross=0，coarse=0
        → fine 仅依赖低×低位带，无跨带泄漏
      - 只保留高 16 位（w_high = w&~0xFFFF）：coarse 不变，fine=0，cross=0
        → coarse 仅依赖高×高位带
      - cross 依赖混合位带（保留高位×低位时非零）
  #24b [E] 尺度分离（能量主导）——观察性：
      - 随机 int32 输入下，各层对融合值的贡献 coarse·2^32 >> cross·2^16 >> fine，
        各层占据分离的幅度带（低冗余的工程含义）

运行: py -3.14 validate_math_msint_h2_orthogonality.py
"""
import sys
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parents[3]
_ENGINE_DIR = Path(__file__).resolve().parents[2]
for _p in [_PROJ_ROOT, _ENGINE_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

from sgn.msint.cpp_backend import CppSplitDot, _PySplitDot, USING_CPP

RNG_SEEDS = [0, 1, 2, 3, 4]
LOW_MASK = 0xFFFF


def _rand_int32(rng, n):
    return [int(rng.randint(-2**31, 2**31 - 1, dtype=np.int64)) for _ in range(n)]


def _layers(w, x):
    """返回 (fine, cross, coarse)，来自 C++ SplitDot.dot_split（int32→int16）。"""
    partials = CppSplitDot.dot_split(w, x, 32, 16)
    assert len(partials) == 3
    return partials[0], partials[1], partials[2]


# ============================================================
# #24a [T] 位带依赖（信息分离/低冗余）— bit-exact
# ============================================================

def test_24a_band_dependency_exact():
    """各层只读取指定输入位带，无跨带泄漏（bit-exact）。"""
    n_fail = 0
    n_low_ok = n_high_ok = 0
    for seed in RNG_SEEDS:
        rng = np.random.RandomState(seed)
        for _ in range(200):
            K = int(rng.randint(1, 300))
            w = _rand_int32(rng, K)
            x = _rand_int32(rng, K)

            fine, cross, coarse = _layers(w, x)

            # --- 只保留低 16 位：fine 必须不变，cross/coarse 必须为 0 ---
            w_low = [v & LOW_MASK for v in w]
            x_low = [v & LOW_MASK for v in x]
            f_low, c_low, co_low = _layers(w_low, x_low)
            if f_low != fine or c_low != 0 or co_low != 0:
                n_fail += 1
                break
            n_low_ok += 1

            # --- 只保留高 16 位：coarse 必须不变，fine/cross 必须为 0 ---
            w_high = [v & ~LOW_MASK for v in w]
            x_high = [v & ~LOW_MASK for v in x]
            f_high, c_high, co_high = _layers(w_high, x_high)
            if f_high != 0 or c_high != 0 or co_high != coarse:
                n_fail += 1
                break
            n_high_ok += 1

    assert n_fail == 0, f"#24a 位带依赖失败 {n_fail} 次"
    print(f"[PASS] #24a [T] 位带依赖（信息分离）："
          f"{n_low_ok} 组 low-only→fine 不变且 cross=coarse=0，"
          f"{n_high_ok} 组 high-only→coarse 不变且 fine=cross=0（bit-exact，5 seed）")


def test_24a_cross_mixed_band():
    """cross 依赖混合位带：保留 高位×低位 时 cross 非零。"""
    rng = np.random.RandomState(7)
    K = 100
    w = _rand_int32(rng, K)
    x = _rand_int32(rng, K)
    _, cross_full, _ = _layers(w, x)
    # 构造只含 高位(左)×低位(右) 的输入：w_high 与 x_low
    w_high = [v & ~LOW_MASK for v in w]
    x_low = [v & LOW_MASK for v in x]
    f, c, co = _layers(w_high, x_low)
    # 该组合只贡献 cross 项中的 w_h·x_l 部分
    expect_cross = sum(
        (a >> 16) * (b & LOW_MASK) for a, b in zip(w, x)
    )
    assert c == expect_cross, f"cross 混合项不符 {c} vs {expect_cross}"
    assert f == 0 and co == 0, "高位×低位不应产生 fine/coarse"
    # 对称：低位(左)×高位(右) 贡献 cross 中 w_l·x_h 部分
    w_low = [v & LOW_MASK for v in w]
    x_high = [v & ~LOW_MASK for v in x]
    _, c2, _ = _layers(w_low, x_high)
    expect_cross2 = sum(
        (a & LOW_MASK) * (b >> 16) for a, b in zip(w, x)
    )
    assert c2 == expect_cross2
    print("[PASS] #24a cross 混合位带：高位×低位 与 低位×高位 分别精确贡献 cross 项，fine/coarse 不受影响")


# ============================================================
# #24b [E] 尺度分离（能量主导）— 观察性
# ============================================================

def test_24b_scale_separation():
    """随机 int32 输入下，各层对融合值贡献按 2^16 尺度分离（coarse 主导）。"""
    ratios = []
    cross_ratios = []
    for seed in RNG_SEEDS:
        rng = np.random.RandomState(seed)
        for _ in range(100):
            K = int(rng.randint(10, 500))
            w = _rand_int32(rng, K)
            x = _rand_int32(rng, K)
            fine, cross, coarse = _layers(w, x)
            # 各层对融合值 fused = fine + cross·2^16 + coarse·2^32 的贡献幅度
            mag_fine = abs(fine)
            mag_cross = abs(cross << 16)
            mag_coarse = abs(coarse << 32)
            if mag_fine > 0:
                ratios.append(mag_coarse / mag_fine)
                if mag_cross > 0:
                    cross_ratios.append(mag_coarse / mag_cross)

    med_coarse_fine = float(np.median(ratios))
    med_coarse_cross = float(np.median(cross_ratios))
    # 期望：coarse·2^32 相对 fine 主导（K 较大时 coarse 原始量级不低于 fine，再乘 2^32）
    assert med_coarse_fine > 2**20, \
        f"coarse 贡献应主导 fine，实测 median={med_coarse_fine:.1f}"
    assert med_coarse_cross > 2**8, \
        f"coarse 贡献应主导 cross·2^16，实测 median={med_coarse_cross:.1f}"
    print(f"[PASS] #24b [E] 尺度分离（能量主导）："
          f"coarse·2^32 / fine 贡献中位 {med_coarse_fine:.2e}，"
          f"coarse·2^32 / cross·2^16 中位 {med_coarse_cross:.2e}"
          f" → 各层占据分离幅度带（低冗余工程含义）")


if __name__ == "__main__":
    print("=" * 78)
    print("MSint 多输出层信息分离验证（H2：正交/低冗余）")
    print(f"C++ 模式: {USING_CPP}")
    print("=" * 78)
    print()

    test_24a_band_dependency_exact()
    test_24a_cross_mixed_band()
    test_24b_scale_separation()

    print()
    print("=" * 78)
    print("=== All H2 信息分离验证 PASSED ===")
    print("  - #24a [T] 位带依赖：fine/coarse/cross 各自只读指定输入位带，无跨带泄漏")
    print("  - #24a cross 混合位带：精确贡献交叉修正项")
    print("  - #24b [E] 尺度分离：coarse 贡献主导，各层幅度带分离")
    print("=" * 78)
