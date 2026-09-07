# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""低位对角裁剪最小实验（4 位档 int32 截断 bit-exact 验证）

背景：n=8（split_bits=4）时，n²=64 个 (a,c) 点积只聚合出 2n-1=15 个 partial[m]。
融合 fused = Σ_m partial[m] · 2^(4m)。

对「int32 截断」目标（只保留结果的低 32 位）：
  - m≥8 的项 partial[m]·2^(4m) 均为 2^32 的倍数 → 对低 32 位无贡献 → 可裁剪
  - m≤7 的项贡献低 32 位（2^0..2^28）→ 必须保留
  → 正确裁剪方向：保留 m=0..7，裁剪 m=8..14（28/64 = 43.75% ALU）

本实验验证两个候选方向，并统计一致率与 ALU 节省：
  A. 保留 m≤7、裁剪 m≥8 → 期望 bit-exact 一致（PASS）
  B. 保留 m≥7、裁剪 m≤6 → 期望不一致（反方向对照，FAIL）

分布覆盖：全幅均匀 int32 / 高斯量化 / 稀疏 / 真实权重定标量化
  × 多 seed × 多 K。参考实现用 Python 任意精度，保证判定无歧义。

运行: py -3.14 validate_low_diag_trim_int32.py
"""
import sys
from pathlib import Path

# 添加项目根目录和 engine/ 到 path
_PROJ_ROOT = Path(__file__).resolve().parents[4]
_ENGINE_DIR = Path(__file__).resolve().parents[3]
for _p in [_PROJ_ROOT, _ENGINE_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

from sgn.msint.cpp_backend import CppSplitDot, USING_CPP

SPLIT_BITS = 4
TOTAL_BITS = 32
N = TOTAL_BITS // SPLIT_BITS      # 8
N_PARTIALS = 2 * N - 1            # 15
M_KEEP_A = list(range(0, 8))      # 保留 m=0..7（裁剪高位）
M_KEEP_B = list(range(7, 15))     # 保留 m=7..14（裁剪低位，反方向对照）
KEEP_A_COUNT = sum(min(m + 1, 2 * N - 1 - m) for m in M_KEEP_A)  # 36 对
KEEP_B_COUNT = sum(min(m + 1, 2 * N - 1 - m) for m in M_KEEP_B)  # 36 对
TOTAL_PAIRS = N * N               # 64


def _to_i32(x):
    """把任意精度整数 x 截断为有符号 int32（保留低 32 位位模式）。"""
    lo = x & 0xFFFFFFFF
    return lo - 0x100000000 if lo >= 0x80000000 else lo


def _fuse_partials(partials, keep_m):
    """按保留对角 m 集合融合（Python 任意精度）。"""
    return sum(partials[m] << (m * SPLIT_BITS) for m in keep_m)


# ============================================================
# 数据分布生成
# ============================================================

def _uniform_int32(rng, n):
    """全幅均匀 int32（含边界极端值）。"""
    return [int(rng.randint(-2**31, 2**31 - 1, dtype=np.int64)) for _ in range(n)]


def _gauss_quant(rng, n, scale=512):
    """高斯分布定标取整，模拟真实量化权重的集中值域。"""
    return [int(np.clip(round(rng.normal(0, scale)), -2**31, 2**31 - 1)) for _ in range(n)]


def _sparse(rng, n, p=0.9):
    """稀疏：p 概率为 0，其余全幅随机。"""
    vals = []
    for _ in range(n):
        if rng.random() < p:
            vals.append(0)
        else:
            vals.append(int(rng.randint(-2**31, 2**31 - 1, dtype=np.int64)))
    return vals


def _real_quant_weights():
    """加载真实 float 权重并定标量化为 int32（模拟量化权重分布）。"""
    import torch
    # 注：真实权重 .pt 原位于 legacy/traditional/stage_2_3_int_path/results/，
    # 2026-08-16 legacy 独立：已迁至 tests/refs/data/。
    # 安全审计 2026-08-16 决策项 5：修复断链——原路径缺 legacy/ 前缀，加载必失败。
    p = _PROJ_ROOT / "engine" / "sgn" / "tests" / "refs" / "data" / "sbe_cnn_cifar10_wef_triple.pt"
    d = torch.load(str(p), map_location="cpu", weights_only=True)
    sd = d["model_state_dict"]
    w = sd["conv1.weight"].reshape(-1).numpy()   # 864 个 float32
    scale = 2.0 ** 20                            # 模拟 ~2^20 定标
    wq = np.clip(np.round(w * scale), -2**31, 2**31 - 1).astype(np.int64)
    return [int(v) for v in wq]


# ============================================================
# 单组验证
# ============================================================

def _run_case(w, x, label, stats):
    """对一组 (w, x) 验证方案 A / B，返回 (A 是否一致, B 是否一致)。"""
    exact = sum(int(a) * int(b) for a, b in zip(w, x))
    exact_i32 = _to_i32(exact)

    # C++（或 Python 参考）拆分点积 → 15 个 partial[m]
    partials = CppSplitDot.dot_split(w, x, TOTAL_BITS, SPLIT_BITS)
    assert len(partials) == N_PARTIALS, f"{label}: partials 数量 {len(partials)} != {N_PARTIALS}"

    # 完整融合应精确等于 exact（回归护栏）
    full_i32 = _to_i32(_fuse_partials(partials, range(N_PARTIALS)))
    assert full_i32 == exact_i32, f"{label}: 全量融合 {full_i32} != 精确 {exact_i32}"

    a_ok = _to_i32(_fuse_partials(partials, M_KEEP_A)) == exact_i32
    b_ok = _to_i32(_fuse_partials(partials, M_KEEP_B)) == exact_i32

    stats["total"] += 1
    stats["A_ok"] += 1 if a_ok else 0
    stats["B_ok"] += 1 if b_ok else 0
    if not a_ok:
        stats["A_fail"].append(label)
    if not b_ok:
        stats["B_fail"].append(label)
    return a_ok, b_ok


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 78)
    print("低位对角裁剪最小实验（4 位档 int32 截断 bit-exact）")
    print(f"C++ 后端: {USING_CPP}   split_bits={SPLIT_BITS}  n={N}  partials={N_PARTIALS}")
    print("=" * 78)
    print()
    print(f"ALU 统计: 总点积对 {TOTAL_PAIRS}；方案 A 保留 {KEEP_A_COUNT} 对"
          f"（裁剪 {TOTAL_PAIRS - KEEP_A_COUNT}，{-100*(TOTAL_PAIRS-KEEP_A_COUNT)/TOTAL_PAIRS:.2f}%）；"
          f"方案 B 保留 {KEEP_B_COUNT} 对（裁剪 {TOTAL_PAIRS - KEEP_B_COUNT}，"
          f"{-100*(TOTAL_PAIRS-KEEP_B_COUNT)/TOTAL_PAIRS:.2f}%）")
    print()

    stats = {"total": 0, "A_ok": 0, "B_ok": 0, "A_fail": [], "B_fail": []}
    rng = np.random.RandomState(20260814)

    # 1) 合成分布：多 seed × 多 K
    for seed in range(5):
        r2 = np.random.RandomState(seed)
        for K in (64, 256, 1024, 4096):
            w = _uniform_int32(r2, K)
            x = _uniform_int32(r2, K)
            _run_case(w, x, f"均匀 seed{seed} K{K}", stats)

            w = _gauss_quant(r2, K)
            x = _uniform_int32(r2, K)
            _run_case(w, x, f"高斯w seed{seed} K{K}", stats)

            w = _sparse(r2, K)
            x = _uniform_int32(r2, K)
            _run_case(w, x, f"稀疏w seed{seed} K{K}", stats)

    # 2) 真实权重定标量化（conv1 全连接行 × 均匀激活）
    real_w = _real_quant_weights()
    for seed in range(3):
        r2 = np.random.RandomState(seed)
        for K in (256, 512):
            w = real_w[:K]
            x = _uniform_int32(r2, K)
            _run_case(w, x, f"真实权重 seed{seed} K{K}", stats)

    # 3) 边界值组合（0 / ±1 / ±2^31-1 / -2^31 / 满幅 nibble 边界）
    edges = [0, 1, -1, 2**31 - 1, -2**31, 2**16 - 1, -2**16, 2**15, -2**15, 15, 16, -16]
    for seed in range(3):
        r2 = np.random.RandomState(seed)
        K = 64
        w = [edges[r2.randint(len(edges))] for _ in range(K)]
        x = [edges[r2.randint(len(edges))] for _ in range(K)]
        _run_case(w, x, f"边界 seed{seed} K{K}", stats)

    print(f"统计: 共 {stats['total']} 组")
    print(f"  方案 A（保留 m<=7，裁剪 m>=8）: {stats['A_ok']}/{stats['total']} 一致")
    print(f"  方案 B（保留 m>=7，裁剪 m<=6）: {stats['B_ok']}/{stats['total']} 一致")
    if stats["A_fail"]:
        print(f"  A 失败组: {stats['A_fail'][:10]}")
    if stats["B_fail"]:
        print(f"  B 失败组: {stats['B_fail'][:10]}")
    print()
    print("结论:")
    a_pass = stats["A_ok"] == stats["total"] and stats["total"] > 0
    print(f"  [{'PASS' if a_pass else 'FAIL'}] 裁剪 m>=8（保留 m<=7）："
          f"{'-44% ALU' if a_pass else '不一致'}")
    print(f"  [{'PASS' if stats['B_ok'] == stats['total'] else 'FAIL'}] 裁剪 m<=6（保留 m>=7）："
          f"{'一致' if stats['B_ok'] == stats['total'] else '不一致（反方向，非 bit-exact）'}")
    return a_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
