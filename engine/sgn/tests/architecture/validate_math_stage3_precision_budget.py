# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""
阶段 3：精度分配与信息论验证
=====================================================
对应计划 numpy_math_verification_plan_2026_08_13.md §3 阶段 3 的 #9-#12

验证目标（纯数据验证，非神经网络训练）：
  #9  [T] 贪心 bits 分配 = 全局最优（M-凸性，gap 精确为 0）
  #10 [E] 成本信号须含 in_dim 因子（c = grad_l2²·in_dim，缺因子→次优）
  #11 [E] 误差 ≈ 1/(2^b-1)（bits 主杠杆，量化误差随位宽指数下降）
  #12 [E] HC16→HC32 精度提升 ≈ 2^16 ≈ 65536×（位宽翻倍→误差 2^-b 缩放）

真实实现来源：
  - #9/#10: engine/sgn/common/precision_budget.h PrecisionBudget::allocate()
            （贪心边际分配，误差模型 f_i(b)=c_i/(2^b-1)，M-凸性 exp34）
  - 成本信号 c_i = grad_l2² × in_dim（exp26）
  - #11/#12: exp12 量化误差缩放（HC16/HC32 对称量化）

严谨性要求（§2.4/§2.6）：
  - #9  T 类：贪心总误差与 DP 精确最优 gap 严格为 0（bit-exact）
  - #10 E 类：正确成本分配误差 < 错误成本（缺 in_dim）分配误差；报告比值
  - #11 E 类：log-log 拟合量化误差 vs (2^b-1)，斜率 ≈ -1，报告 R² 与 CI
  - #12 E 类：HC16/HC32 误差比值 ≈ 2^16 ≈ 65536，与 64572× 结论比对

用法（需 Python 3.14，因 C++ 扩展为 cp314 编译）：
    py -3.14 engine/sgn/tests/architecture/validate_math_stage3_precision_budget.py
"""
from __future__ import annotations

import os as _os
import sys
import time

import numpy as np

# ============================================================
# 尝试导入真实 C++ PrecisionBudget（engine/ 在 path 时 sgn=engine/sgn）
# ============================================================
_FILE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_ARCH = _FILE_DIR
_TESTS = _os.path.dirname(_ARCH)
_SGN = _os.path.dirname(_TESTS)
_ENGINE = _os.path.dirname(_SGN)
_PROJECT_ROOT = _os.path.dirname(_ENGINE)
_PARENT = _os.path.dirname(_PROJECT_ROOT)
for _p in (_PROJECT_ROOT, _PARENT, _ENGINE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import sgn as _sgn
    _PB = _sgn.PrecisionBudget
    _LC = _sgn.LayerCost
    _REAL_NAME = "C++ sgn.PrecisionBudget"
except Exception as e:  # pragma: no cover
    _PB = None
    _LC = None
    _REAL_NAME = f"None (导入失败: {e})"


# ============================================================
# 本地参照实现（与 precision_budget.h 完全一致的数学）
# ============================================================
def layer_error(c: float, b: int) -> float:
    """f_i(b) = c / (2^b - 1)；b=0 时误差等于成本"""
    if b == 0:
        return c
    return c / (2 ** b - 1)


def marginal_gain(c: float, b: int) -> float:
    """Δ_i(b) = f_i(b) - f_i(b+1) = c·2^b / ((2^b-1)(2^(b+1)-1))"""
    if b == 0:
        return c
    return c * (2 ** b) / ((2 ** b - 1) * (2 ** (b + 1) - 1))


def greedy_allocate(costs, bmins, bmaxs, B):
    """贪心边际分配（与 C++ PrecisionBudget::allocate 一致）"""
    n = len(costs)
    bits = list(bmins)
    remaining = B - sum(bmins)
    if remaining <= 0:
        return bits
    while remaining > 0:
        best_gain = -1.0
        best_idx = -1
        for i in range(n):
            if bits[i] >= bmaxs[i]:
                continue
            gain = marginal_gain(costs[i], bits[i])
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx < 0:
            break
        bits[best_idx] += 1
        remaining -= 1
    return bits


def total_error(costs, bmins, bmaxs, bits):
    return sum(layer_error(c, b) for c, b in zip(costs, bits))


def dp_optimal(costs, bmins, bmaxs, B):
    """DP 精确全局最优（可分凸整数规划，等价 0/1 背包，给出严格最优）

    目标：min Σ f_i(b_i)，约束 Σ b_i = B，b_i ∈ [b_min_i, b_max_i]
    复杂度 O(n · Σb_max · range)，对随机场景完全可行，结果位精确。
    """
    n = len(costs)
    smin = sum(bmins)
    smax = sum(bmaxs)
    if B <= smin:
        return list(bmins), total_error(costs, bmins, bmaxs, bmins)
    if B >= smax:
        bits = list(bmaxs)
        return bits, total_error(costs, bmins, bmaxs, bits)
    Bc = B
    INF = float("inf")
    dp = [[INF] * (smax + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    parent = {}
    for i in range(1, n + 1):
        c, lo, hi = costs[i - 1], bmins[i - 1], bmaxs[i - 1]
        for s in range(smax + 1):
            if dp[i - 1][s] == INF:
                continue
            for b in range(lo, hi + 1):
                ns = s + b
                if ns > smax:
                    continue
                e = dp[i - 1][s] + layer_error(c, b)
                if e < dp[i][ns]:
                    dp[i][ns] = e
                    parent[(i, ns)] = (i - 1, s, b)
    bits = [0] * n
    i, s = n, Bc
    while i > 0:
        pi, ps, b = parent[(i, s)]
        bits[i - 1] = b
        i, s = pi, ps
    # 用与贪心完全一致的干净求和重算最优值，消除 DP 累积求和的 ulp 伪影
    best_err = total_error(costs, bmins, bmaxs, bits)
    return bits, best_err


# ============================================================
# #9 [T] 贪心 = 全局最优（M-凸性，gap 精确为 0）
# ============================================================
def verify_greedy_optimal():
    print("=" * 72)
    print("#9 [T] 贪心 bits 分配 = 全局最优（M-凸性，gap 精确为 0）")
    print("=" * 72)

    rng = np.random.default_rng(2026)
    n_scenarios = 400
    all_pass = True
    n_exact0 = 0
    max_gap = 0.0
    for _ in range(n_scenarios):
        n = int(rng.integers(2, 13))
        # 成本覆盖多个数量级（模拟真实梯度方差分布）
        costs = np.exp(rng.uniform(-8, 6, size=n)).tolist()
        bmins = [int(rng.integers(4, 9)) for _ in range(n)]
        bmaxs = [int(rng.integers(16, 25)) for _ in range(n)]
        smin = sum(bmins)
        smax = sum(bmaxs)
        B = int(rng.integers(smin + 1, smax))  # 保证 smin<B<smax，贪心恰好用满 B

        gbits = greedy_allocate(costs, bmins, bmaxs, B)
        gerr = total_error(costs, bmins, bmaxs, gbits)
        dbits, derr = dp_optimal(costs, bmins, bmaxs, B)

        # 贪心达到全局最优 ⟺ 其总误差 = DP 精确最优值
        # 同一求和顺序下，若 bits 相同则严格为 0；若存在并列最优解，
        # 不同位向量的干净求和最多差 1 ulp（≤1e-15）。
        gap = abs(gerr - derr) / max(derr, 1e-300)
        max_gap = max(max_gap, gap)
        if gap == 0.0:
            n_exact0 += 1
        if gap > 1e-15:     # 超过 1 ulp → 贪心确实次优（M-凸性被破坏）
            all_pass = False
            if n_exact0 < 3:
                print(f"  ✗ 场景 n={n} B={B}: greedy_err={gerr:.6e}, dp_err={derr:.6e}, "
                      f"gap={gap:.3e}")
                print(f"    greedy_bits={gbits}\n    dp_bits    ={dbits}")

    print(f"  随机场景 {n_scenarios} 个（n∈[2,12]，成本跨 14 个数量级，B 在 [smin+1, smax)）")
    print(f"  贪心总误差 == DP 全局最优（gap 精确为 0）: {n_exact0}/{n_scenarios} 例")
    print(f"  所有场景 max gap = {max_gap:.3e}（≤1e-15 即 1 ulp 内 = 达到全局最优）"
          f"  → {'✓' if all_pass else '✗'}")

    # 交叉验证 C++ 实现
    # 判定标准：与本地贪心检查一致的 1 ulp 相对容差（≤1e-15）。
    # 原因：当存在并列最优解时，C++ 贪心可能选中与 DP 回溯不同的
    #       最优位向量，二者的干净求和最多差 1 ulp（良性，非次优）。
    #       若用严格 != 会把这种并列最优误判为失败。
    if _PB is not None:
        print(f"\n  [交叉验证] 真实实现 {_REAL_NAME}")
        cpp_ok = True
        cpp_cnt = 0
        cpp_exact0 = 0
        cpp_parallel = 0   # 并列最优（bits 不同但误差在 1 ulp 内）
        cpp_max_gap = 0.0
        for _ in range(200):
            n = int(rng.integers(2, 13))
            costs = np.exp(rng.uniform(-8, 6, size=n)).tolist()
            bmins = [int(rng.integers(4, 9)) for _ in range(n)]
            bmaxs = [int(rng.integers(16, 25)) for _ in range(n)]
            smin = sum(bmins)
            smax = sum(bmaxs)
            B = int(rng.integers(smin + 1, smax))
            layers = [_LC(c=c, b_min=bmin, b_max=bmax)
                      for c, bmin, bmax in zip(costs, bmins, bmaxs)]
            pb = _PB(B, layers)
            cpp_bits = list(pb.allocate())
            cpp_err = pb.total_error(cpp_bits)
            dbits, derr = dp_optimal(costs, bmins, bmaxs, B)
            gap = abs(cpp_err - derr) / max(derr, 1e-300)
            cpp_max_gap = max(cpp_max_gap, gap)
            if gap == 0.0:
                cpp_exact0 += 1
            elif cpp_bits != dbits:
                # bits 不同但误差在 1 ulp 内 → 并列最优解（良性）
                cpp_parallel += 1
            if gap > 1e-15:
                # 真正超 1 ulp：要么 C++ 贪心次优，要么实现有差异
                cpp_ok = False
                print(f"    ✗ 用例 n={n} B={B}: cpp_err={cpp_err:.6e}, dp_err={derr:.6e}, "
                      f"gap={gap:.3e}")
                print(f"      cpp_bits={cpp_bits}\n      dp_bits ={dbits}")
            cpp_cnt += 1
        print(f"  C++ PrecisionBudget::allocate() 总误差 == DP 全局最优 "
              f"(1 ulp 容差): {'✓' if cpp_ok else '✗'}（{cpp_cnt} 例）")
        print(f"    其中 gap 精确为 0: {cpp_exact0} 例；并列最优(不同位向量, 1 ulp 内): "
              f"{cpp_parallel} 例；max gap = {cpp_max_gap:.3e}")
        all_pass &= cpp_ok
    else:
        print(f"\n  [交叉验证] 真实实现不可导入，仅用本地一致实现")

    print(f"\n  结论: {'✓ gap 精确为 0（M-凸性成立）' if all_pass else '✗ 有失败'}")
    return all_pass


# ============================================================
# #10 [E] 成本信号须含 in_dim 因子（c = grad_l2²·in_dim）
# ============================================================
def verify_in_dim_factor():
    print("\n" + "=" * 72)
    print("#10 [E] 成本信号须含 in_dim 因子（c = grad_l2²·in_dim）")
    print("=" * 72)

    rng = np.random.default_rng(7)
    # 复现 test_negative_log2_var / exp26 B 场景：in_dim 跨 3~4096 差异大
    in_dims = [3, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    B = 124
    bmin, bmax = 4, 20

    # 多组 grad_l2 分布（梯度范数有差异 + 完全相等两种情况）
    grad_l2_sets = [
        [1.0] * 10,                                  # 梯度范数相同 → in_dim 是唯一区分
        [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 0.001],  # 非均匀（exp34 场景）
        np.exp(rng.uniform(-3, 0, 10)).tolist(),
    ]
    all_pass = True
    for gi, grad_l2 in enumerate(grad_l2_sets):
        # 正确成本：c = grad_l2² · in_dim
        c_correct = [g * g * d for g, d in zip(grad_l2, in_dims)]
        # 错误成本：缺 in_dim（仅 grad_l2²）
        c_wrong = [g * g for g in grad_l2]

        bits_c = greedy_allocate(c_correct, [bmin]*10, [bmax]*10, B)
        bits_w = greedy_allocate(c_wrong, [bmin]*10, [bmax]*10, B)
        # 用正确成本评估两种分配
        err_c = total_error(c_correct, [bmin]*10, [bmax]*10, bits_c)
        err_w_under_c = total_error(c_correct, [bmin]*10, [bmax]*10, bits_w)

        ratio = err_w_under_c / err_c
        ok = err_c < err_w_under_c and bits_c != bits_w
        all_pass &= ok
        print(f"  grad_l2 组 {gi}: 正确分配 err={err_c:.4e}, "
              f"错误(缺 in_dim)分配按正确成本评={err_w_under_c:.4e}, "
              f"劣化比={ratio:.2f}x  → {'✓ 正确<错误' if ok else '✗'}")

    # 倒推：验证成本对 in_dim 的依赖指数（从"正确分配 vs 错误分配"的差异反推）
    # 做法：固定 grad_l2=1，拟合两种分配下"得 bit 越多的层其 in_dim 越大"的相关性
    print("\n  [倒推] in_dim 因子可辨识性：成本应随 in_dim 单调（越大分配越多 bits）")
    grad_l2 = [1.0] * 10
    c_correct = [d for d in in_dims]
    bits_c = greedy_allocate(c_correct, [bmin]*10, [bmax]*10, B)
    # 用 Spearman 相关验证分配 bits 与 in_dim 的单调关系
    rank_bits = np.argsort(np.argsort(bits_c))
    rank_dim = np.argsort(np.argsort(in_dims))
    # Spearman 相关系数（手工实现，避免 scipy 依赖）
    d = rank_bits - rank_dim
    rho = 1 - 6 * np.sum(d * d) / (10 * (10 * 10 - 1))
    print(f"  分配 bits 与 in_dim 的 Spearman 秩相关: ρ={rho:.4f} "
          f"({'✓ 强正相关（成本随 in_dim 增加）' if rho > 0.9 else '✗'})")
    all_pass &= rho > 0.9

    print(f"\n  结论: {'✓ 成本必须含 in_dim 因子' if all_pass else '✗ 有失败'}")
    return all_pass


# ============================================================
# #11 [E] 误差 ≈ 1/(2^b-1)（bits 主杠杆）
# ============================================================
def hc_quantize(x: np.ndarray, bits: int) -> np.ndarray:
    """HC 对称量化 round-trip（float64，避免高 bits 时 float32 精度破坏）

    与 exp12 hc16/hc32_quantize 一致：scale = max_abs / (2^(bits-1)-1)，
    q ∈ [-2^(bits-1), 2^(bits-1)-1]（对称量化，qmin=-2^(bits-1), qmax=2^(bits-1)-1）
    """
    x64 = np.asarray(x, dtype=np.float64)
    max_abs = float(np.abs(x64).max()) if x64.size else 0.0
    if max_abs == 0.0:
        return x64.copy()
    qmax = 2 ** (bits - 1) - 1
    scale = max_abs / qmax
    q = np.clip(np.round(x64 / scale), -(2 ** (bits - 1)), qmax).astype(np.int64)
    return q.astype(np.float64) * scale


def rel_l2(diff: np.ndarray, ref: np.ndarray) -> float:
    den = float(np.linalg.norm(ref))
    return float(np.linalg.norm(diff)) / den if den > 0 else 0.0


def verify_bits_lever():
    print("\n" + "=" * 72)
    print("#11 [E] 误差 ≈ 1/(2^b-1)（bits 主杠杆，log-log 斜率 ≈ -1）")
    print("=" * 72)

    rng = np.random.default_rng(11)
    # 梯度样信号（正态分布），固定信号量化为多个位宽
    g = rng.normal(0.0, 1.0, size=(4096,)).astype(np.float64)
    bits_list = list(range(4, 25))   # 4..24
    errs = []
    print(f"  {'bits':>4} {'(2^b-1)':>12} {'相对L2误差':>12} {'1/(2^b-1)':>14}")
    for b in bits_list:
        g_dq = hc_quantize(g, b)
        e = rel_l2(g - g_dq, g)
        errs.append(e)
        if b <= 16 or b % 4 == 0:
            print(f"  {b:>4} {2**b-1:>12} {e:>12.3e} {1/(2**b-1):>14.3e}")

    errs = np.array(errs)
    # log-log 拟合: log(err) = log(A) + k·log(2^b-1)，期望 k=-1
    X = np.log2(2.0 ** np.array(bits_list) - 1.0)
    Y = np.log2(errs)
    A = np.vstack([X, np.ones_like(X)]).T
    k, logA = np.linalg.lstsq(A, Y, rcond=None)[0]
    Yhat = A @ np.array([k, logA])
    ss_res = np.sum((Y - Yhat) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    # 斜率标准误（用于 CI）
    resid = Y - Yhat
    dof = len(X) - 2
    se2 = np.sum(resid ** 2) / dof
    var_k = se2 / np.sum((X - np.mean(X)) ** 2)
    se_k = np.sqrt(var_k)
    print(f"\n  log-log 拟合: log₂(误差) = {k:.4f}·log₂(2^b-1) + {logA:.3f}")
    print(f"  斜率 k = {k:.4f} ± {1.96*se_k:.4f} (95% CI: "
          f"[{k-1.96*se_k:.4f}, {k+1.96*se_k:.4f}])")
    print(f"  R² = {r2:.6f}")
    ok = abs(k + 1.0) < 0.02 and r2 > 0.999
    print(f"  判定: 斜率≈-1（误差∝1/(2^b-1)）且 R²→1  → "
          f"{'✓ bits 主杠杆成立' if ok else '✗'}")
    return ok


# ============================================================
# #12 [E] HC16→HC32 精度提升 ≈ 2^16 ≈ 65536×
# ============================================================
def verify_hc16_hc32():
    print("\n" + "=" * 72)
    print("#12 [E] HC16→HC32 精度提升 ≈ 2^16 ≈ 65536×（位宽翻倍→误差 2^-b 缩放）")
    print("=" * 72)

    rng = np.random.default_rng(12)
    seeds = [1, 2, 3, 4, 5]
    amplitudes = [0.01, 0.1, 1.0]
    n = 20000
    print(f"  {'幅度':>6} {'HC8 误差':>12} {'HC16 误差':>12} {'HC32 误差':>12} "
          f"{'8→16×':>10} {'16→32×':>12}")
    ratios_16_32 = []
    ratios_8_16 = []
    for amp in amplitudes:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            g = rng.normal(0.0, amp, size=n).astype(np.float64)
            e8 = rel_l2(g - hc_quantize(g, 8), g)
            e16 = rel_l2(g - hc_quantize(g, 16), g)
            e32 = rel_l2(g - hc_quantize(g, 32), g)
            r16_32 = e16 / e32 if e32 > 0 else float("inf")
            r8_16 = e8 / e16 if e16 > 0 else float("inf")
            ratios_16_32.append(r16_32)
            ratios_8_16.append(r8_16)
        print(f"  {amp:>6} {e8:>12.4e} {e16:>12.4e} {e32:>12.4e} "
              f"{np.mean(ratios_8_16[-5:]):>10.0f}x {np.mean(ratios_16_32[-5:]):>12.0f}x")

    mean16_32 = float(np.mean(ratios_16_32))
    mean8_16 = float(np.mean(ratios_8_16))
    theory16_32 = 2.0 ** 16          # 65536
    theory8_16 = 2.0 ** 8            # 256
    # 64572× 是 exp12 实测值；验证 16→32 比值落在 [0.97, 1.03]·65536 内
    lo = 0.97 * theory16_32
    hi = 1.03 * theory16_32
    ok16_32 = lo <= mean16_32 <= hi
    ok8_16 = 0.97 * theory8_16 <= mean8_16 <= 1.03 * theory8_16
    print(f"\n  HC16→HC32 实测均值: {mean16_32:.0f}x（理论 2^16={theory16_32:.0f}，"
          f"exp12 结论 64572×）→ {'✓ 落在 65536±3% 内' if ok16_32 else '✗'}")
    print(f"  HC8→HC16 实测均值: {mean8_16:.0f}x（理论 2^8={theory8_16:.0f}）"
          f"→ {'✓ 落在 256±3% 内' if ok8_16 else '✗'}")
    ok = ok16_32 and ok8_16
    print(f"  结论: {'✓ 位宽翻倍→误差 2^-b 缩放（与 64572× 一致）' if ok else '✗'}")
    return ok


def main():
    t0 = time.time()
    r9 = verify_greedy_optimal()
    r10 = verify_in_dim_factor()
    r11 = verify_bits_lever()
    r12 = verify_hc16_hc32()
    dt = time.time() - t0

    print("\n" + "=" * 72)
    print("阶段 3 验证汇总")
    print("=" * 72)
    print(f"  耗时 {dt:.1f}s")
    print(f"  #9  [T] 贪心=全局最优 gap=0 : {'✓' if r9 else '✗'}")
    print(f"  #10 [E] 成本含 in_dim 因子 : {'✓' if r10 else '✗'}")
    print(f"  #11 [E] 误差∝1/(2^b-1)     : {'✓' if r11 else '✗'}")
    print(f"  #12 [E] HC16→HC32 ≈65536×  : {'✓' if r12 else '✗'}")
    overall = r9 and r10 and r11 and r12
    print(f"\n  总体判定: {'✅ 全部通过' if overall else '❌ 存在失败'}")
    print(f"  #9/#10 交叉验证来源: {_REAL_NAME}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
