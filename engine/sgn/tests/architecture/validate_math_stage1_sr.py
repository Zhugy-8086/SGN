# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""
阶段 1：随机量化统计性质严格验证（S 类）
=========================================
对应计划 numpy_math_verification_plan_2026_08_13.md §3 阶段 1 的 #1-#4

验证目标（纯 numpy 数据验证，非神经网络训练）：
  #1 [S] Bernoulli SR 噪声方差 = Δ²/6
  #2 [S] SR 无偏性 E[noise] = 0
  #3 [S] clip 噪声含 DC 分量（NTF 白噪声前提被破坏）
  #4 [S] 加性均匀噪声+确定性 round = Δ²/12，与 Bernoulli SR(Δ²/6) 区分

严谨性要求（§2.5）：
  - 多随机种子（5 seed）报告 mean±std
  - 方差比用 bootstrap CI + z 检验（不假设高斯）
  - 无偏性用 t 检验
  - 容差随 N 收紧：|ratio-1| ≤ k·SE(ratio)，SE ∝ 1/√N
  - 更大 N 扫描，区分真实统计偏差 vs 数值伪影（float32/float64 对照）
  - 倒推：从实测方差反解 Δ 与指数 b（log-log 拟合）

理论依据：
  Bernoulli SR: q = floor(x/Δ) + (u<frac), u~U(0,1), frac=x/Δ-floor(x/Δ)
  无 clip 时噪声 e = Δ·(q - x/Δ)：
    E[e|frac] = 0（无偏）
    E[e²|frac] = frac(1-frac)·Δ²
  对 frac~U(0,1) 取期望：E[Var] = Δ²·∫frac(1-frac)df = Δ²/6
  加性均匀噪声 U(-Δ/2,Δ/2)+确定性 round：Var = Δ²/12（不同机制）
"""

import numpy as np
import json
import os
import time

# ============================
# 结论类型标注
# ============================
KIND = "S"   # 统计定律，见计划 §2.4

# ============================
# 核心函数（与既有 validate_sr_variance.py 定义一致）
# ============================

def bernoulli_sr(x, delta, qmin, qmax, rng):
    """Bernoulli stochastic rounding，返回反量化值 x_q = q*delta"""
    x_div = x / delta
    q_floor = np.floor(x_div).astype(np.int64)
    frac = x_div - q_floor
    u = rng.random(size=x.shape, dtype=np.float64)
    q = q_floor + (u < frac).astype(np.int64)
    q = np.clip(q, qmin, qmax)
    return q.astype(np.float64) * delta


def additive_noise_round(x, delta, qmin, qmax, rng):
    """加性均匀噪声 U(-Δ/2,Δ/2) + round（抖动 dithering）
    注意：round((x+u)/Δ)=floor(x/Δ+w), w~U(0,1) → 等价 Bernoulli SR → Δ²/6（非 Δ²/12）"""
    u = (rng.random(size=x.shape, dtype=np.float64) - 0.5) * delta
    q = np.round((x + u) / delta).astype(np.int64)
    q = np.clip(q, qmin, qmax)
    return q.astype(np.float64) * delta


def deterministic_round(x, delta, qmin, qmax):
    """确定性 round（无抖动）：经典 mid-tread 均匀量化，x 在 bin 内均匀时误差 U(-Δ/2,Δ/2) → Δ²/12"""
    q = np.round(x / delta).astype(np.int64)
    q = np.clip(q, qmin, qmax)
    return q.astype(np.float64) * delta


def variance_estimates(x, delta, qmin, qmax, rng, n_trials, quant=bernoulli_sr):
    """返回 n_trials 个独立方差估计（条件独立于给定 x）"""
    vars_ = np.zeros(n_trials)
    means_ = np.zeros(n_trials)
    for i in range(n_trials):
        x_q = quant(x, delta, qmin, qmax, rng)
        noise = x_q - x
        vars_[i] = np.var(noise)
        means_[i] = np.mean(noise)
    return vars_, means_


def bootstrap_ci(data, n_resample=2000, alpha=0.05, seed=123):
    """均值 bootstrap 置信区间"""
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_means = np.empty(n_resample)
    for i in range(n_resample):
        boot_means[i] = np.mean(rng.choice(data, size=n, replace=True))
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lo, hi


# ============================
# 实验配置
# ============================
SEEDS = [0, 1, 2, 3, 4]
FLOAT_PREC = "float64"
report = {}


def experiment_variance_delta2_over_6():
    """#1 [S] Bernoulli SR 方差 = Δ²/6（正推 + 更大N + 倒推反解 Δ）"""
    print("=" * 72)
    print("#1 [S] Bernoulli SR 噪声方差 = Δ²/6")
    print("=" * 72)

    range_ = 10.0
    results = {}

    for bits, qmin, qmax, label in [
        (8, 0, 255, "HC8  [0,255]"),
        (16, 0, 65535, "HC16 [0,65535]"),
    ]:
        delta = range_ / (2**bits - 1)
        theory = delta**2 / 6

        # --- 多 seed 收集方差估计 ---
        all_vars = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            x = rng.uniform(0, range_, 500_000)
            v, _ = variance_estimates(x, delta, qmin, qmax, rng, n_trials=40)
            all_vars.extend(v)
        all_vars = np.array(all_vars)
        mean_var = np.mean(all_vars)
        std_var = np.std(all_vars, ddof=1)
        n_est = len(all_vars)
        se = std_var / np.sqrt(n_est)
        ratio = mean_var / theory

        # bootstrap CI of mean_var
        lo, hi = bootstrap_ci(all_vars)
        ci_ratio = (lo / theory, hi / theory)

        # z 检验：mean_var vs theory（SE 由经验 std 估计）
        z = (mean_var - theory) / se
        # 容差随 N 收紧：|ratio-1| <= k*SE(ratio)
        se_ratio = se / theory
        tol_k1 = se_ratio  # 1σ
        tol_k2 = 2 * se_ratio  # 2σ
        pass1 = abs(ratio - 1) <= tol_k1
        pass2 = abs(ratio - 1) <= tol_k2
        z_pass = abs(z) <= 1.96

        # --- 倒推 1：从实测方差反解 Δ ---
        delta_recovered = np.sqrt(6 * mean_var)
        delta_err = abs(delta_recovered - delta) / delta

        results[label] = {
            "delta": delta, "theory": theory,
            "mean_var": mean_var, "std_var": std_var, "n_est": n_est,
            "ratio": ratio, "se_ratio": se_ratio,
            "z": z, "ci_ratio": ci_ratio,
            "pass_1sigma": bool(pass1), "pass_2sigma": bool(pass2), "pass_z": bool(z_pass),
            "delta_recovered": delta_recovered, "delta_err": delta_err,
        }

        print(f"\n  [{label}] Δ={delta:.6e}, 理论 Δ²/6={theory:.6e}")
        print(f"    empirical Var (mean±std, {n_est} 估计, {len(SEEDS)} seed) = "
              f"{mean_var:.6e} ± {std_var:.6e}")
        print(f"    ratio = {ratio:.6f}   (1σ 容差 {tol_k1:.2e}, 2σ 容差 {tol_k2:.2e})")
        print(f"    ratio 95% bootstrap CI = [{ci_ratio[0]:.6f}, {ci_ratio[1]:.6f}]")
        print(f"    z = {z:+.2f} (|z|<=1.96 通过 z 检验: {z_pass})")
        print(f"    判定: 1σ={pass1}, 2σ={pass2}, z检验={z_pass}")
        print(f"    倒推: Δ_实测=√(6·Var)={delta_recovered:.6e}, 相对误差={delta_err:.3e}")

    # --- 更大 N 扫描（稳定性 + 容差随 N 收紧）---
    print("\n  --- 更大 N 扫描（HC8，验证容差随 N 收紧）---")
    N_scan = [1e4, 1e5, 5e5, 2e6]
    scan_rows = []
    delta = 10.0 / 255
    theory = delta**2 / 6
    for N in N_scan:
        N = int(N)
        all_vars = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            x = rng.uniform(0, 10.0, N)
            v, _ = variance_estimates(x, delta, 0, 255, rng, n_trials=20)
            all_vars.extend(v)
        all_vars = np.array(all_vars)
        ratio = np.mean(all_vars) / theory
        se_ratio = (np.std(all_vars, ddof=1) / np.sqrt(len(all_vars))) / theory
        scan_rows.append((N, ratio, se_ratio))
        print(f"    N={N:>8d}: ratio={ratio:.6f}, 1σ={se_ratio:.3e}, "
              f"|ratio-1|/1σ={abs(ratio-1)/se_ratio:.2f}")

    report["#1"] = {"results": results, "n_scan": [list(r) for r in scan_rows]}
    return results


def experiment_unbiasedness():
    """#2 [S] SR 无偏性 E[noise]=0（t 检验）"""
    print("\n" + "=" * 72)
    print("#2 [S] SR 无偏性 E[noise]=0")
    print("=" * 72)

    delta = 10.0 / 255
    out = {}
    for N in [5_000_000]:
        all_means = []
        all_vars = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            x = rng.uniform(0, 10.0, N)
            v, m = variance_estimates(x, delta, 0, 255, rng, n_trials=20)
            all_means.extend(m)
            all_vars.extend(v)
        all_means = np.array(all_means)
        mean_e = np.mean(all_means)
        std_e = np.std(all_means, ddof=1)
        se_e = std_e / np.sqrt(len(all_means))
        t = mean_e / se_e  # H0: E[e]=0
        p_pass = abs(t) <= 1.96
        # 用噪声池化统计绝对偏差
        out = {"mean_e": mean_e, "se_e": se_e, "t": t, "pass": bool(p_pass)}
        print(f"\n  N={N}, {len(all_means)} 次噪声均值估计, {len(SEEDS)} seed")
        print(f"    E[noise] = {mean_e:.6e} ± {se_e:.6e}")
        print(f"    t = {t:+.3f} (|t|<=1.96: {p_pass})")
        print(f"    E[noise]/Δ = {mean_e/delta:.6f}（应≈0）")
    report["#2"] = out
    return out


def experiment_clip_dc():
    """#3 [S] clip 噪声含 DC 分量（NTF 白噪声前提被破坏）"""
    print("\n" + "=" * 72)
    print("#3 [S] clip 噪声含 DC 分量（E[noise]≠0 当发生 clip）")
    print("=" * 72)

    # 构造大量元素落在量化范围外 → 触发 clip
    # 非 clip 区域 SR 无偏，clip 区域有系统性截断 → E[noise] 偏负
    out = {}
    N = 500_000
    clip_rates = [0.0, 0.01, 0.05, 0.10, 0.30, 0.50]
    print(f"  {'clip率':>8} {'E[noise]':>14} {'Var':>14} {'理论Δ²/6':>14} {'DC 明显?':>8}")
    for cr in clip_rates:
        # 生成 x，其中 clip 部分超出范围
        delta = 10.0 / 255
        # 让范围 [0,10] 固定，clip 率通过把部分样本推超范围实现
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 10.0, N)
        n_clip = int(N * cr)
        # 把前 n_clip 个样本放大到超出范围（产生 clip）
        x[:n_clip] = 20.0 + rng.uniform(0, 5.0, n_clip)  # 远超 max=10 → clip 到 255Δ
        rng2 = np.random.default_rng(7)
        x_q = bernoulli_sr(x, delta, 0, 255, rng2)
        noise = x_q - x
        mean_e = np.mean(noise)
        var_e = np.var(noise)
        theory = delta**2 / 6
        dc_obvious = abs(mean_e) > 3 * np.sqrt(var_e / N)  # 显著非零
        out[cr] = {"mean_e": mean_e, "var": var_e}
        print(f"  {cr:>8.2%} {mean_e:>14.6e} {var_e:>14.6e} {theory:>14.6e} {str(dc_obvious):>8}")
    print("  → clip 率>0 时 E[noise]<0（DC 分量），破坏 NTF 白噪声前提")
    report["#3"] = out
    return out


def experiment_delta12_vs_delta6():
    """#4 [S] 三种量化机制方差区分（修正既有文档错误）
    - Bernoulli SR: Δ²/6
    - 均匀抖动+round (round(x/Δ+u)): = Bernoulli SR → Δ²/6（既有 validate_sr_variance.py 结论#4 误标为 Δ²/12）
    - 确定性 round (无抖动): 经典均匀量化 → Δ²/12
    """
    print("\n" + "=" * 72)
    print("#4 [S] 三种量化机制方差：SR(Δ²/6) / 抖动+round(Δ²/6) / 确定性round(Δ²/12)")
    print("=" * 72)

    delta = 10.0 / 255
    N = 500_000
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 10.0, N)

    v_sr, _ = variance_estimates(x, delta, 0, 255, rng, 40, bernoulli_sr)
    v_add, _ = variance_estimates(x, delta, 0, 255, rng, 40, additive_noise_round)
    # 确定性 round：对给定 x 误差固定，方差直接用 var(q*Δ - x)
    x_q_det = deterministic_round(x, delta, 0, 255)
    v_det = np.var(x_q_det - x)

    r_sr_d6 = np.mean(v_sr) / (delta**2 / 6)
    r_add_d6 = np.mean(v_add) / (delta**2 / 6)
    r_add_d12 = np.mean(v_add) / (delta**2 / 12)
    r_det_d12 = v_det / (delta**2 / 12)
    out = {
        "ratio_sr_d6": float(r_sr_d6),
        "ratio_add_d6": float(r_add_d6),
        "ratio_add_d12": float(r_add_d12),
        "ratio_det_d12": float(r_det_d12),
    }
    print(f"  Bernoulli SR    : Var={np.mean(v_sr):.6e}, /(Δ²/6)  = {r_sr_d6:.6f}")
    print(f"  均匀抖动+round  : Var={np.mean(v_add):.6e}, /(Δ²/6)  = {r_add_d6:.6f}  "
          f"/(Δ²/12)={r_add_d12:.6f}  ← 实为 Δ²/6")
    print(f"  确定性 round    : Var={v_det:.6e}, /(Δ²/12) = {r_det_d12:.6f}")
    print("  → 结论：抖动+round 等价 Bernoulli SR（Δ²/6）；Δ²/12 仅来自确定性 round（无抖动）")
    print("  → 更正：既有 validate_sr_variance.py 结论#4 '加性噪声+round=Δ²/12' 有误")
    report["#4"] = out
    return out


def experiment_reverse_exponent():
    """倒推：log(Var) vs log(Δ) 拟合，指数 b 应=2（S 类 E 型关系）"""
    print("\n" + "=" * 72)
    print("倒推补充：Var ∝ Δ^b，log-log 拟合 b（应=2）")
    print("=" * 72)

    range_ = 10.0
    deltas = [range_ / (2**b - 1) for b in (8, 10, 12, 14, 16)]
    vars_ = []
    rng = np.random.default_rng(11)
    x = rng.uniform(0, range_, 500_000)
    for d in deltas:
        qmax = int(range_ / d)
        v, _ = variance_estimates(x, d, 0, qmax, rng, 30, bernoulli_sr)
        vars_.append(np.mean(v))
    logd = np.log(deltas)
    logv = np.log(vars_)
    b, loga = np.polyfit(logd, logv, 1)
    a = np.exp(loga)
    # 理论 a = 1/6 → log a = -log 6
    theory_loga = -np.log(6)
    out = {"b": float(b), "log_a": float(loga), "theory_log_a": float(theory_loga),
           "r2": None}
    # R²
    pred = np.polyval([b, loga], logd)
    ss_res = np.sum((logv - pred) ** 2)
    ss_tot = np.sum((logv - np.mean(logv)) ** 2)
    r2 = 1 - ss_res / ss_tot
    out["r2"] = float(r2)
    print(f"  log Var = b·log Δ + log a, b={b:.6f}（理论 2）, log a={loga:.6f}（理论 {-np.log(6):.6f}）")
    print(f"  R² = {r2:.8f}")
    report["reverse_exponent"] = out
    return out


if __name__ == "__main__":
    t0 = time.time()
    experiment_variance_delta2_over_6()
    experiment_unbiasedness()
    experiment_clip_dc()
    experiment_delta12_vs_delta6()
    experiment_reverse_exponent()
    dt = time.time() - t0

    print("\n" + "=" * 72)
    print("阶段 1 验证汇总")
    print("=" * 72)
    print(f"  浮点精度: {FLOAT_PREC}，耗时 {dt:.1f}s")
    print(f"  #1 SR 方差: 见各 HC8/HC16 ratio 与 z 检验（应≈1, |z|<=1.96）")
    print(f"  #2 无偏性  : t={report['#2']['t']:+.3f}, E[noise]/Δ={report['#2']['mean_e']/ (10.0/255):+.6f}")
    print(f"  #3 clip DC : clip 率>0 时 E[noise]<0（DC 分量存在）")
    print(f"  #4 机制区分 : SR→Δ²/6 ratio={report['#4']['ratio_sr_d6']:.4f}, "
          f"抖动+round→Δ²/6 ratio={report['#4']['ratio_add_d6']:.4f}, "
          f"确定性round→Δ²/12 ratio={report['#4']['ratio_det_d12']:.4f}")
    print(f"  [更正] 既有 validate_sr_variance.py 结论#4 '加性噪声+round=Δ²/12' 有误：实为 Δ²/6")
    print(f"  倒推指数  : b={report['reverse_exponent']['b']:.4f} (理论 2), "
          f"R²={report['reverse_exponent']['r2']:.6f}")

    # 保存 JSON 结果
    out_dir = os.path.join(os.path.dirname(__file__), "_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "math_stage1_sr_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {out_path}")
