# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""
阶段 4：频域 / NTF 噪声整形验证
=====================================================
对应计划 numpy_math_verification_plan_2026_08_13.md §3 阶段 4 的 #13-#16

验证目标（纯数据验证，非神经网络训练）：
  #13 [S] EF ≡ delta-sigma：NTF=(1-z⁻¹)^N, STF=1
  #14 [S] 1 阶 EF 低频抑制（梯度=低通，EF=高通互补）
  #15 [E] clip 噪声 NTF 放大 191000×（低频功率）
  #16 [S] 分离残差 EF 修复 clip 场景（不反馈 clip 误差）

真实实现来源：
  - #13/#14: validate_ef_noise_shaping.py（NTF=(1-z⁻¹)^N 频域验证）
  - #15:     ef_noise_shaping_exploration_2026_08_06.md §3.4 实验4
            （同精度 + clip 场景，标准 EF 低频放大 191000×）
  - #16:     validate_ef_separated_and_cascade.py（分离残差 EF）

严谨性要求（§2.4/§2.6）：
  - #13  [S] 确定性 NTF 恒等式 n_t = NTF(η)_t（代数恒等，机器精度）+ 多seed PSD ratio≈1
  - #14  [S] 低频抑制 ratio（多seed，报告 mean±std）+ 权重噪声 TF=(1-z⁻¹)^{N-1}
  - #15  [E] 复现"同精度+clip"场景，实测低频功率放大比 vs 191000×
  - #16  [S] 分离 EF 低频功率 ≈ 无 EF（无放大），标准 EF 放大；多seed

用法（纯 numpy，无需 C++ 扩展）：
    py -3.14 engine/sgn/tests/architecture/validate_math_stage4_ntf_noise_shaping.py
"""
from __future__ import annotations

import time
from math import comb

import numpy as np

SEEDS = [0, 1, 2, 3, 4]     # 多随机种子，§2.5


# ============================================================
# 核心工具：N 阶标准 EF / 分离 EF / NTF 滤波
# ============================================================

def ef_coefficients(N: int):
    """N 阶 EF 反馈系数 a_k = (-1)^{k+1} C(N,k)"""
    return [(-1) ** (k + 1) * comb(N, k) for k in range(1, N + 1)]


def apply_ntf_filter(eta, N: int):
    """直接应用 NTF=(1-z⁻¹)^N：n_t = Σ_{k=0}^{N} (-1)^k C(N,k) η_{t-k}"""
    T = len(eta)
    out = np.zeros(T)
    for k in range(N + 1):
        coeff = (-1) ** k * comb(N, k)
        if k == 0:
            out += coeff * eta
        else:
            out[k:] += coeff * eta[:-k]
    return out


def make_sr_quantizer(delta, clamp, rng):
    """SR 量化器，分离 truncate 与 clip（与 validate_ef_separated_and_cascade 一致）"""
    def quantize(x):
        if np.ndim(x) == 0:
            u = rng.random(dtype=np.float32)
            q_clip, q_trunc = _sr_step(float(x), u)
            return float(q_clip), float(q_trunc)
        u = rng.random(size=x.shape, dtype=np.float32)
        return _sr_step(x, u)

    def _sr_step(x, u):
        x_div = x / delta
        q_floor = np.floor(x_div)
        frac = x_div - q_floor
        q_trunc = (q_floor + (u < frac).astype(np.float64)) * delta
        q_clip = np.clip(q_trunc, -clamp, clamp) if clamp is not None else q_trunc
        return q_clip, q_trunc
    return quantize


def run_ef(g_seq, quantize, scheme: str, N: int = 1):
    """运行 EF，返回输出噪声序列 n_t = q_t - g_t 与各阶残差/原始噪声

    scheme:
      'no_ef'     ：无 EF（纯量化）
      'standard'  ：标准 EF（残差 = ĝ - q，含 clip 误差）
      'separated' ：分离 EF（残差 = ĝ - q_trunc，不含 clip 误差）
    """
    T = len(g_seq)
    if scheme == 'no_ef':
        n = np.zeros(T)
        for t in range(T):
            q, _ = quantize(g_seq[t])
            n[t] = q - g_seq[t]
        return n, None
    # EF 方案（g 为标量信号，残差用 Python 标量）
    e_buffers = [0.0 for _ in range(N)]
    coeffs = ef_coefficients(N)
    n = np.zeros(T)
    eta = np.zeros(T)          # 原始量化噪声 η_t = q_t - ĝ_t（标准 EF 用）
    for t in range(T):
        g = g_seq[t]
        g_hat = float(g)
        for k in range(N):
            g_hat = g_hat + coeffs[k] * e_buffers[k]
        q, q_trunc = quantize(g_hat)
        q = float(q)
        q_trunc = float(q_trunc)
        n[t] = q - g
        eta[t] = q - g_hat           # 原始量化噪声（standard/separated 均填充）
        if scheme == 'standard':
            e_new = g_hat - q        # 含 clip 误差
        else:                        # separated
            e_new = g_hat - q_trunc  # 只含 truncate 误差
        for k in range(N - 1, 0, -1):
            e_buffers[k] = e_buffers[k - 1]
        e_buffers[0] = e_new
    return n, eta


def psd_of(x):
    """归一化单边 PSD（|FFT|²/T）"""
    T = len(x)
    fft = np.fft.rfft(x)
    return np.abs(fft) ** 2 / T, np.fft.rfftfreq(T, d=1.0)


def welch_psd(x, nperseg=4096):
    """Welch 平均 PSD（分段加窗平均，显著降低估计方差）

    用于 #13 的高阶 NTF 统计验证：原始周期图在 |NTF|^N 极小的低频段
    方差过大，导致 N=2/3 的 ratio 数值发散；Welch 平均后 ratio 稳定。
    """
    T = len(x)
    nseg = T // nperseg
    w = np.hanning(nperseg)
    acc = np.zeros(nperseg // 2 + 1)
    for i in range(nseg):
        seg = x[i * nperseg:(i + 1) * nperseg]
        seg = seg - seg.mean()
        fft = np.fft.rfft(seg * w)
        acc += np.abs(fft) ** 2 / (nperseg * (w ** 2).mean())
    return acc / nseg, np.fft.rfftfreq(nperseg)


def lowfreq_power(x, cutoff=0.05):
    psd, freqs = psd_of(x)
    omegas = 2 * np.pi * freqs
    return float(np.mean(psd[omegas < cutoff]))


def ar1_signal(T, rho, sigma, rng):
    """AR(1) 平稳信号：g_t = ρ g_{t-1} + N(0, σ√(1-ρ²))"""
    g = np.zeros(T)
    g[0] = float(rng.normal(0, sigma))
    for t in range(1, T):
        g[t] = rho * g[t - 1] + float(rng.normal(0, sigma * np.sqrt(1 - rho ** 2)))
    return g


# ============================================================
# #13 [S] EF ≡ delta-sigma：NTF=(1-z⁻¹)^N, STF=1
# ============================================================
def verify_ntf_identity():
    print("=" * 72)
    print("#13 [S] EF ≡ delta-sigma：NTF=(1-z⁻¹)^N, STF=1")
    print("=" * 72)

    T = 100_000
    DELTA = 0.1
    rng = np.random.default_rng(2026)
    g = ar1_signal(T, 0.99, 0.5, rng)

    # ---- 1) 确定性 NTF 恒等式：n_t = NTF(η)_t（标准 EF，含 clip 也能成立）----
    # 代数推导：n_t = q_t - g_t = e_{t-1} - e_t；e_t = -η_t ⇒ n_t = η_t - η_{t-1}
    print("  [1] 确定性恒等式 n = NTF(η)（标准 EF，机器精度）")
    ident_ok = True
    for N in [1, 2, 3]:
        # 无 clip：η 是纯 truncate 白噪声
        quant = make_sr_quantizer(DELTA, None, rng)
        n, eta = run_ef(g, quant, 'standard', N=N)
        ntf = apply_ntf_filter(eta, N)
        max_diff = float(np.max(np.abs(n - ntf)))
        exact = float(np.mean(n == ntf))
        ok = max_diff < 1e-9
        ident_ok &= ok
        print(f"    N={N}: max|n-NTF(η)|={max_diff:.2e}, 逐点精确比例={exact:.4f}"
              f"  → {'✓' if ok else '✗'}")

    # ---- 2) STF=1：信号分量无损通过 ----
    print("  [2] STF=1：信号路径无损（纯正弦输入，输出同频幅度/相位不变）")
    fs = 2 * np.pi * 0.05
    sig = 1.0 * np.sin(fs * np.arange(T)) + 0.3
    quant = make_sr_quantizer(DELTA, None, rng)
    n, eta = run_ef(sig, quant, 'standard', N=1)
    out = sig + n                       # 输出 = 信号 + 整形噪声
    # 用 DFT 在同频 bin 提取幅度（相干检测），STF = |out@e^{-jωt}|/|sig@e^{-jωt}|
    probe = np.exp(-1j * fs * np.arange(T))
    sig_amp = np.abs(np.dot(sig, probe))
    out_amp = np.abs(np.dot(out, probe))
    stf = out_amp / sig_amp if sig_amp > 0 else float('nan')
    stf_ok = abs(stf - 1.0) < 1e-3
    print(f"    同频探测: STF 幅度 = {stf:.6f}  → {'✓ 信号无损通过' if stf_ok else '✗'}")

    # ---- 3) 统计 PSD ratio（多 seed，常量信号 + Welch 平均，中频段）----
    # 原实验（方向 C 方法 B）用非零常量信号使 η 保持白噪声；
    # 高阶 NTF 在低频 |NTF|^N 极小，原始周期图方差过大 → 改用 Welch 平均。
    print("  [3] PSD ratio = 实测/理论 |NTF|²（多 seed，常量信号 + Welch，中频段）")
    ratios = {1: [], 2: [], 3: []}
    for seed in SEEDS:
        rng_s = np.random.default_rng(seed)
        g_const = np.full(T, 0.0037)        # 常量信号 → η 保持白噪声
        for N in [1, 2, 3]:
            quant_s = make_sr_quantizer(DELTA, None, rng_s)
            n, eta = run_ef(g_const, quant_s, 'standard', N=N)
            psd, freqs = welch_psd(n)
            omegas = 2 * np.pi * freqs
            theory = (DELTA ** 2 / 6) * (4 * np.sin(omegas / 2) ** 2) ** N
            mid = (omegas > 0.05) & (omegas < 0.4)
            ratios[N].append(float(np.mean(psd[mid] / theory[mid])))
    all_ok = True
    for N in [1, 2, 3]:
        r = np.array(ratios[N])
        ok = abs(r.mean() - 1.0) < 0.05
        all_ok &= ok
        print(f"    N={N}: ratio = {r.mean():.4f} ± {r.std():.4f} (n={len(r)} seed)"
              f"  → {'✓' if ok else '✗'}")

    ok = ident_ok and stf_ok and all_ok
    print(f"  结论: {'✓ EF ≡ delta-sigma（NTF/STF 成立）' if ok else '✗ 有失败'}")
    return ok


# ============================================================
# #14 [S] 1 阶 EF 低频抑制（梯度=低通，EF=高通互补）
# ============================================================
def verify_lowfreq_suppression():
    print("\n" + "=" * 72)
    print("#14 [S] 1 阶 EF 低频抑制（梯度=低通，EF=高通互补）")
    print("=" * 72)

    T = 100_000
    DELTA = 0.1
    theory_var = DELTA ** 2 / 6

    # ---- 1) NTF=(1-z⁻¹) 是高通：低频被抑制 ----
    print("  [1] NTF=(1-z⁻¹) 高通整形：低频 PSD 抑制（多 seed）")
    suppressions = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        g = ar1_signal(T, 0.99, 0.5, rng)
        quant = make_sr_quantizer(DELTA, None, rng)
        n, eta = run_ef(g, quant, 'standard', N=1)
        low = lowfreq_power(n)              # 整形后低频功率
        white_low = theory_var              # 未整形白噪声低频功率基准
        suppressions.append(white_low / max(low, 1e-30))
    s = np.array(suppressions)
    ok1 = np.all(s > 10)                    # 低频抑制 > 10×
    print(f"    低频抑制 = {s.mean():.0f}× ± {s.std():.0f}× (n={len(s)})"
          f"  → {'✓ 显著抑制' if ok1 else '✗'}")

    # ---- 2) 权重噪声 TF = (1-z⁻¹)^{N-1}（梯度积分器 1/(1-z⁻¹) 抵消一阶 NTF）----
    print("  [2] 权重噪声 TF=(1-z⁻¹)^{N-1}：SGD 积分器(低通)与 NTF(高通)互补")
    print("      （N=1→白噪声平坦；N=2→一阶高通 DC=0；N=3→二阶高通）")
    ok2 = True
    for N in [1, 2, 3]:
        rng = np.random.default_rng(2026)
        eta = rng.uniform(-DELTA / 2, DELTA / 2, T)   # 注入白噪声（均匀，方差 Δ²/12）
        shaped = apply_ntf_filter(eta, N)
        wnoise = np.cumsum(shaped)                    # 权重 = Σ 整形噪声（积分器）
        psd, freqs = psd_of(wnoise)
        omegas = 2 * np.pi * freqs
        uvar = DELTA ** 2 / 12                        # 注入均匀噪声的方差（非 Δ²/6）
        theory = uvar * (4 * np.sin(omegas / 2) ** 2) ** (N - 1)
        mid = (omegas > 0.02) & (omegas < 0.4)
        ratio = np.mean(psd[mid] / theory[mid])
        dc = psd[0]
        okN = abs(ratio - 1.0) < 0.1
        print(f"    N={N}: 权重噪声 PSD/理论 ratio={ratio:.3f}, DC={dc:.2e}"
              f"  → {'✓' if okN else '✗'}")
        ok2 &= okN

    # ---- 3) 高阶 DC 抑制对比（与报告 10385× 一致）----
    rng = np.random.default_rng(2026)
    eta = rng.uniform(-DELTA / 2, DELTA / 2, T)
    dc1 = float(psd_of(np.cumsum(apply_ntf_filter(eta, 1)))[0][0])   # 1 阶权重噪声 DC 功率
    dc2 = float(psd_of(np.cumsum(apply_ntf_filter(eta, 2)))[0][0])   # 2 阶权重噪声 DC 功率
    r12 = dc1 / max(dc2, 1e-30)
    print(f"  [3] 1阶→2阶 权重 DC 功率比 = {r12:.0f}×（报告 10385×，量级对比）")

    ok = ok1 and ok2
    print(f"  结论: {'✓ 1 阶 EF 低频抑制 + 高低通互补成立' if ok else '✗ 有失败'}")
    return ok


# ============================================================
# #15 [E] clip 噪声 NTF 放大 191000×（低频功率）
# ============================================================
def verify_clip_amplification():
    print("\n" + "=" * 72)
    print("#15 [E] clip 噪声 NTF 放大 191000×（低频功率）")
    print("=" * 72)
    print("  复现 ef_noise_shaping_exploration §3.4 实验4：同精度+clip 场景")
    print("  （零均值 AR(1) 信号 → 间歇 clip；delta 极小 → truncate 噪声≈0，")
    print("    clip 是唯一噪声源。标准 EF 的原始量化噪声 η=q-ĝ 被反馈放大）")

    T = 100_000
    DELTA = 1e-3          # 极小步长 ≈ 同精度（truncate 噪声≈0）
    CLAMP = 0.3
    RHO = 0.99            # 零均值 AR(1)，边际 std=0.5 → 间歇 clip（~55%）

    ratios = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        g = ar1_signal(T, RHO, 0.5, rng)
        quant = make_sr_quantizer(DELTA, CLAMP, rng)
        n_noef, _ = run_ef(g, quant, 'no_ef')
        n_std, eta_std = run_ef(g, quant, 'standard', N=1)
        lp_noef = lowfreq_power(n_noef)     # 无 EF 原始噪声（=η，因无反馈）
        lp_std = lowfreq_power(eta_std)     # 标准 EF 原始量化噪声 η=q-ĝ
        ratios.append(lp_std / max(lp_noef, 1e-30))
        if seed == SEEDS[0]:
            cr = float(np.mean(np.abs(g) > CLAMP))
            print(f"    [seed{seed}] clip 率 = {cr*100:.1f}%")
            print(f"    无 EF  低频功率 = {lp_noef:.4e}")
            print(f"    标准EF 低频功率 = {lp_std:.4e}")

    ratios = np.array(ratios)
    mean_r = float(np.mean(ratios))
    theory = 191000.0
    # [E] 经验标度：验证"极大放大"量级（与 191000× 同量级，±10× 内）
    ok = theory / 10 <= mean_r <= theory * 10
    print(f"    低频功率放大比 = {mean_r:.0f}× ± {ratios.std():.0f}× (n={len(ratios)} seed)")
    print(f"    冻结结论 191000× → {'✓ 同量级（机制复现）' if ok else '✗ 偏差过大'}")
    print(f"  结论: {'✓ clip 噪声 NTF 放大（白噪声前提被破坏）' if ok else '✗'}")
    return ok


# ============================================================
# #16 [S] 分离残差 EF 修复 clip 场景
# ============================================================
def verify_separated_ef():
    print("\n" + "=" * 72)
    print("#16 [S] 分离残差 EF 修复 clip 场景（不反馈 clip 误差）")
    print("=" * 72)

    T = 100_000
    DELTA = 1e-3
    CLAMP = 0.3
    RHO = 0.99

    # 多 seed：对比三种方案的原始量化噪声 η=q-ĝ 低频功率（相对无 EF）
    # 标准 EF 放大（对照）；分离 EF 只反馈 truncate 误差，不放大 clip
    sep_ratios = []
    std_ratios = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        g = ar1_signal(T, RHO, 0.5, rng)
        quant = make_sr_quantizer(DELTA, CLAMP, rng)
        n_noef, _ = run_ef(g, quant, 'no_ef')
        _, eta_std = run_ef(g, quant, 'standard', N=1)
        _, eta_sep = run_ef(g, quant, 'separated', N=1)
        lp_noef = lowfreq_power(n_noef)
        sep_ratios.append(lowfreq_power(eta_sep) / max(lp_noef, 1e-30))
        std_ratios.append(lowfreq_power(eta_std) / max(lp_noef, 1e-30))

    sep = np.array(sep_ratios)
    std = np.array(std_ratios)
    ok_sep = np.all(np.abs(sep - 1.0) < 0.1)     # 分离 EF 无放大
    ok_std = np.all(std > 10)                     # 标准 EF 显著放大（对照）
    print(f"    分离 EF 低频功率 / 无 EF = {sep.mean():.3f} ± {sep.std():.3f}"
          f"  → {'✓ 无放大' if ok_sep else '✗'}")
    print(f"    标准 EF 低频功率 / 无 EF = {std.mean():.0f} ± {std.std():.0f}"
          f"  → {'✓ 放大（对照成立）' if ok_std else '✗'}")
    ok = ok_sep and ok_std
    print(f"  结论: {'✓ 分离残差 EF 修复 clip 场景' if ok else '✗ 有失败'}")
    return ok


def main():
    t0 = time.time()
    r13 = verify_ntf_identity()
    r14 = verify_lowfreq_suppression()
    r15 = verify_clip_amplification()
    r16 = verify_separated_ef()
    dt = time.time() - t0

    print("\n" + "=" * 72)
    print("阶段 4 验证汇总")
    print("=" * 72)
    print(f"  耗时 {dt:.1f}s")
    print(f"  #13 [S] EF≡delta-sigma NTF/STF : {'✓' if r13 else '✗'}")
    print(f"  #14 [S] 1阶EF低频抑制+高低通互补 : {'✓' if r14 else '✗'}")
    print(f"  #15 [E] clip 噪声 NTF 放大 191000× : {'✓' if r15 else '✗'}")
    print(f"  #16 [S] 分离残差 EF 修复 clip   : {'✓' if r16 else '✗'}")
    overall = r13 and r14 and r15 and r16
    print(f"\n  总体判定: {'✅ 全部通过' if overall else '❌ 存在失败'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
