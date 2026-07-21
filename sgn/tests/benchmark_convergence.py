#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 [zhugy-8086]
"""SGN v5.1.7 补充: 收敛速度基准测试

4 个测试维度：
  1. 噪声鲁棒性（噪声等级 0.0~0.3）
  2. 网络规模（神经元数 64~1024）
  3. 多层 vs 单层
  4. 静默开启 vs 关闭

用法:
  python tests/benchmark_convergence.py           # 交互式菜单
  python tests/benchmark_convergence.py --all     # 全部运行
  python tests/benchmark_convergence.py --test 1  # 单项运行
"""
import os
import sys
import json
import time
import random
from datetime import datetime

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ============================================================
# 合成样本生成器（自包含，不依赖外部数据集）
# ============================================================

def generate_synthetic_samples(num_labels=4, samples_per_label=50, d=64, seed=42):
    """生成合成样本：每个标签有一个原型模板，样本 = 原型 + 翻转噪声

    注意：强度值使用 0/255（不是 0/1），否则 `_verify` 的
    交叉相乘检查无法区分标记/未标记区域，验证永不通过。

    Args:
        num_labels: 标签数量（用 A, B, C, ... 表示）
        samples_per_label: 每个标签的样本数
        d: 窗口像素数（8×8=64）
        seed: 随机种子
    Returns:
        [(intensity_list, label), ...]
    """
    rng = random.Random(seed)
    labels = [chr(ord('A') + i) for i in range(num_labels)]
    samples = []
    for label in labels:
        # 原型模板：每个标签一个固定的位掩码
        prototype = [rng.randint(0, 1) for _ in range(d)]
        for _ in range(samples_per_label):
            # 样本 = 原型 + 翻转噪声（默认 10% 翻转），映射到 0/255 范围
            intensity = [255 if (b ^ (1 if rng.random() < 0.1 else 0)) else 0 for b in prototype]
            samples.append((intensity, label))
    rng.shuffle(samples)
    return samples


def add_noise(intensity, noise_prob, seed=None):
    """给样本添加翻转噪声

    Args:
        intensity: 原始强度图（0/1 列表）
        noise_prob: 每个像素的翻转概率
        seed: 随机种子（None 表示不固定）
    Returns:
        加噪后的强度图
    """
    rng = random.Random(seed) if seed is not None else random
    return [b ^ (1 if rng.random() < noise_prob else 0) for b in intensity]


# ============================================================
# 评估函数
# ============================================================

def evaluate_accuracy(core, test_samples):
    """在测试集上评估准确率

    Args:
        core: SGNCore 实例
        test_samples: [(intensity, label), ...]
    Returns:
        accuracy (0.0~1.0)
    """
    if not test_samples:
        return 0.0
    correct = 0
    from engine.layers import classify_multi_layer, classify_sample
    multi_layer = getattr(core, 'multi_layer_enabled', False)
    d = getattr(core, 'D', 64)
    for intensity, label in test_samples:
        if multi_layer:
            pred, _ = classify_multi_layer(core, intensity)
        else:
            pred, _ = classify_sample(core, intensity, d)
        if pred == label:
            correct += 1
    return correct / len(test_samples)


# ============================================================
# 核心测试函数
# ============================================================

def _make_core(overrides=None):
    """创建 SGNCore 实例，允许覆盖配置"""
    from engine.config import CONFIG, ConfigRegistry
    # 先重置到默认值，避免之前测试的残留影响
    saved = {}
    if overrides:
        for k, v in overrides.items():
            saved[k] = (ConfigRegistry._values.get(k), CONFIG.get(k))
            ConfigRegistry._values[k] = v
            CONFIG[k] = v
    from engine.core import SGNCore
    core = SGNCore(seed=CONFIG.get("SEED", 42))
    # 恢复修改（使 _make_core 不污染全局配置）
    if overrides:
        for k, (old_v, old_c) in saved.items():
            if old_v is not None:
                ConfigRegistry._values[k] = old_v
                CONFIG[k] = old_v
            else:
                ConfigRegistry._values.pop(k, None)
                CONFIG.pop(k, None)
    return core


def _train_and_track(core, train_samples, test_samples, max_steps,
                     eval_interval=100, target_acc=0.9, label="",
                     test_index=1, total_tests=1):
    """训练并跟踪准确率，记录首次达到目标准确率的步数

    Args:
        label: 用于进度输出的描述标签（如 "noise=0.10"）
        test_index: 当前测试序号（1-based），用于整体进度显示
        total_tests: 本轮测试总数
    Returns:
        dict: {
            'first_target_step': 首次达目标的步数（None=未达到），
            'final_acc': 最终准确率,
            'accuracy_history': [(step, acc), ...],
            'avg_step_ms': 平均每步耗时,
            'participation_rate': 最终参与率（多层模式）,
        }
    """
    from engine.config import CONFIG
    multi_layer = getattr(core, 'multi_layer_enabled', False)
    batch_enabled = CONFIG.get("BATCH_TRAIN_ENABLED", True) and multi_layer
    batch_size = CONFIG.get("BATCH_SIZE", 32)

    # 检测终端能力：非 tty 时自动切多行模式
    _is_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    _multi_line = os.environ.get("SGN_BENCH_MODE") == "multi" or not _is_tty

    accuracy_history = []
    first_target_step = None
    total_ms = 0.0
    step = 0
    next_eval_step = eval_interval  # 按实际步数触发评估
    t_start = time.time()

    while step < max_steps:
        if batch_enabled:
            batch = train_samples[step % len(train_samples):][:batch_size]
            if len(batch) < batch_size:
                batch = train_samples[:batch_size]
            t0 = time.time()
            core.train_batch(batch)
            total_ms += (time.time() - t0) * 1000
            step += len(batch)
        else:
            intensity, label = train_samples[step % len(train_samples)]
            t0 = time.time()
            core.train(intensity, label)
            total_ms += (time.time() - t0) * 1000
            step += 1

        # 周期性评估：基于实际步数，而非 batch 对齐
        if step >= next_eval_step or step >= max_steps:
            acc = evaluate_accuracy(core, test_samples)
            accuracy_history.append((step, acc))

            # 显示步数（capped，batch 跳步可能超出）
            display_step = min(step, max_steps)

            # 速度 & ETA
            elapsed = time.time() - t_start
            steps_per_sec = step / max(1, elapsed)
            eta = max(0, (max_steps - step) / max(1, steps_per_sec))

            # 进度条
            bar_width = 20
            filled = min(bar_width, int(bar_width * display_step / max_steps))
            bar = "█" * filled + "░" * (bar_width - filled)
            pct = min(100, display_step * 100 // max_steps)

            # 趋势箭头
            if len(accuracy_history) >= 2:
                prev_acc = accuracy_history[-2][1]
                trend = "↑" if acc > prev_acc + 0.005 else ("↓" if acc < prev_acc - 0.005 else "→")
            else:
                trend = ""

            # 输出
            if _multi_line:
                print(f"  [{test_index}/{total_tests}] {label} | {bar} {pct:>3}% | "
                      f"步={display_step}/{max_steps} | acc={acc:.1%}{trend} | "
                      f"{steps_per_sec:.0f}步/秒 | 剩余 {eta:.0f}秒")
            else:
                line = (f"\r  [{test_index}/{total_tests}] {label} | {bar} {pct:>3}% | "
                        f"步={display_step}/{max_steps} | acc={acc:.1%}{trend} | "
                        f"{steps_per_sec:.0f}步/秒 | 剩余 {eta:.0f}秒  ")
                sys.stdout.write(line)
                sys.stdout.flush()

            if first_target_step is None and acc >= target_acc:
                first_target_step = step
            next_eval_step += eval_interval

    final_acc = accuracy_history[-1][1] if accuracy_history else 0.0
    # 换行结束进度条
    if _multi_line:
        print(f"  [{test_index}/{total_tests}] {label} 完成")
    else:
        print()
    state = core.get_state()
    participation_rate = state.get('avg_participation_rate', 0.0) if multi_layer else 0.0
    avg_step_ms = total_ms / step if step > 0 else 0.0

    return {
        'first_target_step': first_target_step,
        'final_acc': final_acc,
        'accuracy_history': accuracy_history,
        'avg_step_ms': round(avg_step_ms, 3),
        'participation_rate': round(participation_rate, 4),
    }


# ============================================================
# 测试 1: 噪声鲁棒性
# ============================================================

def test_noise_robustness(d=64, max_steps=2000):
    """测试不同噪声等级下的收敛速度"""
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
    total = len(noise_levels)
    print(f"\n[测试1] 噪声鲁棒性")
    print(f"  参数: d={d}, max_steps={max_steps}, noise_levels={noise_levels}")
    print(f"  [{total} 组，预计 {total * max_steps} 步，按 Ctrl+C 可中断]")
    print("  运行中...")

    results = []
    for i, noise_prob in enumerate(noise_levels):
        from engine.config import CONFIG, ConfigRegistry
        ConfigRegistry._values["FLIP_PROB"] = noise_prob
        CONFIG["FLIP_PROB"] = noise_prob

        # 生成训练集和测试集
        train_samples = generate_synthetic_samples(
            num_labels=4, samples_per_label=50, d=d, seed=42)
        test_samples = generate_synthetic_samples(
            num_labels=4, samples_per_label=20, d=d, seed=999)

        core = _make_core({"D": d})
        r = _train_and_track(core, train_samples, test_samples, max_steps,
                             label=f"noise={noise_prob:.2f}",
                             test_index=i + 1, total_tests=total)
        r['noise_prob'] = noise_prob
        results.append(r)
        status = f"步数={r['first_target_step']}" if r['first_target_step'] else "未收敛"
        print(f"  noise={noise_prob:.2f} → {status}, final_acc={r['final_acc']:.1%}")

    return results


# ============================================================
# 测试 2: 网络规模
# ============================================================

def test_network_scale(d=64, max_steps=2000):
    """测试不同神经元数量下的收敛速度"""
    scales = [64, 128, 256, 512, 1024]
    total = len(scales)
    print(f"\n[测试2] 网络规模")
    print(f"  参数: d={d}, max_steps={max_steps}, scales={scales}")
    print(f"  [{total} 组，预计 {total * max_steps} 步，按 Ctrl+C 可中断]")
    print("  运行中...")

    results = []
    for i, neuron_count in enumerate(scales):
        from engine.config import CONFIG, ConfigRegistry
        # 多层模式：L0 和 L1 各占一半
        l0 = neuron_count // 2
        l1 = neuron_count - l0
        ConfigRegistry._values["NEURON_LAYER_0_COUNT"] = l0
        ConfigRegistry._values["NEURON_LAYER_1_COUNT"] = l1
        CONFIG["NEURON_LAYER_0_COUNT"] = l0
        CONFIG["NEURON_LAYER_1_COUNT"] = l1

        train_samples = generate_synthetic_samples(
            num_labels=4, samples_per_label=50, d=d, seed=42)
        test_samples = generate_synthetic_samples(
            num_labels=4, samples_per_label=20, d=d, seed=999)

        core = _make_core({"D": d, "NEURON_LAYER_0_COUNT": l0, "NEURON_LAYER_1_COUNT": l1})
        r = _train_and_track(core, train_samples, test_samples, max_steps,
                             label=f"neurons={neuron_count}",
                             test_index=i + 1, total_tests=total)
        r['neuron_count'] = neuron_count
        results.append(r)
        status = f"步数={r['first_target_step']}" if r['first_target_step'] else "未收敛"
        print(f"  neurons={neuron_count} → {status}, final_acc={r['final_acc']:.1%}, "
              f"avg_step={r['avg_step_ms']}ms")

    return results


# ============================================================
# 测试 3: 多层 vs 单层
# ============================================================

def test_multi_vs_single(d=64, max_steps=2000):
    """测试多层模式和单层模式的收敛对比"""
    modes = [("多层", True), ("单层", False)]
    total = len(modes)
    print(f"\n[测试3] 多层 vs 单层")
    print(f"  参数: d={d}, max_steps={max_steps}")
    print(f"  [{total} 组，预计 {total * max_steps} 步，按 Ctrl+C 可中断]")
    print("  运行中...")

    from engine.config import CONFIG, ConfigRegistry
    results = []

    for i, (mode_name, multi_layer) in enumerate(modes):
        ConfigRegistry._values["ENABLE_MULTI_LAYER_NEURON"] = multi_layer
        CONFIG["ENABLE_MULTI_LAYER_NEURON"] = multi_layer

        train_samples = generate_synthetic_samples(
            num_labels=4, samples_per_label=50, d=d, seed=42)
        test_samples = generate_synthetic_samples(
            num_labels=4, samples_per_label=20, d=d, seed=999)

        core = _make_core({"D": d, "ENABLE_MULTI_LAYER_NEURON": multi_layer})
        r = _train_and_track(core, train_samples, test_samples, max_steps,
                             label=f"{mode_name}",
                             test_index=i + 1, total_tests=total)
        r['mode'] = mode_name
        results.append(r)
        status = f"步数={r['first_target_step']}" if r['first_target_step'] else "未收敛"
        print(f"  {mode_name} → {status}, final_acc={r['final_acc']:.1%}, "
              f"avg_step={r['avg_step_ms']}ms")

    return results


# ============================================================
# 测试 4: 静默开启 vs 关闭
# ============================================================

def test_silence_on_off(d=64, max_steps=2000):
    """测试静默机制开启和关闭的收敛对比"""
    modes = [("静默开启", True), ("静默关闭", False)]
    total = len(modes)
    print(f"\n[测试4] 静默开启 vs 关闭")
    print(f"  参数: d={d}, max_steps={max_steps}")
    print(f"  [{total} 组，预计 {total * max_steps} 步，按 Ctrl+C 可中断]")
    print("  运行中...")

    from engine.config import CONFIG, ConfigRegistry
    results = []

    for i, (silence_name, silence_on) in enumerate(modes):
        ConfigRegistry._values["ENABLE_ADAPTIVE_SILENCE"] = silence_on
        CONFIG["ENABLE_ADAPTIVE_SILENCE"] = silence_on

        train_samples = generate_synthetic_samples(
            num_labels=4, samples_per_label=50, d=d, seed=42)
        test_samples = generate_synthetic_samples(
            num_labels=4, samples_per_label=20, d=d, seed=999)

        core = _make_core({"D": d, "ENABLE_ADAPTIVE_SILENCE": silence_on})
        r = _train_and_track(core, train_samples, test_samples, max_steps,
                             label=f"{silence_name}",
                             test_index=i + 1, total_tests=total)
        r['silence'] = silence_name
        results.append(r)
        status = f"步数={r['first_target_step']}" if r['first_target_step'] else "未收敛"
        print(f"  {silence_name} → {status}, final_acc={r['final_acc']:.1%}, "
              f"参与率={r['participation_rate']:.1%}")

    return results


# ============================================================
# 报告输出
# ============================================================

def _acc_color(acc):
    """准确率颜色: >=90%绿, >=50%黄, <50%红"""
    from engine.utils import C
    if acc >= 0.9:
        return C.GRN
    elif acc >= 0.5:
        return C.YEL
    else:
        return C.RED


def _step_str(r):
    """收敛步数字符串（带颜色）"""
    from engine.utils import C
    if r['first_target_step'] is not None:
        return f"{C.GRN}{r['first_target_step']}{C.RST}"
    return f"{C.RED}未收敛{C.RST}"


def _print_table(title, headers, rows, summary=None):
    """打印带颜色的表格

    Args:
        title: 表格标题
        headers: 列名列表
        rows: 行数据列表（每行是已格式化的字符串列表）
        summary: 摘要字符串（可选）
    """
    from engine.utils import C
    import re, unicodedata

    def _visible_len(s):
        """可见显示宽度（中文占2列，ANSI码不计）"""
        s = re.sub(r'\033\[[0-9;]*m', '', s)
        return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in s)

    def _pad(s, width):
        """按可见长度填充空格（处理 ANSI 颜色码）"""
        pad_count = width - _visible_len(s)
        return s + " " * max(0, pad_count)

    col_widths = [_visible_len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            cl = _visible_len(cell)
            if cl > col_widths[i]:
                col_widths[i] = cl

    # 打印
    print(f"\n  {C.BOLD}{C.CYN}[{title}]{C.RST}")
    header_line = " │ ".join(_pad(h, col_widths[i]) for i, h in enumerate(headers))
    sep_len = _visible_len(header_line)
    print(f"  {C.DIM}{'─' * sep_len}{C.RST}")
    print(f"  {C.BOLD}{header_line}{C.RST}")
    print(f"  {C.DIM}{'─' * sep_len}{C.RST}")
    for row in rows:
        line = " │ ".join(_pad(cell, col_widths[i]) for i, cell in enumerate(row))
        print(f"  {line}")
    if summary:
        print(f"  {C.DIM}{'─' * sep_len}{C.RST}")
        print(f"  {C.DIM}摘要: {summary}{C.RST}")


def print_report(all_results):
    """打印控制台表格摘要（v5.1.7 优化: 带颜色+box表格+摘要行）"""
    from engine.utils import C

    print(f"\n  {C.BOLD}{C.CYN}╔{'═'*56}╗{C.RST}")
    print(f"  {C.BOLD}{C.CYN}║{C.RST} {C.BOLD}{'SGN 收敛基准测试报告':^54}{C.RST} {C.BOLD}{C.CYN}║{C.RST}")
    print(f"  {C.BOLD}{C.CYN}╚{'═'*56}╝{C.RST}")

    if 'test1' in all_results:
        headers = ['noise_prob', '收敛步数', '最终准确率', '每步耗时']
        rows = []
        converged = 0
        best = None
        for r in all_results['test1']:
            acc_c = _acc_color(r['final_acc'])
            rows.append([
                f"{r['noise_prob']:.2f}",
                _step_str(r),
                f"{acc_c}{r['final_acc']:.1%}{C.RST}",
                f"{r['avg_step_ms']}ms",
            ])
            if r['first_target_step'] is not None:
                converged += 1
                if best is None or r['first_target_step'] < best[1]:
                    best = (r['noise_prob'], r['first_target_step'])
        summary = f"{converged}/{len(rows)} 收敛"
        if best:
            summary += f", 最佳 noise={best[0]:.2f}({best[1]}步)"
        _print_table("测试1 噪声鲁棒性", headers, rows, summary)

    if 'test2' in all_results:
        headers = ['neurons', '收敛步数', '最终准确率', '每步耗时']
        rows = []
        converged = 0
        for r in all_results['test2']:
            acc_c = _acc_color(r['final_acc'])
            rows.append([
                str(r['neuron_count']),
                _step_str(r),
                f"{acc_c}{r['final_acc']:.1%}{C.RST}",
                f"{r['avg_step_ms']}ms",
            ])
            if r['first_target_step'] is not None:
                converged += 1
        summary = f"{converged}/{len(rows)} 收敛"
        _print_table("测试2 网络规模", headers, rows, summary)

    if 'test3' in all_results:
        headers = ['模式', '收敛步数', '最终准确率', '每步耗时']
        rows = []
        for r in all_results['test3']:
            acc_c = _acc_color(r['final_acc'])
            rows.append([
                r['mode'],
                _step_str(r),
                f"{acc_c}{r['final_acc']:.1%}{C.RST}",
                f"{r['avg_step_ms']}ms",
            ])
        # 找出更优的模式
        converged_modes = [r for r in all_results['test3'] if r['first_target_step'] is not None]
        if len(converged_modes) == 2:
            faster = min(converged_modes, key=lambda x: x['first_target_step'])
            summary = f"更优: {faster['mode']}({faster['first_target_step']}步)"
        elif len(converged_modes) == 1:
            summary = f"仅 {converged_modes[0]['mode']} 收敛"
        else:
            summary = "均未收敛"
        _print_table("测试3 多层 vs 单层", headers, rows, summary)

    if 'test4' in all_results:
        headers = ['静默', '收敛步数', '最终准确率', '参与率']
        rows = []
        for r in all_results['test4']:
            acc_c = _acc_color(r['final_acc'])
            rows.append([
                r['silence'],
                _step_str(r),
                f"{acc_c}{r['final_acc']:.1%}{C.RST}",
                f"{r['participation_rate']:.1%}",
            ])
        converged_modes = [r for r in all_results['test4'] if r['first_target_step'] is not None]
        if len(converged_modes) == 2:
            faster = min(converged_modes, key=lambda x: x['first_target_step'])
            summary = f"更优: {faster['silence']}({faster['first_target_step']}步)"
        elif len(converged_modes) == 1:
            summary = f"仅 {converged_modes[0]['silence']} 收敛"
        else:
            summary = "均未收敛"
        _print_table("测试4 静默开启 vs 关闭", headers, rows, summary)

    print()


def save_json_report(all_results, results_dir=None):
    """保存 JSON 详细报告"""
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), "benchmark_results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f"benchmark_{timestamp}.json")

    # 序列化（处理 None 和 tuple）
    def _serialize(obj):
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_serialize(v) for v in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        return obj

    report = {
        'timestamp': timestamp,
        'results': _serialize(all_results),
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nJSON 报告已保存: {filepath}")
    return filepath


def list_history_results(results_dir=None):
    """列出历史基准测试结果"""
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), "benchmark_results")
    if not os.path.exists(results_dir):
        print("  暂无历史结果")
        return []

    files = sorted([f for f in os.listdir(results_dir) if f.endswith('.json')],
                   reverse=True)
    if not files:
        print("  暂无历史结果")
        return []

    print("\n=== 历史基准测试结果 ===")
    for i, f in enumerate(files[:10]):  # 最近 10 个
        filepath = os.path.join(results_dir, f)
        try:
            with open(filepath, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            ts = data.get('timestamp', f)
            print(f"  [{i+1}] {ts}  ({f})")
        except Exception:
            print(f"  [{i+1}] {f} (读取失败)")

    return files[:10]


def show_history_detail(results_dir=None):
    """查看历史结果详情"""
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), "benchmark_results")
    files = list_history_results(results_dir)
    if not files:
        return

    try:
        choice = input("\n选择编号查看详情（回车返回）: ").strip()
        if not choice:
            return
        idx = int(choice) - 1
        if idx < 0 or idx >= len(files):
            print("  无效编号")
            return
        filepath = os.path.join(results_dir, files[idx])
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("\n" + "=" * 60)
        print(f"报告时间: {data.get('timestamp', '?')}")
        print("=" * 60)
        print_report(data.get('results', {}))
    except (ValueError, IndexError):
        print("  无效输入")


# ============================================================
# 交互式菜单
# ============================================================

def interactive_menu():
    """交互式基准测试菜单"""
    from engine.config import CONFIG

    while True:
        from engine.utils import C, box, hr
        box("基准测试")

        # 显示当前网络配置
        multi = CONFIG.get("ENABLE_MULTI_LAYER_NEURON", False)
        mode_str = "多层模式" if multi else "单层模式"
        neuron_count = CONFIG.get("MAX_NEURONS", 0)
        if multi:
            l0 = CONFIG.get("NEURON_LAYER_0_COUNT", 0)
            l1 = CONFIG.get("NEURON_LAYER_1_COUNT", 0)
            neuron_count = l0 + l1
        noise = CONFIG.get("NOISE_TYPE", "composite")
        silence = "开" if CONFIG.get("ENABLE_ADAPTIVE_SILENCE", True) else "关"

        print(f"  当前网络: {mode_str}  神经元: {neuron_count}  噪声: {noise}  静默: {silence}")
        hr(50)
        print(f"  {C.CYN}[1]{C.RST}噪声鲁棒性  (5级: 0.0~0.3)")
        print(f"  {C.CYN}[2]{C.RST}网络规模    (5档: 64~1024)")
        print(f"  {C.CYN}[3]{C.RST}多层vs单层  (2组对比)")
        print(f"  {C.CYN}[4]{C.RST}静默开vs关  (2组对比)")
        print(f"  {C.CYN}[a]{C.RST}全部运行    {C.CYN}[h]{C.RST}历史结果")
        print(f"  {C.RED}[q]{C.RST}返回扩展功能")
        hr(50)

        try:
            choice = input("选择: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return

        if choice == "q" or choice == "":
            return

        all_results = {}
        if choice == "1":
            all_results['test1'] = test_noise_robustness()
        elif choice == "2":
            all_results['test2'] = test_network_scale()
        elif choice == "3":
            all_results['test3'] = test_multi_vs_single()
        elif choice == "4":
            all_results['test4'] = test_silence_on_off()
        elif choice == "a":
            all_results['test1'] = test_noise_robustness()
            all_results['test2'] = test_network_scale()
            all_results['test3'] = test_multi_vs_single()
            all_results['test4'] = test_silence_on_off()
        elif choice == "h":
            show_history_detail()
            continue
        else:
            print(f"  {C.RED}未知命令: '{choice}'{C.RST}")
            continue

        if all_results:
            print_report(all_results)
            save_json_report(all_results)


# ============================================================
# 命令行入口
# ============================================================

def main():
    """命令行入口"""
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--all":
            all_results = {}
            all_results['test1'] = test_noise_robustness()
            all_results['test2'] = test_network_scale()
            all_results['test3'] = test_multi_vs_single()
            all_results['test4'] = test_silence_on_off()
            print_report(all_results)
            save_json_report(all_results)
            return
        elif arg == "--test" and len(sys.argv) > 2:
            test_id = sys.argv[2]
            all_results = {}
            if test_id == "1":
                all_results['test1'] = test_noise_robustness()
            elif test_id == "2":
                all_results['test2'] = test_network_scale()
            elif test_id == "3":
                all_results['test3'] = test_multi_vs_single()
            elif test_id == "4":
                all_results['test4'] = test_silence_on_off()
            else:
                print(f"未知测试编号: {test_id}（可选 1-4）")
                return
            print_report(all_results)
            save_json_report(all_results)
            return
        elif arg in ("--help", "-h"):
            print(__doc__)
            return

    # 默认交互式
    interactive_menu()


if __name__ == "__main__":
    main()
