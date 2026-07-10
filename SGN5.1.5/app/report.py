#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 [zhugy-8086]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SGN-Lite v5.0 报告与图表导出模块 —— 从 sgn_visual 拆分

包含：图表导出中心、matplotlib 遗留函数、模型导出
"""

from __future__ import annotations

import os
from typing import Optional


def do_plot(core):
    """图表导出中心 - v4.3 使用 BackendRegistry 自动选择后端"""
    from app.backends import BackendRegistry
    from engine.config import CONFIG
    from engine.utils import C, hr

    t = len(core.history)
    if t == 0:
        print(f"\n  {C.warn('⚠')} 无历史数据，无法绘制曲线")
        return

    cfg_backend = CONFIG.get("CHART_BACKEND", "auto")
    if cfg_backend != "auto":
        backend = BackendRegistry.get(cfg_backend)
        if backend is None:
            backend = BackendRegistry.auto_select()
            print(f"  {C.YEL}⚠ 配置后端 '{cfg_backend}' 不可用，已回退到 auto{C.RST}")
    else:
        backend = BackendRegistry.auto_select()
    has_mpl = BackendRegistry.get("matplotlib") is not None and BackendRegistry.get("matplotlib").is_available()

    print(f"\n  {C.BOLD}图表导出中心{C.RST}  (当前后端: {C.val(backend.name)})")
    hr(46)
    if backend.name == "ascii" and not has_mpl:
        print(f"  {C.YEL}⚠ matplotlib 未安装，已降级为 ASCII 模式{C.RST}")
        print(f"  {C.info('ℹ')} 安装命令: pip install matplotlib{C.RST}")
    print(f"  {C.CYN}[1]{C.RST} 累计准确率曲线")
    print(f"  {C.CYN}[2]{C.RST} ASCII 学习曲线  (终端显示)")
    if has_mpl:
        print(f"  {C.CYN}[3]{C.RST} 神经元状态分布  (matplotlib)")
        print(f"  {C.CYN}[4]{C.RST} 模板增长曲线    (matplotlib)")
        print(f"  {C.CYN}[5]{C.RST} 综合面板        (matplotlib)")
    print(f"  {C.CYN}[6]{C.RST} CSV 数据导出")
    print(f"  {C.RED}[q]{C.RST} 取消")
    hr(46)

    try:
        choice = input("  选择图表 [1-6/q]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return

    if choice == "q" or choice == "":
        print(f"  {C.DIM}已取消{C.RST}")
        return

    if choice == "1":
        path = backend.plot_accuracy(core.history)
        if path:
            print(f"  {C.ok('✓')} 已保存: {C.val(path)}")
    elif choice == "2":
        ascii_backend = BackendRegistry.get("ascii") or backend
        ascii_backend.plot_ascii(core.history)
    elif choice == "3":
        if has_mpl:
            mpl = BackendRegistry.get("matplotlib")
            path = mpl.plot_neurons(core)
            if path:
                print(f"  {C.ok('✓')} 已保存: {C.val(path)}")
        else:
            print(f"  {C.err('✗')} 该图表需要 matplotlib")
    elif choice == "4":
        if has_mpl:
            mpl = BackendRegistry.get("matplotlib")
            path = mpl.plot_templates(core.history)
            if path:
                print(f"  {C.ok('✓')} 已保存: {C.val(path)}")
        else:
            print(f"  {C.err('✗')} 该图表需要 matplotlib")
    elif choice == "5":
        if has_mpl:
            mpl = BackendRegistry.get("matplotlib")
            path = mpl.plot_comprehensive(core.history, core)
            if path:
                print(f"  {C.ok('✓')} 已保存: {C.val(path)}")
        else:
            print(f"  {C.err('✗')} 该图表需要 matplotlib")
    elif choice == "6":
        csv_backend = BackendRegistry.get("csv")
        if csv_backend:
            path = csv_backend.plot_accuracy(core.history)
            if path:
                print(f"  {C.ok('✓')} 已导出 CSV: {C.val(path)}")
    else:
        print(f"  {C.err('✗')} 无效选择: {choice}")


def do_export(core):
    """导出详细模型报告"""
    from app.persist import save_model
    from engine.config import CONFIG
    from engine.utils import C

    st = core.get_state()
    print(f"\n  {C.BOLD}模型状态导出{C.RST}")
    print(f"  神经元: {C.val(st['active'])}活跃 {C.val(st['locked'])}锁定 {C.val(st['encouraged'])}鼓励中")
    print(f"  模板: {C.val(st['templates'])}/{CONFIG['MAX_TEMPLATES']}")
    save_model(core)


# ============================================================
# matplotlib 遗留函数（内部使用，不直接暴露给用户）
# ============================================================

def _plot_accuracy(core):
    """图表1: 累计准确率曲线"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = len(core.history)
    steps = list(range(1, t + 1))
    ver = [1 if x["V"] else 0 for x in core.history]
    acc = []
    s = 0
    for i, v_step in enumerate(ver):
        s += v_step
        acc.append(s / (i + 1) * 100)
    plt.figure(figsize=(10, 5))
    plt.plot(steps, acc, "g-", linewidth=0.8, label="Cumulative Accuracy")
    plt.xlabel("Step")
    plt.ylabel("Accuracy (%)")
    plt.title("SGN-Lite v5.0 累计准确率曲线")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 105)
    path = os.path.normpath("sgn_plot_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()

    from engine.utils import C
    print(f"  {C.ok('✓')} 已保存: {C.val(path)}")


def _plot_ascii(core):
    """图表2: ASCII 终端学习曲线"""
    t = len(core.history)
    v = sum(1 for x in core.history if x["V"])
    pct = v / t * 100

    from engine.utils import C

    print(f"\n  {C.BOLD}ASCII 学习曲线 (最近100步){C.RST}")
    recent = core.history[-100:]
    chunk_size = 10
    for i in range(0, len(recent), chunk_size):
        chunk = recent[i:i+chunk_size]
        v_count = sum(1 for x in chunk if x["V"])
        bar = f"{C.GRN}{'█'*v_count}{C.RST}{C.DIM}{'░'*(chunk_size-v_count)}{C.RST}"
        start_step = max(1, t - len(recent) + i + 1)
        end_step = start_step + len(chunk) - 1
        print(f"  步{start_step:>4}-{end_step:>4}: [{bar}] {v_count:>2}/{len(chunk)}")
    print(f"  累计: {C.GRN}{v}{C.RST}/{t} ({pct:.1f}%)")


def _to_float(val):
    """将 DiscreteCoordinate 或数值转为 float（仅可视化用）"""
    if hasattr(val, 'index') and hasattr(val, 'scale'):
        return val.index / val.scale
    return float(val)


def _plot_neurons(core):
    """图表3: 神经元状态分布"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    active = [_to_float(n["base"]) for n in core.N if not n["L"]]
    locked = [_to_float(n["base"]) for n in core.N if n["L"]]
    labels = ["Active", "Locked"]
    counts = [len(active), len(locked)]
    avg_base_active = sum(active) / len(active) if active else 0
    avg_base_locked = sum(locked) / len(locked) if locked else 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#2ecc71", "#e74c3c"]
    axes[0].bar(labels, counts, color=colors, edgecolor="black")
    axes[0].set_ylabel("Neuron Count")
    axes[0].set_title("神经元活跃/锁定数量")
    for i, v in enumerate(counts):
        axes[0].text(i, v + 1, str(v), ha="center", va="bottom", fontweight="bold")
    data = [active, locked] if active and locked else ([active] if active else [locked])
    lbls = (["Active", "Locked"] if active and locked else
            (["Active"] if active else ["Locked"]))
    axes[1].boxplot(data, labels=lbls)
    axes[1].set_ylabel("Base Speed")
    axes[1].set_title("基础速度分布")
    plt.tight_layout()
    path = os.path.normpath("sgn_plot_neurons.png")
    plt.savefig(path, dpi=150)
    plt.close()

    from engine.utils import C
    print(f"  {C.ok('✓')} 已保存: {C.val(path)}")
    print(f"  {C.DIM}   活跃神经元平均速度: {avg_base_active:.3f}  锁定: {avg_base_locked:.3f}{C.RST}")


def _plot_templates(core):
    """图表4: 模板增长与校验通过率"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = len(core.history)
    steps = list(range(1, t + 1))
    tpl_counts = [info.get("templates", 0) for info in core.history]
    window = 50
    ver_rates = []
    for i in range(t):
        start = max(0, i - window + 1)
        chunk = core.history[start:i+1]
        rate = sum(1 for x in chunk if x["V"]) / len(chunk) * 100 if chunk else 0
        ver_rates.append(rate)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color1 = "#3498db"
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Template Count", color=color1)
    ax1.plot(steps, tpl_counts, color=color1, linewidth=1.2, label="Templates")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "#e67e22"
    ax2.set_ylabel("Verify Rate (%)", color=color2)
    ax2.plot(steps, ver_rates, color=color2, linewidth=1.0, linestyle="--", label="Verify Rate")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 105)

    plt.title("SGN-Lite v5.0 模板增长与校验通过率")
    fig.tight_layout()
    path = os.path.normpath("sgn_plot_templates.png")
    plt.savefig(path, dpi=150)
    plt.close()

    from engine.utils import C
    print(f"  {C.ok('✓')} 已保存: {C.val(path)}")


def _plot_comprehensive(core):
    """图表5: 综合面板 (2×2 子图)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = len(core.history)
    steps = list(range(1, t + 1))
    ver = [1 if x["V"] else 0 for x in core.history]
    acc = []
    s = 0
    for i, v_step in enumerate(ver):
        s += v_step
        acc.append(s / (i + 1) * 100)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(steps, acc, "g-", linewidth=0.8)
    axes[0, 0].set_title("累计准确率")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 105)

    tpl_counts = [info.get("templates", 0) for info in core.history]
    axes[0, 1].plot(steps, tpl_counts, "b-", linewidth=1.0)
    axes[0, 1].set_title("模板库增长")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Template Count")
    axes[0, 1].grid(True, alpha=0.3)

    active_hist = [info.get("active", 0) for info in core.history]
    locked_hist = [info.get("locked", 0) for info in core.history]
    axes[1, 0].plot(steps, active_hist, "g-", linewidth=0.8, label="Active")
    axes[1, 0].plot(steps, locked_hist, "r-", linewidth=0.8, label="Locked")
    axes[1, 0].set_title("神经元状态变化")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    recent = core.history[-50:]
    r_steps = list(range(t - len(recent) + 1, t + 1))
    r_ver = [1 if x["V"] else 0 for x in recent]
    axes[1, 1].scatter(r_steps, r_ver, c=["g" if v else "r" for v in r_ver], s=20, alpha=0.7)
    axes[1, 1].set_title("最近50步校验结果")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Verified (1=Yes, 0=No)")
    axes[1, 1].set_ylim(-0.2, 1.2)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("SGN-Lite v5.0 综合监控面板", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.normpath("sgn_plot_comprehensive.png")
    plt.savefig(path, dpi=150)
    plt.close()

    from engine.utils import C
    print(f"  {C.ok('✓')} 已保存: {C.val(path)}")
    print(f"  {C.DIM}   包含: 准确率曲线 | 模板增长 | 神经元状态 | 最近校验散点{C.RST}")
