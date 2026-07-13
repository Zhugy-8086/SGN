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
"""SGN-Lite v5.0 可视化模块 —— 拆分后瘦身版

职责：
  - 模板可视化 (do_visualize)
  - 热力图 (do_heatmap)
  - 统计信息 (do_stats)
  - 仪表盘 (do_gauge)

注意：所有测试/报告/绘图函数已拆分到：
  sgn_test.py    — 推理/批量/混淆/噪声测试
  sgn_report.py  — 图表导出/报告
  sgn_draw.py    — 基础绘图工具
"""

from __future__ import annotations

from typing import Optional


def do_visualize(core):
    """ASCII模板可视化（分页 + 多层 OR 合并 + 强度模式）"""
    from app.draw import draw_binary_grid
    from engine.config import CONFIG
    from engine.layers import extract_layers, combine_layers
    from engine.utils import C, clear_stdin_buffer

    graph_mode = getattr(core, 'graph_mode', False)

    def _classify_shape(mask, grid_size):
        """分析掩码结构，标注 filled/ring/line/unknown"""
        if mask == 0:
            return "empty"
        # 找边界框
        rows, cols = [], []
        for i in range(grid_size * grid_size):
            if (mask >> i) & 1:
                rows.append(i // grid_size)
                cols.append(i % grid_size)
        if not rows:
            return "empty"
        r_min, r_max = min(rows), max(rows)
        c_min, c_max = min(cols), max(cols)
        h = r_max - r_min + 1
        w = c_max - c_min + 1
        bits = bin(mask).count('1')
        area = h * w
        if area == 0:
            return "unknown"
        fill_ratio = bits / area
        aspect = h / w if w > 0 else 99
        # 判断形状
        if aspect > 3 or aspect < 0.33:
            return "line"
        if fill_ratio > 0.6:
            return "filled"
        elif fill_ratio < 0.45 and bits >= 6:
            return "ring"
        return "unknown"

    if graph_mode:
        # 图模式：显示图结构信息
        if not hasattr(core, 'graphs') or not core.graphs:
            print(f"  {C.warn('⚠')} 无图数据可显示")
            return
        print(f"\n  {C.BOLD}图结构可视化{C.RST} {C.CYN}[图模式]{C.RST}")
        for label, g in sorted(core.graphs.items()):
            print(f"\n  {C.BOLD}标签 '{label}'{C.RST}: {g.get_total_nodes()}节点, 最高层 L{g.get_max_layer()}")
            layer_dist = {}
            for n in g.nodes.values():
                layer_dist[n.layer] = layer_dist.get(n.layer, 0) + 1
            for layer, count in sorted(layer_dist.items()):
                print(f"    L{layer}: {count}个节点")
            # 显示前5个L0节点的特征
            l0_nodes = g.get_layer_nodes(0)[:5]
            if l0_nodes:
                print(f"    L0节点示例 (前{len(l0_nodes)}个):")
                for n in l0_nodes:
                    pos = n.position_norm
                    feat_str = ",".join(f"L{dc.level}:I{dc.index}" for dc in n.feature_vector[:3])
                    print(f"      node{n.node_id}: pos=({pos[0]},{pos[1]}) act={n.activation} feat=[{feat_str}...]")
        return

    if not core.templates:
        print(f"  {C.warn('⚠')} 无模板可显示")
        return

    d = getattr(core, 'D', CONFIG.get("D", 64))
    gs = int(d**0.5)
    total = len(core.templates)
    page_size = 10
    page = 0
    intensity_mode = False

    sorted_tmpls = sorted(core.templates, key=lambda x: x[2], reverse=True)
    total_pages = (total + page_size - 1) // page_size

    def _draw_template_intensity(mask, hc, grid_size, indent="    "):
        """用 hit_count 着色的强度风格模板渲染"""
        for r in range(grid_size):
            row = []
            for c in range(grid_size):
                idx = r * grid_size + c
                on = (mask >> idx) & 1
                if not on:
                    row.append(f"{C.DIM}··{C.RST}")
                elif hc >= 200:
                    row.append(f"{C.BOLD}{C.WHT}██{C.RST}")
                elif hc >= 150:
                    row.append(f"{C.WHT}██{C.RST}")
                elif hc >= 100:
                    row.append(f"{C.DIM}{C.WHT}▓▓{C.RST}")
                elif hc >= 50:
                    row.append(f"{C.DIM}▒▒{C.RST}")
                else:
                    row.append(f"{C.DIM}░░{C.RST}")
            print(f"{indent}{''.join(row)}")

    while True:
        start = page * page_size
        end = min(start + page_size, total)
        mode_tag = f" {C.CYN}[强度模式]{C.RST}" if intensity_mode else ""
        print(f"\n  {C.BOLD}模板可视化{C.RST} ({gs}×{gs}点阵){mode_tag} "
              f"{C.DIM}[{start+1}-{end}/{total} 页{page+1}/{total_pages}]{C.RST}")

        for idx, (lb, tm, sc, hc) in enumerate(sorted_tmpls[start:end], start=start+1):
            shape = _classify_shape(tm, gs)
            shape_tag = {"filled": "filled", "ring": "ring", "line": "line"}.get(shape, "")
            tag_str = f" {C.CYN}[{shape_tag}]{C.RST}" if shape_tag else ""
            print(f"  {C.BOLD}{lb}{C.RST} #{idx} sc={sc} hc={hc}{tag_str}")
            if intensity_mode:
                _draw_template_intensity(tm, hc, gs)
            else:
                draw_binary_grid(tm, grid_size=gs)
            print()

        print(f"  {C.DIM}[n]下一页 [p]上一页 [a]显示全部 [i]强度/二值切换 [q]退出{C.RST}")
        if intensity_mode:
            print(f"  {C.DIM}【强度模式】█=高命中(hc≥200) ▓=中(≥100) ▒=低(≥50) ░=极低 <50 ··=off{C.RST}")
        else:
            print(f"  {C.DIM}【提示】模板仅显示单层 OR 合并掩码。按 [i] 切换强度模式查看命中置信度。{C.RST}")
        try:
            clear_stdin_buffer()
            cmd = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return

        if cmd == "n":
            page = min(page + 1, total_pages - 1)
        elif cmd == "p":
            page = max(page - 1, 0)
        elif cmd == "i":
            intensity_mode = not intensity_mode
        elif cmd == "a":
            for idx, (lb, tm, sc, hc) in enumerate(sorted_tmpls, 1):
                shape = _classify_shape(tm, gs)
                shape_tag = {"filled": "filled", "ring": "ring", "line": "line"}.get(shape, "")
                tag_str = f" {C.CYN}[{shape_tag}]{C.RST}" if shape_tag else ""
                print(f"  {C.BOLD}{lb}{C.RST} #{idx} sc={sc} hc={hc}{tag_str}")
                if intensity_mode:
                    _draw_template_intensity(tm, hc, gs)
                else:
                    draw_binary_grid(tm, grid_size=gs)
                print()
            print(f"  {C.DIM}共 {total} 个模板已全部显示{C.RST}")
            return
        elif cmd == "q":
            return


def do_heatmap(core):
    """频率热力图（窗口大小自适应）"""
    from engine.config import CONFIG
    from engine.utils import C

    if getattr(core, 'graph_mode', False):
        print(f"\n  {C.BOLD}热力图{C.RST}")
        print(f"  {C.YEL}⚠ 图模式下热力图暂不支持（模板为空）{C.RST}")
        if hasattr(core, 'graphs') and core.graphs:
            total_nodes = sum(g.get_total_nodes() for g in core.graphs.values())
            print(f"  {C.DIM}当前有 {len(core.graphs)} 个图，共 {total_nodes} 个节点{C.RST}")
        return

    counts = {}
    d = getattr(core, 'D', CONFIG.get("D", 64))
    grid_size = int(d ** 0.5)
    for lb, tm, sc, hc in core.templates:
        for i in range(d):
            if (tm >> i) & 1:
                counts[i] = counts.get(i, 0) + 1
    print(f"\n  {C.BOLD}模板频率热力图{C.RST}")
    max_c = max(counts.values()) if counts else 1
    for r in range(grid_size):
        row = []
        for c in range(grid_size):
            idx = r * grid_size + c
            v = counts.get(idx, 0)
            if v > max_c * 0.90:
                ch = f"{C.RED}██{C.RST}"
            elif v > max_c * 0.75:
                ch = f"{C.MAG}██{C.RST}"
            elif v > max_c * 0.60:
                ch = f"{C.YEL}██{C.RST}"
            elif v > max_c * 0.45:
                ch = f"{C.GRN}▓▓{C.RST}"
            elif v > max_c * 0.30:
                ch = f"{C.CYN}▓▓{C.RST}"
            elif v > max_c * 0.15:
                ch = f"{C.BLU}▒▒{C.RST}"
            elif v > 0:
                ch = f"{C.DIM}░░{C.RST}"
            else:
                ch = f"{C.DIM}  {C.RST}"
            row.append(ch)
        print(f"    {''.join(row)}")


def do_stats(core):
    """统计信息"""
    from engine.config import CONFIG
    from engine.utils import C

    print(f"\n  {C.BOLD}统计信息{C.RST}")
    st = core.get_state()

    if getattr(core, 'graph_mode', False):
        # 图模式显示
        total_nodes = sum(g.get_total_nodes() for g in core.graphs.values()) if hasattr(core, 'graphs') else 0
        max_layer = max((g.get_max_layer() for g in core.graphs.values()), default=0) if hasattr(core, 'graphs') and core.graphs else 0
        print(f"  {C.CYN}[图模式]{C.RST}")
        print(f"  神经元: {C.val(st['neurons'])} 活跃:{C.GRN}{st['active']}{C.RST}")
        print(f"  图: {C.val(st['templates'])} 个标签")
        print(f"  节点: {C.val(total_nodes)} 最高层: L{max_layer}")
        # 各图节点分布
        if hasattr(core, 'graphs') and core.graphs:
            for label, g in sorted(core.graphs.items()):
                layer_dist = {}
                for n in g.nodes.values():
                    layer_dist[n.layer] = layer_dist.get(n.layer, 0) + 1
                dist_str = " ".join(f"L{k}={v}" for k, v in sorted(layer_dist.items()))
                print(f"    [{label}] {g.get_total_nodes()}节点 ({dist_str})")
    else:
        # 模板模式显示
        lock_pct = st['locked'] / st['neurons'] * 100 if st['neurons'] else 0
        lock_bar = "█" * int(lock_pct / 5) + "░" * (20 - int(lock_pct / 5))
        lock_col = C.GRN if lock_pct < 50 else C.YEL if lock_pct < 80 else C.RED
        print(f"  神经元: {C.val(st['neurons'])} 活跃:{C.GRN}{st['active']}{C.RST} "
              f"锁定:{lock_col}{st['locked']}{C.RST} 鼓励中:{C.val(st['encouraged'])}")
        print(f"  锁定率: {lock_col}{lock_pct:.0f}%{C.RST} [{lock_bar}]")
        print(f"  模板: {C.val(st['templates'])}/{CONFIG['MAX_TEMPLATES']}")
        avg_b = st['avg_base']
        base_str = f'{avg_b}' if isinstance(avg_b, int) else f'{avg_b:.3f}'
        print(f"  平均基础速度: {C.val(base_str)}")
        if core.templates:
            labels = sorted(set(t[0] for t in core.templates))
            print(f"  已知字符: {C.val(''.join(labels))}")

    v = sum(1 for x in core.history if x["V"])
    t = len(core.history)
    pct = v / t * 100 if t else 0
    print(f"  校验: {C.GRN}{v}{C.RST}/{C.val(t)} ({pct:.1f}%)")


def do_gauge(core):
    """仪表盘"""
    from engine.config import CONFIG
    from engine.utils import C, box, hr

    graph_mode = getattr(core, 'graph_mode', False)
    print(f"\n  {C.BOLD}SGN-Lite v5.0 仪表盘{C.RST}")
    st = core.get_state()
    hr(46)

    if graph_mode:
        total_nodes = sum(g.get_total_nodes() for g in core.graphs.values()) if hasattr(core, 'graphs') else 0
        print(f"  {C.CYN}[图模式]{C.RST}")
        print(f"  神经元: {C.GRN}{st['active']}{C.RST} 活跃 | {st['neurons']} 总数")
        print(f"  图: {C.CYN}{st['templates']}{C.RST} 个标签 | 节点: {C.val(total_nodes)}")
    else:
        print(f"  神经元: {C.GRN}{st['active']}{C.RST} 活跃 | {C.YEL}{st['locked']}{C.RST} 锁定 | {st['neurons']} 总数")
        print(f"  模板: {C.CYN}{st['templates']}{C.RST}/{CONFIG['MAX_TEMPLATES']}")

    t = len(core.history)
    v = sum(1 for x in core.history if x["V"])
    pct = v / t * 100 if t else 0
    print(f"  校验: {C.GRN}{v}{C.RST}/{t} ({pct:.1f}%)")

    if not graph_mode:
        avg_b = st['avg_base']
        base_str = f'{avg_b}' if isinstance(avg_b, int) else f'{avg_b:.3f}'
        print(f"  平均速度: {C.val(base_str)}")
        if core.templates:
            labels = sorted(set(t[0] for t in core.templates))
            print(f"  已知字符: {C.BOLD}{''.join(labels)}{C.RST}")
    else:
        if hasattr(core, 'graphs') and core.graphs:
            labels = sorted(core.graphs.keys())
            print(f"  已知字符: {C.BOLD}{''.join(labels)}{C.RST}")

    hr(46)
    if core.history:
        print(f"  {C.BOLD}最近10步:{C.RST}")
        for i, info in enumerate(core.history[-10:], max(1, t-9)):
            v_mark = f"{C.GRN}✓{C.RST}" if info["V"] else f"{C.RED}✗{C.RST}"
            base_fmt = info.get('base', 0)
            base_str2 = f'{base_fmt}' if isinstance(base_fmt, int) else f'{base_fmt:.3f}'
            match_val = info.get('match', 0)
            label_val = info.get('label', '?')
            print(f"  步{i:>4}: {v_mark} L={label_val} match={match_val:>3} base={base_str2}")


# ============================================================
# 向后兼容转发层
# ============================================================

from app.test import do_inference, do_batch_test, do_confusion, do_noise_test
from app.report import do_plot, do_export, _plot_accuracy, _plot_ascii, _plot_neurons, _plot_templates, _plot_comprehensive
from app.draw import draw_binary_grid, draw_index_grid, draw_intensity_grid
