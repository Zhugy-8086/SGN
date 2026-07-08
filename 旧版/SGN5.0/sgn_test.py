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
"""SGN-Lite v5.0 测试模块 —— 从 sgn_visual 拆分

包含：推理测试、批量测试、混淆矩阵、噪声测试
"""

from __future__ import annotations

import random
from typing import List, Tuple, Optional


def _classify(core, intensity, d=None):
    """统一分类入口 - 自动适配模板模式和图模式"""
    if d is None:
        from sgn_config import CONFIG
        d = getattr(core, 'D', CONFIG.get("D", 16))
    if getattr(core, 'graph_mode', False):
        from sgn_graph_match import classify_with_graph
        return classify_with_graph(core, intensity, d)
    else:
        from sgn_layers import classify_sample
        return classify_sample(core, intensity, d)


def do_inference(core, source=None):
    """交互推理 - 支持连续测试，按 q 退出"""
    from sgn_config import PATTERNS, LABELS, CONFIG
    from sgn_layers import extract_layers
    from sgn_draw import draw_binary_grid, draw_index_grid, draw_intensity_grid

    # 获取可用标签
    if source is not None and hasattr(source, 'patterns'):
        available_labels = list(source.patterns.keys())
        available_patterns = source.patterns
    elif source is not None and hasattr(source, '_samples') and source._samples:
        available_labels = sorted(set(lb for _, lb in source._samples))
        available_patterns = None
    elif source is not None and hasattr(source, 'formula_type'):
        # 【v4.3-fix】VectorPatternSource 支持：动态生成标签和标准模板
        labels_map = {
            "line": ["LINE"], "circle": ["CIRCLE"],
            "sine": ["SINE"], "catear": ["CATEAR"],
            "mixed": ["LINE", "CIRCLE", "SINE", "CATEAR"]
        }
        available_labels = labels_map.get(source.formula_type, ["UNKNOWN"])
        available_patterns = {}
        # 生成一批样本，按标签提取作为"标准模板"
        try:
            samples = source.generate_batch(len(available_labels) * 3)
            for lb in available_labels:
                for s, sl in samples:
                    if sl == lb:
                        available_patterns[lb] = s
                        break
                else:
                    available_patterns[lb] = [0] * core.D
        except Exception:
            available_patterns = {lb: [0] * core.D for lb in available_labels}
    else:
        available_labels = list(LABELS)
        available_patterns = PATTERNS

    from sgn_utils import C

    while True:
        print(f"\n  {C.BOLD}推理测试{C.RST}")
        print(f"  输入字符({','.join(available_labels)})或回车随机，q退出: ", end="")
        try:
            inp = input().strip().upper()
        except (EOFError, KeyboardInterrupt):
            return
        if inp == 'Q':
            return
        if not inp or inp not in available_labels:
            inp = random.choice(available_labels)

        # 获取原始模板
        if available_patterns is not None and inp in available_patterns:
            original = available_patterns[inp][:]
        elif source is not None and hasattr(source, '_samples'):
            matches = [s for s, l in source._samples if l == inp]
            original = matches[0] if matches else [0] * core.D
        else:
            original = PATTERNS.get(inp, [0] * core.D)[:]

        d = getattr(core, 'D', CONFIG.get("D", 16))
        grid_size = int(d ** 0.5)

        # 【fix】二维居中扩展（与 sgn_blackbox.py 一致）
        if d > len(original):
            old_gs = int(len(original) ** 0.5)
            intensity = [0] * d
            row_off = (grid_size - old_gs) // 2
            col_off = (grid_size - old_gs) // 2
            for r in range(old_gs):
                for c in range(old_gs):
                    idx_old = r * old_gs + c
                    idx_new = (r + row_off) * grid_size + (c + col_off)
                    intensity[idx_new] = original[idx_old]
        else:
            intensity = original[:]

        noise_indices = []
        for i in range(d):
            if random.random() < 0.1:
                old_v = intensity[i]
                intensity[i] = random.randint(0, 255)
                if intensity[i] != old_v:
                    noise_indices.append(i)

        layers, lc = extract_layers(intensity, d=d)
        if lc == 0:
            print(f"  {C.err('✗')} 无法提取层")
            continue

        pred, best_s = _classify(core, intensity, d)

        conf_str = f"{C.GRN}高{C.RST}" if best_s >= 80 else f"{C.YEL}中{C.RST}" if best_s >= 50 else f"{C.RED}低{C.RST}"

        print(f"\n  {C.BOLD}{'─'*44}{C.RST}")
        print(f"  {C.BOLD}输入字符: {C.CYN}{inp}{C.RST}  →  网络识别: {C.CYN}{pred}{C.RST}  匹配度: {C.val(best_s)}%  置信: {conf_str}")
        print(f"  {C.BOLD}{'─'*44}{C.RST}")
        print(f"\n  {C.BOLD}① 数据位置索引图 (intensity[0]~intensity[{d-1}]){C.RST}")
        draw_index_grid(grid_size=int(d**0.5))
        print(f"\n  {C.BOLD}② 标准模板 '{inp}' 原始强度值{C.RST}")
        draw_intensity_grid(original, title="")
        print(f"\n  {C.BOLD}③ 实际输入强度 (含10%随机噪声){C.RST}")
        if noise_indices:
            print(f"  {C.YEL}⚠ 噪声位置: {', '.join(str(i) for i in noise_indices)}{C.RST}")
        else:
            print(f"  {C.GRN}✓ 本次无噪声{C.RST}")
        draw_intensity_grid(intensity, title="", highlight_changed=set(noise_indices) if noise_indices else None)
        print(f"\n  {C.BOLD}④ 二值化提取结果 (layers[0]){C.RST}")
        print(f"  {C.DIM}┌─ 二值化教学注释 ─────────────────────────────────┐{C.RST}")
        print(f"  {C.DIM}│ 第②③步是 0~255 的灰度强度（像照片有亮有暗）。{C.RST}")
        print(f"  {C.DIM}│ SGN 核心只认开/关两种状态（像黑白线稿）。{C.RST}")
        print(f"  {C.DIM}│ 转换规则：挑出最亮的像素 → 变成 1(██)，其余 0(··)。{C.RST}")
        print(f"  {C.DIM}│ 这样就把灰度图转描成黑白掩码，供神经网络匹配。{C.RST}")
        print(f"  {C.DIM}│ 对比：看本图亮块位置，应与第③步高亮区域一致。{C.RST}")
        print(f"  {C.DIM}└────────────────────────────────────────────────┘{C.RST}")
        draw_binary_grid(layers[0] if lc > 0 else 0, grid_size=int(d**0.5))
        print(f"\n  {C.BOLD}{'─'*44}{C.RST}")


def do_batch_test(core, test_samples=None):
    """批量测试"""
    from sgn_utils import C

    if test_samples is None:
        print(f"\n  {C.RED}✗ 错误：未提供测试样本 (test_samples=None){C.RST}")
        print(f"  {C.DIM}提示：批量测试需要独立的测试样本，用于客观评估泛化能力。{C.RST}")
        print(f"  {C.DIM}      请确保训练流程正确传递了 test_samples 参数。{C.RST}")
        print(f"\n  {C.DIM}按 Enter 返回菜单...{C.RST}")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        return

    print(f"\n  {C.BOLD}批量测试{C.RST}")
    if getattr(core, 'graph_mode', False):
        print(f"  {C.CYN}[图模式]{C.RST}")
    correct = 0
    total = 0
    for intensity, label in test_samples:
        pred_lb, best_s = _classify(core, intensity)
        if pred_lb == label and best_s >= 80:
            correct += 1
        total += 1
    pct = correct / total * 100 if total else 0
    col = C.GRN if pct >= 90 else C.YEL if pct >= 70 else C.RED
    print(f"  识别率: {col}{correct}/{total} ({pct:.1f}%){C.RST}")


def do_confusion(core, test_samples=None):
    """混淆矩阵 —— 动态标签自适应（支持矢量/文件/内置模式）"""
    from sgn_utils import mini_bar, C

    if test_samples is None:
        print(f"\n  {C.RED}✗ 错误：未提供测试样本 (test_samples=None){C.RST}")
        print(f"  {C.DIM}提示：混淆矩阵需要测试样本才能生成。{C.RST}")
        print(f"  {C.DIM}      请确保训练流程正确传递了 test_samples 参数。{C.RST}")
        print(f"\n  {C.DIM}按 Enter 返回菜单...{C.RST}")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        return

    if getattr(core, 'graph_mode', False):
        print(f"  {C.CYN}[图模式]{C.RST}")

    # 【v4.3-fix】动态提取标签，彻底移除硬编码 LABELS 依赖
    # 【v4.3-fix2】加入 '?' 未识别列，避免丢弃未识别样本虚高准确率
    all_labels = sorted(set(lb for _, lb in test_samples))
    lb2i = {lb: i for i, lb in enumerate(all_labels)}
    size = len(all_labels)
    cm = [[0] * (size + 1) for _ in range(size)]  # +1 列给 '?'
    per_class_total = {lb: 0 for lb in all_labels}
    per_class_correct = {lb: 0 for lb in all_labels}

    for intensity, label in test_samples:
        pred, best_sim = _classify(core, intensity)
        per_class_total[label] += 1
        if pred in lb2i:
            cm[lb2i[label]][lb2i[pred]] += 1
            if pred == label:
                per_class_correct[label] += 1
        else:
            cm[lb2i[label]][size] += 1  # '?' 列

    print(f"\n  {C.BOLD}混淆矩阵{C.RST} ({size} 类)")
    # 动态表头（含 '?' 列）
    header = "    " + " ".join(f"{lb:>2}" for lb in all_labels) + "  ?"
    print(f"  {C.DIM}{header}{C.RST}")
    for i, lb in enumerate(all_labels):
        total = sum(cm[i])
        correct = cm[i][i]
        row = f"  {C.BOLD}{lb}{C.RST}  "
        for j in range(size + 1):
            v = cm[i][j]
            if j == size:
                col = C.RED if v > 0 else C.DIM
            else:
                col = C.GRN if i == j and v > 0 else C.YEL if v > 0 else C.DIM
            row += f"{col}{v:>2}{C.RST} "
        acc = correct / total * 100 if total else 0
        bar = mini_bar(acc, 100, 10)
        print(f"{row} {C.BOLD}{acc:.0f}%{C.RST} {bar}")


def do_noise_test(core, test_samples=None, noise_model=None, source=None):
    """噪声鲁棒性测试（支持所有输入模式和噪声类型选择）"""
    from sgn_config import CONFIG, LABELS, PATTERNS
    from sgn_utils import hr, C

    print(f"\n  {C.BOLD}噪声鲁棒性测试{C.RST}")
    if getattr(core, 'graph_mode', False):
        print(f"  {C.CYN}[图模式]{C.RST}")

    # 噪声类型选择
    default_test_type = CONFIG.get("NOISE_TEST_TYPE", "composite")
    if noise_model is None:
        print(f"\n  {C.BOLD}选择噪声类型:{C.RST}")
        print(f"  {C.DIM}【说明】本测试评估网络对不同噪声类型的泛化能力，")
        print(f"         与训练阶段使用的噪声配置相互独立。{C.RST}")
        print(f"  {C.CYN}[1]{C.RST} 复合噪声 (SGN三层: 翻转/强抖动/弱抖动){C.GRN} ★{C.RST}" if default_test_type == "composite" else f"  {C.CYN}[1]{C.RST} 复合噪声 (SGN三层: 翻转/强抖动/弱抖动)")
        print(f"  {C.CYN}[2]{C.RST} 高斯噪声 (连续随机扰动){C.GRN} ★{C.RST}" if default_test_type == "gaussian" else f"  {C.CYN}[2]{C.RST} 高斯噪声 (连续随机扰动)")
        print(f"  {C.CYN}[3]{C.RST} 椒盐噪声 (极端离散故障){C.GRN} ★{C.RST}" if default_test_type == "salt_pepper" else f"  {C.CYN}[3]{C.RST} 椒盐噪声 (极端离散故障)")
        print(f"  {C.CYN}[4]{C.RST} 块遮挡噪声 (大面积缺失){C.GRN} ★{C.RST}" if default_test_type == "block" else f"  {C.CYN}[4]{C.RST} 块遮挡噪声 (大面积缺失)")
        print(f"  {C.RED}[q]{C.RST} 取消")
        hr(46)
        try:
            choice = input("  选择: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if choice == "q" or choice == "":
            print(f"  {C.DIM}已取消{C.RST}")
            return

        grid_size = int(getattr(core, 'grid_size', CONFIG.get("D", 16) ** 0.5))

        if choice == "1":
            from sgn_input import DefaultCompositeNoise
            noise_cls = lambda p: DefaultCompositeNoise(noise_prob=p)
            param_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
            fmt = lambda v: f"{v:>5.2f}"
        elif choice == "2":
            from sgn_input import GaussianNoise
            is_vector = (source is not None and
                        (hasattr(source, 'formula_type') or
                         (hasattr(source, 'sources') and
                          any(hasattr(s, 'formula_type') for s in getattr(source, 'sources', [])))))
            base_prob = 0.15 if is_vector else 1.0
            noise_cls = lambda p, bp=base_prob: GaussianNoise(noise_prob=bp, sigma=p)
            param_values = [0, 10, 20, 30, 40, 50, 60, 80, 100]
            fmt = lambda v: f"{v:>5.0f}"
        elif choice == "3":
            from sgn_input import SaltPepperNoise
            noise_cls = lambda p: SaltPepperNoise(noise_prob=p)
            param_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
            fmt = lambda v: f"{v:>5.2f}"
        elif choice == "4":
            from sgn_input import BlockNoise
            noise_cls = lambda p: BlockNoise(block_size=max(2, grid_size // 4), prob=p, grid_size=grid_size)
            param_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
            fmt = lambda v: f"{v:>5.2f}"
        else:
            print(f"  {C.RED}无效选择: '{choice}'{C.RST}")
            return
    else:
        noise_cls = type(noise_model)
        param_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        fmt = lambda v: f"{v:>5.2f}"

    # 确定标签和基础样本来源
    samples_per_level = 40

    if source is not None and hasattr(source, 'patterns'):
        test_labels = list(source.patterns.keys())
        base_patterns = source.patterns
        source_type = "pattern"
    elif source is not None and hasattr(source, 'formula_type'):
        if source.formula_type == "mixed":
            test_labels = ["LINE", "CIRCLE", "SINE"]
        else:
            test_labels = {"line": ["LINE"], "circle": ["CIRCLE"], "sine": ["SINE"], "catear": ["CATEAR"]}.get(source.formula_type, ["UNKNOWN"])
        source_type = "vector"
    elif source is not None and hasattr(source, 'sources'):
        test_labels = []
        for src in source.sources:
                    test_labels.extend({"line": ["LINE"], "circle": ["CIRCLE"], "sine": ["SINE"], "catear": ["CATEAR"]}.get(src.formula_type, ["UNKNOWN"]))
        source_type = "mixed_vector"
    elif source is not None and hasattr(source, '_samples') and source._samples:
        test_labels = sorted(set(lb for _, lb in source._samples))
        source_type = "file"
    elif test_samples is not None:
        test_labels = sorted(set(lb for _, lb in test_samples))
        source_type = "inferred"
    else:
        if core.templates:
            test_labels = sorted(set(t[0] for t in core.templates))
        else:
            test_labels = list(LABELS)
        source_type = "inferred"

    # 测试循环
    results = []
    for param_val in param_values:
        nm = noise_cls(param_val)

        # 生成测试样本
        level_samples = []
        if source_type == "pattern":
            for lbl in test_labels:
                base = base_patterns.get(lbl)
                if base is None:
                    continue
                for _ in range(samples_per_level):
                    level_samples.append((nm.apply(base.copy()), lbl))
        elif source_type in ("vector", "mixed_vector"):
            from sgn_input import VectorPatternSource
            if source_type == "vector":
                temp = VectorPatternSource(
                    source.formula_type, source.grid_size,
                    nm, samples_per_label=samples_per_level, seed=42
                )
                level_samples = temp.generate_batch(samples_per_level * len(test_labels), split='all')
            else:
                for src in source.sources:
                    temp = VectorPatternSource(
                        src.formula_type, src.grid_size,
                        nm, samples_per_label=samples_per_level, seed=42
                    )
                    level_samples.extend(temp.generate_batch(samples_per_level, split='all'))
        elif source_type == "file":
            n_per = max(1, samples_per_level // max(1, len(source._samples)))
            for intensity, lbl in source._samples:
                for _ in range(n_per):
                    level_samples.append((nm.apply(intensity.copy()), lbl))
        elif source_type == "inferred":
            for intensity, lbl in test_samples:
                level_samples.append((nm.apply(intensity.copy()), lbl))

        # 执行识别
        correct = 0
        total = 0
        for intensity, label in level_samples:
            pred_lb, best_s = _classify(core, intensity)
            if pred_lb == label and best_s >= 80:
                correct += 1
            total += 1
        pct = correct / total * 100 if total else 0
        results.append((param_val, pct))

        bar_len = int(pct / 5)
        bar = f"{C.GRN}{'█'*bar_len}{C.RST}{C.DIM}{'░'*(20-bar_len)}{C.RST}"
        pct_col = C.GRN if pct >= 80 else C.YEL if pct >= 50 else C.RED
        print(f"  {fmt(param_val)} {pct_col}{pct:>6.1f}%{C.RST}  [{bar}]")

    # 汇总
    hr(46)
    max_pct = max(pct for _, pct in results)
    min_pct = min(pct for _, pct in results)
    print(f"  最高识别率: {C.GRN}{max_pct:.1f}%{C.RST}  最低: {C.RED}{min_pct:.1f}%{C.RST}")
    print(f"  {C.DIM}【说明】本测试为独立泛化能力评估，非旧版鲁棒性准确率。")
    print(f"         测试噪声与训练噪声相互独立，曲线反映网络对未见噪声的泛化率。{C.RST}")
    if source_type in ("vector", "mixed_vector"):
        print(f"  {C.DIM}注: 矢量模式每次渲染参数随机，测试反映对随机变形的鲁棒性{C.RST}")
