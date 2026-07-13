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
"""SGN-Lite v5.0 黑箱验证模块 —— 从 sgn_interactive 拆分

黑箱模式手动验证阶段 —— 增强可视化，显示输入点阵、噪声位置、二值化层
"""

from __future__ import annotations

import random
from typing import Optional


def do_blackbox_verify(core, source=None):
    """黑箱模式手动验证阶段"""
    from engine.layers import extract_layers, combine_layers, match_bits
    from app.draw import draw_binary_grid, draw_index_grid, draw_intensity_grid
    from engine.config import CONFIG, PATTERNS, LABELS
    from engine.utils import C, box

    # 获取可用标签
    if source is not None and hasattr(source, 'patterns'):
        available_labels = list(source.patterns.keys())
        available_patterns = source.patterns
    elif source is not None and hasattr(source, '_samples') and source._samples:
        available_labels = sorted(set(lb for _, lb in source._samples))
        available_patterns = None
    elif source is not None and hasattr(source, 'formula_type'):
        # 【v4.3-fix】VectorPatternSource 支持
        labels_map = {
            "line": ["LINE"], "circle": ["CIRCLE"],
            "sine": ["SINE"], "arch": ["ARCH"], "leaf": ["LEAF"],
            "mixed": ["LINE", "CIRCLE", "SINE", "ARCH", "LEAF"]
        }
        available_labels = labels_map.get(source.formula_type, ["UNKNOWN"])
        available_patterns = {}
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
    elif core.templates:
        available_labels = sorted(set(t[0] for t in core.templates))
        available_patterns = None
    else:
        available_labels = list(LABELS)
        available_patterns = PATTERNS

    if not core.templates:
        print(f"\n  {C.YEL}⚠ 网络尚无模板，无法验证。{C.RST}")
        print(f"  {C.DIM}提示：黑箱训练可能未完成，或模板库为空。{C.RST}")
        return

    box("黑箱验证阶段")
    print(f"  {C.DIM}请输入字符({','.join(available_labels)})验证网络，或回车随机，q退出{C.RST}")
    print(f"  {C.DIM}此阶段检验网络在640步黑箱训练后，对未知输入的真实推理能力{C.RST}")
    print(f"  {C.DIM}注意：SGN没有严格的训练/推理分界，验证即继续推演{C.RST}\n")

    while True:
        try:
            inp = input(f"  输入字符 [{','.join(available_labels)}/Enter随机/q退出]: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            break
        if not inp:
            inp = random.choice(available_labels)
        if inp.lower() == "q":
            break
        if inp not in available_labels:
            print(f"  {C.err('✗')} 无效字符: {inp}")
            continue

        # 获取原始模板
        if available_patterns is not None and inp in available_patterns:
            original = available_patterns[inp][:]
        elif source is not None and hasattr(source, '_samples'):
            matches = [s for s, l in source._samples if l == inp]
            original = matches[0] if matches else [0] * core.D
        else:
            original = PATTERNS.get(inp, [0] * core.D)[:]

        d = getattr(core, 'D', CONFIG.get("D", 64))  # 默认 64 = 8×8
        grid_size = int(d ** 0.5)

        # 二维居中扩展
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
            if random.random() < 0.05:
                old_v = intensity[i]
                intensity[i] = max(0, min(255, intensity[i] + random.randint(-64, 64)))
                if intensity[i] != old_v:
                    noise_indices.append(i)

        layers, lc = extract_layers(intensity, d=d)
        if lc == 0:
            print(f"  {C.err('✗')} 无法提取特征")
            continue

        signature = combine_layers(layers)
        best_s, pred = -1, '?'
        for tlb, tm, ts, thc in core.templates:
            s = match_bits(tm, signature, d=d)
            if s > best_s:
                best_s, pred = s, tlb

        conf_str = f"{C.GRN}高{C.RST}" if best_s >= 80 else f"{C.YEL}中{C.RST}" if best_s >= 50 else f"{C.RED}低{C.RST}"
        match_bar = f"{C.GRN}{'█'*(best_s//10)}{C.RST}{C.DIM}{'░'*((100-best_s)//10)}{C.RST}"

        print(f"\n  {C.BOLD}{'─'*44}{C.RST}")
        print(f"  {C.BOLD}输入字符: {C.CYN}{inp}{C.RST}  →  网络识别: {C.CYN}{pred}{C.RST}  匹配度: {C.val(best_s)}%  置信: {conf_str}")
        print(f"  {C.BOLD}{'─'*44}{C.RST}")

        print(f"\n  {C.BOLD}① 数据位置索引图 (intensity[0]~intensity[{d-1}]){C.RST}")
        draw_index_grid(grid_size=grid_size)

        print(f"\n  {C.BOLD}② 标准模板 '{inp}' 原始强度值{C.RST}")
        draw_intensity_grid(original, title="")

        print(f"\n  {C.BOLD}③ 实际输入强度 (含5%随机噪声){C.RST}")
        if noise_indices:
            print(f"  {C.YEL}⚠ 噪声位置: {', '.join(str(i) for i in noise_indices)}{C.RST}")
        else:
            print(f"  {C.GRN}✓ 本次无噪声{C.RST}")
        draw_intensity_grid(intensity, title="", highlight_changed=set(noise_indices) if noise_indices else None)

        print(f"\n  {C.BOLD}④ 二值化提取结果 (分层 + 合并签名){C.RST}")
        print(f"  {C.DIM}┌─ 二值化教学注释 ─────────────────────────────────┐{C.RST}")
        print(f"  {C.DIM}│ 第②③步是 0~255 的灰度强度（像照片有亮有暗）。{C.RST}")
        print(f"  {C.DIM}│ SGN 核心只认开/关两种状态（像黑白线稿）。{C.RST}")
        print(f"  {C.DIM}│ 转换规则：按亮度排序，每层取最亮的~50%像素，{C.RST}")
        print(f"  {C.DIM}│ 逐层提取直到所有亮像素被覆盖（最多4层）。{C.RST}")
        print(f"  {C.DIM}│ 合并签名 = 所有层 OR 组合，即完整的二值特征。{C.RST}")
        print(f"  {C.DIM}│ 对比：合并签名应与第③步高亮区域一致。{C.RST}")
        print(f"  {C.DIM}└────────────────────────────────────────────────┘{C.RST}")
        # 显示各层
        for li in range(lc):
            layer_bits = bin(layers[li]).count('1')
            print(f"  {C.DIM}Layer {li} ({layer_bits}px):{C.RST}")
            draw_binary_grid(layers[li], grid_size=grid_size)
        # 显示合并签名
        combined_bits = bin(signature).count('1')
        active_count = sum(1 for v in intensity if v > 0)
        print(f"  {C.BOLD}合并签名 ({combined_bits}px / {active_count}亮像素, 覆盖率{combined_bits/max(active_count,1)*100:.0f}%):{C.RST}")
        draw_binary_grid(signature, grid_size=grid_size)

        print(f"\n  匹配度: {C.val(best_s)}% [{match_bar}]  置信: {conf_str}")
        hit_templates = []
        for label, indices in core._template_index.items():
            for idx in indices:
                tlb, tm, ts, thc = core.templates[idx]
                s = match_bits(tm, signature, d=d)
                if s >= 50:
                    hit_templates.append((tlb, s))
        if hit_templates:
            hit_templates.sort(key=lambda x: x[1], reverse=True)
            top3 = hit_templates[:3]
            refs = ", ".join([f"{C.DIM}{tlb}={s}%{C.RST}" for tlb, s in top3])
            print(f"  模板参考: {refs}")

        try:
            judge = input(f"\n  判断正确? [y/n/s跳过]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if judge == "s":
            continue
        correct = (judge == "y")

        core.blackbox_log.append({
            "input": inp,
            "pred": pred,
            "correct": correct,
            "match": best_s,
            "timestamp": len(core.blackbox_log) + 1,
        })

        if correct:
            print(f"  {C.GRN}✓ 记录：正确识别{C.RST}")
        else:
            print(f"  {C.RED}✗ 记录：识别错误{C.RST}")

    # 验证报告
    if core.blackbox_log:
        total = len(core.blackbox_log)
        correct_cnt = sum(1 for x in core.blackbox_log if x["correct"])
        pct = correct_cnt / total * 100
        box("黑箱验证报告")
        print(f"  验证次数: {C.val(total)}")
        print(f"  用户判定正确: {C.GRN}{correct_cnt}{C.RST} ({C.GRN}{pct:.1f}%{C.RST})")
        print(f"  用户判定错误: {C.RED}{total - correct_cnt}{C.RST}")
        char_stats = {}
        for rec in core.blackbox_log:
            ch = rec["input"]
            if ch not in char_stats:
                char_stats[ch] = {"total": 0, "correct": 0}
            char_stats[ch]["total"] += 1
            if rec["correct"]:
                char_stats[ch]["correct"] += 1
        print(f"\n  {C.BOLD}各字符验证情况:{C.RST}")
        for ch in sorted(char_stats.keys()):
            s = char_stats[ch]
            p = s["correct"] / s["total"] * 100
            col = C.GRN if p >= 80 else C.YEL if p >= 50 else C.RED
            print(f"  {C.BOLD}{ch}{C.RST}: {col}{s['correct']}/{s['total']}{C.RST} ({col}{p:.0f}%{C.RST})")
    else:
        print(f"\n  {C.DIM}未记录任何验证数据{C.RST}")

    print(f"\n  {C.DIM}黑箱验证结束。这些记录反映了网络在真实人工判断下的表现。{C.RST}")
    if core.blackbox_log:
        print(f"\n  {C.info('ℹ')} 输入 [o] 保存模型，验证记录将随模型一并持久化{C.RST}")
        print(f"  {C.DIM}按 Enter 返回菜单...{C.RST}")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
