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
"""SGN-Lite v5.0 基础绘图工具模块 —— 从 sgn_visual 拆分

纯终端 ASCII 绘图函数，不依赖核心引擎，可被任何模块安全导入。
"""

from __future__ import annotations

from typing import List, Optional, Set


def draw_binary_grid(mask, indent="    ", grid_size=None):
    """绘制二进制网格（窗口大小自适应）"""
    if grid_size is None:
        from sgn_config import CONFIG
        d = CONFIG.get("D", 16)
        grid_size = int(d ** 0.5)

    # 延迟导入颜色类，避免循环
    from sgn_utils import C

    for r in range(grid_size):
        row = []
        for c in range(grid_size):
            idx = r * grid_size + c
            on = (mask >> idx) & 1
            row.append(f"{C.GRN}██{C.RST}" if on else f"{C.DIM}··{C.RST}")
        print(f"{indent}{''.join(row)}")


def draw_index_grid(indent="    ", grid_size=None):
    """绘制位置索引图（窗口大小自适应）"""
    if grid_size is None:
        from sgn_config import CONFIG
        d = CONFIG.get("D", 16)
        grid_size = int(d ** 0.5)

    from sgn_utils import C

    cell = "────"
    top = "┌" + "┬".join([cell] * grid_size) + "┐"
    mid = "├" + "┼".join([cell] * grid_size) + "┤"
    bot = "└" + "┴".join([cell] * grid_size) + "┘"
    print(f"{indent}{C.DIM}{top}{C.RST}")
    for r in range(grid_size):
        row_vals = []
        for c in range(grid_size):
            idx = r * grid_size + c
            row_vals.append(f"{C.CYN}{idx:>2}{C.RST}")
        print(f"{indent}{C.DIM}│{C.RST} {' │ '.join(row_vals)} {C.DIM}│{C.RST}")
        if r < grid_size - 1:
            print(f"{indent}{C.DIM}{mid}{C.RST}")
    print(f"{indent}{C.DIM}{bot}{C.RST}")


def draw_intensity_grid(intensity, title="", indent="    ", highlight_changed=None, grid_size=None):
    """绘制强度值可视化点阵图（窗口大小自适应）"""
    if grid_size is None:
        from sgn_config import CONFIG
        d = CONFIG.get("D", 16)
        grid_size = int(d ** 0.5)

    from sgn_utils import C

    if title:
        print(f"{indent}{C.BOLD}{title}{C.RST}")

    def intensity_block(v, is_changed=False):
        if is_changed:
            return f"{C.BOLD}{C.YEL}██{C.RST}"
        if v >= 240:
            return f"{C.BOLD}{C.WHT}██{C.RST}"
        elif v >= 210:
            return f"{C.WHT}██{C.RST}"
        elif v >= 180:
            return f"{C.DIM}{C.WHT}██{C.RST}"
        elif v >= 150:
            return f"{C.BOLD}{C.WHT}▓▓{C.RST}"
        elif v >= 120:
            return f"{C.WHT}▓▓{C.RST}"
        elif v >= 90:
            return f"{C.DIM}{C.WHT}▓▓{C.RST}"
        elif v >= 60:
            return f"{C.BOLD}{C.WHT}▒▒{C.RST}"
        elif v >= 30:
            return f"{C.WHT}▒▒{C.RST}"
        elif v >= 10:
            return f"{C.DIM}░░{C.RST}"
        else:
            return f"{C.DIM}··{C.RST}"

    cell = "────────"
    top = "┌" + "┬".join([cell] * grid_size) + "┐"
    mid = "├" + "┼".join([cell] * grid_size) + "┤"
    bot = "└" + "┴".join([cell] * grid_size) + "┘"
    print(f"{indent}{C.DIM}{top}{C.RST}")
    for r in range(grid_size):
        row_blocks = []
        for c in range(grid_size):
            idx = r * grid_size + c
            v = intensity[idx] if idx < len(intensity) else 0
            changed = False
            if highlight_changed and idx in highlight_changed:
                changed = True
            row_blocks.append(intensity_block(v, changed))
        print(f"{indent}{C.DIM}│{C.RST} {' │ '.join(row_blocks)} {C.DIM}│{C.RST}")

        row_nums = []
        for c in range(grid_size):
            idx = r * grid_size + c
            v = intensity[idx] if idx < len(intensity) else 0
            changed = False
            if highlight_changed and idx in highlight_changed:
                changed = True
            if changed:
                row_nums.append(f"{C.YEL}{v:>3}{C.RST}")
            elif v >= 200:
                row_nums.append(f"{C.WHT}{v:>3}{C.RST}")
            elif v >= 100:
                row_nums.append(f"{C.DIM}{v:>3}{C.RST}")
            else:
                row_nums.append(f"{C.DIM}{v:>3}{C.RST}")
        print(f"{indent}{C.DIM}│{C.RST} {' │ '.join(row_nums)} {C.DIM}│{C.RST}")

        if r < grid_size - 1:
            print(f"{indent}{C.DIM}{mid}{C.RST}")
    print(f"{indent}{C.DIM}{bot}{C.RST}")

    legend = (
        f"{indent}{C.BOLD}{C.WHT}██{C.RST}=240+ "
        f"{C.WHT}██{C.RST}=210 "
        f"{C.DIM}{C.WHT}██{C.RST}=180 "
        f"{C.BOLD}{C.WHT}▓▓{C.RST}=150 "
        f"{C.WHT}▓▓{C.RST}=120 "
        f"{C.DIM}{C.WHT}▓▓{C.RST}=90 "
        f"{C.BOLD}{C.WHT}▒▒{C.RST}=60 "
        f"{C.WHT}▒▒{C.RST}=30 "
        f"{C.DIM}░░{C.RST}=10 "
        f"{C.DIM}··{C.RST}=0"
    )
    if highlight_changed:
        legend += f"  {C.BOLD}{C.YEL}██{C.RST}=噪声点"
    print(legend)
