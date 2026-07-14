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
"""SGN-Lite v5.0 交互调度模块 —— 拆分后瘦身版

职责：
  - menu() 交互菜单调度（调用 CommandRegistry 执行命令）
  - 统一提醒服务（_on_config_changed 回调）
  - 向后兼容：转发到 sgn_panel / sgn_help / sgn_training / sgn_blackbox

注意：所有具体实现已拆分到：
  sgn_panel.py      — 控制面板/扩展菜单/高级选项
  sgn_help.py       — 帮助手册
  sgn_training.py   — 训练循环
  sgn_blackbox.py   — 黑箱验证
  sgn_cmd_registry  — 命令注册
"""

from __future__ import annotations

import sys
from app.commands import CommandRegistry, CommandContext

from app.draw import draw_binary_grid, draw_index_grid, draw_intensity_grid


def menu(core, samples, step, max_step, test_samples=None, source=None):
    """交互菜单 - v4.3 使用 CommandRegistry 动态调度

    所有命令回调已注册到 CommandRegistry（由 sgn_cmd_registry 聚合）。
    """
    from engine.utils import C, hr, progress_bar, clear_stdin_buffer

    while True:
        hr(46)
        print(f"  {C.CYN}步{step}/{max_step}{C.RST} {progress_bar(step, max_step)}")

        # 动态生成菜单显示
        cat_order = ["训练", "测试", "可视化", "系统"]
        registered_cats = set()
        for c in CommandRegistry.list_commands():
            registered_cats.add(c.category)
        cat_order = [c for c in cat_order if c in registered_cats] + sorted(registered_cats - set(cat_order))

        for cat in cat_order:
            cmds = CommandRegistry.list_commands(category=cat)
            if cmds:
                if cat == "训练":
                    entries = []
                    for c in cmds:
                        if c.hotkey == "":
                            entries.append(f"{C.GRN}[Enter]{C.RST}{c.description}")
                        elif c.hotkey == "a":
                            entries.append(f"{C.CYN}[{c.hotkey}]{C.RST}{c.description}")
                        elif c.hotkey == "r":
                            entries.append(f"{C.YEL}[{c.hotkey}]{C.RST}{c.description}")
                        else:
                            entries.append(f"{C.CYN}[{c.hotkey}]{C.RST}{c.description}")
                    print(f"  {'  '.join(entries)}")
                elif cat == "测试":
                    entries = [f"{C.CYN}[{c.hotkey}]{C.RST}{c.description}" for c in cmds]
                    print(f"  {'  '.join(entries[:4])}")
                    if len(entries) > 4:
                        print(f"  {'  '.join(entries[4:])}")
                elif cat == "可视化":
                    entries = [f"{C.CYN}[{c.hotkey}]{C.RST}{c.description}" for c in cmds]
                    print(f"  {'  '.join(entries[:4])}")
                    if len(entries) > 4:
                        print(f"  {'  '.join(entries[4:])}")
                else:  # 系统
                    entries = []
                    for c in cmds:
                        if c.hotkey == "q":
                            entries.append(f"{C.RED}[{c.hotkey}]{C.RST}{c.description}")
                        elif c.hotkey in ("w", "r", "b"):
                            entries.append(f"{C.YEL}[{c.hotkey}]{C.RST}{c.description}")
                        else:
                            entries.append(f"{C.CYN}[{c.hotkey}]{C.RST}{c.description}")
                    if len(entries) <= 6:
                        print(f"  {'  '.join(entries)}")
                    else:
                        print(f"  {'  '.join(entries[:6])}")
                        print(f"  {'  '.join(entries[6:])}")

        try:
            try:
                clear_stdin_buffer()
            except Exception:
                # 【v4.3-fix】clear_stdin_buffer 在伪终端下可能底层崩溃，
                # 跳过清空不影响功能，只是可能残留回车。
                pass
            c = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "quit"

        ctx = CommandContext(core=core, samples=samples, test_samples=test_samples, source=source, step=step, max_step=max_step)
        handled = CommandRegistry.execute(c, ctx)
        if handled:
            result = ctx.get("_result")
            if result is not None:
                return result
            continue

        print(f"  {C.RED}未知命令: '{c}'{C.RST}")
        print(f"  {C.DIM}输入 [h] 查看帮助{C.RST}")


# ============================================================
# 统一提醒服务
# ============================================================

def _on_config_changed(key, old, new):
    """配置变更统一提醒回调

    注册到 HookRegistry "sgn:on_config_changed" 事件，
    无论配置从哪个入口变更都会触发。
    """
    from engine.config import ConfigRegistry, CONFIG
    from engine.utils import C, log_print

    item = ConfigRegistry.get_schema(key)
    if not item:
        return

    if item.requires_rebuild:
        log_print(f"  {C.YEL}⚠ {key} 是架构参数，修改后需重建网络才生效{C.RST}")
        log_print(f"  {C.DIM}   返回控制面板或直接训练时将自动重建{C.RST}")

    if key == "CHART_BACKEND" and new != "auto":
        log_print(f"  {C.info('ℹ')} 图表后端已切换为 {C.val(new)}")
    elif key == "STORAGE_BACKEND":
        log_print(f"  {C.info('ℹ')} 存储后端已切换为 {C.val(new)}")
    elif key == "AUTOSAVE_STRATEGY":
        log_print(f"  {C.info('ℹ')} 自动保存策略已切换为 {C.val(new)}")
    elif key == "COLOR_OUTPUT":
        status = "启用" if new else "禁用"
        log_print(f"  {C.info('ℹ')} 彩色输出已{status}")

    if hasattr(new, "level") and hasattr(old, "level"):
        if not callable(getattr(new, "index", None)) and not callable(getattr(old, "index", None)):
            if new.level != old.level or new.index != old.index:
                log_print(f"  {C.DIM}   {key} = {old} → {new}{C.RST}")


# 模块加载时注册（强引用，避免弱引用被回收）
try:
    from engine.hooks import HookRegistry
    HookRegistry.register("sgn:on_config_changed", _on_config_changed, weak=False)
except ImportError:
    pass


# ============================================================
# 向后兼容转发层
# ============================================================

# 控制面板
from app.panel import control_panel, homepage, _extension_menu, _input_source_menu, _show_category_menu

# 帮助手册
from app.help import do_help, do_why_accuracy

# 训练循环
from app.training import run_training_loop, header, out_step

# 黑箱验证
from app.blackbox import do_blackbox_verify
