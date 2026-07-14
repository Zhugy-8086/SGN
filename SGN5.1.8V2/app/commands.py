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
"""SGN-Lite v5.0 命令注册表 - CommandRegistry

阶段3重构：消灭 sgn_interactive.py 里 menu() 和 control_panel() 的巨型 if/elif。
允许外部插件注册新命令，在不改源码的情况下扩展交互功能。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, OrderedDict as OrderedDictType
from collections import OrderedDict


# ============================================================
# CommandContext - 命令上下文的类型约定
# ============================================================

class CommandContext(dict):
    """命令上下文字典，传递 core, samples, step 等共享状态

    插件回调应接受 **kwargs 以兼容未来扩展。
    """
    pass


# ============================================================
# Command - 命令描述
# ============================================================

@dataclass
class Command:
    """单个命令的元数据

    Attributes:
        hotkey: 热键，如 "i", "t", "s"
        description: 人类可读的描述
        category: 分类（"训练"/"测试"/"可视化"/"系统"/"扩展"）
        callback: 回调函数，签名应为 fn(core, **kwargs) 或 fn(**kwargs)
        requires_trained: 是否要求网络已训练（templates > 0）
        order: 在同一 category 中的排序权重（越小越靠前）
    """
    hotkey: str
    description: str
    category: str
    callback: Callable
    requires_trained: bool = True  # 默认更安全
    order: int = 99


# ============================================================
# CommandRegistry
# ============================================================

class CommandRegistry:
    """命令注册表 - 消灭巨型 if/elif

    核心热键保护锁：q, r, h, Enter 等核心命令不允许插件覆盖。
    冲突时抛 KeyError 并打印冲突双方。
    """

    # 受保护的核心热键，插件无法覆盖
    _PROTECTED_KEYS = {"", "q", "r", "h", "\r", "\n"}

    _commands: OrderedDictType[str, Command] = OrderedDict()
    # 双结构：同时维护 list 保持插入顺序
    _command_list: List[Command] = []

    @classmethod
    def register(cls, cmd: Command, force: bool = False) -> None:
        """注册命令，冲突时抛 KeyError

        Args:
            force: 若为 True，跳过受保护热键检查（仅供系统内部注册使用）
        """
        hk = cmd.hotkey.lower()
        # 【v4.3-fix】核心热键保护锁实际生效，但系统内部可用 force=True 注册
        if hk in cls._PROTECTED_KEYS and hk != "" and not force:
            raise KeyError(
                f"热键保护: [{hk}] 是核心系统热键，不允许插件覆盖"
            )
        if hk in cls._commands:
            existing = cls._commands[hk]
            raise KeyError(
                f"热键冲突: [{hk}] 已被 '{existing.description}' ({existing.category}) 占用，"
                f"'{cmd.description}' ({cmd.category}) 无法注册"
            )
        cls._commands[hk] = cmd
        cls._command_list.append(cmd)
        # 按 order 排序
        cls._command_list.sort(key=lambda c: (c.category, c.order, c.hotkey))

    @classmethod
    def unregister(cls, hotkey: str) -> bool:
        """注销指定热键的命令"""
        hk = hotkey.lower()
        if hk not in cls._commands:
            return False
        cmd = cls._commands.pop(hk)
        cls._command_list.remove(cmd)
        return True

    @classmethod
    def execute(cls, hotkey: str, context: CommandContext) -> bool:
        """执行命令，返回是否找到并执行了命令

        Args:
            hotkey: 用户输入的热键
            context: 传递 core, samples, step 等状态
                      回调可通过 context["_result"] = value 设置返回值
        """
        hk = hotkey.lower()
        # 特殊处理 Enter 键（可能传入空字符串）
        if hk == "" and "" not in cls._commands:
            return False
        cmd = cls._commands.get(hk)
        if not cmd:
            return False
        # requires_trained 检查：放行已训练（history 非空）的网络
        if cmd.requires_trained:
            core = context.get("core")
            if core:
                has_templates = bool(getattr(core, "templates", None))
                has_graphs = bool(getattr(core, "graphs", None))
                has_history = bool(getattr(core, "history", None))
                if not has_templates and not has_graphs and not has_history:
                    from engine.utils import C
                    print(f"  {C.YEL}⚠ 请先训练或加载模型{C.RST}")
                    return True
                if not has_templates and not has_graphs and has_history:
                    from engine.utils import C
                    print(f"  {C.YEL}⚠ 当前无模板/图（图形特征未被学习），仍可继续操作{C.RST}")
        try:
            # 回调签名兼容：优先传 **context，回调自行决定接收哪些参数
            result = cmd.callback(**context)
            if result is not None:
                context["_result"] = result
        except TypeError as e:
            # 若回调不接受 **kwargs，尝试传 (core, **context)
            core = context.get("core")
            if core:
                try:
                    result = cmd.callback(core, **{k: v for k, v in context.items() if k != "core"})
                    if result is not None:
                        context["_result"] = result
                except TypeError:
                    from engine.utils import C
                    print(f"  {C.RED}[命令错误] {cmd.hotkey}: 回调签名不匹配: {e}{C.RST}")
            else:
                from engine.utils import C
                print(f"  {C.RED}[命令错误] {cmd.hotkey}: {e}{C.RST}")
        return True

    @classmethod
    def generate_help(cls) -> str:
        """按 category 分组自动生成帮助文本"""
        lines = []
        categories = OrderedDict()
        for cmd in cls._command_list:
            if cmd.category not in categories:
                categories[cmd.category] = []
            categories[cmd.category].append(cmd)
        for cat, cmds in categories.items():
            entries = "  ".join(f"[{C.CYN}{c.hotkey}{C.RST}]{c.description}" for c in cmds)
            lines.append(f"  {C.BOLD}{cat}{C.RST}: {entries}")
        return "\n".join(lines)

    @classmethod
    def generate_menu_lines(cls) -> List[str]:
        """生成菜单显示用的行列表（供 menu() 使用）"""
        lines = []
        categories = OrderedDict()
        for cmd in cls._command_list:
            if cmd.category not in categories:
                categories[cmd.category] = []
            categories[cmd.category].append(cmd)
        for cat, cmds in categories.items():
            entries = "  ".join(f"[{c.hotkey}]{c.description}" for c in cmds)
            lines.append(f"  {cat}: {entries}")
        return lines

    @classmethod
    def get_menu_hotkeys(cls) -> List[str]:
        """返回所有已注册的菜单热键"""
        return [cmd.hotkey for cmd in cls._command_list]

    @classmethod
    def list_commands(cls, category: str = None) -> List[Command]:
        """返回已注册的命令列表（可选按 category 过滤）

        替代直接访问 _command_list，保持封装。
        """
        if category:
            return [cmd for cmd in cls._command_list if cmd.category == category]
        return list(cls._command_list)

    @classmethod
    def clear(cls) -> None:
        """清空所有命令（用于单测隔离）"""
        cls._commands.clear()
        cls._command_list.clear()
