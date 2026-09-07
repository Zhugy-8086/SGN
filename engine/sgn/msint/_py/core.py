#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
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
"""MSInt ABI 第二层：核心实现

包含：
  - MetadataEngine: 元数据引擎（槽位表 + 视角注册表 + 任务注册表）
  - TaskContext: 任务上下文（视角过滤器）
  - ViewEngine: 视角引擎（基础视角 + 命名视角 + 视角链解析）
  - MSInt: MSInt 核心类（实现 MSIntProtocol 协议）

设计原则：
  - 视角是临时解码的，不占持久存储
  - 基础视角从槽位表自动生成
  - 命名视角由消费者注册
  - 任务上下文是视角过滤器，不改变底层存储

详见: 内部档案
"""

from __future__ import annotations

import ast as _ast
import re
from difflib import get_close_matches
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .backends import (
    LazyBackend,
    PackedBackend,
    SlotBackend,
    StorageBackend,
    auto_select_backend,
    get_backend_class,
)
from .protocol import (
    ABI_VERSION,
    MSIntProtocol,
    SlotInfo,
    SlotNotFoundError,
    SlotValidationError,
    ViewDef,
    ViewNotFoundError,
    ViewNotAllowedError,
    VIEW_TYPE_BITSPLIT,
    VIEW_TYPE_CONCAT,
    VIEW_TYPE_DERIVE,
    VIEW_TYPE_FORMULA,
    VIEW_TYPE_MAP,
    VIEW_TYPE_SLOT,
)
from .version import CURRENT_ABI_VERSION


# ============================================================
# v5.3.2 第一阶段：错误信息建议辅助函数
# ============================================================

def _format_suggestion(name: str, available: List[str], *,
                       noun: str = "视角") -> str:
    """格式化错误信息建议字符串

    用 `difflib.get_close_matches` 在 `available` 中找最近匹配，
    附在错误信息末尾，帮助使用者快速定位拼写错误或了解可用选项。

    Args:
        name: 使用者输入的（不存在的）名称
        available: 可用名称列表
        noun: 名词（"视角" / "槽位"），用于措辞

    Returns:
        形如以下的建议字符串：
            ''
            或
            '。\\n  Did you mean: \\'category\\'?\\n  Available: category, level, seq'

    设计：
      - 返回空串当 available 为空（无可建议）
      - `get_close_matches(n=3, cutoff=0.6)` 标准库即可，零依赖
      - `available` 按字典序排序，保证错误信息稳定（便于测试断言）
      - 最多列 10 个可用项，避免错误信息过长
    """
    if not available:
        return ""
    sorted_available = sorted(available)
    close = get_close_matches(name, sorted_available, n=3, cutoff=0.6)
    parts: List[str] = []
    if close:
        quoted = ", ".join(f"'{c}'" for c in close)
        parts.append(f"  Did you mean: {quoted}")
    # 最多列 10 个，避免过长
    shown = sorted_available[:10]
    avail_str = ", ".join(shown)
    if len(sorted_available) > 10:
        avail_str += f", ... ({len(sorted_available)} total)"
    parts.append(f"  Available: {avail_str}")
    return "。\n" + "\n".join(parts)


# ============================================================
# 公式视角安全求值器（安全审计 2026-08-16 T1，替代 eval）
# ============================================================

# 仅允许的二元运算符（不含 ** / 真除 / 比较 / 布尔运算）
_FORMULA_BINOPS = {
    _ast.Add: lambda a, b: a + b,
    _ast.Sub: lambda a, b: a - b,
    _ast.Mult: lambda a, b: a * b,
    _ast.FloorDiv: lambda a, b: a // b,
    _ast.Mod: lambda a, b: a % b,
    _ast.BitAnd: lambda a, b: a & b,
    _ast.BitOr: lambda a, b: a | b,
    _ast.BitXor: lambda a, b: a ^ b,
    _ast.LShift: None,  # 特殊处理（移位宽度上限防护）
    _ast.RShift: None,
}
_FORMULA_UNARYOPS = {
    _ast.UAdd: lambda v: +v,
    _ast.USub: lambda v: -v,
    _ast.Invert: lambda v: ~v,
}
# 移位宽度上限（防 1 << 10**9 类内存 DoS；槽位 bits 上限 64，4096 绰绰有余）
_MAX_SHIFT = 4096


def _eval_formula_expr(expr: str, namespace: Dict[str, int]) -> int:
    """AST 白名单求值：仅 BinOp/UnaryOp/Name/整型 Constant，其他节点一律拒绝"""
    try:
        tree = _ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"语法错误: {e}") from e

    def ev(node: "_ast.AST") -> int:
        if isinstance(node, _ast.Expression):
            return ev(node.body)
        if isinstance(node, _ast.BinOp):
            op_type = type(node.op)
            if op_type not in _FORMULA_BINOPS:
                raise ValueError(f"不支持的运算符: {op_type.__name__}")
            left, right = ev(node.left), ev(node.right)
            if op_type is _ast.LShift or op_type is _ast.RShift:
                if right < 0 or right > _MAX_SHIFT:
                    raise ValueError(f"移位宽度越界 (0..{_MAX_SHIFT}): {right}")
                return left << right if op_type is _ast.LShift else left >> right
            return _FORMULA_BINOPS[op_type](left, right)
        if isinstance(node, _ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _FORMULA_UNARYOPS:
                raise ValueError(f"不支持的一元运算符: {op_type.__name__}")
            return _FORMULA_UNARYOPS[op_type](ev(node.operand))
        if isinstance(node, _ast.Constant):
            if isinstance(node.value, int) and not isinstance(node.value, bool):
                return node.value
            raise ValueError(f"仅支持整型常量，得到 {type(node.value).__name__}")
        if isinstance(node, _ast.Name):
            if node.id in namespace:
                return namespace[node.id]
            raise ValueError(f"未知变量 '{node.id}'（可用: {sorted(namespace)}）")
        raise ValueError(f"不支持的表达式节点: {type(node).__name__}")

    return ev(tree)


# ============================================================
# 内置映射函数
# ============================================================

_BUILTIN_MAPPINGS: Dict[str, Callable[[int], int]] = {
    "identity": lambda x: x,
    "log": lambda x: (int.bit_length(x) if x > 0 else 0),
    "sqrt_int": lambda x: int(x ** 0.5),
    "negate": lambda x: -x,
}


def register_mapping(name: str, fn: Callable[[int], int]) -> None:
    """注册自定义映射函数

    Args:
        name: 映射函数名
        fn: 映射函数（int → int）
    """
    _BUILTIN_MAPPINGS[name] = fn


def get_mapping(name: str) -> Callable[[int], int]:
    """获取映射函数

    Raises:
        ValueError: 映射函数未注册（v5.3.2 P4 修复：从 KeyError 改为 ValueError，
            统一 register_view(strict=True) 的异常类型契约，与视角定义非法的
            ``ValueError`` 一致）
    """
    if name not in _BUILTIN_MAPPINGS:
        raise ValueError(
            f"未知映射函数 '{name}'，已注册: {list(_BUILTIN_MAPPINGS.keys())}"
        )
    return _BUILTIN_MAPPINGS[name]


# ============================================================
# 元数据引擎
# ============================================================

class MetadataEngine:
    """元数据引擎

    管理槽位表、视角注册表、任务注册表。

    职责：
      1. 槽位表管理：存储和查询槽位定义
      2. 视角注册表：存储和查询命名视角
      3. 任务注册表：存储和查询任务上下文
      4. 自描述输出：生成 describe() 的元数据部分
    """

    def __init__(self, slot_table: List[SlotInfo]):
        """初始化

        Args:
            slot_table: 槽位定义列表
        """
        self._slot_table: List[SlotInfo] = list(slot_table)
        self._slot_indices: Dict[str, int] = {
            s.name: i for i, s in enumerate(self._slot_table)
        }
        self._views: Dict[str, ViewDef] = {}
        self._tasks: Dict[str, List[str]] = {}

    # ---- 槽位表 ----
    @property
    def slot_table(self) -> List[SlotInfo]:
        """槽位定义列表（副本）"""
        return list(self._slot_table)

    def slot_names(self) -> List[str]:
        """所有槽位名"""
        return [s.name for s in self._slot_table]

    def slot_index(self, name: str) -> int:
        """根据槽位名查找索引

        Raises:
            SlotNotFoundError: 槽位名不存在
        """
        if name not in self._slot_indices:
            suggestion = _format_suggestion(
                name, self.slot_names(), noun="槽位"
            )
            raise SlotNotFoundError(f"槽位 '{name}' 不存在{suggestion}")
        return self._slot_indices[name]

    def get_slot(self, name: str) -> SlotInfo:
        """根据槽位名获取槽位定义

        Raises:
            SlotNotFoundError: 槽位名不存在
        """
        idx = self.slot_index(name)
        return self._slot_table[idx]

    def has_slot(self, name: str) -> bool:
        """检查槽位是否存在"""
        return name in self._slot_indices

    # ---- 视角注册表 ----
    def register_view(self, view_def: ViewDef) -> None:
        """注册一个命名视角

        Args:
            view_def: 视角定义

        Raises:
            SlotNotFoundError: 视角引用的槽位不存在
            ValueError: 视角定义非法
        """
        # 校验源槽位存在
        for slot_name in view_def.slots:
            if not self.has_slot(slot_name):
                suggestion = _format_suggestion(
                    slot_name, self.slot_names(), noun="槽位"
                )
                raise SlotNotFoundError(
                    f"视角 '{view_def.name}' 引用的槽位 '{slot_name}' "
                    f"不存在{suggestion}"
                )
        # 校验 bitsplit 的 target_bits
        if view_def.view_type == VIEW_TYPE_BITSPLIT:
            if view_def.target_bits not in (8, 16, 32, 64):
                raise ValueError(
                    f"bitsplit 视角 '{view_def.name}' 的 target_bits "
                    f"必须是 8/16/32/64，得到 {view_def.target_bits}"
                )
            if not view_def.slots:
                raise ValueError(
                    f"bitsplit 视角 '{view_def.name}' 必须指定源槽位"
                )
            source_slot = self.get_slot(view_def.slots[0])
            if view_def.target_bits > source_slot.bits:
                raise ValueError(
                    f"bitsplit 视角 '{view_def.name}' 的 target_bits "
                    f"({view_def.target_bits}) 不能大于源槽位 "
                    f"'{source_slot.name}' 的 bits ({source_slot.bits})"
                )
        # 校验 map 的 mapping
        if view_def.view_type == VIEW_TYPE_MAP:
            if not view_def.mapping:
                raise ValueError(
                    f"map 视角 '{view_def.name}' 必须指定 mapping"
                )
        # 校验 formula/derive 的 expr
        if view_def.view_type in (VIEW_TYPE_FORMULA, VIEW_TYPE_DERIVE):
            if not view_def.expr:
                raise ValueError(
                    f"{view_def.view_type} 视角 '{view_def.name}' 必须指定 expr"
                )

        self._views[view_def.name] = view_def

    def get_view(self, name: str) -> Optional[ViewDef]:
        """获取命名视角定义（不存在返回 None）"""
        return self._views.get(name)

    def has_view(self, name: str) -> bool:
        """检查命名视角是否已注册"""
        return name in self._views

    def unregister_view(self, name: str) -> None:
        """v5.3.2 第三阶段 #4.3：撤销命名视角注册

        用于 register_view(strict=True) 校验失败时回滚。
        视角不存在则静默忽略（幂等）。

        Args:
            name: 视角名称
        """
        self._views.pop(name, None)

    def view_names(self) -> List[str]:
        """所有命名视角名"""
        return list(self._views.keys())

    # ---- 任务注册表 ----
    def register_task(self, task: str, view_names: List[str]) -> None:
        """注册一个任务上下文

        Args:
            task: 任务名
            view_names: 任务可见的视角名列表

        Raises:
            ViewNotFoundError: 视角名既不是命名视角也不是基础视角
        """
        if not task:
            raise ValueError("任务名不能为空")
        # 校验视角名有效（命名视角或槽位名）
        for vn in view_names:
            if not self.has_view(vn) and not self.has_slot(vn):
                suggestion = _format_suggestion(
                    vn, self.view_names() + self.slot_names()
                )
                raise ViewNotFoundError(
                    f"任务 '{task}' 引用的视角 '{vn}' 不存在{suggestion}"
                )
        self._tasks[task] = list(view_names)

    def get_task_views(self, task: str) -> List[str]:
        """获取任务可见的视角列表

        Raises:
            KeyError: 任务未注册
        """
        if task not in self._tasks:
            raise KeyError(f"任务 '{task}' 未注册")
        return list(self._tasks[task])

    def has_task(self, task: str) -> bool:
        """检查任务是否已注册"""
        return task in self._tasks

    def task_names(self) -> List[str]:
        """所有任务名"""
        return list(self._tasks.keys())

    # ---- 自描述 ----
    def describe(self, backend_name: str, abi_version: str) -> Dict[str, Any]:
        """生成自描述元数据"""
        return {
            "abi_version": abi_version,
            "backend": backend_name,
            "slots": [s.to_dict() for s in self._slot_table],
            "views": {name: vdef.to_dict() for name, vdef in self._views.items()},
            "tasks": {task: list(views) for task, views in self._tasks.items()},
        }

    # ---- 序列化 ----
    def serialize(self) -> Dict[str, Any]:
        """序列化元数据（不含后端状态）"""
        return {
            "slot_table": [s.to_dict() for s in self._slot_table],
            "views": {name: vdef.to_dict() for name, vdef in self._views.items()},
            "tasks": {task: list(views) for task, views in self._tasks.items()},
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "MetadataEngine":
        """从字典反序列化"""
        slot_table = [SlotInfo.from_dict(s) for s in data["slot_table"]]
        engine = cls(slot_table)
        for name, vdef_data in data.get("views", {}).items():
            engine.register_view(ViewDef.from_dict(vdef_data))
        for task, view_names in data.get("tasks", {}).items():
            engine.register_task(task, view_names)
        return engine


# ============================================================
# 任务上下文
# ============================================================

class TaskContext:
    """任务上下文——视角过滤器

    绑定一个任务名，只允许访问该任务注册的视角。
    不改变底层存储，只改变可见的视角集合。
    """

    def __init__(self, msint: "MSInt", task: str):
        """初始化

        Args:
            msint: 所属的 MSInt 实例
            task: 任务名

        Raises:
            KeyError: 任务未注册
        """
        self._msint = msint
        self._task = task
        self._allowed_views = msint._metadata.get_task_views(task)

    @property
    def task(self) -> str:
        """任务名"""
        return self._task

    def view(self, view_name: str) -> Union[int, List[int]]:
        """请求视角（只允许任务注册的视角）

        Raises:
            ViewNotAllowedError: 视角不被任务允许
            ViewNotFoundError: 视角不存在
        """
        if view_name not in self._allowed_views:
            # v5.3.2 第一阶段 #1.3：添加"如何注册到任务"的建议
            # 区分两种情况：视角存在但未被任务允许 / 视角根本不存在
            allowed_sorted = sorted(self._allowed_views)
            ms = self._msint
            if ms.has_view(view_name) or ms._metadata.has_slot(view_name):
                # 视角存在但未被任务允许 → 建议重新注册任务
                msg = (
                    f"视角 '{view_name}' 不被任务 '{self._task}' 允许。\n"
                    f"  任务 '{self._task}' 当前允许的视角: {', '.join(allowed_sorted) or '(空)'}\n"
                    f"  To allow this view, re-register the task:\n"
                    f"    ms.register_task('{self._task}', "
                    f"{allowed_sorted + [view_name]!r})"
                )
            else:
                # 视角根本不存在 → 建议最近匹配
                all_views = ms.views()
                suggestion = _format_suggestion(view_name, all_views)
                msg = (
                    f"视角 '{view_name}' 不被任务 '{self._task}' 允许"
                    f"（且该视角未注册）{suggestion}"
                )
            raise ViewNotAllowedError(msg)
        return self._msint.view(view_name)

    def views(self) -> List[str]:
        """返回任务可见的视角列表"""
        return list(self._allowed_views)

    def has_view(self, view_name: str) -> bool:
        """检查视角是否被任务允许"""
        return view_name in self._allowed_views

    # 代理 MSInt 的元数据方法
    def describe(self) -> Dict[str, Any]:
        """返回元数据（只含任务可见的视角）

        views 字典包含：
          - 任务允许的命名视角（含完整 ViewDef）
          - 任务允许的槽位名（作为基础视角，ViewDef 为 slot 类型）
        """
        full = self._msint.describe()
        filtered_views = {}
        for name in self._allowed_views:
            if name in full["views"]:
                # 命名视角
                filtered_views[name] = full["views"][name]
            elif self._msint._metadata.has_slot(name):
                # 基础视角（槽位读取）
                filtered_views[name] = {
                    "name": name,
                    "view_type": VIEW_TYPE_SLOT,
                    "slots": [name],
                    "expr": "",
                    "target_bits": 0,
                    "mapping": "",
                    "index": None,
                }
        full["views"] = filtered_views
        full["current_task"] = self._task
        return full

    @property
    def abi_version(self) -> str:
        """ABI 版本号"""
        return self._msint.abi_version


# ============================================================
# 视角引擎
# ============================================================

# 视角链表达式解析的正则
_CONCAT_PATTERN = re.compile(r"^concat\(([^)]+)\)$")
_MAP_PATTERN = re.compile(r"^map\(([^,]+),\s*([^)]+)\)$")
_BITSPLIT_INDEX_PATTERN = re.compile(r"^([^.]+)\.int(\d+)\[(\d+)\]$")
_BITSPLIT_PATTERN = re.compile(r"^([^.]+)\.int(\d+)$")


class ViewEngine:
    """视角引擎

    负责把"视角名"翻译成"解码操作"。

    支持三阶视角：
      1. 基础视角（自动生成）：
         - slot_name → 槽位读取
         - slot_name.intN → 位拆分
         - slot_name.intN[k] → 位拆分索引
         - concat(s1, s2, ...) → 拼接
      2. 命名视角（手动注册）：从 metadata 查询
      3. 视角链（组合）：map(slot.intN, mapping) 等
    """

    def __init__(self, metadata: MetadataEngine):
        self._metadata = metadata

    def resolve(self, view_name: str, backend: StorageBackend
                ) -> Union[int, List[int]]:
        """解析视角名并返回解码值

        Args:
            view_name: 视角名或视角链表达式
            backend: 存储后端

        Returns:
            解码后的值

        Raises:
            ViewNotFoundError: 视角不存在
        """
        # 1. 命名视角优先
        vdef = self._metadata.get_view(view_name)
        if vdef is not None:
            return self._resolve_view_def(vdef, backend)

        # 2. 视角链：map(...)
        m = _MAP_PATTERN.match(view_name)
        if m:
            inner = m.group(1).strip()
            mapping_name = m.group(2).strip()
            inner_value = self.resolve(inner, backend)
            # 内部结果可能是 int 或 list
            if isinstance(inner_value, list):
                # 对每个元素应用映射
                mapping_fn = get_mapping(mapping_name)
                return [mapping_fn(v) for v in inner_value]
            else:
                mapping_fn = get_mapping(mapping_name)
                return mapping_fn(inner_value)

        # 3. 拼接视角：concat(s1, s2, ...)
        m = _CONCAT_PATTERN.match(view_name)
        if m:
            slot_names = [s.strip() for s in m.group(1).split(",")]
            return self._resolve_concat(slot_names, backend)

        # 4. 位拆分索引：expr.int8[1]（expr 可以是槽位名或复合表达式）
        m = _BITSPLIT_INDEX_PATTERN.match(view_name)
        if m:
            expr = m.group(1)
            target_bits = int(m.group(2))
            idx = int(m.group(3))
            raw = self._resolve_expr_or_slot(expr, target_bits, backend)
            parts = self._bitsplit_value(raw, target_bits)
            if idx < 0 or idx >= len(parts):
                raise IndexError(
                    f"位拆分索引 {idx} 超出范围 [0, {len(parts)})"
                )
            return parts[idx]

        # 5. 位拆分：expr.int8（expr 可以是槽位名或复合表达式）
        m = _BITSPLIT_PATTERN.match(view_name)
        if m:
            expr = m.group(1)
            target_bits = int(m.group(2))
            raw = self._resolve_expr_or_slot(expr, target_bits, backend)
            return self._bitsplit_value(raw, target_bits)

        # 6. 槽位读取
        if self._metadata.has_slot(view_name):
            idx = self._metadata.slot_index(view_name)
            return backend.get(idx)

        # v5.3.2 第一阶段 #1.1：添加最近匹配 + 可用视角列表
        suggestion = _format_suggestion(
            view_name, self.list_available_views()
        )
        raise ViewNotFoundError(f"视角 '{view_name}' 不存在{suggestion}")

    def _resolve_view_def(self, vdef: ViewDef,
                          backend: StorageBackend) -> Union[int, List[int]]:
        """解析命名视角"""
        if vdef.view_type == VIEW_TYPE_SLOT:
            if not vdef.slots:
                raise ValueError(
                    f"slot 视角 '{vdef.name}' 必须指定源槽位"
                )
            idx = self._metadata.slot_index(vdef.slots[0])
            return backend.get(idx)

        elif vdef.view_type == VIEW_TYPE_BITSPLIT:
            if vdef.target_bits == 0:
                raise ValueError(
                    f"bitsplit 视角 '{vdef.name}' 必须指定 target_bits"
                )
            result = self._resolve_bitsplit(
                vdef.slots[0], vdef.target_bits, backend
            )
            if vdef.index is not None:
                if vdef.index < 0 or vdef.index >= len(result):
                    raise IndexError(
                        f"bitsplit 视角 '{vdef.name}' 的 index {vdef.index} "
                        f"超出范围 [0, {len(result)})"
                    )
                return result[vdef.index]
            return result

        elif vdef.view_type == VIEW_TYPE_CONCAT:
            return self._resolve_concat(vdef.slots, backend)

        elif vdef.view_type == VIEW_TYPE_FORMULA:
            return self._resolve_formula(vdef, backend)

        elif vdef.view_type == VIEW_TYPE_MAP:
            if not vdef.slots:
                raise ValueError(
                    f"map 视角 '{vdef.name}' 必须指定源槽位"
                )
            idx = self._metadata.slot_index(vdef.slots[0])
            raw = backend.get(idx)
            mapping_fn = get_mapping(vdef.mapping)
            return mapping_fn(raw)

        elif vdef.view_type == VIEW_TYPE_DERIVE:
            return self._resolve_derive(vdef, backend)

        else:
            raise ValueError(
                f"未知视角类型 '{vdef.view_type}'（视角 '{vdef.name}'）"
            )

    def _resolve_bitsplit(self, slot_name: str, target_bits: int,
                          backend: StorageBackend) -> List[int]:
        """位拆分：把槽位值按 target_bits 拆分成多个分片"""
        if not self._metadata.has_slot(slot_name):
            suggestion = _format_suggestion(
                slot_name, self._metadata.slot_names(), noun="槽位"
            )
            raise ViewNotFoundError(f"槽位 '{slot_name}' 不存在{suggestion}")
        slot = self._metadata.get_slot(slot_name)
        if target_bits > slot.bits:
            raise ValueError(
                f"target_bits({target_bits}) 不能大于源槽位 "
                f"'{slot_name}' 的 bits({slot.bits})"
            )
        idx = self._metadata.slot_index(slot_name)
        raw = backend.get(idx)
        if raw < 0:
            # 负数转补码表示
            raw = raw & ((1 << slot.bits) - 1)
        # 拆分
        mask = (1 << target_bits) - 1
        result = []
        remaining = raw
        for _ in range((slot.bits + target_bits - 1) // target_bits):
            result.append(remaining & mask)
            remaining >>= target_bits
        return result

    def _resolve_bitsplit_index(self, slot_name: str, target_bits: int,
                                idx: int, backend: StorageBackend) -> int:
        """位拆分索引：取第 idx 个分片"""
        parts = self._resolve_bitsplit(slot_name, target_bits, backend)
        if idx < 0 or idx >= len(parts):
            raise IndexError(
                f"位拆分索引 {idx} 超出范围 [0, {len(parts)})"
            )
        return parts[idx]

    def _resolve_expr_or_slot(self, expr: str, target_bits: int,
                              backend: StorageBackend) -> int:
        """解析表达式或槽位名为一个 int 值（用于视角链的中间步骤）

        如果 expr 是已知槽位名，直接读取；否则递归解析为复合表达式
        （如 concat(a,b)），再返回其 int 值。

        Args:
            expr: 槽位名或复合表达式（如 "concat(category, level)"）
            target_bits: 期望的位宽（用于校验槽位位宽是否足够）
            backend: 存储后端

        Returns:
            解析后的 int 值

        Raises:
            ValueError: 槽位位宽不足以支持 target_bits 拆分
            ViewNotFoundError: 表达式无法解析
        """
        if self._metadata.has_slot(expr):
            slot = self._metadata.get_slot(expr)
            if target_bits > slot.bits:
                raise ValueError(
                    f"target_bits({target_bits}) 不能大于源槽位 "
                    f"'{expr}' 的 bits({slot.bits})"
                )
            idx = self._metadata.slot_index(expr)
            return backend.get(idx)
        # 递归解析复合表达式
        inner_value = self.resolve(expr, backend)
        if isinstance(inner_value, int):
            return inner_value
        if isinstance(inner_value, list):
            raise ViewNotFoundError(
                f"表达式 '{expr}' 返回了列表，无法对列表做位拆分——"
                f"请先对单个元素取索引，如 'concat(a,b).int8[0]'"
            )
        raise ViewNotFoundError(
            f"表达式 '{expr}' 返回了 {type(inner_value).__name__}，无法做位拆分"
        )

    @staticmethod
    def _bitsplit_value(raw: int, target_bits: int) -> List[int]:
        """对任意 int 值做位拆分（不依赖槽位定义）

        Args:
            raw: 原始 int 值
            target_bits: 每个分片的位宽

        Returns:
            分片列表（低位在前）
        """
        mask = (1 << target_bits) - 1
        result = []
        remaining = raw
        total_bits = max(raw.bit_length(), 1)
        for _ in range((total_bits + target_bits - 1) // target_bits):
            result.append(remaining & mask)
            remaining >>= target_bits
        return result

    def _resolve_concat(self, slot_names: List[str],
                        backend: StorageBackend) -> int:
        """拼接：把多个槽位值按位宽拼接为一个 int

        语义：第一个槽位在高位，最后一个槽位在低位。
        例：concat(category, level) → category 在 bits 8-15，level 在 bits 0-7
        """
        result = 0
        # 从前往后拼接：第一个槽位最后左移（成为最高位）
        for slot_name in slot_names:
            if not self._metadata.has_slot(slot_name):
                suggestion = _format_suggestion(
                    slot_name, self._metadata.slot_names(), noun="槽位"
                )
                raise ViewNotFoundError(f"槽位 '{slot_name}' 不存在{suggestion}")
            slot = self._metadata.get_slot(slot_name)
            idx = self._metadata.slot_index(slot_name)
            raw = backend.get(idx)
            if raw < 0:
                raw = raw & ((1 << slot.bits) - 1)
            # 先把已有结果左移，再或上新值
            result = (result << slot.bits) | (raw & ((1 << slot.bits) - 1))
        return result

    def _resolve_formula(self, vdef: ViewDef,
                         backend: StorageBackend) -> int:
        """公式视角：按表达式求值

        表达式中用 a, b, c, ... 引用 slots 中的第 1, 2, 3, ... 个槽位。
        仅支持简单的算术表达式（+ - * // % & | ^ << >>）。

        安全审计 2026-08-16 T1：原实现字符白名单 + eval——`**`（两个合法
        `*`）可通过白名单，`9**99999999` 等表达式构成 CPU/内存 DoS。
        改为 AST 白名单求值器（仅 BinOp/UnaryOp/Name/整型 Constant），
        移位宽度另有上限防护。
        """
        if not vdef.slots:
            raise ValueError(
                f"formula 视角 '{vdef.name}' 必须指定源槽位"
            )
        # 收集槽位值
        values = []
        for slot_name in vdef.slots:
            idx = self._metadata.slot_index(slot_name)
            values.append(backend.get(idx))
        # 构造命名空间
        var_names = [chr(ord('a') + i) for i in range(len(values))]
        namespace = dict(zip(var_names, values))
        try:
            result = _eval_formula_expr(vdef.expr, namespace)
        except ValueError as e:
            raise ValueError(
                f"formula 视角 '{vdef.name}' 的 expr 求值失败: {e}"
            ) from e
        if not isinstance(result, int):
            raise ValueError(
                f"formula 视角 '{vdef.name}' 的 expr 结果必须是 int，"
                f"得到 {type(result).__name__}"
            )
        return result

    def _resolve_derive(self, vdef: ViewDef,
                        backend: StorageBackend) -> int:
        """推导视角：与 formula 相同的求值逻辑"""
        return self._resolve_formula(vdef, backend)

    def list_available_views(self) -> List[str]:
        """列出所有可用的视角名（命名视角 + 槽位名）

        不包含视角链表达式（如 concat(...)、map(...)）。
        """
        views = list(self._metadata.view_names())
        views.extend(self._metadata.slot_names())
        return views

    # ---- v5.3.2 第三阶段 #4.1：视角链静态校验 ----
    def validate(self, view_name: str) -> None:
        """静态校验视角表达式可解析（不求值）

        用 DryRunBackend（每槽位返回全 1 值）跑一遍 resolve 路径，覆盖：
          - 命名视角存在性
          - 视角链表达式语法（concat / map / intN / intN[k]）
          - 引用的槽位/视角存在
          - bitsplit target_bits 合法且 index 在静态范围内
            （基于位宽推断：长度 = ceil(slot.bits / target_bits)）
          - map mapping 已注册
          - formula expr 编译通过（字符白名单 + 求值）
          - derive 链路可达

        不依赖真实存储值，可在视角注册后立即调用。

        局限性（v5.3.2 P7 文档化）：
          - **formula / derive / map 的 dry-run 可能存在 false positive / false negative**：
            dry-run 用全 1 值跑 eval，某些在 dry-run 下抛异常的 expr 在真实值下
            可能合法（false negative，如 ``a // (a - b)`` 在 a=b=0xFF 时 ZeroDivisionError，
            但 a=10, b=3 时合法），反之亦然（false positive，如 ``100 // a`` 在 a=0xFF
            合法但 a=0 时 ZeroDivisionError）。
          - 这些视角的最终正确性仍依赖真实运行时校验。validate 只保证"表达式语法
            + 引用结构"正确，不保证"所有真实值下不抛异常"。
          - **不校验任务上下文可见性**：本方法只校验视角可解析，不校验视角是否被
            任务允许。任务级校验请用 ``TaskContext``。

        Args:
            view_name: 视角名称或视角链表达式

        Raises:
            ViewNotFoundError: 视角不存在
            ValueError: 视角定义非法（如 target_bits 过大、expr 含非法字符、
                strict=True 时 mapping 未注册等）
            IndexError: bitsplit index 超出静态范围
        """
        dryrun = _DryRunBackend(self._metadata.slot_table)
        self.resolve(view_name, dryrun)


class _DryRunBackend(StorageBackend):
    """视角校验用 backend：对每个槽位返回全 1 值

    用 (1<<bits)-1 作为槽位值，保证：
      - bitsplit 长度 = ceil(slot.bits / target_bits)（与真实运行时一致）
      - map mapping 能正常调用（非空输入）
      - formula expr 求值不会因 0 触发意外（如 0//0）
      - concat 拼接出全 1 的合法 int

    v5.3.2 P5 修复：继承 ``StorageBackend`` 满足 ``ViewEngine.resolve`` 的类型标注
    （``backend: StorageBackend``），防止未来 mypy 报错或接口扩展时遗漏。
    set / serialize / deserialize 显式抛 ``NotImplementedError`` 表明 dry-run 不可写。
    """

    name = "dryrun"
    immutable = True

    def __init__(self, slot_table: List[SlotInfo]):
        super().__init__(slot_table)

    def get(self, index: int) -> int:
        # P6 修复：显式边界检查，与 SlotBackend.get 行为一致
        if index < 0 or index >= len(self._slot_table):
            raise IndexError(
                f"槽位索引 {index} 超出范围 [0, {len(self._slot_table)})"
            )
        slot = self._slot_table[index]
        return (1 << slot.bits) - 1

    def set(self, index: int, value: int) -> None:
        """dry-run 后端不可写"""
        raise NotImplementedError("_DryRunBackend 不可写，仅用于视角校验")

    def serialize(self) -> Dict[str, Any]:
        """dry-run 后端不可序列化"""
        raise NotImplementedError("_DryRunBackend 不可序列化，仅用于视角校验")

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "StorageBackend":
        """dry-run 后端不可反序列化"""
        raise NotImplementedError("_DryRunBackend 不可反序列化，仅用于视角校验")


# ============================================================
# MSInt 核心类
# ============================================================

class MSInt(MSIntProtocol):
    """多语义整数存储（ABI 协议实现）

    不是 int 的子类，不参与运算。职责：
      1. 存储：多后端存储动态值
      2. 解码：视角引擎临时解码，不占持久存储
      3. 描述：元数据自描述
      4. 任务：多任务上下文隔离

    程序运算用普通 int，不经过 MSInt。

    线程安全（v5.3.2 P3 文档化）：
      - MSInt 实例的 ``register_view`` / ``unregister_view`` / ``register_task``
        等写入方法 **未加锁**，设计为单线程使用。
      - ``register_view(strict=True)`` 的"注册→校验→失败回滚"三步不是原子的，
        多线程并发调用时存在竞态窗口（其他线程可能看到"已注册但无法解析"的视角）。
      - 跨线程共享 MSInt 实例时，调用方需自行加锁，或避免并发写入。
      - LevelScheduler 的线程安全（v5.3.1 P1.5）不覆盖 MSInt 实例——MSInt 是
        LevelScheduler 的"消费者"，不是其内部状态。

    Example:
        >>> ms = MSInt.from_slots({"category": 5, "level": 2})
        >>> ms.view("category")
        5
        >>> ms.as_int()  # 拼接所有槽位
        1282
    """

    ABI_VERSION = CURRENT_ABI_VERSION

    def __init__(self, backend: StorageBackend,
                 metadata: MetadataEngine,
                 abi_version: str = CURRENT_ABI_VERSION):
        """初始化

        Args:
            backend: 存储后端
            metadata: 元数据引擎
            abi_version: ABI 版本号
        """
        self._backend = backend
        self._metadata = metadata
        self._abi_version = abi_version
        self._view_engine = ViewEngine(metadata)

    # ---- 工厂方法 ----
    @classmethod
    def from_slots(cls, slots: Dict[str, int], *,
                   slot_defs: Optional[List[SlotInfo]] = None,
                   backend: Optional[str] = None) -> "MSInt":
        """从命名槽位字典创建 MSInt

        Args:
            slots: {"category": 5, "level": 2, ...}
            slot_defs: 槽位定义列表（可选，不提供则按值自动推断 bits）
            backend: 存储后端名（可选，不提供则自动选择）

        Returns:
            MSInt 实例
        """
        names = list(slots.keys())
        values = list(slots.values())

        # 构造槽位定义
        if slot_defs is None:
            # 自动推断：根据值范围推断 bits
            slot_defs = []
            for name, val in slots.items():
                if val < 0:
                    raise ValueError(
                        f"自动推断不支持负值（槽位 '{name}' = {val}），"
                        f"请显式提供 slot_defs"
                    )
                if val <= 0xFF:
                    bits = 8
                elif val <= 0xFFFF:
                    bits = 16
                elif val <= 0xFFFFFFFF:
                    bits = 32
                else:
                    bits = 64
                slot_defs.append(SlotInfo(name=name, bits=bits, signed=False))
        else:
            # 校验 slot_defs 与 slots 名一致
            def_names = [s.name for s in slot_defs]
            if def_names != names:
                # 允许 slot_defs 顺序与 slots 不同
                if set(def_names) != set(names):
                    raise ValueError(
                        f"slot_defs 名({def_names}) 与 slots 名({names}) 不一致"
                    )
                # 按 slots 顺序重排 slot_defs
                slot_def_map = {s.name: s for s in slot_defs}
                slot_defs = [slot_def_map[n] for n in names]

        # 选择后端
        if backend is None:
            backend = auto_select_backend(len(slot_defs))
        backend_cls = get_backend_class(backend)

        # 创建后端实例
        if backend_cls is LazyBackend:
            raise ValueError(
                "LazyBackend 不支持 from_slots，请用 from_lazy"
            )
        if backend_cls is PackedBackend:
            backend_inst = PackedBackend.from_values(slot_defs, values)
        else:
            backend_inst = backend_cls(slot_table=slot_defs, values=values)

        # 创建元数据引擎
        metadata = MetadataEngine(slot_defs)

        return cls(backend=backend_inst, metadata=metadata)

    @classmethod
    def from_values(cls, values: List[int], *,
                    slot_defs: Optional[List[SlotInfo]] = None,
                    backend: Optional[str] = None) -> "MSInt":
        """从值列表创建 MSInt（无命名槽位，按位置索引）

        Args:
            values: 值列表
            slot_defs: 槽位定义列表（可选，不提供则按位置自动生成 slot_0, slot_1, ...）
            backend: 存储后端名（可选）
        """
        if slot_defs is None:
            slot_defs = []
            for i, val in enumerate(values):
                if val < 0:
                    raise ValueError(
                        f"自动推断不支持负值（位置 {i} = {val}），"
                        f"请显式提供 slot_defs"
                    )
                if val <= 0xFF:
                    bits = 8
                elif val <= 0xFFFF:
                    bits = 16
                elif val <= 0xFFFFFFFF:
                    bits = 32
                else:
                    bits = 64
                slot_defs.append(SlotInfo(name=f"slot_{i}", bits=bits))
        if len(slot_defs) != len(values):
            raise ValueError(
                f"slot_defs 长度({len(slot_defs)}) 与 values 长度({len(values)}) 不匹配"
            )
        slots_dict = {s.name: v for s, v in zip(slot_defs, values)}
        return cls.from_slots(slots_dict, slot_defs=slot_defs, backend=backend)

    @classmethod
    def from_lazy(cls, slot_defs: List[SlotInfo],
                  compute_fn: Callable[[], List[int]]) -> "MSInt":
        """从惰性计算函数创建 MSInt（LazyBackend）

        Args:
            slot_defs: 槽位定义列表
            compute_fn: 计算函数，返回所有槽位值列表
        """
        backend = LazyBackend(slot_table=slot_defs, compute_fn=compute_fn)
        metadata = MetadataEngine(slot_defs)
        return cls(backend=backend, metadata=metadata)

    @classmethod
    def from_packed(cls, packed_value: int,
                    slot_defs: List[SlotInfo]) -> "MSInt":
        """从打包的 int 值重建 MSInt（int → MSInt 往返转换）

        这是 MSInt → int → MSInt 往返转换的关键入口。
        当程序持有 raw int 值和槽位定义时，可用此方法重建 MSInt
        并正常使用 view() 读取各语义。

        Args:
            packed_value: 打包后的整数值（通常来自 ms.as_int()）
            slot_defs: 槽位定义列表（必须与编码时一致，按 as_int 的高位在前顺序）

        Returns:
            MSInt 实例

        Example:
            >>> sd = [SlotInfo("cat", 8), SlotInfo("lvl", 8)]
            >>> ms = MSInt.from_slots({"cat": 5, "lvl": 2}, slot_defs=sd)
            >>> raw = ms.as_int()
            >>> ms2 = MSInt.from_packed(raw, sd)
            >>> ms2.view("cat")  # 5
        """
        # as_int() 的布局：第一个槽位在高位
        # 按 slot_defs 顺序从高位到低位解包
        slots: Dict[str, int] = {}
        total_bits = sum(s.bits for s in slot_defs)
        shift = total_bits
        for slot in slot_defs:
            shift -= slot.bits
            mask = (1 << slot.bits) - 1
            slots[slot.name] = (packed_value >> shift) & mask
        return cls.from_slots(slots, slot_defs=slot_defs)

    @classmethod
    def auto_split(cls, raw: int, bits: int = 8, *,
                   total_bits: int = 64,
                   prefix: str = "slot") -> "MSInt":
        """自动拆分:把 raw 按固定 bits 拆成多个等宽槽位

        无需预声明字段名,自动生成 N 个等宽 SlotInfo。
        作为 SplitTree 废弃的过渡方案(SplitTree 只支持 2 的幂等宽二分,
        auto_split 支持任意 bits)。

        布局:第一个槽位在高位(与 as_int 一致),便于 from_packed 往返。

        Args:
            raw: 原始整数值(必须 >= 0)
            bits: 每个槽位的位宽(默认 8,支持任意正整数)
            total_bits: 总位宽(默认 64,必须能被 bits 整除)
            prefix: 槽位名前缀(默认 "slot",生成 slot_0, slot_1, ...)

        Returns:
            MSInt 实例,包含 N = total_bits // bits 个等宽槽位

        Raises:
            ValueError: bits <= 0, total_bits <= 0, 或 total_bits 不能被 bits 整除

        Example:
            >>> ms = MSInt.auto_split(0x12345678, bits=8, total_bits=32)
            >>> ms.view("slot_0")  # 0x12 (高 8 位)
            >>> ms.view("slot_3")  # 0x78 (低 8 位)
            >>> ms.as_int()        # 0x12345678 (往返一致)
        """
        if bits <= 0:
            raise ValueError(f"bits 必须 > 0,得到 {bits}")
        if total_bits <= 0:
            raise ValueError(f"total_bits 必须 > 0,得到 {total_bits}")
        if total_bits % bits != 0:
            raise ValueError(
                f"total_bits({total_bits}) 必须能被 bits({bits}) 整除"
            )
        if raw < 0:
            raise ValueError(f"raw 必须 >= 0,得到 {raw}")

        n_slots = total_bits // bits
        slot_defs = [
            SlotInfo(name=f"{prefix}_{i}", bits=bits, signed=False)
            for i in range(n_slots)
        ]
        # 按 as_int 布局(高位在前)解包
        slots: Dict[str, int] = {}
        shift = total_bits
        mask = (1 << bits) - 1
        for i in range(n_slots):
            shift -= bits
            slots[f"{prefix}_{i}"] = (raw >> shift) & mask
        return cls.from_slots(slots, slot_defs=slot_defs)

    # ---- ABI 协议接口 ----
    def view(self, view_name: str) -> Union[int, List[int]]:
        """请求一个视角，临时解码（不持久化）

        Args:
            view_name: 视角名称或视角链表达式

        Returns:
            解码后的值

        Raises:
            ViewNotFoundError: 视角不存在
        """
        return self._view_engine.resolve(view_name, self._backend)

    def validate_view(self, view_name: str) -> None:
        """v5.3.2 第三阶段 #4.1：静态校验视角表达式可解析（不求值）

        用全 1 dry-run backend 跑一遍 resolve 路径，校验：
          - 视角名/视角链表达式语法
          - 引用的槽位/视角存在
          - bitsplit target_bits 合法且 index 在静态范围内
          - map mapping 已注册
          - formula expr 编译通过
          - derive 链路可达

        不修改任何状态，可重复调用。常用于：
          1. 注册视角后立即校验（与 ``register_view(strict=True)`` 等价）
          2. 反序列化后验证视角定义完整
          3. 测试中替代 ``ms.view(name)`` 检查视角可用性

        Args:
            view_name: 视角名称或视角链表达式

        Raises:
            ViewNotFoundError: 视角不存在
            ValueError: 视角定义非法
            IndexError: bitsplit index 超出静态范围
        """
        self._view_engine.validate(view_name)

    def views(self) -> List[str]:
        """返回所有可用视角名称（命名视角 + 槽位名，不含视角链表达式）"""
        return self._view_engine.list_available_views()

    def has_view(self, view_name: str) -> bool:
        """检查视角是否存在（命名视角或槽位名）"""
        return self._metadata.has_view(view_name) or self._metadata.has_slot(view_name)

    def describe(self) -> Dict[str, Any]:
        """返回完整元数据"""
        return self._metadata.describe(
            backend_name=self._backend.name,
            abi_version=self._abi_version,
        )

    def slot_table(self) -> List[SlotInfo]:
        """返回槽位表"""
        return self._metadata.slot_table

    def slot_names(self) -> List[str]:
        """返回所有槽位名列表"""
        return self._metadata.slot_names()

    def with_context(self, task: str) -> TaskContext:
        """返回绑定任务上下文的视图代理

        Raises:
            KeyError: 任务未注册
        """
        return TaskContext(self, task)

    def register_task(self, task: str, view_names: List[str]) -> None:
        """注册一个任务上下文"""
        self._metadata.register_task(task, view_names)

    def register_view(self, name: str, view_type: str, *,
                      strict: bool = False, **kwargs: Any) -> None:
        """注册一个命名视角

        Args:
            name: 视角名称
            view_type: 视角类型（见 VALID_VIEW_TYPES）
            strict: v5.3.2 第三阶段 #4.3 —— 注册后立即调用 validate_view 校验
                视角可解析（target_bits 合法、index 在范围内、mapping 已注册、
                formula expr 编译通过等）。默认 False 保持向后兼容。
            **kwargs: 视角定义参数（slots, expr, target_bits, mapping, index)

        Raises:
            SlotNotFoundError: 视角引用的槽位不存在
            ValueError: 视角定义非法（含 strict=True 时 mapping 未注册、
                formula expr 非法等）
            IndexError: bitsplit index 超出静态范围（仅 strict=True 时）
            ViewNotFoundError: 视角链引用的子视角不存在（仅 strict=True 时）

        Note:
            strict=True 路径用 try/finally 模式撤销注册，即使 ``KeyboardInterrupt`` /
            ``SystemExit`` 等 ``BaseException`` 打断校验也能回滚，保证不会留下
            "看起来注册了但实际无法解析"的视角。
        """
        vdef = ViewDef(name=name, view_type=view_type, **kwargs)
        self._metadata.register_view(vdef)
        if strict:
            # P2 修复：用 try/finally + 标志位替代 except Exception，
            # 保证 BaseException（如 KeyboardInterrupt）也能触发回滚
            ok = False
            try:
                self.validate_view(name)
                ok = True
            finally:
                if not ok:
                    # v5.3.2 第三阶段 #4.3：strict 校验失败时撤销注册
                    self._metadata.unregister_view(name)

    @property
    def abi_version(self) -> str:
        """ABI 版本号"""
        return self._abi_version

    # ---- 存储访问 ----
    def as_int(self) -> int:
        """临时解码为单语义 int（拼接所有槽位）

        把所有槽位按位宽拼接为一个 int。
        """
        slot_names = self._metadata.slot_names()
        if not slot_names:
            return 0
        return self._view_engine._resolve_concat(slot_names, self._backend)

    def update_slot(self, name: str, value: int) -> None:
        """更新单个槽位的值

        Raises:
            SlotNotFoundError: 槽位不存在
            NotImplementedError: 后端不可变
            SlotValidationError: 值超范围
        """
        idx = self._metadata.slot_index(name)
        try:
            self._backend.set(idx, value)
        except NotImplementedError:
            raise

    @property
    def backend_name(self) -> str:
        """当前使用的存储后端名"""
        return self._backend.name

    @property
    def backend(self) -> StorageBackend:
        """底层存储后端（供适配器使用）"""
        return self._backend

    @property
    def metadata(self) -> MetadataEngine:
        """元数据引擎（供适配器使用）"""
        return self._metadata

    # ---- 序列化辅助 ----
    def serialize(self) -> Dict[str, Any]:
        """序列化整个 MSInt 状态

        Returns:
            包含 backend_state 和 metadata 的字典
        """
        return {
            "abi_version": self._abi_version,
            "backend": self._backend.name,
            "backend_state": self._backend.serialize(),
            "metadata": self._metadata.serialize(),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "MSInt":
        """从字典反序列化

        Args:
            data: serialize() 输出的字典
        """
        metadata = MetadataEngine.deserialize(data["metadata"])
        backend_name = data["backend"]
        backend_cls = get_backend_class(backend_name)
        backend_state = data["backend_state"]
        # 不同后端的反序列化签名不同
        if backend_cls is SlotBackend:
            backend_inst = SlotBackend.deserialize(
                backend_state, metadata.slot_table
            )
        elif backend_cls is PackedBackend:
            backend_inst = PackedBackend.deserialize(
                backend_state, metadata.slot_table
            )
        elif backend_cls is LazyBackend:
            backend_inst = LazyBackend.deserialize(
                backend_state, metadata.slot_table
            )
        else:
            # 自定义后端：尝试通用接口
            backend_inst = backend_cls.deserialize(backend_state)
        return cls(
            backend=backend_inst,
            metadata=metadata,
            abi_version=data["abi_version"],
        )

    # ---- 协议字符串 ----
    def __repr__(self) -> str:
        return (
            f"MSInt(backend={self._backend.name}, "
            f"slots={self._metadata.slot_names()}, "
            f"abi_version={self._abi_version})"
        )


# ============================================================
# v5.3.2 第三阶段 #3.6：MSInt.preset 类属性
# ============================================================
# 延迟导入避免循环（presets.py 顶部需要 import MSInt），
# 这里在 MSInt 类定义完整后绑定 preset 命名空间。
from .presets import _PresetNamespace  # noqa: E402

MSInt.preset = _PresetNamespace
