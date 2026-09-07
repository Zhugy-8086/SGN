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
"""MSInt ABI 第一层：协议定义

本模块定义 MSInt ABI 的抽象协议，不包含任何具体实现。

设计原则：
  - 协议层不依赖任何具体存储后端
  - 协议层不 import SGN 任何模块
  - 协议层只定义接口（dataclass + Protocol），不实现行为

详见: 内部档案
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable


# ============================================================
# ABI 版本号
# ============================================================

ABI_VERSION = "1.0"


# ============================================================
# 槽位定义
# ============================================================

@dataclass(frozen=True)
class SlotInfo:
    """槽位定义（不可变）

    描述一个物理槽位的元数据。

    Attributes:
        name: 槽位名称（如 "category", "level", "sequence"）
        bits: 位宽（8 / 16 / 32 / 64）。v5.4.0 阶段0.3：可选，None 时从 value_range 自动推导
        signed: 是否有符号
        value_range: 值范围 (min, max)，None 表示按 bits 自动推断
        unit: 语义单位（如 "class_id", "level", "index"），可选
        description: 人类可读描述，可选

    v5.4.0 阶段0.3 方案 E：自动 bit 推导
        当 bits=None 时，从 value_range 自动计算最小位宽：
            bits = ceil(log2(max - min + 1))  （无符号）
            bits = ceil(log2(max - min + 1)) + 1  （有符号，需补码符号位）
        示例：
            SlotInfo("category", bits=None, value_range=(0, 35))  → 6 bit（36 类只需 6 bit）
            SlotInfo("level", bits=None, value_range=(1, 100))     → 7 bit
            SlotInfo("flags", bits=8)                              → 8 bit（显式指定）
    """
    name: str
    bits: Optional[int] = None
    signed: bool = False
    value_range: Optional[Tuple[int, int]] = None
    unit: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SlotInfo.name 不能为空")
        # v5.4.0 阶段0.3: bits=None 时从 value_range 自动推导
        if self.bits is None:
            derived = self._derive_bits_from_range()
            object.__setattr__(self, "bits", derived)
        if self.bits <= 0:
            raise ValueError(
                f"SlotInfo.bits 必须是正整数,得到 {self.bits}"
            )

    def _derive_bits_from_range(self) -> int:
        """v5.4.0 阶段0.3 方案 E：从 value_range 推导最小 bit 数

        Returns:
            最小需要的 bit 数

        Raises:
            ValueError: bits=None 且 value_range=None，无法推导
        """
        if self.value_range is None:
            raise ValueError(
                f"SlotInfo '{self.name}' 的 bits=None 时必须提供 value_range"
            )
        lo, hi = self.value_range
        if lo > hi:
            raise ValueError(
                f"SlotInfo '{self.name}' 的 value_range 下界 > 上界: ({lo}, {hi})"
            )
        span = hi - lo + 1
        if span <= 0:
            raise ValueError(
                f"SlotInfo '{self.name}' 的 value_range span 非正: {span}"
            )
        # 计算需要的 bit 数
        import math
        unsigned_bits = math.ceil(math.log2(span)) if span > 1 else 1
        if self.signed:
            # 有符号需要额外的符号位
            return unsigned_bits + 1
        return unsigned_bits

    @property
    def max_value(self) -> int:
        """最大值（按 bits 和 signed 推断）"""
        if self.value_range is not None:
            return self.value_range[1]
        if self.signed:
            return (1 << (self.bits - 1)) - 1
        return (1 << self.bits) - 1

    @property
    def min_value(self) -> int:
        """最小值（按 bits 和 signed 推断）"""
        if self.value_range is not None:
            return self.value_range[0]
        if self.signed:
            return -(1 << (self.bits - 1))
        return 0

    def validate(self, value: int) -> None:
        """校验值是否在合法范围

        Args:
            value: 待校验的值

        Raises:
            ValueError: 值超出合法范围
        """
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                f"槽位 '{self.name}' 的值必须是 int，得到 {type(value).__name__}"
            )
        if value < self.min_value or value > self.max_value:
            raise ValueError(
                f"槽位 '{self.name}' 的值 {value} 超出范围 "
                f"[{self.min_value}, {self.max_value}]"
            )

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self.name,
            "bits": self.bits,
            "signed": self.signed,
            "value_range": list(self.value_range) if self.value_range else None,
            "unit": self.unit,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SlotInfo":
        """从字典反序列化"""
        return cls(
            name=data["name"],
            bits=data.get("bits"),
            signed=data.get("signed", False),
            value_range=tuple(data["value_range"]) if data.get("value_range") else None,
            unit=data.get("unit", ""),
            description=data.get("description", ""),
        )


# ============================================================
# 视角定义
# ============================================================

# 视角类型常量
VIEW_TYPE_SLOT = "slot"          # 槽位读取
VIEW_TYPE_BITSPLIT = "bitsplit"  # 位拆分
VIEW_TYPE_CONCAT = "concat"      # 拼接
VIEW_TYPE_FORMULA = "formula"    # 公式
VIEW_TYPE_MAP = "map"            # 映射
VIEW_TYPE_DERIVE = "derive"      # 推导

VALID_VIEW_TYPES = {
    VIEW_TYPE_SLOT, VIEW_TYPE_BITSPLIT, VIEW_TYPE_CONCAT,
    VIEW_TYPE_FORMULA, VIEW_TYPE_MAP, VIEW_TYPE_DERIVE,
}


@dataclass
class ViewDef:
    """视角定义（可变，用于注册时构造）

    描述一个命名视角如何从槽位表解码。

    Attributes:
        name: 视角名称（如 "role_id", "priority"）
        view_type: 视角类型（见 VALID_VIEW_TYPES）
        slots: 源槽位名列表（如 ["category", "level"]）
        expr: 表达式（formula / derive 类型用，如 "a * 10 + b"）
        target_bits: 目标位宽（bitsplit 类型用，如 8 表示 int8）
        mapping: 映射函数名（map 类型用，如 "log"）
        index: 索引（bitsplit 类型用，取第 k 个分片，None 表示返回全部）
    """
    name: str
    view_type: str
    slots: List[str] = field(default_factory=list)
    expr: str = ""
    target_bits: int = 0
    mapping: str = ""
    index: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ViewDef.name 不能为空")
        if self.view_type not in VALID_VIEW_TYPES:
            raise ValueError(
                f"ViewDef.view_type 必须是 {VALID_VIEW_TYPES}，得到 {self.view_type}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self.name,
            "view_type": self.view_type,
            "slots": list(self.slots),
            "expr": self.expr,
            "target_bits": self.target_bits,
            "mapping": self.mapping,
            "index": self.index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ViewDef":
        """从字典反序列化"""
        return cls(
            name=data["name"],
            view_type=data["view_type"],
            slots=list(data.get("slots", [])),
            expr=data.get("expr", ""),
            target_bits=data.get("target_bits", 0),
            mapping=data.get("mapping", ""),
            index=data.get("index"),
        )


# ============================================================
# 异常类型
# ============================================================

class MSIntError(Exception):
    """MSInt 异常基类"""


class ViewNotFoundError(MSIntError):
    """视角不存在"""


class ViewNotAllowedError(MSIntError):
    """视角不被任务上下文允许"""


class SlotNotFoundError(MSIntError):
    """槽位不存在"""


class VersionIncompatibleError(MSIntError):
    """ABI 版本不兼容"""


class SlotValidationError(MSIntError):
    """槽位值校验失败"""


# ============================================================
# ABI 协议接口
# ============================================================

@runtime_checkable
class MSIntProtocol(Protocol):
    """MSInt ABI 协议接口

    任何 MSInt 实现必须满足此协议。协议本身不依赖任何具体实现。
    """

    # ---- 视角请求 ----
    def view(self, view_name: str) -> Union[int, List[int]]:
        """请求一个视角，返回临时解码值（不持久化）

        Args:
            view_name: 视角名称或视角链表达式

        Returns:
            解码后的值（int 或 List[int]）

        Raises:
            ViewNotFoundError: 视角不存在
        """
        ...

    def views(self) -> List[str]:
        """返回所有可用视角的名称列表（不含视角链表达式）"""
        ...

    def has_view(self, view_name: str) -> bool:
        """检查视角是否存在（命名视角或基础视角）"""
        ...

    # ---- 元数据 ----
    def describe(self) -> Dict[str, Any]:
        """返回完整元数据（槽位表 + 视角定义 + 任务定义 + 版本信息）"""
        ...

    def slot_table(self) -> List[SlotInfo]:
        """返回槽位表"""
        ...

    def slot_names(self) -> List[str]:
        """返回所有槽位名列表"""
        ...

    # ---- 任务上下文 ----
    def with_context(self, task: str) -> Any:
        """返回绑定任务上下文的视图代理"""
        ...

    def register_task(self, task: str, view_names: List[str]) -> None:
        """注册一个任务上下文（指定任务可见的视角列表）"""
        ...

    # ---- 视角注册（消费者自主） ----
    def register_view(self, name: str, view_type: str, **kwargs: Any) -> None:
        """注册一个命名视角"""
        ...

    # ---- 版本 ----
    @property
    def abi_version(self) -> str:
        """返回 ABI 版本号（如 "1.0"）"""
        ...

    # ---- 存储访问 ----
    def as_int(self) -> int:
        """临时解码为单语义 int（拼接所有槽位）"""
        ...

    def update_slot(self, name: str, value: int) -> None:
        """更新单个槽位的值"""
        ...

    @property
    def backend_name(self) -> str:
        """当前使用的存储后端名"""
        ...


# ============================================================
# 存储后端协议
# ============================================================

@runtime_checkable
class StorageBackendProtocol(Protocol):
    """存储后端协议

    所有存储后端必须满足此协议。
    """

    name: str

    def get(self, index: int) -> int:
        """读取第 index 个槽位"""
        ...

    def set(self, index: int, value: int) -> None:
        """写入第 index 个槽位"""
        ...

    def get_all(self) -> List[int]:
        """读取所有槽位"""
        ...

    def slot_count(self) -> int:
        """槽位数量"""
        ...

    def serialize(self) -> Dict[str, Any]:
        """序列化后端状态"""
        ...

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "StorageBackendProtocol":
        """反序列化后端状态"""
        ...
