#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""MSInt 进阶层导出（v5.3.2 第二阶段 #2.1）

分层导出策略的第二层：视角与映射函数。
入门层（`engine.ms_int`）只暴露 `MSInt` / `SlotInfo` / `ViewDef` / 异常即可覆盖
绝大多数使用场景；本模块暴露视角引擎和映射注册函数，供需要自定义视角或
扩展映射函数的进阶使用者使用。

== 何时用 engine.ms_int.views ==

- 自定义视角类型（除 slot/bitsplit/concat/formula/map/derive 之外）
- 注册自定义映射函数（register_mapping）
- 直接操作 MetadataEngine / ViewEngine / TaskContext（高级组合）

入门使用者不需要 import 本模块——`MSInt` 类已封装了常用视角操作。

== 导出符号 ==

视角引擎：
  - MetadataEngine: 元数据引擎（槽位/视角/任务注册）
  - ViewEngine: 视角解析引擎
  - TaskContext: 任务上下文（限制可见视角）

视角类型常量：
  - VALID_VIEW_TYPES: 所有合法视角类型集合
  - VIEW_TYPE_SLOT / VIEW_TYPE_BITSPLIT / VIEW_TYPE_CONCAT
  - VIEW_TYPE_FORMULA / VIEW_TYPE_MAP / VIEW_TYPE_DERIVE

映射函数：
  - register_mapping(name, fn): 注册自定义映射
  - get_mapping(name): 查询映射函数

详见: 内部档案 §四
"""

from __future__ import annotations

from .core import (
    MetadataEngine,
    TaskContext,
    ViewEngine,
    get_mapping,
    register_mapping,
)
from .protocol import (
    VALID_VIEW_TYPES,
    VIEW_TYPE_BITSPLIT,
    VIEW_TYPE_CONCAT,
    VIEW_TYPE_DERIVE,
    VIEW_TYPE_FORMULA,
    VIEW_TYPE_MAP,
    VIEW_TYPE_SLOT,
)

__all__ = [
    # 视角引擎
    "MetadataEngine",
    "ViewEngine",
    "TaskContext",
    # 视角类型常量
    "VALID_VIEW_TYPES",
    "VIEW_TYPE_SLOT",
    "VIEW_TYPE_BITSPLIT",
    "VIEW_TYPE_CONCAT",
    "VIEW_TYPE_FORMULA",
    "VIEW_TYPE_MAP",
    "VIEW_TYPE_DERIVE",
    # 映射函数
    "register_mapping",
    "get_mapping",
]
