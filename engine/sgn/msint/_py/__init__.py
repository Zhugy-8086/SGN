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
"""MSInt ABI —— 多语义整数存储协议

== 使用者决策树（v5.3.2 第一阶段 #5）==

你想做什么？
├─ 存多字段数据              → MSInt.from_slots
├─ 跨进程传递                → ms.as_int() + MSInt.from_packed
├─ 等宽拆分（替代 SplitTree） → MSInt.auto_split
├─ 给不同消费者不同视角       → ms.register_view
├─ 限制任务可见的视角         → ms.register_task + ms.with_context
└─ 接入 SGN                  → SGNMSIntAdapter（见 engine.ms_int.internals）

入门只需：from_slots + view
进阶：engine.ms_int.views（视角引擎 + 映射注册）
高级：engine.ms_int.internals（存储后端 + 版本引擎 + SGN 适配器）
文档：内部档案

== 架构 ==

三层架构：
  第一层：protocol.py    - ABI 协议定义（SlotInfo, ViewDef, MSIntProtocol）
  第二层：core.py        - MSInt 核心（视角引擎 + 元数据引擎 + 任务引擎）
         backends.py    - 存储后端（SlotBackend / PackedBackend / LazyBackend
                          / PersistentBackend / SparseBackend / StreamBackend）
         version.py     - 版本引擎
  第三层：adapters.py    - 消费者适配（SGNMSIntAdapter, MSIntSerializer）

设计原则：
  - 程序内部仍然是普通 int 运算（性能无损，指令集正常）
  - 动态生成的 int 数据通过 MSInt ABI 获得多语义存储能力
  - 多地址存入、多视角读取、临时解码不占持久存储
  - MSInt 不继承 int，不参与运算，不重写 int 魔法方法

详见: 内部档案
"""

from __future__ import annotations

# 第一层：协议定义
from .protocol import (
    ABI_VERSION,
    MSIntError,
    MSIntProtocol,
    SlotInfo,
    SlotNotFoundError,
    SlotValidationError,
    StorageBackendProtocol,
    VALID_VIEW_TYPES,
    VersionIncompatibleError,
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

# 第二层：核心实现
from .backends import (
    LazyBackend,
    PackedBackend,
    PersistentBackend,
    SlotBackend,
    SparseBackend,
    StorageBackend,
    StreamBackend,
    auto_select_backend,
    get_backend_class,
    register_backend,
)
from .core import (
    MSInt,
    MetadataEngine,
    TaskContext,
    ViewEngine,
    get_mapping,
    register_mapping,
)
from .version import CURRENT_ABI_VERSION, VersionEngine

# 第三层：消费者适配
from .adapters import (
    MSIntSerializer,
    SGNMSIntAdapter,
    from_sgn_neuron,
)

__all__ = [
    # 协议层
    "ABI_VERSION",
    "MSIntProtocol",
    "SlotInfo",
    "ViewDef",
    "StorageBackendProtocol",
    "VALID_VIEW_TYPES",
    "VIEW_TYPE_SLOT",
    "VIEW_TYPE_BITSPLIT",
    "VIEW_TYPE_CONCAT",
    "VIEW_TYPE_FORMULA",
    "VIEW_TYPE_MAP",
    "VIEW_TYPE_DERIVE",
    # 异常
    "MSIntError",
    "ViewNotFoundError",
    "ViewNotAllowedError",
    "SlotNotFoundError",
    "SlotValidationError",
    "VersionIncompatibleError",
    # 核心实现
    "MSInt",
    "MetadataEngine",
    "TaskContext",
    "ViewEngine",
    # 存储后端
    "StorageBackend",
    "SlotBackend",
    "PackedBackend",
    "LazyBackend",
    "PersistentBackend",
    "SparseBackend",
    "StreamBackend",
    "auto_select_backend",
    "get_backend_class",
    "register_backend",
    # 映射函数
    "register_mapping",
    "get_mapping",
    # 版本
    "CURRENT_ABI_VERSION",
    "VersionEngine",
    # 适配器
    "SGNMSIntAdapter",
    "MSIntSerializer",
    "from_sgn_neuron",
]
