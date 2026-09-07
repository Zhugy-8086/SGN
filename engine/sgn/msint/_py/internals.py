#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""MSInt 高级层导出（v5.3.2 第二阶段 #2.2）

分层导出策略的第三层：存储后端、版本引擎、SGN 适配器。
这些符号面向框架开发者（需要自定义存储后端、对接 SGN 神经元、做版本兼容），
应用层使用者不应该 import 本模块。

== 何时用 engine.ms_int.internals ==

- 自定义存储后端（继承 StorageBackend，注册到 register_backend）
- 按槽位规模选择后端（auto_select_backend）
- 跨进程/跨版本数据兼容（VersionEngine）
- 把 MSInt 接入 SGN 训练循环（SGNMSIntAdapter, from_sgn_neuron）
- 序列化 MSInt 为可持久化格式（MSIntSerializer）

应用层使用者只需要 `engine.ms_int`（入门）或 `engine.ms_int.views`（进阶）。

== 导出符号 ==

存储后端：
  - StorageBackend: 后端基类
  - SlotBackend: 槽位存储后端（默认）
  - PackedBackend: 打包存储后端（跨进程）
  - LazyBackend: 延迟求值后端
  - register_backend / get_backend_class / auto_select_backend

版本引擎：
  - VersionEngine: ABI 版本兼容引擎

SGN 适配器：
  - SGNMSIntAdapter: 把 SGN 神经元接入 MSInt
  - MSIntSerializer: MSInt 序列化器
  - from_sgn_neuron: 从 SGN 神经元构造 MSInt

详见: 内部档案 §四
"""

from __future__ import annotations

from .adapters import (
    MSIntSerializer,
    SGNMSIntAdapter,
    from_sgn_neuron,
)
from .backends import (
    LazyBackend,
    PackedBackend,
    SlotBackend,
    StorageBackend,
    auto_select_backend,
    get_backend_class,
    register_backend,
)
from .version import VersionEngine

__all__ = [
    # 存储后端
    "StorageBackend",
    "SlotBackend",
    "PackedBackend",
    "LazyBackend",
    "register_backend",
    "get_backend_class",
    "auto_select_backend",
    # 版本引擎
    "VersionEngine",
    # SGN 适配器
    "SGNMSIntAdapter",
    "MSIntSerializer",
    "from_sgn_neuron",
]
