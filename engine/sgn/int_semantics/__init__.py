#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""int_semantics — 整数语义层包

独立于 SGN 精度调度的整数语义工具集。

设计原则（§6.1）：
  - 本包每个模块不 import engine 中的调度/匹配逻辑符号
  - 仅依赖 engine.ms_int（独立 ABI）和标准库
  - 消费者通过本包获取整数语义工具，不直接依赖 engine.level

v5.3.0 Step 2：首批迁入 DualInterpreter（依赖 MSInt，不再依赖 SplitTree）
"""

from .dual_interpreter import DualInterpreter

__all__ = ["DualInterpreter"]
