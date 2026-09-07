#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SGN Engine Package — Stage 3.0

新版引擎入口，仅包含 engine/sgn/（C++ 扩展 + Python 子包）。
MSInt 与 int_semantics 已物理迁入 engine/sgn/ 内部（engine/sgn/msint/_py/、
engine/sgn/int_semantics/），实现自包含，不再依赖 SGN Lite 旧包。

注意：本包顶层不导出符号，避免触发 engine.sgn 顶层加载（其 models.py 依赖
顶层 `import sgn`，在 engine.sgn 身份下不可解析）。请直接使用子包：
    from engine.sgn.int_semantics import DualInterpreter
    from engine.sgn.msint import MSInt

旧版 SGN-Lite L0/L1 已移入 legacy/_legacy/（废弃归档），**不建议使用**——
活跃功能统一走 engine.sgn 子包：
    import engine.sgn as sgn
    from engine.sgn.nn import Linear, Conv2d
    from engine.sgn.level import LevelContext
（安全审计 2026-08-16 决策项 6：原 docstring 引导 `from legacy._legacy import
SGNCore, LevelScheduler` 属废弃代码引导，已改为指向活跃入口。）
"""

# 版本信息
__version__ = "0.9.0"

__all__ = []