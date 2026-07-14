#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SGN-Lite v5.1.5 Engine Package

Core engine components:
  - SGNCore: Main engine class
  - LevelScheduler: Level scheduling system
  - ConfigRegistry: Configuration management
"""

# 核心组件
from .core import SGNCore
from .level import LevelScheduler, OperationType, LevelContext
from .config import CONFIG, ConfigRegistry, DiscreteCoordinate
from .hooks import HookRegistry

# 版本信息
__version__ = "5.1.5"

# 便捷导出
__all__ = [
    "SGNCore",
    "LevelScheduler",
    "OperationType",
    "LevelContext",
    "ConfigRegistry",
    "DiscreteCoordinate",
    "HookRegistry",
]
