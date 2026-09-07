# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""SGN MSint 子包 — 多语义整数存储协议

Stage 3.0 统一抽象层入口。Python 实现已物理迁入本目录 `_py/` 子包
（原 engine/ms_int，SGN Lite 废弃代码，迁入后自包含）。

导入方式:
    from sgn.msint import MSInt, SlotInfo
    # 等价于（内部 Python 实现）
    from sgn.msint._py import MSInt, SlotInfo
"""
from __future__ import annotations

# 相对导入 _py 子包（禁用绝对 import，避免 sgn / engine.sgn 双模块实例）
from . import _py as _ms_int

# re-export 公开 API。
# 安全审计 2026-08-16 R1：原 dir() fallback 会把子模块名（backends/core/
# protocol 等模块对象）一并 re-export——强制 __all__，缺失时显式报错
# 而非静默泄漏命名空间
for _name in _ms_int.__all__:
    globals()[_name] = getattr(_ms_int, _name)

# 显式导出常用类型（便于 IDE 提示）
MSInt = _ms_int.MSInt
SlotInfo = _ms_int.SlotInfo
CURRENT_ABI_VERSION = _ms_int.CURRENT_ABI_VERSION

__all__ = [
    "MSInt",
    "SlotInfo",
    "CURRENT_ABI_VERSION",
    "VersionEngine",
    "register_mapping",
    "register_view",
    "register_task",
    "from_slots",
    "auto_split",
    "from_packed",
]
