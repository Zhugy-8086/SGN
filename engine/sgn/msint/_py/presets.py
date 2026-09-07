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
"""MSInt ABI 第三层：preset 系统（v5.3.2 第三阶段 #3）

预设常用场景的槽位布局 + 视角注册模板，减少样板代码。

四个 preset：
  - role_id:                游戏角色 ID 4 槽位（server_id/category/level/seq）
  - cross_process_id:       跨进程 packed int 重建
  - auto_split_equivalent:  SplitTree 等价替代（等宽拆分）
  - sgn_neuron_view:        SGN 神经元桥接（自动注册映射 + 位拆分视角）

使用方式：
    ms = MSInt.preset.role_id(slots={"server_id": 10, ...})
    ms = MSInt.preset.cross_process_id(packed=0x12345678)
    ms = MSInt.preset.auto_split_equivalent(0x12345678, bits=8, total_bits=32)
    ms = MSInt.preset.sgn_neuron_view(scheduler, neuron_id=0, slots={...})

设计原则：
  - preset 只是工厂函数集合，不引入新逻辑
  - 不修改 MSInt 核心代码（除添加 preset 类属性外）
  - sgn_neuron_view 用延迟导入，避免顶层依赖 SGN

详见: 内部档案 §五.1
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .protocol import SlotInfo

if TYPE_CHECKING:
    from .core import MSInt


# ============================================================
# 内部共享：role_id 布局
# ============================================================

# 4 槽位：server_id(8) | category(8) | level(8) | seq(16) = 40 位
# 第一个槽位在高位（与 from_packed / as_int 的拼接顺序一致）
_ROLE_ID_SLOT_DEFS: List[SlotInfo] = [
    SlotInfo(name="server_id", bits=8, signed=False, unit="server", description="服务器 ID"),
    SlotInfo(name="category",  bits=8, signed=False, unit="class",  description="角色类别"),
    SlotInfo(name="level",     bits=8, signed=False, unit="level",  description="角色等级"),
    SlotInfo(name="seq",       bits=16, signed=False, unit="index", description="序列号"),
]


def _role_id_view_defs() -> List[Dict[str, Any]]:
    """role_id preset 自动注册的视角定义"""
    return [
        {
            "name": "role_id",
            "view_type": "concat",
            "slots": ["server_id", "category", "level"],
        },
        {
            "name": "full_id",
            "view_type": "concat",
            "slots": ["server_id", "category", "level", "seq"],
        },
    ]


# ============================================================
# Preset 命名空间
# ============================================================

class _PresetNamespace:
    """MSInt.preset 命名空间

    通过 ``MSInt.preset.<preset_name>(...)`` 调用，等价于：
      1. 用预设的槽位布局构造 MSInt
      2. 自动注册 preset 内置的命名视角

    所有方法都是静态方法，不持有状态。
    """

    # ---- role_id ----
    @staticmethod
    def role_id(
        slots: Dict[str, int],
        *,
        slot_defs: Optional[List[SlotInfo]] = None,
    ) -> "MSInt":
        """游戏角色 ID preset（4 槽位 40 位）

        默认槽位布局：
          - server_id (8 位):  服务器 ID
          - category  (8 位):  角色类别
          - level     (8 位):  角色等级
          - seq       (16 位): 序列号

        自动注册视角：
          - role_id: concat(server_id, category, level)  —— 24 位标识符
          - full_id: concat(server_id, category, level, seq)  —— 40 位全局 ID

        Args:
            slots: 槽位值字典，至少包含上述 4 个槽位
            slot_defs: 自定义槽位定义（默认使用 role_id 标准布局）

        Returns:
            已注册 role_id / full_id 视角的 MSInt 实例

        Example:
            >>> ms = MSInt.preset.role_id({"server_id": 10, "category": 1,
            ...                            "level": 5, "seq": 100})
            >>> ms.view("role_id")  # 24 位标识符
            >>> ms.view("full_id")  # 40 位全局 ID
        """
        from .core import MSInt  # 延迟导入避免循环

        defs = slot_defs if slot_defs is not None else _ROLE_ID_SLOT_DEFS
        ms = MSInt.from_slots(slots, slot_defs=defs, backend="slot")
        # S1: preset 出厂即保证视角可解析，用 strict=True 自校验
        for view_kwargs in _role_id_view_defs():
            ms.register_view(**view_kwargs, strict=True)
        return ms

    # ---- cross_process_id ----
    @staticmethod
    def cross_process_id(
        packed: int,
        *,
        slot_defs: Optional[List[SlotInfo]] = None,
        register_views: bool = False,
    ) -> "MSInt":
        """跨进程 packed int 重建 preset

        把 ``ms.as_int()`` 输出的打包 int 重建为 MSInt，常用于跨进程传递。
        默认使用 role_id 标准布局，可通过 slot_defs 自定义。

        与 ``role_id`` preset 的差异（v5.3.2 P8 文档化）：
          - 本 preset 默认 **不** 注册 ``role_id`` / ``full_id`` 命名视角，
            只重建存储。接收方只能用基础视角（``ms.view("server_id")`` 等）。
          - 如需命名视角，传 ``register_views=True``（会自动注册并 strict 校验），
            或直接用 ``role_id`` preset（但 role_id 接收的是 slots 字典不是 packed int）。
          - 典型用法：发送方 ``role_id`` preset 创建 + as_int 打包，
            接收方 ``cross_process_id(raw, register_views=True)`` 重建 + 启用命名视角。

        Args:
            packed: 打包后的 int（来自另一个 MSInt.as_int()）
            slot_defs: 自定义槽位定义（默认使用 role_id 标准布局）
            register_views: 是否自动注册 role_id / full_id 视角（默认 False 保持
                向后兼容；传 True 与 role_id preset 对称）

        Returns:
            重建的 MSInt 实例（与原 MSInt 槽位值一致）

        Example:
            >>> ms1 = MSInt.preset.role_id({"server_id": 10, ...})
            >>> raw = ms1.as_int()
            >>> ms2 = MSInt.preset.cross_process_id(raw)
            >>> ms2.view("server_id") == ms1.view("server_id")
            True
            >>> # 若需要命名视角，传 register_views=True
            >>> ms3 = MSInt.preset.cross_process_id(raw, register_views=True)
            >>> ms3.view("role_id") == ms1.view("role_id")
            True
        """
        from .core import MSInt

        defs = slot_defs if slot_defs is not None else _ROLE_ID_SLOT_DEFS
        ms = MSInt.from_packed(packed, defs)
        if register_views:
            # P8: 可选注册命名视角，与 role_id preset 对称
            for view_kwargs in _role_id_view_defs():
                ms.register_view(**view_kwargs, strict=True)
        return ms

    # ---- auto_split_equivalent ----
    @staticmethod
    def auto_split_equivalent(
        raw: int,
        bits: int = 8,
        *,
        total_bits: int = 64,
        prefix: str = "slot",
    ) -> "MSInt":
        """SplitTree 等价替代 preset

        把一个整数等宽拆分为多个槽位，每个槽位 ``bits`` 位。
        支持 ``bits`` 为任意正整数（SplitTree 只支持 2 的幂）。

        Args:
            raw: 原始整数
            bits: 每个槽位的位宽（默认 8）
            total_bits: 总位宽（默认 64，与 ``MSInt.auto_split`` 一致），必须能被 bits 整除
            prefix: 槽位名前缀（默认 "slot"，生成 slot_0 / slot_1 / ...）

        Returns:
            等宽拆分后的 MSInt 实例

        Example:
            >>> ms = MSInt.preset.auto_split_equivalent(0x12345678, bits=8, total_bits=32)
            >>> ms.view("slot_0")  # 高 8 位
            >>> ms.view("slot_3")  # 低 8 位
            >>>
            >>> # SplitTree 无法表达——12 不是 2 的幂
            >>> ms = MSInt.preset.auto_split_equivalent(0xFFF_AAA_555_000, bits=12, total_bits=48)
        """
        from .core import MSInt

        return MSInt.auto_split(raw, bits=bits, total_bits=total_bits, prefix=prefix)

    # ---- sgn_neuron_view ----
    @staticmethod
    def sgn_neuron_view(
        scheduler: Any,
        neuron_id: int,
        slots: Dict[str, int],
        *,
        slot_defs: Optional[List[SlotInfo]] = None,
    ) -> "MSInt":
        """SGN 神经元桥接 preset

        从 SGN LevelScheduler 的神经元配置创建 MSInt，并自动注册：
          - "{slot}.mapped" 视角：按 neuron 的 mapping 函数映射
          - "{slot}.intN"   视角：按 neuron 的 split_config 等宽拆分

        Args:
            scheduler: SGN LevelScheduler 实例
            neuron_id: 神经元 ID
            slots: 槽位值字典
            slot_defs: 槽位定义（可选，默认由 from_slots 推断）

        Returns:
            已注册 SGN 视角的 MSInt 实例

        Example:
            >>> sched = LevelScheduler()
            >>> ms = MSInt.preset.sgn_neuron_view(
            ...     sched, neuron_id=0,
            ...     slots={"feature_0": 100, "feature_1": 200},
            ... )
            >>> ms.view("feature_0.mapped")
            >>> ms.view("feature_0.int8")
        """
        from .adapters import SGNMSIntAdapter
        from .core import MSInt

        ms = MSInt.from_slots(slots, slot_defs=slot_defs)
        adapter = SGNMSIntAdapter(scheduler)
        return adapter.adapt(ms, neuron_id=neuron_id)

    # ---- 元信息 ----
    @classmethod
    def list_presets(cls) -> List[str]:
        """列出所有可用 preset 名"""
        return [
            "role_id",
            "cross_process_id",
            "auto_split_equivalent",
            "sgn_neuron_view",
        ]

    @classmethod
    def describe(cls, name: str) -> Dict[str, Any]:
        """返回指定 preset 的描述信息

        Args:
            name: preset 名（见 list_presets()）

        Returns:
            包含 name / docstring / slot_defs（若适用）的字典

        Raises:
            KeyError: preset 名不存在
        """
        presets = {
            "role_id": cls.role_id,
            "cross_process_id": cls.cross_process_id,
            "auto_split_equivalent": cls.auto_split_equivalent,
            "sgn_neuron_view": cls.sgn_neuron_view,
        }
        if name not in presets:
            raise KeyError(
                f"preset '{name}' 不存在，可用: {sorted(presets.keys())}"
            )
        method = presets[name]
        doc = (method.__doc__ or "").strip()
        info: Dict[str, Any] = {
            "name": name,
            "doc": doc,
        }
        if name in ("role_id", "cross_process_id"):
            info["slot_defs"] = [s.to_dict() for s in _ROLE_ID_SLOT_DEFS]
            if name == "role_id":
                info["views"] = _role_id_view_defs()
        return info


# S2: __all__ 不导出下划线私有名（_PresetNamespace 和 _ROLE_ID_SLOT_DEFS 仅供 core.py 显式 import）
__all__: List[str] = []
