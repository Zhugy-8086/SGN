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
"""MSInt ABI 第三层：消费者适配层

实现两种适配器：
  - SGNMSIntAdapter: 把 SGN LevelScheduler 配置翻译成 MSInt 视角
  - MSIntSerializer: MSInt ←→ JSON 序列化适配

设计原则：
  - 适配器是外部的，不修改 MSInt 核心代码
  - MSInt 协议本身不依赖 SGN（通过 TYPE_CHECKING 避免运行时导入）
  - 适配器是可选的，不使用适配器时 MSInt 完全独立

详见: 内部档案
"""

from __future__ import annotations

import logging as _logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .core import MSInt, register_mapping
from .protocol import (
    SlotInfo,
    ViewDef,
    ViewNotFoundError,
    VIEW_TYPE_BITSPLIT,
    VIEW_TYPE_MAP,
)
from .version import CURRENT_ABI_VERSION, VersionEngine

if TYPE_CHECKING:
    from ..level import LevelScheduler, MappingFunction, SplitTree

_logger = _logging.getLogger(__name__)


# ============================================================
# SGN 适配器
# ============================================================

# SGN 映射函数名到 MSInt 内置映射名的映射
_SGN_MAPPING_TO_MSINT = {
    "linear": "identity",
    "log": "log",
    "threshold": "identity",  # 阈值映射退化为 identity（阈值逻辑在 SGN 侧）
}


class SGNMSIntAdapter:
    """SGN LevelScheduler ←→ MSInt 桥接适配器

    职责：
      1. 把 SGN 的神经元配置翻译成 MSInt 的视角注册
      2. 把 SGN 的 MappingFunction 翻译成 MSInt 的映射视角
      3. 把 SGN 的 SplitTree 翻译成 MSInt 的位拆分视角

    使用方式：
        adapter = SGNMSIntAdapter(scheduler)
        ms = adapter.adapt(ms, neuron_id=0)

    关键：
      - 适配器是外部的，不修改 MSInt 源码
      - MSInt 不知道 SGN 的存在
      - 原始 ms 不变，返回的是同一个 ms（视角注册是 in-place 的）
    """

    def __init__(self, scheduler: "LevelScheduler"):
        """初始化

        Args:
            scheduler: SGN LevelScheduler 实例
        """
        self._scheduler = scheduler

    def adapt(self, ms: MSInt, neuron_id: int) -> MSInt:
        """把 SGN 神经元配置应用到 MSInt

        v5.2.0 Step 3: 读取 scheduler 中 neuron_id 的 mapping / split_config，
        在 MSInt 上注册对应的命名视角。

        Args:
            ms: 待适配的 MSInt 实例
            neuron_id: 神经元 ID

        Returns:
            返回同一个 ms（视角注册是 in-place 的）
        """
        mapping = self._scheduler.get_mapping(neuron_id)
        split_config = self._scheduler.get_split_config(neuron_id)

        # 注册映射视角：对每个槽位添加 "{slot}.mapped" 视角
        if mapping is not None:
            # mapping.name 可能是方法或属性
            name_attr = getattr(mapping, "name", None)
            if callable(name_attr):
                mapping_name = name_attr()
            elif isinstance(name_attr, str):
                mapping_name = name_attr
            else:
                mapping_name = "linear"
            msint_mapping = _SGN_MAPPING_TO_MSINT.get(mapping_name, "identity")
            # 安全审计 2026-08-16 V1：未知 mapping 退化 identity 时给出
            # warning（原静默退化，数值语义变化难排查）
            if mapping_name not in _SGN_MAPPING_TO_MSINT:
                _logger.warning(
                    "SGN mapping '%s' 无对应 MSInt 实现，各槽位的 "
                    "'*.mapped' 视角退化为 identity", mapping_name,
                )
            for slot_name in ms.slot_names():
                view_name = f"{slot_name}.mapped"
                if not ms.has_view(view_name):
                    ms.register_view(
                        name=view_name,
                        view_type=VIEW_TYPE_MAP,
                        slots=[slot_name],
                        mapping=msint_mapping,
                    )

        # 注册位拆分视角：按 split_config 的等宽叶子位宽拆分
        if split_config is not None:
            leaf_widths = self._extract_leaf_widths_from_config(split_config)
            for slot_name in ms.slot_names():
                slot = ms._metadata.get_slot(slot_name)
                for target_bits in leaf_widths:
                    if target_bits <= slot.bits:
                        view_name = f"{slot_name}.int{target_bits}"
                        if not ms.has_view(view_name):
                            ms.register_view(
                                name=view_name,
                                view_type=VIEW_TYPE_BITSPLIT,
                                slots=[slot_name],
                                target_bits=target_bits,
                            )

        return ms

    @staticmethod
    def _extract_leaf_widths_from_config(
        split_config: Tuple[int, int]
    ) -> List[int]:
        """从 split_config 提取叶子节点的位宽列表

        v5.2.0 Step 3: 替代 _extract_leaf_widths(SplitTree)。
        split_config = (split_bits, total_bits)，等宽拆分只有一个唯一叶子位宽。

        Args:
            split_config: (split_bits, total_bits) 元组

        Returns:
            去重后的叶子位宽列表（升序），例如 [8]
        """
        split_bits, _total_bits = split_config
        # 等宽拆分：所有叶子位宽相同，去重后只有一个值
        return [split_bits] if split_bits > 0 else [8]

    @staticmethod
    def _extract_leaf_widths(split_tree: "SplitTree") -> List[int]:
        """[已废弃] 从 SplitTree 提取叶子节点的位宽列表

        v5.2.0 Step 3: 改用 _extract_leaf_widths_from_config。
        本方法保留用于向后兼容。
        """
        # SplitTree 可能有 leaf_widths 或类似属性
        widths = set()
        # 尝试常见属性名
        if hasattr(split_tree, "leaf_widths"):
            widths.update(split_tree.leaf_widths)
        elif hasattr(split_tree, "leaf_bits_list"):
            widths.update(split_tree.leaf_bits_list)
        elif hasattr(split_tree, "leaves"):
            for leaf in split_tree.leaves:
                if hasattr(leaf, "bits"):
                    widths.add(leaf.bits)
        else:
            # 退化策略：默认 8 位拆分
            widths.add(8)
        return sorted(widths) if widths else [8]


# ============================================================
# 序列化适配器
# ============================================================

class MSIntSerializer:
    """MSInt ←→ JSON 序列化适配器

    职责：
      1. 把 MSInt 序列化为可 JSON 化的字典
      2. 从字典反序列化重建 MSInt

    注意：
      - LazyBackend 反序列化后失去惰性特性（变为固定值）
      - 自定义映射函数无法序列化（只保留内置映射名）
    """

    @staticmethod
    def to_dict(ms: MSInt) -> Dict[str, Any]:
        """序列化 MSInt 为可 JSON 化的字典

        Args:
            ms: MSInt 实例

        Returns:
            包含 abi_version、backend、backend_state、metadata 的字典
        """
        return ms.serialize()

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> MSInt:
        """从字典反序列化 MSInt

        Args:
            data: to_dict 输出的字典

        Returns:
            重建的 MSInt 实例

        Raises:
            VersionIncompatibleError: ABI 版本不兼容
        """
        # 版本协商
        data_version = data.get("abi_version", "1.0.0")
        VersionEngine.negotiate(data_version, CURRENT_ABI_VERSION)

        return MSInt.deserialize(data)

    @staticmethod
    def to_compact_dict(ms: MSInt) -> Dict[str, Any]:
        """序列化为紧凑字典（不含元数据细节）

        适用于：仅需重建存储，不需要视角和任务信息的场景。
        """
        return {
            "abi_version": ms.abi_version,
            "backend": ms.backend_name,
            "backend_state": ms.backend.serialize(),
        }

    @staticmethod
    def verify_roundtrip(ms: MSInt) -> bool:
        """验证序列化往返一致性

        Args:
            ms: MSInt 实例

        Returns:
            True 如果序列化→反序列化后视角值一致
        """
        data = MSIntSerializer.to_dict(ms)
        ms2 = MSIntSerializer.from_dict(data)

        # 校验 ABI 版本
        if ms.abi_version != ms2.abi_version:
            return False

        # 校验后端名
        if ms.backend_name != ms2.backend_name:
            return False

        # 校验槽位名
        if ms.slot_names() != ms2.slot_names():
            return False

        # 校验所有视角值
        for view_name in ms.views():
            try:
                v1 = ms.view(view_name)
                v2 = ms2.view(view_name)
                if v1 != v2:
                    return False
            except ViewNotFoundError:
                return False

        # 校验 as_int
        if ms.as_int() != ms2.as_int():
            return False

        return True


# ============================================================
# 便捷工厂函数
# ============================================================

def from_sgn_neuron(scheduler: "LevelScheduler", neuron_id: int,
                    slots: Dict[str, int],
                    slot_defs: Optional[List[SlotInfo]] = None) -> MSInt:
    """便捷工厂：从 SGN 神经元配置 + 槽位值创建 MSInt

    Args:
        scheduler: SGN LevelScheduler 实例
        neuron_id: 神经元 ID
        slots: 槽位值字典
        slot_defs: 槽位定义列表（可选）

    Returns:
        适配后的 MSInt 实例
    """
    ms = MSInt.from_slots(slots, slot_defs=slot_defs)
    adapter = SGNMSIntAdapter(scheduler)
    return adapter.adapt(ms, neuron_id=neuron_id)
