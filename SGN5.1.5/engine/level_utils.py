#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 [zhugy-8086]
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
"""
SGN-Lite v5.1.5 Level 调度器便捷函数

提供更简洁的 API，让上层建筑（竞争、赫布、图匹配）可以一行代码获取 level。
"""

from typing import Optional
from engine.level import (
    LevelScheduler, OperationType, LevelContext,
    get_global_scheduler, set_global_scheduler,
    get_level_for_add, get_level_for_sub,
    get_level_for_compare, get_level_for_response_speed,
    update_neuron_stats, get_neuron_level
)


def get_level_for_hebbian(neuron_id: Optional[int] = None) -> int:
    """获取赫布学习的 level（增强）"""
    return get_level_for_add(neuron_id)


def get_level_for_weaken(neuron_id: Optional[int] = None) -> int:
    """获取赫布学习的 level（削弱）"""
    return get_level_for_sub(neuron_id)


def get_level_for_template_merge(neuron_id: Optional[int] = None) -> int:
    """获取模板合并的 level"""
    return get_level_for_compare(neuron_id)


def get_level_for_graph_match(neuron_id: Optional[int] = None) -> int:
    """获取图匹配的 level"""
    return get_level_for_compare(neuron_id)


def get_level_for_response(neuron_id: Optional[int] = None) -> int:
    """获取响应速度计算的 level"""
    return get_level_for_response_speed(neuron_id)


def get_level_for_verify(neuron_id: Optional[int] = None) -> int:
    """获取校验的 level"""
    return get_level_for_compare(neuron_id)


class LevelHelper:
    """Level 调度器辅助类 - 提供面向对象的便捷接口"""

    def __init__(self, scheduler: Optional[LevelScheduler] = None):
        self._scheduler = scheduler or get_global_scheduler()

    def get_level(
        self,
        operation: OperationType,
        neuron_id: Optional[int] = None,
        source: str = ""
    ) -> int:
        """获取指定运算的 level"""
        ctx = self._scheduler.get_context(operation, neuron_id, source)
        return ctx.target_level

    def get_context(
        self,
        operation: OperationType,
        neuron_id: Optional[int] = None,
        source: str = ""
    ) -> LevelContext:
        """获取完整的运算上下文"""
        return self._scheduler.get_context(operation, neuron_id, source)

    def update_neuron(
        self,
        neuron_id: int,
        match: int,
        verified: bool
    ) -> None:
        """更新神经元统计"""
        self._scheduler.update_stats(neuron_id, match, verified)

    def get_neuron_info(self, neuron_id: int) -> dict:
        """获取神经元信息"""
        stats = self._scheduler.get_stats(neuron_id)
        level = self._scheduler.get_neuron_level(neuron_id)
        return {
            "level": level,
            "match_variance": stats.match_variance if stats else 0,
            "verification_rate": stats.verification_rate if stats else 0,
            "total_count": stats.total_count if stats else 0,
            "level_change_count": stats.level_change_count if stats else 0,
        }

    def get_all_info(self) -> dict:
        """获取所有神经元信息摘要"""
        stats = self._scheduler.get_all_stats()
        level_counts = {}
        for nid, stat in stats.items():
            level = stat.current_level
            level_counts[level] = level_counts.get(level, 0) + 1

        return {
            "total_neurons": len(stats),
            "level_distribution": level_counts,
            "adapted_count": sum(1 for s in stats.values() if s.level_change_count > 0),
        }


def create_level_helper(scheduler: Optional[LevelScheduler] = None) -> LevelHelper:
    """创建 Level 辅助实例"""
    return LevelHelper(scheduler)
