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
SGN-Lite v5.1.3 Level 调度器模块

Phase 1: 从 sgn_core.py 抽离 level 相关逻辑，建立独立的 level 调度系统。

设计原则：
  - level 管"怎么算"，不管"存什么"
  - 调度器管"什么时候用什么算"，不管"算的是什么"
  - 存储和运算分离，策略和数值分离，上层完全无感

核心组件：
  - LevelContext: 运算上下文（level 语境）
  - LevelStrategy: 策略接口（可热替换）
  - LevelScheduler: 调度器核心（管理所有策略）
  - NeuronLevelStats: 神经元统计（用于自适应）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


# ============================================================
# 运算类型枚举
# ============================================================

class OperationType(Enum):
    """运算类型枚举"""
    ADD = "add"          # 加法（赫布学习增强）
    SUB = "sub"          # 减法（赫布学习削弱）
    MUL = "mul"          # 乘法（响应速度计算）
    COMPARE = "compare"  # 比较（排序/匹配）
    ASSIGN = "assign"    # 赋值（初始化/重置）


# ============================================================
# LevelContext - 运算上下文
# ============================================================

@dataclass
class LevelContext:
    """运算上下文 - 管理单次运算的 level 语境

    Attributes:
        target_level: 目标层级
        operation: 运算类型
        source: 来源标识（用于调试/日志）
    """
    target_level: int
    operation: OperationType = OperationType.ASSIGN
    source: str = ""

    def __repr__(self):
        return f"LevelContext(level={self.target_level}, op={self.operation.value}, src={self.source})"


# ============================================================
# NeuronLevelStats - 神经元统计
# ============================================================

@dataclass
class NeuronLevelStats:
    """神经元 level 统计 - 用于自适应策略

    Attributes:
        neuron_id: 神经元 ID
        current_level: 当前工作的 level
        match_history: 匹配值历史（用于计算方差）
        verified_count: 验证通过次数
        total_count: 总参与次数
        level_change_count: level 切换次数
        last_match: 最近一次匹配值
    """
    neuron_id: int
    current_level: int = 2
    match_history: List[int] = field(default_factory=list)
    verified_count: int = 0
    total_count: int = 0
    level_change_count: int = 0
    last_match: int = 0

    @property
    def match_variance(self) -> float:
        """计算匹配值方差（用于自适应判断）"""
        if len(self.match_history) < 2:
            return 0.0
        mean = sum(self.match_history) / len(self.match_history)
        return sum((x - mean) ** 2 for x in self.match_history) / len(self.match_history)

    @property
    def verification_rate(self) -> float:
        """验证通过率"""
        if self.total_count == 0:
            return 0.0
        return self.verified_count / self.total_count

    def update(self, match: int, verified: bool) -> None:
        """更新统计"""
        self.last_match = match
        self.total_count += 1
        if verified:
            self.verified_count += 1
        # 保留最近 100 个匹配值
        self.match_history.append(match)
        if len(self.match_history) > 100:
            self.match_history.pop(0)


# ============================================================
# LevelStrategy - 策略接口
# ============================================================

class LevelStrategy(ABC):
    """level 策略接口 - 可热替换

    所有策略实现必须提供：
      - 运算时的 level 决策
      - 自适应建议（可选）
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        pass

    @property
    @abstractmethod
    def default_level(self) -> int:
        """默认 level"""
        pass

    @abstractmethod
    def get_level_for_operation(
        self,
        operation: OperationType,
        neuron_id: Optional[int] = None,
        stats: Optional[NeuronLevelStats] = None
    ) -> int:
        """获取指定运算的 level"""
        pass

    def suggest_adaptation(self, stats: NeuronLevelStats) -> Optional[int]:
        """根据统计建议新的 level（可选实现）

        Returns:
            建议的 level，或 None 表示不调整
        """
        return None


# ============================================================
# 内置策略实现
# ============================================================

class StandardStrategy(LevelStrategy):
    """标准策略 - 固定 level，不自适应

    适用场景：
      - 默认 L0 神经元
      - 需要稳定精度的场景
    """

    def __init__(self, level: int = 2):
        self._level = level

    @property
    def name(self) -> str:
        return f"standard(L{self._level})"

    @property
    def default_level(self) -> int:
        return self._level

    def get_level_for_operation(
        self,
        operation: OperationType,
        neuron_id: Optional[int] = None,
        stats: Optional[NeuronLevelStats] = None
    ) -> int:
        return self._level


class AdaptiveStrategy(LevelStrategy):
    """自适应策略 - 根据神经元统计动态调整 level

    规则：
      - 匹配值方差 < 阈值 → 建议细粒度 level（更精确）
      - 匹配值方差 > 阈值 → 建议粗粒度 level（更稳定）
      - 验证率下降 → 回退到原 level
    """

    def __init__(
        self,
        base_level: int = 2,
        variance_threshold: float = 100.0,
        history_window: int = 50
    ):
        self._base_level = base_level
        self._variance_threshold = variance_threshold
        self._history_window = history_window

    @property
    def name(self) -> str:
        return f"adaptive(base=L{self._base_level})"

    @property
    def default_level(self) -> int:
        return self._base_level

    def get_level_for_operation(
        self,
        operation: OperationType,
        neuron_id: Optional[int] = None,
        stats: Optional[NeuronLevelStats] = None
    ) -> int:
        if stats is None:
            return self._base_level
        return stats.current_level

    def suggest_adaptation(self, stats: NeuronLevelStats) -> Optional[int]:
        """根据统计建议新的 level"""
        if len(stats.match_history) < self._history_window:
            return None

        variance = stats.match_variance
        current = stats.current_level

        # 方差小 → 细粒度（level 增大）
        if variance < self._variance_threshold / 4:
            suggested = min(current + 1, 4)  # 最高 level=4
            if suggested != current:
                return suggested

        # 方差大 → 粗粒度（level 减小）
        if variance > self._variance_threshold * 4:
            suggested = max(current - 1, 0)  # 最低 level=0
            if suggested != current:
                return suggested

        return None


class LayerAwareStrategy(LevelStrategy):
    """层级感知策略 - 根据神经元所在层自动选择 level

    L0 神经元：标准精度（level=2）
    L1 神经元：粗精度（level=1），因为图特征已经过抽象
    """

    def __init__(self):
        self._layer_levels = {
            0: 2,  # L0: 标准精度
            1: 1,  # L1: 粗精度
        }

    @property
    def name(self) -> str:
        return "layer_aware"

    @property
    def default_level(self) -> int:
        return 2

    def get_level_for_operation(
        self,
        operation: OperationType,
        neuron_id: Optional[int] = None,
        stats: Optional[NeuronLevelStats] = None
    ) -> int:
        # 从统计中推断层级（如果可用）
        if stats and hasattr(stats, 'layer'):
            return self._layer_levels.get(stats.layer, self.default_level)
        return self.default_level


# ============================================================
# LevelScheduler - 调度器核心
# ============================================================

class LevelScheduler:
    """level 调度器 - 管理运算精度语境

    核心职责：
      1. 定策略 - 给定运算类型和神经元，决定用什么 level
      2. 管自适应 - 根据神经元训练历史，动态调整 level
      3. 控输出 - 运算结果的 level 由策略决定，不由输入决定

    性能优化：
      - 使用 LRU 缓存加速 get_context() 调用
      - 缓存键为 (operation, neuron_id)
      - 缓存自动失效于 update_stats() 时

    使用方式：
      scheduler = LevelScheduler()
      ctx = scheduler.get_context(neuron_id=0, operation=OperationType.ADD)
      # ctx.target_level 就是该用的 level
    """

    def __init__(self, cache_size: int = 1024):
        # 策略注册表
        self._strategies: Dict[str, LevelStrategy] = {}
        # 神经元 → 策略 映射
        self._neuron_strategy: Dict[int, str] = {}
        # 神经元统计
        self._neuron_stats: Dict[int, NeuronLevelStats] = {}
        # 默认策略
        self._default_strategy: Optional[LevelStrategy] = None
        # 自适应检查间隔
        self._adapt_interval: int = 100
        self._step_counter: int = 0

        # v5.1.5 Phase 4: 性能优化缓存
        self._cache_size = cache_size
        self._context_cache: Dict[Tuple[OperationType, Optional[int]], LevelContext] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # 注册内置策略
        self._register_builtin_strategies()

    def _register_builtin_strategies(self) -> None:
        """注册内置策略"""
        self.register_strategy(StandardStrategy(2))
        self.register_strategy(StandardStrategy(1))
        self.register_strategy(StandardStrategy(0))
        self.register_strategy(AdaptiveStrategy(2))
        self.register_strategy(LayerAwareStrategy())
        # 设置默认策略
        self._default_strategy = StandardStrategy(2)

    # ---- 策略管理 ----

    def register_strategy(self, strategy: LevelStrategy) -> None:
        """注册策略"""
        self._strategies[strategy.name] = strategy

    def get_strategy(self, name: str) -> Optional[LevelStrategy]:
        """获取策略"""
        return self._strategies.get(name)

    def set_default_strategy(self, name: str) -> bool:
        """设置默认策略"""
        strategy = self._strategies.get(name)
        if strategy:
            self._default_strategy = strategy
            return True
        return False

    # ---- 神经元绑定 ----

    def bind_neuron(
        self,
        neuron_id: int,
        strategy_name: str,
        initial_level: Optional[int] = None
    ) -> None:
        """绑定神经元到指定策略"""
        self._neuron_strategy[neuron_id] = strategy_name
        strategy = self._strategies.get(strategy_name)
        level = initial_level if initial_level is not None else (
            strategy.default_level if strategy else 2
        )
        self._neuron_stats[neuron_id] = NeuronLevelStats(
            neuron_id=neuron_id,
            current_level=level
        )

    def bind_layer(
        self,
        layer: int,
        strategy_name: str,
        neuron_ids: List[int]
    ) -> None:
        """批量绑定同一层的神经元"""
        for nid in neuron_ids:
            self.bind_neuron(nid, strategy_name)

    # ---- 核心接口 ----

    def get_context(
        self,
        operation: OperationType,
        neuron_id: Optional[int] = None,
        source: str = ""
    ) -> LevelContext:
        """获取运算上下文

        Args:
            operation: 运算类型
            neuron_id: 神经元 ID（可选）
            source: 来源标识（调试用）

        Returns:
            LevelContext 包含目标 level

        性能优化：使用 LRU 缓存加速重复调用
        """
        # v5.1.5 Phase 4: 检查缓存
        cache_key = (operation, neuron_id)
        if cache_key in self._context_cache:
            self._cache_hits += 1
            cached = self._context_cache[cache_key]
            # 返回新的 LevelContext，避免共享引用问题
            return LevelContext(
                target_level=cached.target_level,
                operation=cached.operation,
                source=source
            )

        self._cache_misses += 1

        # 获取策略
        strategy = self._get_strategy_for_neuron(neuron_id)

        # 获取统计
        stats = self._neuron_stats.get(neuron_id) if neuron_id is not None else None

        # 调用策略获取 level
        level = strategy.get_level_for_operation(operation, neuron_id, stats)

        ctx = LevelContext(
            target_level=level,
            operation=operation,
            source=source
        )

        # 存入缓存（限制大小）
        if len(self._context_cache) < self._cache_size:
            self._context_cache[cache_key] = ctx

        return ctx

    def resolve_binary_op(
        self,
        op: OperationType,
        left_level: int,
        right_level: int,
        neuron_id: Optional[int] = None
    ) -> int:
        """二元运算的 level 决策

        Args:
            op: 运算类型
            left_level: 左操作数 level
            right_level: 右操作数 level
            neuron_id: 神经元 ID（可选）

        Returns:
            统一后的 target level
        """
        # 获取策略
        strategy = self._get_strategy_for_neuron(neuron_id)
        stats = self._neuron_stats.get(neuron_id) if neuron_id is not None else None

        # 策略决定目标 level
        target = strategy.get_level_for_operation(op, neuron_id, stats)

        # 如果策略返回的 level 与输入不匹配，取更精细的
        # （保留精度，宁可多算一点）
        if left_level != right_level:
            # 取更精细的 level（数值更小 = 更精细）
            target = min(left_level, right_level, target)

        return target

    def update_stats(
        self,
        neuron_id: int,
        match: int,
        verified: bool
    ) -> None:
        """更新神经元统计

        Args:
            neuron_id: 神经元 ID
            match: 匹配值
            verified: 是否验证通过
        """
        if neuron_id not in self._neuron_stats:
            self._neuron_stats[neuron_id] = NeuronLevelStats(neuron_id=neuron_id)

        stats = self._neuron_stats[neuron_id]
        stats.update(match, verified)

        # v5.1.5 Phase 4: 清除该神经元的缓存（因为 level 可能变化）
        self._invalidate_cache(neuron_id)

        # 定期检查自适应
        self._step_counter += 1
        if self._step_counter % self._adapt_interval == 0:
            self._check_adaptation(neuron_id)

    def _check_adaptation(self, neuron_id: int) -> None:
        """检查并执行自适应调整"""
        strategy = self._get_strategy_for_neuron(neuron_id)
        stats = self._neuron_stats.get(neuron_id)

        if stats is None:
            return

        suggested = strategy.suggest_adaptation(stats)
        if suggested is not None and suggested != stats.current_level:
            # 记录变化
            old_level = stats.current_level
            stats.current_level = suggested
            stats.level_change_count += 1

            # v5.1.5 Phase 4: 清除缓存（level 变化）
            self._invalidate_cache(neuron_id)

            # 可选：发射事件通知上层
            try:
                from engine.hooks import HookRegistry
                HookRegistry.emit(
                    "sgn:level_adapted",
                    neuron_id=neuron_id,
                    old_level=old_level,
                    new_level=suggested,
                    variance=stats.match_variance
                )
            except ImportError:
                pass

    def _invalidate_cache(self, neuron_id: Optional[int] = None) -> None:
        """清除缓存

        Args:
            neuron_id: 指定神经元 ID，或 None 清除所有缓存
        """
        if neuron_id is None:
            self._context_cache.clear()
        else:
            # 清除该神经元相关的缓存
            keys_to_remove = [
                key for key in self._context_cache.keys()
                if key[1] == neuron_id
            ]
            for key in keys_to_remove:
                del self._context_cache[key]

    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / max(1, total),
            "cache_size": len(self._context_cache),
            "max_size": self._cache_size,
        }

    def clear_cache(self) -> None:
        """清除所有缓存"""
        self._context_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_strategy_for_neuron(self, neuron_id: Optional[int]) -> LevelStrategy:
        """获取神经元对应的策略"""
        if neuron_id is None:
            return self._default_strategy

        strategy_name = self._neuron_strategy.get(neuron_id)
        if strategy_name:
            return self._strategies.get(strategy_name, self._default_strategy)

        return self._default_strategy

    # ---- 查询接口 ----

    def get_stats(self, neuron_id: int) -> Optional[NeuronLevelStats]:
        """获取神经元统计"""
        return self._neuron_stats.get(neuron_id)

    def get_all_stats(self) -> Dict[int, NeuronLevelStats]:
        """获取所有神经元统计"""
        return dict(self._neuron_stats)

    def get_neuron_level(self, neuron_id: int) -> int:
        """获取神经元当前 level"""
        stats = self._neuron_stats.get(neuron_id)
        if stats:
            return stats.current_level
        return self._default_strategy.default_level

    # ---- 序列化（预留） ----

    def serialize(self) -> Dict[str, Any]:
        """序列化调度器状态（用于持久化）"""
        return {
            "neuron_strategy": dict(self._neuron_strategy),
            "neuron_stats": {
                nid: {
                    "current_level": s.current_level,
                    "verified_count": s.verified_count,
                    "total_count": s.total_count,
                    "level_change_count": s.level_change_count,
                }
                for nid, s in self._neuron_stats.items()
            },
            "default_strategy": self._default_strategy.name if self._default_strategy else None,
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        """反序列化调度器状态"""
        # 恢复神经元绑定
        for nid_str, strategy_name in data.get("neuron_strategy", {}).items():
            nid = int(nid_str)
            self.bind_neuron(nid, strategy_name)

        # 恢复统计
        for nid_str, stats_data in data.get("neuron_stats", {}).items():
            nid = int(nid_str)
            if nid in self._neuron_stats:
                stats = self._neuron_stats[nid]
                stats.current_level = stats_data.get("current_level", stats.current_level)
                stats.verified_count = stats_data.get("verified_count", 0)
                stats.total_count = stats_data.get("total_count", 0)
                stats.level_change_count = stats_data.get("level_change_count", 0)

        # 恢复默认策略
        default_name = data.get("default_strategy")
        if default_name:
            self.set_default_strategy(default_name)


# ============================================================
# 全局调度器实例（单例模式）
# ============================================================

_global_scheduler: Optional[LevelScheduler] = None


def get_global_scheduler() -> LevelScheduler:
    """获取全局调度器实例"""
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = LevelScheduler()
    return _global_scheduler


def set_global_scheduler(scheduler: LevelScheduler) -> None:
    """设置全局调度器实例"""
    global _global_scheduler
    _global_scheduler = scheduler


# ============================================================
# 便捷函数（上层建筑调用）
# ============================================================

def get_level_for_add(neuron_id: Optional[int] = None) -> int:
    """获取加法运算的 level"""
    return get_global_scheduler().get_context(
        OperationType.ADD, neuron_id, "hebbian_enhance"
    ).target_level


def get_level_for_sub(neuron_id: Optional[int] = None) -> int:
    """获取减法运算的 level"""
    return get_global_scheduler().get_context(
        OperationType.SUB, neuron_id, "hebbian_weaken"
    ).target_level


def get_level_for_compare(neuron_id: Optional[int] = None) -> int:
    """获取比较运算的 level"""
    return get_global_scheduler().get_context(
        OperationType.COMPARE, neuron_id, "competition"
    ).target_level


def get_level_for_response_speed(neuron_id: Optional[int] = None) -> int:
    """获取响应速度计算的 level"""
    return get_global_scheduler().get_context(
        OperationType.MUL, neuron_id, "response_speed"
    ).target_level


def update_neuron_stats(neuron_id: int, match: int, verified: bool) -> None:
    """更新神经元统计（上层调用）"""
    get_global_scheduler().update_stats(neuron_id, match, verified)


def get_neuron_level(neuron_id: int) -> int:
    """获取神经元当前 level"""
    return get_global_scheduler().get_neuron_level(neuron_id)
