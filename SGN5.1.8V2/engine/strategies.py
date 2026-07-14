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
"""SGN-Lite v5.0 核心策略抽象 —— 解耦硬编码阈值（v5.1.5：基线从 4×4 提升到 8×8）

LayerStrategy: 控制二值化层提取逻辑（mark_count, layer_max, min_marked）
VerifyStrategy: 控制校验通过条件（max_span, intensity_diff_thresh）

策略包按窗口尺寸匹配：
  DefaultLayerStrategy    → 兼容 8×8 实验参数，随 D 自适应
  SparseLineStrategy      → 细线条/矢量图形（16×16+），降低标记阈值
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from engine.config import CONFIG


# ============================================================
# LayerStrategy —— 层提取策略
# ============================================================

class LayerStrategy(ABC):
    """层提取策略抽象基类

    控制 extract_layers 等价逻辑中的阈值参数：
      - mark_count: 每层应标记多少位
      - layer_max: 最大层数
      - min_marked: 校验阶段最小标记数（稀疏保护）
    """

    @property
    @abstractmethod
    def applicable_range(self) -> Tuple[int, int]:
        """返回适用的 D 范围 (min_d, max_d)，0 表示无上限"""
        pass

    @abstractmethod
    def get_mark_count(self, d: int, active_pixels: int) -> int:
        """给定窗口大小 d 和当前活跃像素数，返回本层应标记多少位"""
        pass

    @abstractmethod
    def get_layer_max(self, d: int) -> int:
        """给定窗口大小，返回最大层数"""
        pass

    @abstractmethod
    def get_min_marked(self, d: int) -> int:
        """校验阶段的最小标记数阈值（稀疏保护）"""
        pass

    @abstractmethod
    def get_min_bits(self, d: int) -> int:
        """模板 AND 合并时的最小保留位数"""
        pass

    def is_applicable(self, d: int) -> bool:
        """检查当前策略是否适用于给定 D"""
        min_d, max_d = self.applicable_range
        return d >= min_d and (max_d == 0 or d <= max_d)


class DefaultLayerStrategy(LayerStrategy):
    """默认策略：兼容当前 8×8 实验参数，随 D 线性自适应

    原硬编码逻辑：
      mark_count = max(max(3, d//8), active_pixels//2)
      layer_max  = CONFIG[LAYER_MAX]（默认 4）
      min_marked = max(MIN_MARKED_CNT, d//8)
      min_bits   = 3
    """

    @property
    def applicable_range(self) -> Tuple[int, int]:
        return (8, 255)  # 8×8 到 16×16，256+ 归 SparseLineStrategy

    def get_mark_count(self, d: int, active_pixels: int) -> int:
        # 原逻辑: max(max(3, d//8), active_pixels//2)
        return max(max(3, d // 8), active_pixels // 2)

    def get_layer_max(self, d: int) -> int:
        return CONFIG.get("LAYER_MAX", 4)

    def get_min_marked(self, d: int) -> int:
        # Bug #2 修复: d//8 对 8x8 网格(D=64)要求至少 8 个标记像素太严格，
        # 初始随机神经元难以通过，改为 d//16 (D=64→4) 更宽松
        return max(CONFIG.get("MIN_MARKED_CNT", 2), d // 16)

    def get_min_bits(self, d: int) -> int:
        # 原硬编码 3，对大窗口自适应放大
        return max(3, d // 16)


class SparseLineStrategy(LayerStrategy):
    """细线条策略：适用于矢量图形（16×16+），降低标记阈值

    矢量图形（直线/圆/正弦）的亚像素渲染通常只覆盖少量像素，
    若用 DefaultLayerStrategy 的 d//8 阈值，细线条会被稀疏保护误杀。
    """

    @property
    def applicable_range(self) -> Tuple[int, int]:
        return (256, 0)  # 16×16 以上，无上限

    def get_mark_count(self, d: int, active_pixels: int) -> int:
        # 细线条活跃像素少，不强制 d//8，取活跃像素的更高比例
        # 【v4.3-fix】80% 标记率确保覆盖图形主体，避免直线断裂
        return max(2, active_pixels * 4 // 5)

    def get_layer_max(self, d: int) -> int:
        # 大窗口可能需要更多层来捕获细节
        return min(6, max(2, d // 16))

    def get_min_marked(self, d: int) -> int:
        # 细线条可能只有 3-5 个像素亮，不能要求 d//8
        return max(2, min(d // 16, 5))

    def get_min_bits(self, d: int) -> int:
        # 细线条模板位少，合并阈值相应降低
        return max(2, d // 32)


# ============================================================
# VerifyStrategy —— 校验策略
# ============================================================

class VerifyStrategy(ABC):
    """校验策略抽象基类

    控制 _verify 的通过条件：
      - max_span: 空间跨度上限
      - intensity_diff_thresh: 强度差异阈值
    """

    @abstractmethod
    def get_max_span(self, grid_size: int) -> int:
        """空间跨度上限（像素坐标差的平方和）"""
        pass

    @abstractmethod
    def get_intensity_diff_thresh(self) -> int:
        """标记/未标记区域强度差异阈值"""
        pass


class DefaultVerifyStrategy(VerifyStrategy):
    """默认校验策略：兼容当前 8×8 行为"""

    def get_max_span(self, grid_size: int) -> int:
        # 原逻辑: (grid_size - 1)^2 + (grid_size - 1)^2
        return (grid_size - 1) ** 2 + (grid_size - 1) ** 2

    def get_intensity_diff_thresh(self) -> int:
        return CONFIG.get("INTENSITY_DIFF_THRESH", 1)


class RelaxedVerifyStrategy(VerifyStrategy):
    """宽松校验策略：适用于高噪声/低分辨率场景

    降低强度差异阈值，允许更弱的对比度通过。
    """

    def get_max_span(self, grid_size: int) -> int:
        # 与默认相同
        return (grid_size - 1) ** 2 + (grid_size - 1) ** 2

    def get_intensity_diff_thresh(self) -> int:
        # 阈值减半，更宽松
        return max(1, CONFIG.get("INTENSITY_DIFF_THRESH", 1) // 2)


# ============================================================
# StrategyRegistry —— 策略注册与自动选择
# ============================================================

class StrategyRegistry:
    """策略注册表 —— 按窗口尺寸自动匹配最佳策略"""

    _layer_strategies: list = []
    _verify_strategies: list = []

    @classmethod
    def register_layer(cls, strategy: LayerStrategy) -> None:
        cls._layer_strategies.append(strategy)

    @classmethod
    def register_verify(cls, strategy: VerifyStrategy) -> None:
        cls._verify_strategies.append(strategy)

    @classmethod
    def auto_select_layer(cls, d: int) -> LayerStrategy:
        """按 D 自动选择第一个适用的 LayerStrategy"""
        for s in cls._layer_strategies:
            if s.is_applicable(d):
                return s
        return DefaultLayerStrategy()  # 保底

    @classmethod
    def auto_select_verify(cls, grid_size: int) -> VerifyStrategy:
        """返回默认 VerifyStrategy（当前无尺寸区分）"""
        if cls._verify_strategies:
            return cls._verify_strategies[0]
        return DefaultVerifyStrategy()

    @classmethod
    def list_layer_strategies(cls):
        return [s.__class__.__name__ for s in cls._layer_strategies]

    @classmethod
    def clear(cls):
        cls._layer_strategies.clear()
        cls._verify_strategies.clear()


# 自动注册默认策略
StrategyRegistry.register_layer(DefaultLayerStrategy())
StrategyRegistry.register_layer(SparseLineStrategy())
StrategyRegistry.register_verify(DefaultVerifyStrategy())


# ============================================================
# 匹配策略（v5.1.6: 旧 GateMatchStrategy 已删除，由自适应静默取代）
# ============================================================

class MatchStrategy(ABC):
    """匹配策略抽象基类

    控制神经元与输入的匹配方式：
      - GlobalMatchStrategy: 传统全局 XOR 匹配
    """

    @abstractmethod
    def compute(self, n, layers: List[int], layer_count: int, d: int) -> int:
        """计算神经元与输入的匹配度（0~100）"""
        pass


class GlobalMatchStrategy(MatchStrategy):
    """传统全局 XOR 匹配（默认策略，向后兼容）"""

    def compute(self, n, layers: List[int], layer_count: int, d: int) -> int:
        from engine.layers import popcount
        t = sum(
            d - popcount(n["T"][ll] ^ layers[ll])
            for ll in range(min(layer_count, CONFIG.get("LAYER_MAX", 4)))
        )
        denom = min(layer_count, CONFIG.get("LAYER_MAX", 4)) * d
        return (t * 100) // denom if denom > 0 else 0
