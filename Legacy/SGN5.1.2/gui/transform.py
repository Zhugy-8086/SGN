#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gui/transform.py — 图形变换引擎（TransformEngine）"""
from __future__ import annotations

import random
from typing import List, Tuple, Dict

from gui.utils import apply_transform


class TransformEngine:
    """图形变换引擎：管理旋转变换参数，生成变体

    移动适配的核心：用户只画一个基础图形，引擎负责生成多种变体。
    """

    def __init__(self):
        self.angle = 0.0
        self.offset_x = 0
        self.offset_y = 0
        self.scale = 1.0

    def randomize(self, grid_size: int):
        """随机生成一组变换参数"""
        self.angle = random.uniform(-45, 45)
        max_off = grid_size // 3
        self.offset_x = random.randint(-max_off, max_off)
        self.offset_y = random.randint(-max_off, max_off)
        self.scale = random.uniform(0.6, 1.4)

    def apply(self, intensity: List[int], grid_size: int) -> List[int]:
        """应用当前变换参数"""
        return apply_transform(
            intensity, grid_size,
            self.angle, self.offset_x, self.offset_y, self.scale
        )

    def generate_variants(
        self, intensity: List[int], grid_size: int, count: int
    ) -> List[Tuple[List[int], Dict]]:
        """生成多个随机变体，返回 (变体 intensity, 参数 dict) 列表"""
        variants = []
        original = self.snapshot()
        for _ in range(count):
            self.randomize(grid_size)
            variant = self.apply(intensity, grid_size)
            variants.append((variant, self.snapshot()))
        self.restore(original)
        return variants

    def snapshot(self) -> Dict:
        return {
            "angle": self.angle,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "scale": self.scale,
        }

    def restore(self, state: Dict):
        self.angle = state.get("angle", 0.0)
        self.offset_x = state.get("offset_x", 0)
        self.offset_y = state.get("offset_y", 0)
        self.scale = state.get("scale", 1.0)
