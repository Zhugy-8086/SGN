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
"""SGN-Lite v5.0 输入管道抽象 - InputSource / NoiseModel / FeatureExtractor

阶段2重构：解耦硬编码的 PATTERNS、gen_samples 噪声模型、extract_layers。
允许外部插件接入新输入源，在不改源码的情况下更换噪声模型和特征提取器。
"""

from __future__ import annotations

import random
import csv
import json
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


# ============================================================
# NoiseModel - 噪声模型抽象
# ============================================================

class NoiseModel(ABC):
    """噪声模型抽象基类

    所有实现必须保证返回值在 [0, 255] 范围内且为 int 类型。
    """

    @abstractmethod
    def apply(self, base_pattern: List[int]) -> List[int]:
        """返回施加噪声后的 intensity

        Args:
            base_pattern: 原始模式强度值列表（0-255）

        Returns:
            施加噪声后的强度值列表，每个值在 [0, 255] 范围内
        """
        pass

    def _clamp(self, value: int) -> int:
        """强制将值限制在 [0, 255] 并转为 int"""
        return max(0, min(255, int(value)))


class BaseNoiseModel(NoiseModel):
    """噪声模型基类 —— 提供共用循环框架，避免复制粘贴

    子类只需实现 _perturb_pixel(self, v) -> int 方法。
    """

    def __init__(self, noise_prob: float = 0.15, grid_size: int = None):
        self.noise_prob = noise_prob
        self.grid_size = grid_size  # 可选，供 BlockNoise 等使用

    def apply(self, base_pattern: List[int]) -> List[int]:
        intensity = []
        for v in base_pattern:
            if random.random() < self.noise_prob:
                v = self._perturb_pixel(v)
            intensity.append(self._clamp(v))
        return intensity

    @abstractmethod
    def _perturb_pixel(self, v: int) -> int:
        """子类实现：对单个像素施加噪声，返回新值"""
        pass


class DefaultCompositeNoise(BaseNoiseModel):
    """默认复合噪声模型（v4.2 窗口大小自适应）

    三层复合噪声模型（更接近真实传感器故障）：
      40% 电平翻转:   0↔255，模拟按键接触不良/线路断开（离散故障）
      30% 强抖动:     ±128，模拟 ADC 采样误差/电源纹波（大幅跳变）
      30% 弱抖动:     ±32， 模拟机械振动/温度漂移（小幅干扰）
    """

    def __init__(self, noise_prob: float = 0.15):
        super().__init__(noise_prob)

    def _perturb_pixel(self, v: int) -> int:
        r = random.random()
        if r < 0.40:
            # 电平翻转：离散故障，彻底变脸
            return 255 - v
        elif r < 0.70:
            # 强抖动：可能改变像素排序关系
            return v + random.randint(-128, 128)
        else:
            # 弱抖动：通常不改变排序，但增加数值不确定性
            return v + random.randint(-32, 32)


class GaussianNoise(BaseNoiseModel):
    """高斯噪声模型 —— 插件可注入

    更适合模拟连续随机噪声的场景。
    """

    def __init__(self, noise_prob: float = 0.15, sigma: float = 32.0):
        super().__init__(noise_prob)
        self.sigma = sigma

    def _perturb_pixel(self, v: int) -> int:
        return int(v + random.gauss(0, self.sigma))


class SaltPepperNoise(BaseNoiseModel):
    """椒盐噪声 —— 训练分布外，测极端离散故障"""

    def __init__(self, noise_prob: float = 0.15):
        super().__init__(noise_prob)

    def _perturb_pixel(self, v: int) -> int:
        return 255 if random.random() < 0.5 else 0


class BlockNoise(NoiseModel):
    """随机块遮挡 —— 训练分布外，测大面积缺失

    直接继承 NoiseModel（非 BaseNoiseModel），因为块遮挡需要
    二维坐标计算，无法抽象为单像素 _perturb_pixel。
    """

    def __init__(self, block_size: int = 2, prob: float = 0.2, grid_size: int = 4):
        self.noise_prob = prob
        self.grid_size = grid_size
        self.block_size = block_size

    def apply(self, base_pattern: List[int]) -> List[int]:
        if self.grid_size is None:
            raise ValueError("BlockNoise 需要 grid_size 参数")
        intensity = base_pattern.copy()
        gs = self.grid_size
        if random.random() < self.noise_prob:
            # 随机选块左上角
            max_r = gs - self.block_size
            max_c = gs - self.block_size
            if max_r < 0 or max_c < 0:
                return intensity  # 块太大，不遮挡
            r = random.randint(0, max_r)
            c = random.randint(0, max_c)
            for dr in range(self.block_size):
                for dc in range(self.block_size):
                    idx = (r + dr) * gs + (c + dc)
                    if idx < len(intensity):
                        intensity[idx] = 0
        return intensity


class FeatureExtractor(ABC):
    """特征提取器抽象基类

    从强度值中提取二值化层掩膜。
    所有实现必须返回 (layer_masks, layer_count)，其中 layer_masks 是 int 列表。
    """

    @abstractmethod
    def extract(self, intensity: List[int]) -> Tuple[List[int], int]:
        """返回 (layer_masks, layer_count)

        Args:
            intensity: 输入强度值列表（0-255）

        Returns:
            layer_masks: 二值化层掩膜列表，每个元素是 int
            layer_count: 实际层数
        """
        pass

    def validate_output(self, masks: List[int], layer_count: int, max_layers: int = 4) -> None:
        """验证输出格式是否符合契约

        Raises:
            AssertionError: 若输出格式不符合要求
        """
        assert isinstance(masks, list), f"layer_masks 必须是 list， got {type(masks)}"
        assert layer_count <= max_layers, f"layer_count={layer_count} 超过 LAYER_MAX={max_layers}"
        for i, m in enumerate(masks):
            assert isinstance(m, int), f"layer_masks[{i}] 必须是 int， got {type(m)}"


class DefaultLayerExtractor(FeatureExtractor):
    """默认层提取器（v4.1 extract_layers 逻辑）

    二值化提取最多 layer_max 层 16 位掩膜。
    """

    def __init__(self, layer_max: int = 4, d: int = 16):
        self.layer_max = layer_max
        self.d = d

    def extract(self, intensity: List[int]) -> Tuple[List[int], int]:
        layer_masks = []
        remaining = list(intensity[:self.d])  # 拷贝避免修改原始数据

        # 空输入防御
        if not remaining:
            return [], 0

        # 全亮画布特判：所有像素均为高值时，直接返回全掩膜
        if all(v > 128 for v in remaining):
            return [(1 << self.d) - 1], 1  # 动态位宽：16→0xFFFF, 64→2^64-1

        for _ in range(self.layer_max):
            mark_count = max(max(3, self.d // 8), sum(1 for v in remaining if v > 0) // 2)
            # 修复排序：值降序、值相同时索引升序（取前面的点）
            active = sorted(((v, i) for i, v in enumerate(remaining) if v > 0),
                            key=lambda x: (-x[0], x[1]))
            if not active:
                break
            mask = 0
            for _, idx in active[:mark_count]:
                mask |= (1 << idx)
                remaining[idx] = 0
            layer_masks.append(mask)

        # 契约校验
        self.validate_output(layer_masks, len(layer_masks), self.layer_max)
        return layer_masks, len(layer_masks)


# ============================================================
# InputSource - 输入源抽象
# ============================================================

class InputSource(ABC):
    """输入源抽象基类

    生成训练用的 (intensity, label) 样本批次。
    若数据不足，实现必须循环采样或自动复制。
    """

    @abstractmethod
    def generate_batch(self, count: int, split: str = 'train') -> List[Tuple[List[int], str]]:
        """返回 [(intensity, label), ...]

        Args:
            count: 需要的样本数量

        Returns:
            样本列表，长度 >= count（允许循环采样）
        """
        pass


class PatternInputSource(InputSource):
    """模式输入源（v4.1 默认行为）

    从 PATTERNS 字典循环生成带噪声的训练样本。
    """

    def __init__(self, patterns: dict, noise_model: NoiseModel, labels: Optional[List[str]] = None, test_split: float = 0.0, validation_labels: Optional[List[str]] = None):
        self.patterns = patterns
        self.noise = noise_model
        self.labels = labels or list(patterns.keys())
        self.test_split = test_split
        self.validation_labels = set(validation_labels or [])
        self._train_labels = [l for l in self.labels if l not in self.validation_labels]
        self._test_labels = [l for l in self.labels if l in self.validation_labels]

    def generate_batch(self, count: int, split: str = 'train') -> List[Tuple[List[int], str]]:
        if split == 'test' and self._test_labels:
            label_pool = self._test_labels
        elif split == 'test' and self.test_split > 0:
            # 比例留出模式：由上层统一 stratified split
            # PatternInputSource 本身不维护状态，返回 all 让上层切
            label_pool = self.labels
        else:
            # train 或 all
            label_pool = self._train_labels if self._train_labels else self.labels

        samples = []
        label_idx = 0
        for _ in range(count):
            label = label_pool[label_idx % len(label_pool)]
            label_idx += 1
            base = self.patterns[label]
            # 拷贝避免噪声模型修改原始模板
            noisy = self.noise.apply(base.copy())
            samples.append((noisy, label))
        return samples

    def get_label_distribution(self) -> Dict[str, int]:
        return {lbl: len(self.patterns[lbl]) for lbl in self.labels}


class FileInputSource(InputSource):
    """文件输入源 - 从外部 CSV/JSON 加载样本

    CSV 格式: intensity_0,intensity_1,...,intensity_15,label
    JSON 格式: [{"intensity": [...], "label": "0"}, ...]
    """

    def __init__(self, filepath: str, feature_extractor: Optional[FeatureExtractor] = None):
        self.filepath = filepath
        self.feature_extractor = feature_extractor
        self._samples: List[Tuple[List[int], str]] = []
        self._load()

    def _load(self) -> None:
        """从文件加载样本"""
        if self.filepath.endswith(".csv"):
            self._load_csv()
        elif self.filepath.endswith(".json"):
            self._load_json()
        else:
            raise ValueError(f"不支持的文件格式: {self.filepath}（仅支持 .csv 和 .json）")

    def _load_csv(self) -> None:
        with open(self.filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                # 最后一列是 label，前面是 intensity
                if len(row) < 2:
                    continue
                label = row[-1].strip()
                intensity = [int(float(v)) for v in row[:-1]]
                self._samples.append((intensity, label))

    def _load_json(self) -> None:
        with open(self.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            intensity = item.get("intensity", item.get("data", []))
            label = str(item.get("label", item.get("class", "?")))
            if intensity:
                self._samples.append(([int(v) for v in intensity], label))

    def generate_batch(self, count: int, split: str = 'train') -> List[Tuple[List[int], str]]:
        """循环采样：若数据不足自动循环"""
        if not self._samples:
            raise ValueError(f"文件 {self.filepath} 中没有有效样本")
        result = []
        for i in range(count):
            result.append(self._samples[i % len(self._samples)])
        return result




# ============================================================
# VectorPatternSource - 矢量图形生成器（8×8/16×16 扩展）
# ============================================================

class VectorPatternSource(InputSource):
    """矢量图形输入源 — 整数化数学方程实时渲染为像素网格

    【整数化渲染管线 — 自然灰度，不预设分层】
    全部计算使用 x1000 定点数 + LUT 查表，无浮点运行时依赖。
    抗锯齿基于亚像素覆盖面积计算连续灰度（0~255），
    不人为预设 128/80 等"配合 SGN 分层"的硬编码值。

    核心原则：
      - 渲染只管生成真实的传感器级灰度矩阵
      - SGN 的 extract_layers 自己扫描、排序、截断、分层
      - 两者通过 (intensity, label) 接口联动，互不侵入内部逻辑

    支持的公式类型：
      - line:    y = k*x + b      (直线)
      - circle:  (x-cx)^2 + (y-cy)^2 = r^2  (圆)
      - sine:    y = A*sin(omega*x + phi)   (正弦波)
    """

    # 类级 LUT：360点 sin 查表，值域 x1000，占用 720 字节
    _sin_table = None
    _lut_ready = False

    def __init__(self, formula_type: str = "line", grid_size: int = 8,
                 noise_model: Optional[NoiseModel] = None,
                 samples_per_label: int = 10,
                 seed: int = 42):
        self.formula_type = formula_type
        self.grid_size = grid_size
        self.d = grid_size * grid_size
        self.noise = noise_model or DefaultCompositeNoise(0.15)
        self.samples_per_label = samples_per_label
        self._rng = random.Random(seed)
        if not VectorPatternSource._lut_ready:
            VectorPatternSource._init_lut()

    @classmethod
    def _init_lut(cls):
        """初始化 sin LUT（360点，值域 x1000）"""
        import math  # 仅在 LUT 初始化时使用
        cls._sin_table = [int(math.sin(math.radians(d)) * 1000) for d in range(360)]
        cls._lut_ready = True

    # ---- 整数化坐标映射（x1000） ----
    def _x_coord(self, i: int) -> int:
        """x 坐标映射（x1000 定点数，结果 ∈ [-1000, 1000]）"""
        gs = self.grid_size
        center = (gs - 1) * 1000 // 2
        scale = (gs - 1) * 1000 // 2
        return (i * 1000 - center) * 1000 // scale

    # ---- 自然灰度抗锯齿：亚像素覆盖分配 ----
    def _write_pixel(self, intensity: List[int], i: int, j_precise: int):
        """根据亚像素精确位置 j_precise（x1000）分配连续灰度

        j_precise // 1000 = 主像素行
        j_precise % 1000  = 小数部分（到主像素中心的距离比例）

        主像素灰度 = 255 * (1000 - frac) // 1000
        邻域像素灰度 = 255 * frac // 1000
        两者之和 ≤ 255，能量守恒。
        """
        gs = self.grid_size
        j_main = j_precise // 1000
        frac = j_precise % 1000

        # 主像素（线条主要覆盖区域）
        if 0 <= j_main < gs:
            val = 255 * (1000 - frac) // 1000
            idx = j_main * gs + i
            if val > intensity[idx]:
                intensity[idx] = val

        # 邻域像素（线条边缘覆盖区域）
        if frac > 0 and 0 <= j_main + 1 < gs:
            val = 255 * frac // 1000
            idx = (j_main + 1) * gs + i
            if val > intensity[idx]:
                intensity[idx] = val

    # ---- 整数化渲染：直线（亚像素 DDA） ----
    def _render_line(self, k_x1000: int, b_x1000: int) -> List[int]:
        """整数化渲染直线 y = k*x + b

        对每个 x 列计算精确 y（x1000），再通过 _write_pixel 分配亚像素灰度。
        不人为设 128，灰度由线条与像素中心的距离自然决定。
        """
        gs = self.grid_size
        intensity = [0] * (gs * gs)
        for i in range(gs):
            x = self._x_coord(i)
            y = (k_x1000 * x // 1000) + b_x1000
            # j_precise: y ∈ [-1000, 1000] 映射为 x1000 精度的像素坐标
            j_precise = (1000 - y) * (gs - 1) * 1000 // 2000
            self._write_pixel(intensity, i, j_precise)
        return intensity

    # ---- 整数化渲染：正弦波（亚像素 LUT + 离散方向） ----
    def _render_sine(self, A_x1000: int, omega_x1000: int, phi_x1000: int,
                     orientation: str = "horizontal") -> List[int]:
        """整数化渲染正弦波

        Args:
            orientation: "horizontal"（横向 y=A*sin(ωx+φ)）
                         "vertical"（纵向 x=A*sin(ωy+φ)）
        """
        gs = self.grid_size
        intensity = [0] * (gs * gs)

        if orientation == "vertical":
            # 纵向：交换 x/y 遍历方向
            for j in range(gs):
                y = self._x_coord(j)
                angle = (omega_x1000 * y // 1000) + phi_x1000
                deg = (angle * 360 // 6283) % 360
                sin_val = self._sin_table[deg]
                x_val = A_x1000 * sin_val // 1000
                # x 方向写像素
                i_precise = (1000 + x_val) * (gs - 1) * 1000 // 2000
                i_main = i_precise // 1000
                frac = i_precise % 1000
                if 0 <= i_main < gs:
                    val = 255 * (1000 - frac) // 1000
                    idx = j * gs + i_main
                    if val > intensity[idx]:
                        intensity[idx] = val
                if frac > 0 and 0 <= i_main + 1 < gs:
                    val = 255 * frac // 1000
                    idx = j * gs + i_main + 1
                    if val > intensity[idx]:
                        intensity[idx] = val
        else:
            # 横向：原逻辑
            for i in range(gs):
                x = self._x_coord(i)
                angle = (omega_x1000 * x // 1000) + phi_x1000
                deg = (angle * 360 // 6283) % 360
                sin_val = self._sin_table[deg]
                y = A_x1000 * sin_val // 1000
                j_precise = (1000 - y) * (gs - 1) * 1000 // 2000
                self._write_pixel(intensity, i, j_precise)
        return intensity

    # ---- 整数化渲染：圆（距离场 + 多形态） ----
    def _render_circle(self, cx_x1000: int, cy_x1000: int, r_x1000: int,
                       variant: str = "solid") -> List[int]:
        """整数化渲染圆

        Args:
            variant: 渲染形态
                "solid" — 实心圆，内部填充 + 边界自然衰减
                "thin"  — 细空心环（~1px 边界）
                "thick" — 粗空心环（~3px 边界）
        """
        gs = self.grid_size
        intensity = [0] * (gs * gs)

        if variant == "thin":
            edge_width = 1000   # 1.0 pixel
        elif variant == "thick":
            edge_width = 3000   # 3.0 pixels
        else:
            edge_width = 0      # solid: 不用于边界判断

        for i in range(gs):
            x = self._x_coord(i)
            for j in range(gs):
                y = self._x_coord(j)
                dx = x - cx_x1000
                dy = y - cy_x1000
                dist = int((dx * dx + dy * dy) ** 0.5)
                idx = j * gs + i

                if variant == "solid":
                    if dist <= r_x1000:
                        # 实心：内部全亮，边缘 2px 自然衰减
                        fade = 2000
                        if r_x1000 - dist < fade:
                            val = 255 * (r_x1000 - dist) // fade
                        else:
                            val = 255
                        if val > intensity[idx]:
                            intensity[idx] = val
                else:
                    # 空心：仅填充边界附近
                    d_edge = abs(dist - r_x1000)
                    if d_edge < edge_width:
                        val = 255 * (edge_width - d_edge) // edge_width
                        if val > intensity[idx]:
                            intensity[idx] = val
        return intensity

    # ---- 整数化渲染：猫耳（双圆弧 + 朝向旋转） ----
    def _render_catear(self, ear_spacing_x1000: int, ear_radius_x1000: int,
                       open_angle_deg: int, orientation: str = "up",
                       variant: str = "outline") -> List[int]:
        """整数化渲染猫耳（两只对称圆弧耳）

        Args:
            ear_spacing_x1000: 两耳中心间距 (x1000)
            ear_radius_x1000: 每只耳朵半径 (x1000)
            open_angle_deg: 圆弧开角(度), 120=尖耳 150=中 180=半圆
            orientation: 朝向 up/down/left/right
            variant: outline=轮廓 / filled=填充
        """
        import math as _math
        gs = self.grid_size
        intensity = [0] * (gs * gs)

        half_spacing = ear_spacing_x1000 // 2
        half_angle = open_angle_deg // 2

        # 圆弧角度范围 (以"朝上"为基准, 弧顶在上)
        start_deg = 180 - half_angle
        end_deg = 180 + half_angle

        # 边界宽度 (outline 模式)
        edge_width = max(500, ear_radius_x1000 // 3)

        for i in range(gs):
            x_raw = self._x_coord(i)
            for j in range(gs):
                y_raw = self._x_coord(j)

                # 朝向旋转：将渲染坐标映射到"朝上"基准坐标系
                if orientation == "down":
                    rx, ry = x_raw, -y_raw
                elif orientation == "left":
                    rx, ry = y_raw, x_raw
                elif orientation == "right":
                    rx, ry = -y_raw, x_raw
                else:  # up
                    rx, ry = x_raw, y_raw

                max_val = 0

                for side in (-1, 1):
                    ear_cx = side * half_spacing
                    # 耳朵中心偏移：朝上方向偏移半径，使弧底在原点
                    ear_cy = -ear_radius_x1000
                    dx = rx - ear_cx
                    dy = ry - ear_cy
                    dist = int((dx * dx + dy * dy) ** 0.5)

                    if dist == 0:
                        angle_deg = 90  # 正上方
                    else:
                        # 计算角度: 从耳朵中心指向像素的方向
                        # atan2(-dy, dx) 因为 y 轴向下, 0度=右, 90度=上
                        angle_rad = _math.atan2(-dy, dx)
                        angle_deg = int(angle_rad * 180 / _math.pi)
                        if angle_deg < 0:
                            angle_deg += 360

                    # 检查角度是否在圆弧范围内
                    in_arc = False
                    if start_deg <= end_deg:
                        in_arc = start_deg <= angle_deg <= end_deg
                    else:
                        in_arc = angle_deg >= start_deg or angle_deg <= end_deg

                    if not in_arc:
                        continue

                    if variant == "filled":
                        if dist <= ear_radius_x1000:
                            fade = 1000
                            if ear_radius_x1000 - dist < fade:
                                val = 255 * (ear_radius_x1000 - dist) // fade
                            else:
                                val = 255
                            max_val = max(max_val, val)
                    else:  # outline
                        d_edge = abs(dist - ear_radius_x1000)
                        if d_edge < edge_width:
                            val = 255 * (edge_width - d_edge) // edge_width
                            max_val = max(max_val, val)

                idx = j * gs + i
                intensity[idx] = max_val

        return intensity

    # ---- 参数生成（x1000 定点数） ----
    def _generate_params(self, label: str) -> dict:
        """随机生成公式参数（圆/正弦离散化防背板）"""
        r = self._rng
        if self.formula_type == "line":
            # 离散化防背板：7 种典型斜率 × 3 档截距
            return {
                "k": r.choice([-2000, -1000, -500, 0, 500, 1000, 2000]),
                "b": r.choice([-500, 0, 500]),
            }
        elif self.formula_type == "circle":
            # 离散化防背板：9 种位置 × 3 档半径 × 3 种形态
            cx_choices = [-500, 0, 500]
            cy_choices = [-500, 0, 500]
            r_choices = [300, 500, 700]
            variant_choices = ["solid", "thin", "thick"]
            return {
                "cx": r.choice(cx_choices),
                "cy": r.choice(cy_choices),
                "r": r.choice(r_choices),
                "variant": r.choice(variant_choices),
            }
        elif self.formula_type == "sine":
            # 离散化防背板：方向二选一，振幅/频率/相位离散档位
            return {
                "orientation": r.choice(["horizontal", "vertical"]),
                "A": r.choice([500, 800, 1000]),
                "omega": r.choice([1500, 2500, 3500]),
                "phi": r.choice([0, 1571, 3142, 4712]),  # 0, π/2, π, 3π/2
            }
        elif self.formula_type == "catear":
            # 猫耳：4朝向 × 3间距 × 3半径 × 3开角 × 2形态 = 216 种
            return {
                "orientation": r.choice(["up", "down", "left", "right"]),
                "ear_spacing": r.choice([300, 500, 700]),
                "ear_radius": r.choice([200, 350, 500]),
                "open_angle": r.choice([120, 150, 180]),
                "variant": r.choice(["outline", "filled"]),
            }
        return {}

    def _contrast_stretch(self, intensity: List[int]) -> List[int]:
        """对比度拉伸：将强度值归一化到 0-255 全范围，增强图形与背景差异"""
        min_v = min(intensity)
        max_v = max(intensity)
        if max_v == min_v:
            return intensity[:]
        return [int((v - min_v) * 255 / (max_v - min_v)) for v in intensity]

    def _render(self, label: str) -> List[int]:
        """根据标签和随机参数渲染强度矩阵"""
        params = self._generate_params(label)
        if self.formula_type == "line":
            raw = self._render_line(params["k"], params["b"])
        elif self.formula_type == "circle":
            raw = self._render_circle(params["cx"], params["cy"], params["r"],
                                      variant=params.get("variant", "solid"))
        elif self.formula_type == "sine":
            raw = self._render_sine(params["A"], params["omega"], params["phi"],
                                    orientation=params.get("orientation", "horizontal"))
        elif self.formula_type == "catear":
            raw = self._render_catear(params["ear_spacing"], params["ear_radius"],
                                      params["open_angle"],
                                      orientation=params.get("orientation", "up"),
                                      variant=params.get("variant", "outline"))
        else:
            raw = [0] * self.d
        # 【v4.3-fix】对比度拉伸，增强图形与背景差异，减少噪声干扰
        raw = self._contrast_stretch(raw)
        return self.noise.apply(raw)

    def generate_batch(self, count: int, split: str = 'train') -> List[Tuple[List[int], str]]:
        """生成矢量图形样本批次"""
        labels_map = {
            "line": ["LINE"],
            "circle": ["CIRCLE"],
            "sine": ["SINE"],
            "catear": ["CATEAR"],
        }
        labels = labels_map.get(self.formula_type, ["UNKNOWN"])
        samples = []
        for i in range(count):
            lbl = labels[i % len(labels)]
            intensity = self._render(lbl)
            samples.append((intensity, lbl))
        return samples


# ============================================================
# 社区简化接口
# ============================================================

def create_vector_source(
    formula: str = "line",
    grid_size: int = 8,
    noise_prob: float = 0.15,
    seed: int = 42,
) -> InputSource:
    """创建整数化矢量图形输入源（社区简化接口）

    内部全部整数化渲染，输出自然灰度（0~255），
    不人为预设分层值，由 SGN extract_layers 自行扫描剥离。

    使用示例:
        source = create_vector_source("line", grid_size=8, noise_prob=0.1)
        samples = source.generate_batch(100)
        # SGNCore 首样本自动识别 D=64
    """
    noise = DefaultCompositeNoise(noise_prob)
    return VectorPatternSource(formula, grid_size, noise, seed=seed)


def create_mixed_vector_source(
    formulas: list = None,
    grid_size: int = 8,
    samples_per_formula: int = 100,
    noise_prob: float = 0.15,
    seed: int = 42,
) -> InputSource:
    """创建多公式混合矢量输入源（多类识别训练）"""
    if formulas is None:
        formulas = ["line", "circle", "sine", "catear"]
    noise = DefaultCompositeNoise(noise_prob)
    return _MixedVectorSource(formulas, grid_size, noise, samples_per_formula, seed)


class _MixedVectorSource(InputSource):
    """内部类：多公式混合样本生成器"""

    def __init__(self, formulas, grid_size, noise, samples_per_formula, seed):
        self.sources = []
        for i, formula in enumerate(formulas):
            src = VectorPatternSource(formula, grid_size, noise, samples_per_formula, seed=seed + i)
            self.sources.append(src)
        self.samples_per_formula = samples_per_formula

    def generate_batch(self, count: int, split: str = 'train') -> List[Tuple[List[int], str]]:
        samples = []
        for src in self.sources:
            samples.extend(src.generate_batch(self.samples_per_formula))
        __import__('random').Random(42).shuffle(samples)
        return samples[:count]

# ============================================================
# 默认输入源工厂（用于 main.py 的默认行为）
# ============================================================

def create_default_source(noise_prob: float = None) -> InputSource:
    """创建默认输入源（内置 PATTERNS + 复合噪声）

    用于 main.py 的默认行为，完全兼容 v4.1/v4.2 原有逻辑。

    Args:
        noise_prob: 噪声翻转概率，若为 None 则从 CONFIG['FLIP_PROB'] 读取

    Returns:
        PatternInputSource 实例
    """
    from sgn_config import PATTERNS, CONFIG
    prob = noise_prob if noise_prob is not None else CONFIG.get("FLIP_PROB", 0.15)
    noise = DefaultCompositeNoise(prob)
    return PatternInputSource(PATTERNS, noise)

# ============================================================
# 文件输入源工厂（用于 main.py 的 --input-source file）
# ============================================================

def create_file_source(filepath: str) -> InputSource:
    """从文件创建输入源（CSV/JSON）

    用于 main.py 的 --input-source file --dataset data.csv 参数。
    完全兼容 FileInputSource 的 CSV/JSON 格式。

    Args:
        filepath: 数据文件路径（.csv 或 .json）

    Returns:
        FileInputSource 实例
    """
    return FileInputSource(filepath)


# ============================================================
# 标准字符输入源（8×8 字符库，仅用于测试）
# ============================================================

class StandardCharSource(InputSource):
    """8×8 标准字符输入源 — 仅用于测试，不用于训练

    使用 sgn_config.STANDARD_CHARS_8x8 字符库，支持噪声注入。
    """

    def __init__(self, chars: str = None, noise_model: NoiseModel = None, grid_size: int = 8):
        from sgn_config import STANDARD_CHARS_8x8
        self.char_lib = STANDARD_CHARS_8x8
        if chars is None:
            self.labels = sorted(self.char_lib.keys())
        else:
            self.labels = [c.upper() for c in chars if c.upper() in self.char_lib]
        self.noise = noise_model or DefaultCompositeNoise(0.0)
        self.grid_size = grid_size
        self.d = grid_size * grid_size

    def generate_batch(self, count: int, split: str = 'train') -> List[Tuple[List[int], str]]:
        samples = []
        per_char = max(1, count // len(self.labels))
        for lbl in self.labels:
            base = self.char_lib[lbl]
            for _ in range(per_char):
                noisy = self.noise.apply(base[:])
                samples.append((noisy, lbl))
        random.shuffle(samples)
        return samples[:count]
