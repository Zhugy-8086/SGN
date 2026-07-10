#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gui/utils.py — 工具函数（intensity 转换、变换、图片加载、训练集导出）

输入层涉及外部模拟信号，保留浮点运算（旋转/缩放等）。
核心识别层仍由 SGN 处理，保持整数化。
"""
from __future__ import annotations

from typing import List, Tuple
import pygame
from pygame import Surface, Color


def intensity_to_surface(intensity: List[int], grid_size: int) -> Surface:
    """将 intensity 列表转为 pygame Surface（灰度）"""
    surf = pygame.Surface((grid_size, grid_size))
    for y in range(grid_size):
        for x in range(grid_size):
            v = intensity[y * grid_size + x]
            surf.set_at((x, y), (v, v, v))
    return surf


def surface_to_intensity(surf: Surface, grid_size: int) -> List[int]:
    """将 pygame Surface 转为 intensity 列表"""
    intensity = []
    for y in range(grid_size):
        for x in range(grid_size):
            c = surf.get_at((x, y))
            intensity.append(c[0])
    return intensity


def load_image_to_intensity(path: str, grid_size: int) -> List[int]:
    """从外部图片文件加载并缩放为指定网格大小的 intensity

    1. 用 PIL 读取图片（支持任意格式）
    2. 转为灰度
    3. 缩放为 grid_size × grid_size
    4. 归一化到 0-255

    若未安装 PIL，尝试用 pygame.image.load（支持 bmp/png 等）。
    """
    try:
        from PIL import Image
        img = Image.open(path).convert("L")
        img = img.resize((grid_size, grid_size), Image.LANCZOS)
        pixels = list(img.getdata())
        return [max(0, min(255, int(v))) for v in pixels]
    except ImportError:
        pass

    # 回退到 pygame
    try:
        img = pygame.image.load(path)
        surf = pygame.transform.smoothscale(img, (grid_size, grid_size))
        intensity = []
        for y in range(grid_size):
            for x in range(grid_size):
                c = surf.get_at((x, y))
                # 灰度 = 0.299R + 0.587G + 0.114B（浮点运算）
                gray = int(0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
                intensity.append(max(0, min(255, gray)))
        return intensity
    except Exception as e:
        raise RuntimeError(f"无法加载图片 {path}: {e}")


def export_training_dataset(
    samples: List[Tuple[List[int], str]],
    path: str,
    grid_size: int,
    format: str = "csv",
) -> None:
    """导出自定义训练集到文件

    Args:
        samples: [(intensity, label), ...]
        path: 输出文件路径
        grid_size: 网格边长
        format: "csv" 或 "json"
    """
    if format == "csv":
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = [f"i{i}" for i in range(grid_size * grid_size)] + ["label"]
            writer.writerow(header)
            for intensity, label in samples:
                row = list(intensity) + [label]
                writer.writerow(row)
    elif format == "json":
        import json
        data = []
        for intensity, label in samples:
            data.append({"intensity": intensity, "label": label, "grid_size": grid_size})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"不支持的格式: {format}（仅 csv/json）")


def apply_transform(
    intensity: List[int],
    grid_size: int,
    angle: float,
    offset_x: int,
    offset_y: int,
    scale: float,
) -> List[int]:
    """对 intensity 应用旋转变换（浮点渲染管线 → 输入层保留精度）

    返回变换后的 intensity 列表（长度不变，grid_size × grid_size）
    """
    # 1. 转为 surface
    surf = intensity_to_surface(intensity, grid_size)

    # 2. 缩放（浮点运算）
    if scale != 1.0:
        new_size = max(1, int(grid_size * scale))
        surf = pygame.transform.smoothscale(surf, (new_size, new_size))

    # 3. 旋转（浮点运算，pygame 使用顺时针，所以取负）
    if angle != 0.0:
        surf = pygame.transform.rotate(surf, -angle)

    # 4. 居中 blit 到目标 surface，加上偏移
    result = pygame.Surface((grid_size, grid_size))
    result.fill((0, 0, 0))
    rw, rh = surf.get_size()
    cx = (grid_size - rw) // 2 + offset_x
    cy = (grid_size - rh) // 2 + offset_y
    result.blit(surf, (cx, cy))

    # 5. 转回 intensity
    return surface_to_intensity(result, grid_size)


def contrast_stretch(intensity: List[int]) -> List[int]:
    """对比度拉伸（Percentile clipping）— 保留浮点运算，归一化到 0-255

    参考 sgn_input.py 的 VectorPatternSource._contrast_stretch 逻辑。
    """
    sorted_vals = sorted(intensity)
    n = len(sorted_vals)
    if n < 3:
        return intensity[:]
    low = sorted_vals[n // 50]       # 2% percentile
    high = sorted_vals[n * 49 // 50]  # 98% percentile
    if high <= low:
        return intensity[:]
    # 浮点运算：拉伸到 0-255
    return [max(0, min(255, int((v - low) * 255.0 / (high - low)))) for v in intensity]
