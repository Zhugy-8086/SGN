#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gui/charlib.py — 内置字符库与预设图案（插件化 8×8）

v5.1.5 重构：
  - 移除 4×4 硬编码模板（CHAR_4x4 / _build_8x8）
  - 改用 engine.config.CharRegistry 插件系统作为字符模板源
  - 默认字符集：8×8 标准点阵（0-9, A-Z，共 36 字符）
  - 字体渲染保留为最终回退，支持任意可打印 ASCII 字符

字符查找优先级：
  1. CharRegistry 注册的预定义模板（精确匹配 + 最近邻缩放）
  2. pygame 字体渲染（任意单字符 ASCII 32~126）
  3. 全零 intensity（无法渲染时的安全回退）
"""
from __future__ import annotations

from typing import Dict, List, Optional


# ============================================================
# 字体渲染缓存（用于生成任意 ASCII 字符的 intensity）
# ============================================================

# 字体缓存：size -> pygame.font.Font，避免重复加载系统字体
_font_cache: Dict[int, "pygame.font.Font"] = {}


def _get_font(size: int):
    """获取指定大小的 pygame 字体（延迟导入，带缓存）

    优先使用 Windows 系统中文字体（微软雅黑/黑体/宋体），
    找不到时回退到 pygame 默认字体。

    Args:
        size: 字体像素大小

    Returns:
        pygame.font.Font 实例
    """
    import pygame
    import os
    if size in _font_cache:
        return _font_cache[size]
    if not pygame.get_init():
        pygame.init()
    # Windows 中文字体候选列表（按优先级排序）
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    font = None
    for path in candidates:
        if os.path.exists(path):
            try:
                font = pygame.font.Font(path, size)
                break
            except Exception:
                pass
    if font is None:
        font = pygame.font.SysFont(None, size)
    _font_cache[size] = font
    return font


def _render_char_to_intensity(char: str, grid_size: int) -> List[int]:
    """通过 pygame 字体渲染将字符转为 intensity 列表

    流程：
      1. 用大字号渲染字符为白色 Surface（高分辨率抗锯齿）
      2. 裁剪到非透明边界框（去除周围空白）
      3. 平滑缩放到 grid_size × grid_size
      4. 逐像素取 RGB 平均值作为灰度 intensity

    Args:
        char: 单个字符
        grid_size: 目标网格边长

    Returns:
        长度为 grid_size * grid_size 的 intensity 列表（0-255）
    """
    import pygame

    # 渲染分辨率为目标网格的 4 倍以上，确保缩放后质量
    render_size = max(grid_size * 4, 64)
    font = _get_font(render_size)
    surf = font.render(char, True, (255, 255, 255))

    # 裁剪到字符实际占据的区域（去除透明边距）
    bbox = surf.get_bounding_rect()
    if bbox.width == 0 or bbox.height == 0:
        return [0] * (grid_size * grid_size)

    cropped = surf.subsurface(bbox).copy()

    # 平滑缩放到目标网格大小
    scaled = pygame.transform.smoothscale(cropped, (grid_size, grid_size))
    # surfarray 返回的是 (width, height, 3) 数组，注意轴顺序
    pixels = pygame.surfarray.array3d(scaled)

    # 逐像素计算灰度值（RGB 平均）
    intensity = [0] * (grid_size * grid_size)
    for y in range(grid_size):
        for x in range(grid_size):
            r, g, b = pixels[x][y]
            intensity[y * grid_size + x] = int((int(r) + int(g) + int(b)) // 3)
    return intensity


# ============================================================
# 公共 API
# ============================================================

def get_char(char: str, grid_size: int = 8) -> List[int]:
    """获取指定字符的 intensity 模板

    查找优先级：
      1. CharRegistry 预定义模板（默认 8×8 标准点阵 0-9 A-Z）
         - 若目标 grid_size 与注册集一致，直接返回副本
         - 若不一致，使用最近邻缩放适配
      2. pygame 字体渲染（任意可打印 ASCII 字符）
      3. 全零列表（安全回退）

    Args:
        char: 单个字符（可打印 ASCII）
        grid_size: 目标网格大小（8/16/32/64）

    Returns:
        长度为 grid_size * grid_size 的 intensity 列表（0-255）
    """
    char = char.upper() if len(char) == 1 else char

    # 优先从 CharRegistry 插件系统获取预定义模板
    try:
        from engine.config import CharRegistry
        result = CharRegistry.get_char(char, grid_size)
        if result is not None:
            return result
    except Exception:
        pass  # 引擎未就绪时静默降级到字体渲染

    # 最终回退：用 pygame 字体渲染生成（支持任意 ASCII 字符）
    if len(char) == 1 and 32 <= ord(char) <= 126:
        return _render_char_to_intensity(char, grid_size)

    # 无法渲染时返回全零（安全回退，避免调用方崩溃）
    return [0] * (grid_size * grid_size)


def list_chars() -> List[str]:
    """返回所有可用字符列表

    包含两部分：
      1. CharRegistry 默认字符集的标签（0-9, A-Z）
      2. 其余可打印 ASCII 字符（空格 ~ 之间，排除已注册的）

    Returns:
        字符列表，注册集字符在前，其余 ASCII 按序追加
    """
    # 获取已注册的字符标签
    try:
        from engine.config import CharRegistry
        registered = CharRegistry.get_labels()
    except Exception:
        registered = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    registered_set = set(registered)
    # 追加其余可打印 ASCII 字符（32=空格, 126=~）
    extra = [chr(i) for i in range(32, 127) if chr(i) not in registered_set]
    return registered + extra
