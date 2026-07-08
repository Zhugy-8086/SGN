#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gui/charlib.py — 内置字符库与预设图案

提供 0-9, A-F 等常用字符的网格模板，供 GUI 一键加载到画布。
支持通过 pygame 字体渲染生成任意可打印 ASCII 字符的 intensity。
"""
from __future__ import annotations

from typing import Dict, List, Optional


# 4×4 字符模板（与 sgn_config.PATTERNS 一致，但 8×8 扩展）
CHAR_4x4: Dict[str, List[int]] = {
    "0": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    "1": [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1],
    "2": [0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1],
    "3": [0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
    "4": [1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    "5": [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
    "6": [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0],
    "7": [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    "8": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    "9": [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    "A": [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    "B": [1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    "C": [0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
    "D": [1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0],
    "E": [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
    "F": [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
}


# 8×8 扩展字符模板（更精细）
CHAR_8x8: Dict[str, List[int]] = {}


def _build_8x8():
    """从 4×4 模板放大到 8×8"""
    for ch, pixels in CHAR_4x4.items():
        intensity = [0] * 64
        for y in range(4):
            for x in range(4):
                v = pixels[y * 4 + x] * 255
                for dy in range(2):
                    for dx in range(2):
                        ny = y * 2 + dy
                        nx = x * 2 + dx
                        intensity[ny * 8 + nx] = v
        CHAR_8x8[ch] = intensity


_build_8x8()


# ============================================================
# 字体渲染缓存（用于生成任意 ASCII 字符）
# ============================================================

_font_cache: Dict[int, "pygame.font.Font"] = {}


def _get_font(size: int):
    """获取指定大小的 pygame 字体（延迟导入）"""
    import pygame
    import os
    if size in _font_cache:
        return _font_cache[size]
    if not pygame.get_init():
        pygame.init()
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
    """通过 pygame 字体渲染将字符转为 intensity 列表"""
    import pygame

    render_size = max(grid_size * 4, 64)
    font = _get_font(render_size)
    surf = font.render(char, True, (255, 255, 255))
    sw, sh = surf.get_size()

    # 裁剪到非透明区域
    bbox = surf.get_bounding_rect()
    if bbox.width == 0 or bbox.height == 0:
        return [0] * (grid_size * grid_size)

    cropped = surf.subsurface(bbox).copy()
    cw, ch = cropped.get_size()

    # 缩放到 grid_size x grid_size
    scaled = pygame.transform.smoothscale(cropped, (grid_size, grid_size))
    pixels = pygame.surfarray.array3d(scaled)

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

    Args:
        char: 单个字符 (可打印 ASCII)
        grid_size: 目标网格大小

    Returns:
        intensity 列表，若 grid_size 不匹配则尝试缩放
    """
    char = char.upper() if len(char) == 1 else char

    # 优先使用预定义模板
    if grid_size == 4 and char in CHAR_4x4:
        return [v * 255 for v in CHAR_4x4[char]]
    if grid_size == 8 and char in CHAR_8x8:
        return CHAR_8x8[char][:]

    # 回退：尝试从 8×8 缩放（仅限有模板的字符）
    if char in CHAR_8x8:
        base = CHAR_8x8[char]
        if grid_size in (16, 32, 64):
            return _scale_up(base, 8, grid_size)

    # 最终回退：用字体渲染生成
    if len(char) == 1 and 32 <= ord(char) <= 126:
        return _render_char_to_intensity(char, grid_size)

    return [0] * (grid_size * grid_size)


def _scale_up(intensity: List[int], src_gs: int, dst_gs: int) -> List[int]:
    """最近邻放大"""
    result = [0] * (dst_gs * dst_gs)
    ratio = src_gs / dst_gs
    for y in range(dst_gs):
        for x in range(dst_gs):
            sx = int(x * ratio)
            sy = int(y * ratio)
            result[y * dst_gs + x] = intensity[sy * src_gs + sx]
    return result


def list_chars() -> List[str]:
    """返回所有可用字符列表（可打印 ASCII: 空格到~）"""
    return [chr(i) for i in range(32, 127)]
