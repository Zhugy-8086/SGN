#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gui/theme.py — 主题、常量、字体定义

不依赖任何其他 GUI 子模块，可被任意子模块安全导入。
"""
from __future__ import annotations

import os
import pygame
from pygame import Color

pygame.init()

# ============================================================
# 窗口自适应（最小 8×8，支持 64×64 等超大网格）
# ============================================================
MIN_W, MIN_H = 1024, 700
DEFAULT_W, DEFAULT_H = 1400, 900

# 根据屏幕分辨率自动调整默认窗口大小
try:
    info = pygame.display.Info()
    SCREEN_W, SCREEN_H = info.current_w, info.current_h
    if SCREEN_W > 0 and SCREEN_H > 0:
        DEFAULT_W = min(DEFAULT_W, int(SCREEN_W * 0.9))
        DEFAULT_H = min(DEFAULT_H, int(SCREEN_H * 0.9))
except Exception:
    SCREEN_W, SCREEN_H = 1920, 1080

FPS = 60

# 深色主题（现代风格）
THEME = {
    # 背景层级
    "bg": Color(22, 22, 28),
    "panel": Color(30, 30, 38),
    "panel_alt": Color(36, 36, 46),
    "panel_border": Color(50, 50, 64),

    # 文字
    "text": Color(220, 220, 230),
    "text_dim": Color(130, 130, 150),
    "text_label": Color(160, 160, 180),

    # 强调色
    "accent": Color(56, 132, 244),
    "accent_hover": Color(76, 152, 255),
    "accent_active": Color(40, 110, 210),

    # 状态色
    "success": Color(60, 185, 90),
    "success_hover": Color(80, 200, 110),
    "warning": Color(240, 190, 50),
    "danger": Color(230, 75, 60),
    "danger_hover": Color(240, 95, 80),

    # 网格
    "grid_line": Color(42, 42, 54),
    "grid_bg": Color(16, 16, 22),
    "pixel_on": Color(255, 255, 255),
    "pixel_off": Color(0, 0, 0),

    # 按钮
    "button_bg": Color(44, 44, 58),
    "button_hover": Color(58, 58, 76),
    "button_active": Color(68, 68, 88),
    "button_border": Color(60, 60, 78),

    # 滑块
    "slider_track": Color(50, 50, 66),
    "slider_track_active": Color(56, 132, 244),
    "slider_thumb": Color(220, 220, 230),
    "slider_thumb_hover": Color(240, 240, 250),

    # 输入框
    "input_bg": Color(24, 24, 32),
    "input_border": Color(50, 50, 64),
    "input_active": Color(56, 132, 244),

    # 分隔线
    "separator": Color(44, 44, 58),
}

# 可选网格大小列表（v5.1.5：移除 4×4，最小网格提升到 8×8）
GRID_SIZES = [8, 16, 32, 64]


def load_font(size: int = 14) -> pygame.font.Font:
    """加载支持中文的字体，回退到默认字体"""
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return pygame.font.Font(path, size)
            except Exception:
                pass
    return pygame.font.SysFont(None, size)
