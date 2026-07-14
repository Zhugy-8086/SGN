#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gui/canvas.py — 绘制画布（DrawCanvas）"""
from __future__ import annotations

from typing import List, Tuple, Optional
import pygame
from pygame import Rect, Surface

from gui.theme import THEME


class DrawCanvas:
    """网格绘制画布：管理网格状态，处理鼠标绘制事件"""

    def __init__(self, x: int, y: int, pixel_size: int, grid_size: int = 16):
        self.rect = Rect(x, y, pixel_size * grid_size, pixel_size * grid_size)
        self.pixel_size = pixel_size
        self.grid_size = grid_size
        self.data: List[List[int]] = [[0] * grid_size for _ in range(grid_size)]
        self.drawing = False
        self.erasing = False
        self.last_cell: Optional[Tuple[int, int]] = None
        self.eraser_mode = False

    def resize(self, grid_size: int):
        """调整网格大小，尽量保留已有内容"""
        old_data = self.data
        old_size = self.grid_size
        self.grid_size = grid_size
        self.data = [[0] * grid_size for _ in range(grid_size)]
        # 缩放保留内容
        for y in range(grid_size):
            for x in range(grid_size):
                ox = int(x * old_size / grid_size)
                oy = int(y * old_size / grid_size)
                if 0 <= ox < old_size and 0 <= oy < old_size:
                    self.data[y][x] = old_data[oy][ox]
        self.rect.width = self.pixel_size * grid_size
        self.rect.height = self.pixel_size * grid_size
        self.last_cell = None

    def clear(self):
        """清空画布"""
        for row in self.data:
            for i in range(len(row)):
                row[i] = 0

    def fill_test_pattern(self, pattern: str = "cross"):
        """填充测试图案"""
        self.clear()
        gs = self.grid_size
        if pattern == "cross":
            for i in range(gs):
                self.data[gs // 2][i] = 255
                self.data[i][gs // 2] = 255
        elif pattern == "circle":
            cx, cy = gs // 2, gs // 2
            r = gs // 3
            for y in range(gs):
                for x in range(gs):
                    if abs((x - cx) ** 2 + (y - cy) ** 2 - r ** 2) < gs:
                        self.data[y][x] = 255
        elif pattern == "diagonal":
            for i in range(gs):
                self.data[i][i] = 255
                self.data[i][gs - 1 - i] = 255
        elif pattern == "border":
            for i in range(gs):
                self.data[0][i] = 255
                self.data[gs - 1][i] = 255
                self.data[i][0] = 255
                self.data[i][gs - 1] = 255

    def _cell_from_pos(self, mx: int, my: int) -> Optional[Tuple[int, int]]:
        """将鼠标坐标转换为网格坐标"""
        if not self.rect.collidepoint(mx, my):
            return None
        x = (mx - self.rect.x) // self.pixel_size
        y = (my - self.rect.y) // self.pixel_size
        x = max(0, min(self.grid_size - 1, x))
        y = max(0, min(self.grid_size - 1, y))
        return (x, y)

    def _line_cells(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham 线段"""
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return cells

    def handle_event(self, event: pygame.event.Event):
        """处理 pygame 鼠标事件"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # 左键
                self.drawing = True
                self.erasing = self.eraser_mode
                cell = self._cell_from_pos(*event.pos)
                if cell:
                    self.last_cell = cell
                    self.data[cell[1]][cell[0]] = 0 if self.eraser_mode else 255
            elif event.button == 3:  # 右键擦除
                self.drawing = True
                self.erasing = True
                cell = self._cell_from_pos(*event.pos)
                if cell:
                    self.last_cell = cell
                    self.data[cell[1]][cell[0]] = 0

        elif event.type == pygame.MOUSEMOTION and self.drawing:
            cell = self._cell_from_pos(*event.pos)
            if cell and self.last_cell:
                for cx, cy in self._line_cells(self.last_cell[0], self.last_cell[1], cell[0], cell[1]):
                    if self.erasing:
                        self.data[cy][cx] = 0
                    else:
                        self.data[cy][cx] = 255
                self.last_cell = cell

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button in (1, 3):
                self.drawing = False
                self.erasing = False
                self.last_cell = None

    def draw(self, screen: Surface):
        """绘制画布到 screen"""
        pygame.draw.rect(screen, THEME["grid_bg"], self.rect)
        for i in range(self.grid_size + 1):
            x = self.rect.x + i * self.pixel_size
            pygame.draw.line(screen, THEME["grid_line"], (x, self.rect.y), (x, self.rect.bottom))
            y = self.rect.y + i * self.pixel_size
            pygame.draw.line(screen, THEME["grid_line"], (self.rect.x, y), (self.rect.right, y))
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                v = self.data[y][x]
                if v > 0:
                    px = self.rect.x + x * self.pixel_size
                    py = self.rect.y + y * self.pixel_size
                    color = pygame.Color(v, v, v)
                    pygame.draw.rect(screen, color, (px, py, self.pixel_size, self.pixel_size))
        pygame.draw.rect(screen, THEME["panel_border"], self.rect, 2)

    def to_intensity(self) -> List[int]:
        """导出为 SGN 格式的 intensity 列表（行优先）"""
        intensity = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                intensity.append(self.data[y][x])
        return intensity

    def from_intensity(self, intensity: List[int]):
        """从 intensity 列表导入"""
        d = len(intensity)
        gs = int(d ** 0.5)
        if gs * gs != d:
            return
        self.resize(gs)
        for y in range(gs):
            for x in range(gs):
                self.data[y][x] = intensity[y * gs + x]

    def from_image(self, path: str, grid_size: int):
        """从图片文件加载到画布"""
        from gui.utils import load_image_to_intensity
        intensity = load_image_to_intensity(path, grid_size)
        self.from_intensity(intensity)
