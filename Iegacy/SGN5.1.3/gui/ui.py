#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gui/ui.py — UI 组件（Button、Slider、Label、TextBox）"""
from __future__ import annotations

from typing import Optional, Callable, List
import pygame
from pygame import Rect, Surface, Color

from gui.theme import THEME


class Button:
    """现代风格按钮"""

    def __init__(self, rect: Rect, text: str, font: pygame.font.Font,
                 callback: Optional[Callable] = None,
                 color_key: str = "button_bg",
                 hover_key: str = "button_hover",
                 active_key: str = "button_active"):
        self.rect = rect
        self.text = text
        self.font = font
        self.callback = callback
        self.color_key = color_key
        self.hover_key = hover_key
        self.active_key = active_key
        self.hovered = False
        self.pressed = False
        self._visible = True

    def show(self, visible: bool = True):
        self._visible = visible

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self._visible:
            return False
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                self.pressed = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.pressed and self.rect.collidepoint(event.pos):
                self.pressed = False
                if self.callback:
                    self.callback()
                return True
            self.pressed = False
        return False

    def draw(self, screen: Surface):
        if not self._visible:
            return
        r = self.rect

        if self.pressed:
            bg = THEME[self.active_key]
        elif self.hovered:
            bg = THEME[self.hover_key]
        else:
            bg = THEME[self.color_key]

        # 阴影
        shadow = Color(max(0, bg.r - 15), max(0, bg.g - 15), max(0, bg.b - 15))
        pygame.draw.rect(screen, shadow, (r.x + 1, r.y + 2, r.width, r.height), border_radius=6)

        # 按钮背景
        pygame.draw.rect(screen, bg, r, border_radius=6)

        # 高光（上半部分微亮）
        if not self.pressed:
            highlight = Color(min(255, bg.r + 10), min(255, bg.g + 10), min(255, bg.b + 10))
            hr = Rect(r.x, r.y, r.width, r.height // 2)
            pygame.draw.rect(screen, highlight, hr, border_radius=6)

        # 边框
        border = THEME.get("button_border", THEME["panel_border"])
        if self.hovered or self.pressed:
            border = THEME.get("accent", border)
        pygame.draw.rect(screen, border, r, 1, border_radius=6)

        # 文字
        text_surf = self.font.render(self.text, True, THEME["text"])
        text_rect = text_surf.get_rect(center=r.center)
        screen.blit(text_surf, text_rect)


class Slider:
    """现代风格滑块"""

    def __init__(self, rect: Rect, label: str, font: pygame.font.Font,
                 min_val: float, max_val: float, default: float,
                 step: float = 1.0, int_only: bool = False,
                 callback: Optional[Callable[[float], None]] = None):
        self.rect = rect
        self.label = label
        self.font = font
        self.min_val = min_val
        self.max_val = max_val
        self.value = default
        self.step = step
        self.int_only = int_only
        self.callback = callback
        self.dragging = False
        self.track_h = 4
        self.thumb_r = 7
        self.label_w = 70
        self.value_w = 50

    def _track_rect(self) -> Rect:
        return Rect(
            self.rect.x + self.label_w,
            self.rect.centery - self.track_h // 2,
            self.rect.width - self.label_w - self.value_w,
            self.track_h
        )

    def _thumb_x(self) -> int:
        track = self._track_rect()
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        return track.x + int(ratio * track.width)

    def _value_from_x(self, x: int) -> float:
        track = self._track_rect()
        ratio = max(0.0, min(1.0, (x - track.x) / track.width))
        raw = self.min_val + ratio * (self.max_val - self.min_val)
        if self.int_only:
            raw = round(raw / self.step) * self.step
            return int(raw)
        return round(raw / self.step) * self.step

    def handle_event(self, event: pygame.event.Event) -> bool:
        track = self._track_rect()
        thumb_x = self._thumb_x()
        thumb_rect = Rect(thumb_x - self.thumb_r - 2, track.centery - self.thumb_r - 2,
                          self.thumb_r * 2 + 4, self.thumb_r * 2 + 4)

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and thumb_rect.collidepoint(event.pos):
                self.dragging = True
                return True
            elif event.button == 1 and track.collidepoint(event.pos):
                self.value = self._value_from_x(event.pos[0])
                if self.callback:
                    self.callback(self.value)
                return True
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.value = self._value_from_x(event.pos[0])
            if self.callback:
                self.callback(self.value)
            return True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        return False

    def draw(self, screen: Surface):
        # 标签
        label_surf = self.font.render(self.label, True, THEME["text_label"])
        screen.blit(label_surf, (self.rect.x, self.rect.y + 3))

        track = self._track_rect()
        thumb_x = self._thumb_x()
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)

        # 轨道背景
        pygame.draw.rect(screen, THEME["slider_track"], track, border_radius=2)

        # 已填充部分
        filled = Rect(track.x, track.y, max(0, thumb_x - track.x), track.height)
        if filled.width > 0:
            pygame.draw.rect(screen, THEME["slider_track_active"], filled, border_radius=2)

        # 滑块阴影
        shadow = Color(max(0, THEME["slider_thumb"].r - 30),
                       max(0, THEME["slider_thumb"].g - 30),
                       max(0, THEME["slider_thumb"].b - 30))
        pygame.draw.circle(screen, shadow, (thumb_x + 1, track.centery + 1), self.thumb_r)

        # 滑块
        thumb_color = THEME["slider_thumb"] if not self.dragging else THEME.get("slider_thumb_hover", THEME["slider_thumb"])
        pygame.draw.circle(screen, thumb_color, (thumb_x, track.centery), self.thumb_r)
        pygame.draw.circle(screen, THEME["accent"], (thumb_x, track.centery), self.thumb_r, 1)

        # 数值
        val_str = str(int(self.value)) if self.int_only else f"{self.value:.1f}"
        val_surf = self.font.render(val_str, True, THEME["text"])
        screen.blit(val_surf, (self.rect.right - self.value_w, self.rect.y + 3))


class Label:
    """文本标签（支持多行和自动换行）"""

    def __init__(self, rect: Rect, text: str, font: pygame.font.Font, color_key: str = "text"):
        self.rect = rect
        self.text = text
        self.font = font
        self.color_key = color_key

    def set_text(self, text: str):
        self.text = text

    def draw(self, screen: Surface):
        lines = self.text.split("\n")
        y = self.rect.y
        for line in lines:
            if y >= self.rect.bottom:
                break
            surf = self.font.render(line, True, THEME[self.color_key])
            screen.blit(surf, (self.rect.x, y))
            y += surf.get_height() + 3


class TextBox:
    """现代风格文本输入框"""

    def __init__(self, rect: Rect, default: str, font: pygame.font.Font,
                 callback: Optional[Callable[[str], None]] = None):
        self.rect = rect
        self.text = default
        self.font = font
        self.callback = callback
        self.active = False
        self.cursor_pos = len(default)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                self.active = True
                return True
            else:
                self.active = False
        if self.active and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self.active = False
                if self.callback:
                    self.callback(self.text)
                return True
            elif event.key == pygame.K_BACKSPACE:
                if self.cursor_pos > 0:
                    self.text = self.text[:self.cursor_pos - 1] + self.text[self.cursor_pos:]
                    self.cursor_pos -= 1
                return True
            elif event.unicode.isprintable():
                self.text = self.text[:self.cursor_pos] + event.unicode + self.text[self.cursor_pos:]
                self.cursor_pos += 1
                return True
        return False

    def draw(self, screen: Surface):
        r = self.rect

        # 背景
        pygame.draw.rect(screen, THEME.get("input_bg", THEME["grid_bg"]), r, border_radius=4)

        # 边框
        border_color = THEME.get("input_active", THEME["accent"]) if self.active else THEME.get("input_border", THEME["panel_border"])
        border_w = 2 if self.active else 1
        pygame.draw.rect(screen, border_color, r, border_w, border_radius=4)

        # 文字
        text_surf = self.font.render(self.text, True, THEME["text"])
        text_y = r.centery - text_surf.get_height() // 2
        screen.blit(text_surf, (r.x + 8, text_y))

        # 光标
        if self.active:
            cursor_x = r.x + 8 + text_surf.get_width() + 1
            cursor_h = text_surf.get_height() + 2
            cursor_rect = Rect(cursor_x, r.centery - cursor_h // 2, 2, cursor_h)
            pygame.draw.rect(screen, THEME["accent"], cursor_rect)
