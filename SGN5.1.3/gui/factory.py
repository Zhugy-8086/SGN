#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gui/factory.py — 函数工厂主窗口（GUIFactory）

包含：窗口自适应布局、画布、变换、训练、测试、图片加载、训练集导出、字符库。
"""
from __future__ import annotations

import sys
import os
import random
import tkinter as tk
from tkinter import filedialog
from typing import List, Tuple, Optional, Dict

import pygame
from pygame import Rect, Surface, Color

from gui.theme import THEME, GRID_SIZES, MIN_W, MIN_H, DEFAULT_W, DEFAULT_H, FPS, load_font
from gui.utils import (
    intensity_to_surface, apply_transform, contrast_stretch,
    load_image_to_intensity, export_training_dataset
)
from gui.canvas import DrawCanvas
from gui.transform import TransformEngine
from gui.ui import Button, Slider, Label, TextBox
from gui.bridge import SGNBridge
from gui.charlib import get_char, list_chars
from gui.dataset_store import CustomDatasetStore
from gui.custom_input_source import CustomDatasetInputSource


class GUIFactory:
    """函数工厂主窗口 — 支持窗口大小自适应、64×64 网格、图片加载、训练集导出、多标签训练"""

    def __init__(self, core=None):
        # 窗口初始化（支持自适应大小）
        self.screen = pygame.display.set_mode((DEFAULT_W, DEFAULT_H), pygame.RESIZABLE)
        pygame.display.set_caption("SGN-Lite v5.1 函数工厂")
        self.clock = pygame.time.Clock()
        self.font = load_font(16)
        self.font_small = load_font(13)
        self.font_large = load_font(20)

        self.window_w, self.window_h = DEFAULT_W, DEFAULT_H
        self.margin = 20
        self.left_ratio = 0.45  # 左面板占窗口宽度的比例

        # 画布（默认 16×16）
        self.canvas: Optional[DrawCanvas] = None
        self._recalc_layout()
        self.canvas = self._make_canvas(16)
        self.canvas.fill_test_pattern("cross")

        # 撤销/重做历史
        self._undo_stack: List[List[int]] = []
        self._redo_stack: List[List[int]] = []
        self._max_undo = 50
        self._save_undo_snapshot()

        # 变换引擎
        self.transform = TransformEngine()
        self.preview_intensity: Optional[List[int]] = None
        self.preview_grid_size = 16

        # SGN 桥接
        self.bridge = SGNBridge(core=core)

        # 持久 tkinter 根窗口（用于文件对话框）
        self._tk_root = tk.Tk()
        self._tk_root.withdraw()

        # 字符库覆盖层状态
        self._charlib_overlay_active = False
        self._charlib_page = 0
        self._charlib_cols = 8
        self._charlib_rows = 5
        self._charlib_chars_per_page = self._charlib_cols * self._charlib_rows
        self._charlib_all_chars = list_chars()

        # 训练集管理（多标签共存，不覆盖）
        self.dataset_store = CustomDatasetStore()
        # 尝试自动加载已有的训练集
        if self.dataset_store.load():
            print(f"[函数工厂] 已加载训练集: {self.dataset_store.count()} 条")

        # 训练状态
        self.label = "A"
        self.train_steps = 100
        self.is_training = False
        self.train_target = 0
        self.train_speed = 5
        self.last_train_info: Optional[Dict] = None
        self.last_test_result: Optional[Tuple[str, int]] = None
        self.train_history: List[Tuple[List[int], str]] = []
        self.multi_label_mode = False  # True=循环所有标签训练

        # 组件
        self.ui_components: List = []
        self._build_ui()

    # ============================================================
    # 布局自适应
    # ============================================================

    def _recalc_layout(self):
        """根据当前窗口大小重新计算布局参数"""
        self.window_w, self.window_h = self.screen.get_size()
        self.left_w = int(self.window_w * self.left_ratio)
        self.right_x = self.left_w + self.margin
        self.right_w = self.window_w - self.right_x - self.margin

    def _make_canvas(self, grid_size: int) -> DrawCanvas:
        """根据左面板大小创建合适尺寸的画布"""
        avail_w = self.left_w - self.margin * 2
        avail_h = self.window_h - 200
        pixel_size = max(4, min(32, avail_w // grid_size, avail_h // grid_size))
        canvas_w = pixel_size * grid_size
        canvas_x = self.margin + (avail_w - canvas_w) // 2
        canvas_y = 100
        c = DrawCanvas(canvas_x, canvas_y, pixel_size, grid_size)
        c.eraser_mode = getattr(self.canvas, 'eraser_mode', False) if self.canvas else False
        return c

    # ============================================================
    # UI 构建
    # ============================================================

    def _build_ui(self):
        self.ui_components.clear()
        m = self.margin
        rx = self.right_x
        rw = self.right_w

        # ---- 左面板工具栏（两行） ----
        btn_w, btn_h = 46, 26
        gap = 5
        row1_x = m
        row1_y = 56
        row2_y = row1_y + btn_h + 6

        # 第一行：网格大小 + 绘图工具
        self.grid_buttons = []
        x = row1_x
        for i, gs in enumerate(GRID_SIZES):
            btn = Button(Rect(x, row1_y, btn_w, btn_h), f"{gs}", self.font_small,
                         callback=lambda _gs=gs: self._set_grid_size(_gs))
            self.grid_buttons.append(btn)
            self.ui_components.append(btn)
            x += btn_w + gap

        x += 6
        self.btn_clear = Button(Rect(x, row1_y, 40, btn_h), "清空", self.font_small,
                                callback=self._clear_canvas)
        self.ui_components.append(self.btn_clear)
        x += 40 + gap

        self.btn_fill_cross = Button(Rect(x, row1_y, 40, btn_h), "十字", self.font_small,
                                     callback=lambda: (self._push_undo(), self.canvas.fill_test_pattern("cross")))
        self.ui_components.append(self.btn_fill_cross)
        x += 40 + gap

        self.btn_fill_circle = Button(Rect(x, row1_y, 40, btn_h), "圆形", self.font_small,
                                      callback=lambda: (self._push_undo(), self.canvas.fill_test_pattern("circle")))
        self.ui_components.append(self.btn_fill_circle)
        x += 40 + gap

        self.btn_eraser = Button(Rect(x, row1_y, 50, btn_h), "橡皮擦", self.font_small,
                                 callback=self._toggle_eraser, color_key="warning")
        self.ui_components.append(self.btn_eraser)

        # 第二行：字符库、图片、撤销重做、图案保存加载
        x = row1_x
        self.btn_charlib = Button(Rect(x, row2_y, 56, btn_h), "字符库", self.font_small,
                                  callback=self._show_charlib_dialog)
        self.ui_components.append(self.btn_charlib)
        x += 56 + gap

        self.btn_load_img = Button(Rect(x, row2_y, 56, btn_h), "加载图片", self.font_small,
                                   callback=self._load_image)
        self.ui_components.append(self.btn_load_img)
        x += 56 + gap

        self.btn_undo = Button(Rect(x, row2_y, 40, btn_h), "撤销", self.font_small,
                               callback=self._undo)
        self.ui_components.append(self.btn_undo)
        x += 40 + gap

        self.btn_redo = Button(Rect(x, row2_y, 40, btn_h), "重做", self.font_small,
                               callback=self._redo)
        self.ui_components.append(self.btn_redo)
        x += 40 + gap

        self.btn_save_img = Button(Rect(x, row2_y, 56, btn_h), "保存图片", self.font_small,
                                   callback=self._save_image, color_key="success")
        self.ui_components.append(self.btn_save_img)
        x += 56 + gap

        self.btn_save_pattern = Button(Rect(x, row2_y, 56, btn_h), "保存图案", self.font_small,
                                       callback=self._save_pattern, color_key="success")
        self.ui_components.append(self.btn_save_pattern)
        x += 56 + gap

        self.btn_load_pattern = Button(Rect(x, row2_y, 56, btn_h), "加载图案", self.font_small,
                                       callback=self._load_pattern)
        self.ui_components.append(self.btn_load_pattern)

        # 右面板标题
        self.title_label = Label(Rect(rx, m, 400, 30), "函数工厂控制面板", self.font_large)

        # 变换滑块
        y = m + 40
        sh = 28
        self.sl_angle = Slider(
            Rect(rx, y, rw, sh), "旋转", self.font, -180, 180, 0, step=5, int_only=True,
            callback=lambda v: self._update_transform()
        )
        self.ui_components.append(self.sl_angle)
        y += sh + 6
        self.sl_offset_x = Slider(
            Rect(rx, y, rw, sh), "X偏移", self.font, -8, 8, 0, step=1, int_only=True,
            callback=lambda v: self._update_transform()
        )
        self.ui_components.append(self.sl_offset_x)
        y += sh + 6
        self.sl_offset_y = Slider(
            Rect(rx, y, rw, sh), "Y偏移", self.font, -8, 8, 0, step=1, int_only=True,
            callback=lambda v: self._update_transform()
        )
        self.ui_components.append(self.sl_offset_y)
        y += sh + 6
        self.sl_scale = Slider(
            Rect(rx, y, rw, sh), "缩放", self.font, 0.5, 1.5, 1.0, step=0.1, int_only=False,
            callback=lambda v: self._update_transform()
        )
        self.ui_components.append(self.sl_scale)

        # 变换按钮
        y += sh + 10
        bw = 80
        self.btn_random = Button(
            Rect(rx, y, bw, 30), "随机变换", self.font_small, callback=self._random_transform,
            color_key="accent", hover_key="accent_hover", active_key="accent_active"
        )
        self.ui_components.append(self.btn_random)
        self.btn_reset_transform = Button(
            Rect(rx + bw + 8, y, bw, 30), "重置变换", self.font_small, callback=self._reset_transform
        )
        self.ui_components.append(self.btn_reset_transform)
        # 生成 N 个变体按钮
        self.btn_gen_variants = Button(
            Rect(rx + bw * 2 + 16, y, 100, 30), "生成100变体", self.font_small,
            callback=self._generate_variant_dataset, color_key="success"
        )
        self.ui_components.append(self.btn_gen_variants)

        # 预览区
        y += 40
        preview_size = min(240, rw - 20)
        self.preview_rect = Rect(rx, y, preview_size, preview_size)
        self.preview_label = Label(Rect(rx, y - 22, 200, 20), "变换预览", self.font)

        # 训练控制区
        y += self.preview_rect.height + 30
        self.label_label = Label(Rect(rx, y, 200, 20), "训练标签:", self.font)
        # 标签按钮（两行 8 个）
        self.label_buttons = []
        labels = list("0123456789ABCDEF")
        btn_size = min(40, (rw - 20) // 8)
        for i, lb in enumerate(labels):
            bx = rx + (i % 8) * (btn_size + 4)
            by = y + 22 + (i // 8) * (btn_size + 4)
            btn = Button(
                Rect(bx, by, btn_size, btn_size), lb, self.font,
                callback=lambda _lb=lb: self._set_label(_lb)
            )
            self.label_buttons.append(btn)
            self.ui_components.append(btn)

        # 自定义标签输入框
        y += 2 * (btn_size + 4) + 10
        self.label_input = TextBox(
            Rect(rx, y, 120, 28), self.label, self.font_small,
            callback=lambda text: self._set_label(text.upper())
        )
        self.ui_components.append(self.label_input)

        y += 40
        self.sl_train_steps = Slider(
            Rect(rx, y, rw, sh), "训练步数", self.font, 10, 5000, 100, step=10, int_only=True
        )
        self.ui_components.append(self.sl_train_steps)
        y += sh + 10
        self.btn_train = Button(
            Rect(rx, y, 90, 34), "开始训练", self.font, callback=self._toggle_training,
            color_key="success", hover_key="accent_hover", active_key="accent_active"
        )
        self.ui_components.append(self.btn_train)
        self.btn_test = Button(
            Rect(rx + 100, y, 90, 34), "实时测试", self.font, callback=self._test_current,
            color_key="accent", hover_key="accent_hover", active_key="accent_active"
        )
        self.ui_components.append(self.btn_test)
        self.btn_reset_net = Button(
            Rect(rx + 200, y, 90, 34), "重置网络", self.font, callback=self._reset_network
        )
        self.ui_components.append(self.btn_reset_net)

        # 导出按钮
        y += 44
        self.btn_export_csv = Button(
            Rect(rx, y, 100, 30), "导出CSV", self.font_small,
            callback=lambda: self._export_dataset("csv")
        )
        self.ui_components.append(self.btn_export_csv)
        self.btn_export_json = Button(
            Rect(rx + 108, y, 100, 30), "导出JSON", self.font_small,
            callback=lambda: self._export_dataset("json")
        )
        self.ui_components.append(self.btn_export_json)

        # 模型保存/加载
        y += 38
        self.btn_model_save = Button(
            Rect(rx, y, 100, 30), "保存模型", self.font_small,
            callback=self._save_model, color_key="success"
        )
        self.ui_components.append(self.btn_model_save)
        self.btn_model_load = Button(
            Rect(rx + 108, y, 100, 30), "加载模型", self.font_small,
            callback=self._load_model
        )
        self.ui_components.append(self.btn_model_load)

        # 训练集管理区
        y += 38
        self.dataset_label = Label(Rect(rx, y, rw, 22), "训练集管理", self.font)

        y += 22
        self.btn_add_to_dataset = Button(
            Rect(rx, y, 100, 28), "添加到训练集", self.font_small,
            callback=self._add_to_dataset, color_key="success"
        )
        self.ui_components.append(self.btn_add_to_dataset)
        self.btn_save_dataset = Button(
            Rect(rx + 108, y, 60, 28), "保存", self.font_small,
            callback=self._save_dataset
        )
        self.ui_components.append(self.btn_save_dataset)
        self.btn_load_dataset = Button(
            Rect(rx + 174, y, 60, 28), "加载", self.font_small,
            callback=self._load_dataset
        )
        self.ui_components.append(self.btn_load_dataset)
        self.btn_clear_dataset = Button(
            Rect(rx + 240, y, 60, 28), "清空", self.font_small,
            callback=self._clear_dataset_entries
        )
        self.ui_components.append(self.btn_clear_dataset)

        y += 34
        self.btn_train_multi = Button(
            Rect(rx, y, 140, 30), "多标签训练", self.font_small,
            callback=self._toggle_multi_label_training,
            color_key="accent", hover_key="accent_hover", active_key="accent_active"
        )
        self.ui_components.append(self.btn_train_multi)
        self.btn_train_multi_stop = Button(
            Rect(rx + 148, y, 100, 30), "停止多标签", self.font_small,
            callback=self._stop_multi_label_training, color_key="danger"
        )
        self.ui_components.append(self.btn_train_multi_stop)
        self.btn_train_multi_stop.show(False)  # 初始隐藏

        # 训练集列表显示区
        y += 36
        self.dataset_list_rect = Rect(rx, y, rw, 80)
        self.dataset_list_label = Label(self.dataset_list_rect, "训练集: 空", self.font_small)
        self._update_dataset_list()

        # 状态日志区
        y += 90
        self.status_rect = Rect(rx, y, rw, self.window_h - y - m - 10)
        self.status_label = Label(self.status_rect, "就绪\n绘制图形后添加到训练集，或多标签训练", self.font_small)

        # 默认高亮当前标签和网格大小
        self._set_label(self.label)
        self._update_grid_highlight()
        self._update_transform()

    # ============================================================
    # 功能方法
    # ============================================================

    def _set_grid_size(self, gs: int):
        self._push_undo()
        old_intensity = self.canvas.to_intensity()
        old_gs = self.canvas.grid_size
        old_eraser = self.canvas.eraser_mode
        self.canvas = self._make_canvas(gs)
        self.canvas.eraser_mode = old_eraser
        # 手动缩放旧数据到新网格大小
        if old_gs != gs and old_intensity:
            scaled = [0] * (gs * gs)
            for y in range(gs):
                for x in range(gs):
                    ox = int(x * old_gs / gs)
                    oy = int(y * old_gs / gs)
                    scaled[y * gs + x] = old_intensity[oy * old_gs + ox]
            for y in range(gs):
                for x in range(gs):
                    self.canvas.data[y][x] = scaled[y * gs + x]
        else:
            for y in range(gs):
                for x in range(gs):
                    self.canvas.data[y][x] = old_intensity[y * gs + x]
        self.preview_grid_size = gs
        max_off = gs // 3
        self.sl_offset_x.min_val = -max_off
        self.sl_offset_x.max_val = max_off
        self.sl_offset_y.min_val = -max_off
        self.sl_offset_y.max_val = max_off
        self._update_grid_highlight()
        self._update_transform()
        self._update_status(f"网格已切换为 {gs}×{gs}")

    def _update_grid_highlight(self):
        gs = self.canvas.grid_size
        for i, btn in enumerate(self.grid_buttons):
            if GRID_SIZES[i] == gs:
                btn.color_key = "accent"
                btn.hover_key = "accent_hover"
                btn.active_key = "accent_active"
            else:
                btn.color_key = "button_bg"
                btn.hover_key = "button_hover"
                btn.active_key = "button_active"

    def _clear_canvas(self):
        self._push_undo()
        self.canvas.clear()
        self._update_transform()

    def _update_transform(self):
        self.transform.angle = self.sl_angle.value
        self.transform.offset_x = int(self.sl_offset_x.value)
        self.transform.offset_y = int(self.sl_offset_y.value)
        self.transform.scale = self.sl_scale.value
        self.preview_intensity = self.transform.apply(self.canvas.to_intensity(), self.canvas.grid_size)

    def _random_transform(self):
        self.sl_angle.value = random.randint(-45, 45)
        max_off = self.canvas.grid_size // 3
        self.sl_offset_x.value = random.randint(-max_off, max_off)
        self.sl_offset_y.value = random.randint(-max_off, max_off)
        self.sl_scale.value = round(random.uniform(0.6, 1.4), 1)
        self._update_transform()

    def _reset_transform(self):
        self.sl_angle.value = 0
        self.sl_offset_x.value = 0
        self.sl_offset_y.value = 0
        self.sl_scale.value = 1.0
        self._update_transform()

    def _set_label(self, lb: str):
        self.label = lb[:1] if lb else "A"
        self.label_input.text = self.label
        for btn in self.label_buttons:
            if btn.text == self.label:
                btn.color_key = "accent"
                btn.hover_key = "accent_hover"
                btn.active_key = "accent_active"
            else:
                btn.color_key = "button_bg"
                btn.hover_key = "button_hover"
                btn.active_key = "button_active"

    def _show_charlib_dialog(self):
        """切换字符库覆盖层显示"""
        self._charlib_overlay_active = not self._charlib_overlay_active
        self._charlib_page = 0

    def _load_char(self, char: str):
        """加载字符到画布并关闭覆盖层"""
        gs = self.canvas.grid_size
        intensity = get_char(char, gs)
        self.canvas.from_intensity(intensity)
        self._update_transform()
        if len(char) == 1 and char.isalnum():
            self.label = char.upper()
        else:
            self.label = char
        self._set_label(self.label)
        self._charlib_overlay_active = False
        self._update_status(f"已加载字符: {char}")
        self._push_undo()

    # ============================================================
    # 撤销 / 重做
    # ============================================================

    def _save_undo_snapshot(self):
        """保存当前画布状态到撤销栈"""
        snapshot = [row[:] for row in self.canvas.data]
        flat = [v for row in snapshot for v in row]
        if self._undo_stack and self._undo_stack[-1] == flat:
            return
        self._undo_stack.append(flat)
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _push_undo(self):
        """在操作后推入撤销快照"""
        self._save_undo_snapshot()

    def _undo(self):
        """撤销上一步"""
        if len(self._undo_stack) <= 1:
            self._update_status("没有可撤销的操作")
            return
        current = self._undo_stack.pop()
        self._redo_stack.append(current)
        prev = self._undo_stack[-1]
        gs = self.canvas.grid_size
        for y in range(gs):
            for x in range(gs):
                self.canvas.data[y][x] = prev[y * gs + x]
        self._update_transform()
        self._update_status(f"已撤销 (可重做 {len(self._redo_stack)} 步)")

    def _redo(self):
        """重做"""
        if not self._redo_stack:
            self._update_status("没有可重做的操作")
            return
        state = self._redo_stack.pop()
        self._undo_stack.append(state)
        gs = self.canvas.grid_size
        for y in range(gs):
            for x in range(gs):
                self.canvas.data[y][x] = state[y * gs + x]
        self._update_transform()
        self._update_status(f"已重做 (可撤销 {len(self._undo_stack) - 1} 步)")

    def _save_image(self):
        """将当前画布保存为 PNG 图片"""
        path = filedialog.asksaveasfilename(
            parent=self._tk_root,
            title="保存画布为图片",
            defaultextension=".png",
            initialfile=f"canvas_{self.label}_{self.canvas.grid_size}x{self.canvas.grid_size}.png",
            filetypes=[("PNG 图片", "*.png"), ("所有文件", "*.*")]
        )
        if not path:
            self._update_status("保存已取消")
            return
        try:
            gs = self.canvas.grid_size
            px = max(4, min(32, 512 // gs))
            surf = pygame.Surface((gs * px, gs * px))
            surf.fill((0, 0, 0))
            for y in range(gs):
                for x in range(gs):
                    v = self.canvas.data[y][x]
                    if v > 0:
                        color = pygame.Color(v, v, v)
                        pygame.draw.rect(surf, color, (x * px, y * px, px, px))
            pygame.image.save(surf, path)
            self._update_status(f"画布已保存为图片:\n{path}\n网格: {gs}×{gs}  像素: {px}px")
        except Exception as e:
            self._update_status(f"保存图片失败: {e}")

    # ============================================================
    # 橡皮擦工具
    # ============================================================

    def _toggle_eraser(self):
        """切换橡皮擦模式"""
        self.canvas.eraser_mode = not self.canvas.eraser_mode
        if self.canvas.eraser_mode:
            self.btn_eraser.color_key = "accent"
            self.btn_eraser.hover_key = "accent_hover"
            self.btn_eraser.active_key = "accent_active"
            self._update_status("橡皮擦模式: ON（左键擦除）")
        else:
            self.btn_eraser.color_key = "warning"
            self.btn_eraser.hover_key = "warning"
            self.btn_eraser.active_key = "warning"
            self._update_status("橡皮擦模式: OFF（左键绘制）")

    # ============================================================
    # 图案保存 / 加载（JSON 文件）
    # ============================================================

    def _save_pattern(self):
        """将当前画布保存为图案文件"""
        path = filedialog.asksaveasfilename(
            parent=self._tk_root,
            title="保存图案",
            defaultextension=".json",
            initialfile=f"pattern_{self.label}_{self.canvas.grid_size}x{self.canvas.grid_size}.json",
            filetypes=[("JSON 图案", "*.json"), ("所有文件", "*.*")]
        )
        if not path:
            return
        try:
            import json
            data = {
                "name": self.label,
                "grid_size": self.canvas.grid_size,
                "intensity": self.canvas.to_intensity(),
                "description": f"标签={self.label} 网格={self.canvas.grid_size}x{self.canvas.grid_size}",
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._update_status(f"图案已保存:\n{path}\n标签: {self.label}  网格: {self.canvas.grid_size}×{self.canvas.grid_size}")
        except Exception as e:
            self._update_status(f"保存图案失败: {e}")

    def _load_pattern(self):
        """从图案文件加载到画布"""
        path = filedialog.askopenfilename(
            parent=self._tk_root,
            title="加载图案",
            filetypes=[("JSON 图案", "*.json"), ("所有文件", "*.*")]
        )
        if not path or not os.path.exists(path):
            return
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            intensity = data.get("intensity", [])
            name = data.get("name", "?")
            gs = data.get("grid_size")
            if not intensity or not gs:
                self._update_status("图案文件格式错误")
                return
            self._push_undo()
            self.canvas = self._make_canvas(gs)
            for y in range(gs):
                for x in range(gs):
                    self.canvas.data[y][x] = intensity[y * gs + x]
            self.label = name
            self._set_label(self.label)
            self.preview_grid_size = gs
            self._update_grid_highlight()
            self._update_transform()
            self._update_status(f"已加载图案: {name}\n网格: {gs}×{gs}")
        except Exception as e:
            self._update_status(f"加载图案失败: {e}")

    def _load_image(self):
        """从图片文件加载到画布"""
        path = filedialog.askopenfilename(
            parent=self._tk_root,
            title="选择图片文件",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.gif"), ("所有文件", "*.*")]
        )
        if path and os.path.exists(path):
            try:
                gs = self.canvas.grid_size
                self._push_undo()
                self.canvas.from_image(path, gs)
                self._update_transform()
                self._update_status(f"已加载图片: {os.path.basename(path)}\n网格: {gs}×{gs}")
            except Exception as e:
                self._update_status(f"加载图片失败: {e}")
        else:
            self._update_status("未选择图片")

    def _generate_variant_dataset(self):
        """生成 100 个变体并保存到内存（用于导出）"""
        gs = self.canvas.grid_size
        base = self.canvas.to_intensity()
        original = self.transform.snapshot()
        self.train_history = []
        for _ in range(100):
            self.transform.randomize(gs)
            variant = self.transform.apply(base, gs)
            if self.bridge.noise is not None:
                variant = self.bridge.noise.apply(variant)
            self.train_history.append((variant, self.label))
        self.transform.restore(original)
        self._update_transform()
        self._update_status(f"已生成 100 个变体\n标签: {self.label}  网格: {gs}×{gs}\n点击导出CSV/JSON保存")

    def _export_dataset(self, fmt: str):
        """导出自定义训练集"""
        if not self.train_history:
            # 如果没有预生成变体，用当前画布生成一批
            self._generate_variant_dataset()
        ext = ".csv" if fmt == "csv" else ".json"
        default_name = f"dataset_{self.label}_{self.canvas.grid_size}x{self.canvas.grid_size}{ext}"
        path = filedialog.asksaveasfilename(
            parent=self._tk_root,
            title=f"导出训练集 ({fmt.upper()})",
            defaultextension=ext,
            initialfile=default_name,
            filetypes=[(f"{fmt.upper()} 文件", f"*{ext}"), ("所有文件", "*.*")]
        )
        if not path:
            self._update_status("导出已取消")
            return
        try:
            export_training_dataset(self.train_history, path, self.canvas.grid_size, fmt)
            self._update_status(f"训练集已导出:\n{path}\n样本数: {len(self.train_history)}")
        except Exception as e:
            self._update_status(f"导出失败: {e}")

    def _toggle_training(self):
        if self.is_training:
            self.is_training = False
            self.btn_train.text = "开始训练"
            self.btn_train.color_key = "success"
            self._update_status(f"训练已暂停\n{self.bridge.state_text()}")
        else:
            self.is_training = True
            self.train_target = int(self.sl_train_steps.value)
            self.btn_train.text = "停止训练"
            self.btn_train.color_key = "danger"
            self._update_status(f"训练中... 目标 {self.train_target} 步")

    def _test_current(self):
        if self.preview_intensity is None:
            self._update_transform()
        pred, score = self.bridge.classify(self.preview_intensity)
        self.last_test_result = (pred, score)
        conf = "高" if score >= 80 else "中" if score >= 50 else "低"
        msg = f"实时测试结果:\n  预测标签: {pred}\n  匹配度: {score}%\n  置信: {conf}\n  当前标签: {self.label}"
        self._update_status(msg)

    def _reset_network(self):
        self.bridge.reset(seed=random.randint(0, 1000))
        self._update_status(f"网络已重置\n{self.bridge.state_text()}")

    def _update_status(self, text: str):
        self.status_label.set_text(text)

    # ============================================================
    # 字符库覆盖层（纯 pygame，不阻塞事件循环）
    # ============================================================

    def _handle_charlib_overlay_event(self, event: pygame.event.Event) -> bool:
        """处理字符库覆盖层事件，返回 True 表示事件已被消费"""
        if not self._charlib_overlay_active:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self._charlib_overlay_active = False
                return True
            if event.key == pygame.K_LEFT:
                self._charlib_page = max(0, self._charlib_page - 1)
                return True
            if event.key == pygame.K_RIGHT:
                max_page = (len(self._charlib_all_chars) - 1) // self._charlib_chars_per_page
                self._charlib_page = min(max_page, self._charlib_page + 1)
                return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            overlay_x = (self.window_w - 500) // 2
            overlay_y = (self.window_h - 420) // 2
            # 检查是否点击了覆盖层外部 → 关闭
            if not (overlay_x <= mx <= overlay_x + 500 and overlay_y <= my <= overlay_y + 420):
                self._charlib_overlay_active = False
                return True
            # 检查翻页按钮
            btn_y = overlay_y + 370
            if btn_y <= my <= btn_y + 28:
                if overlay_x + 10 <= mx <= overlay_x + 90:
                    self._charlib_page = max(0, self._charlib_page - 1)
                    return True
                if overlay_x + 400 <= mx <= overlay_x + 490:
                    max_page = (len(self._charlib_all_chars) - 1) // self._charlib_chars_per_page
                    self._charlib_page = min(max_page, self._charlib_page + 1)
                    return True
            # 检查字符按钮
            start = self._charlib_page * self._charlib_chars_per_page
            chars = self._charlib_all_chars[start:start + self._charlib_chars_per_page]
            btn_area_x = overlay_x + 20
            btn_area_y = overlay_y + 50
            btn_size = 50
            btn_gap = 6
            for i, ch in enumerate(chars):
                col = i % self._charlib_cols
                row = i // self._charlib_cols
                bx = btn_area_x + col * (btn_size + btn_gap)
                by = btn_area_y + row * (btn_size + btn_gap)
                if bx <= mx <= bx + btn_size and by <= my <= by + btn_size:
                    self._load_char(ch)
                    return True
        return False

    def _draw_charlib_overlay(self):
        """绘制字符库覆盖层"""
        if not self._charlib_overlay_active:
            return
        # 半透明背景
        overlay = pygame.Surface((self.window_w, self.window_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        panel_w, panel_h = 500, 420
        panel_x = (self.window_w - panel_w) // 2
        panel_y = (self.window_h - panel_h) // 2

        # 面板背景
        pygame.draw.rect(self.screen, THEME["panel"], (panel_x, panel_y, panel_w, panel_h), border_radius=8)
        pygame.draw.rect(self.screen, THEME["accent"], (panel_x, panel_y, panel_w, panel_h), 2, border_radius=8)

        # 标题
        title = self.font_large.render("字符库 — 选择字符加载到画布", True, THEME["text"])
        self.screen.blit(title, (panel_x + (panel_w - title.get_width()) // 2, panel_y + 10))

        # 分页信息
        total = len(self._charlib_all_chars)
        max_page = max(0, (total - 1) // self._charlib_chars_per_page)
        page_info = self.font_small.render(
            f"第 {self._charlib_page + 1}/{max_page + 1} 页  |  左右键翻页  |  ESC 关闭", True, THEME["text_dim"]
        )
        self.screen.blit(page_info, (panel_x + (panel_w - page_info.get_width()) // 2, panel_y + 38))

        # 字符按钮
        start = self._charlib_page * self._charlib_chars_per_page
        chars = self._charlib_all_chars[start:start + self._charlib_chars_per_page]
        btn_area_x = panel_x + 20
        btn_area_y = panel_y + 50
        btn_size = 50
        btn_gap = 6
        mx, my = pygame.mouse.get_pos()

        for i, ch in enumerate(chars):
            col = i % self._charlib_cols
            row = i // self._charlib_cols
            bx = btn_area_x + col * (btn_size + btn_gap)
            by = btn_area_y + row * (btn_size + btn_gap)
            is_hover = bx <= mx <= bx + btn_size and by <= my <= by + btn_size
            color = THEME["accent_hover"] if is_hover else THEME["button_bg"]
            pygame.draw.rect(self.screen, color, (bx, by, btn_size, btn_size), border_radius=4)
            pygame.draw.rect(self.screen, THEME["panel_border"], (bx, by, btn_size, btn_size), 1, border_radius=4)
            # 显示字符（大号）和 ASCII 码（小号）
            ch_surf = self.font.render(ch, True, THEME["text"])
            self.screen.blit(ch_surf, (bx + (btn_size - ch_surf.get_width()) // 2,
                                       by + (btn_size - ch_surf.get_height()) // 2 - 6))
            code = self.font_small.render(str(ord(ch)), True, THEME["text_dim"])
            self.screen.blit(code, (bx + (btn_size - code.get_width()) // 2, by + btn_size - 14))

        # 翻页按钮
        btn_y = panel_y + 370
        # 上一页
        prev_color = THEME["button_hover"] if self._charlib_page > 0 else THEME["button_bg"]
        pygame.draw.rect(self.screen, prev_color, (panel_x + 10, btn_y, 80, 28), border_radius=4)
        prev_text = self.font_small.render("← 上一页", True, THEME["text"])
        self.screen.blit(prev_text, (panel_x + 10 + (80 - prev_text.get_width()) // 2, btn_y + 5))
        # 下一页
        next_color = THEME["button_hover"] if self._charlib_page < max_page else THEME["button_bg"]
        pygame.draw.rect(self.screen, next_color, (panel_x + 400, btn_y, 90, 28), border_radius=4)
        next_text = self.font_small.render("下一页 →", True, THEME["text"])
        self.screen.blit(next_text, (panel_x + 400 + (90 - next_text.get_width()) // 2, btn_y + 5))

    def _generate_training_sample(self) -> Tuple[List[int], str]:
        base = self.canvas.to_intensity()
        gs = self.canvas.grid_size
        engine = TransformEngine()
        engine.randomize(gs)
        transformed = engine.apply(base, gs)
        if self.bridge.noise is not None:
            transformed = self.bridge.noise.apply(transformed)
        return transformed, self.label

    def _do_training_frame(self):
        if not self.is_training and not self.multi_label_mode:
            return
        steps_this_frame = 0
        for _ in range(self.train_speed):
            if self.multi_label_mode:
                if self.bridge.history_length() >= self.train_target:
                    self.multi_label_mode = False
                    self.btn_train_multi.show(True)
                    self.btn_train_multi_stop.show(False)
                    self._update_status(f"多标签训练完成！\n{self.bridge.state_text()}")
                    break
                intensity, label = self._generate_multi_label_sample()
            else:
                if self.bridge.history_length() >= self.train_target:
                    self.is_training = False
                    self.btn_train.text = "开始训练"
                    self.btn_train.color_key = "success"
                    self._update_status(f"训练完成！\n{self.bridge.state_text()}")
                    break
                intensity, label = self._generate_training_sample()
            info = self.bridge.train_step(intensity, label)
            self.last_train_info = info
            self.train_history.append((intensity, label))
            steps_this_frame += 1

        if (self.is_training or self.multi_label_mode) and steps_this_frame > 0:
            step = self.bridge.history_length()
            v = "✓" if info.get("V") else "✗"
            match = info.get("match", 0)
            tmpl = info.get("templates", 0)
            mode = "多标签" if self.multi_label_mode else "单标签"
            self._update_status(f"{mode}训练中... 步 {step}/{self.train_target}\n"
                              f"  校验: {v} 匹配: {match}% 模板: {tmpl}\n"
                              f"{self.bridge.state_text()}")

    # ============================================================
    # 训练集管理（多标签共存）
    # ============================================================

    def _add_to_dataset(self):
        """将当前画布内容添加到训练集（不覆盖，追加）"""
        intensity = self.canvas.to_intensity()
        label = self.label
        idx = self.dataset_store.add(label, intensity, f"网格{self.canvas.grid_size}x{self.canvas.grid_size}")
        self._update_dataset_list()
        self._update_status(f"已添加到训练集: 标签={label} 索引={idx}\n"
                          f"当前共 {self.dataset_store.count()} 条\n"
                          f"标签: {', '.join(self.dataset_store.get_labels())}")

    def _save_model(self):
        """保存 SGN 模型到文件"""
        path = filedialog.asksaveasfilename(
            parent=self._tk_root,
            title="保存模型",
            defaultextension=".json",
            initialfile=f"sgn_model_{self.label}_{self.bridge.history_length()}steps.json",
            filetypes=[("JSON 模型", "*.json"), ("所有文件", "*.*")]
        )
        if not path:
            self._update_status("保存已取消")
            return
        ok = self.bridge.model_save(path)
        if ok:
            self._update_status(f"模型已保存: {path}\n{self.bridge.state_text()}")
        else:
            self._update_status("模型保存失败")

    def _load_model(self):
        """从文件加载 SGN 模型"""
        path = filedialog.askopenfilename(
            parent=self._tk_root,
            title="加载模型",
            filetypes=[("JSON 模型", "*.json"), ("所有文件", "*.*")]
        )
        if not path:
            self._update_status("加载已取消")
            return
        ok = self.bridge.model_load(path)
        if ok:
            self._update_status(f"模型已加载: {path}\n{self.bridge.state_text()}")
        else:
            self._update_status("模型加载失败")

    def _save_dataset(self):
        """保存训练集到文件"""
        path = filedialog.asksaveasfilename(
            parent=self._tk_root,
            title="保存训练集",
            defaultextension=".json",
            initialfile="custom_dataset.json",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")]
        )
        if not path:
            self._update_status("保存已取消")
            return
        try:
            saved_path = self.dataset_store.save(path)
            self._update_status(f"训练集已保存:\n{saved_path}\n"
                              f"条目: {self.dataset_store.count()}  标签: {', '.join(self.dataset_store.get_labels())}")
        except Exception as e:
            self._update_status(f"保存失败: {e}")

    def _load_dataset(self):
        """从文件加载训练集"""
        path = filedialog.askopenfilename(
            parent=self._tk_root,
            title="加载训练集",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")]
        )
        if not path:
            self._update_status("加载已取消")
            return
        try:
            ok = self.dataset_store.load(path)
            if ok:
                self._update_dataset_list()
                self._update_status(f"训练集已加载:\n{path}\n"
                                  f"条目: {self.dataset_store.count()}  标签: {', '.join(self.dataset_store.get_labels())}")
            else:
                self._update_status(f"加载失败: 文件不存在或格式错误")
        except Exception as e:
            self._update_status(f"加载失败: {e}")

    def _clear_dataset_entries(self):
        """清空训练集所有条目"""
        self.dataset_store.clear()
        self._update_dataset_list()
        self._update_status("训练集已清空")

    def _update_dataset_list(self):
        """更新训练集列表显示"""
        if not self.dataset_store.entries:
            self.dataset_list_label.set_text("训练集: 空")
            return
        lines = ["训练集:"]
        for i, e in enumerate(self.dataset_store.entries[:10]):
            lines.append(f"  {i+1}. [{e['label']}] {e['description']}")
        if self.dataset_store.count() > 10:
            lines.append(f"  ... 共 {self.dataset_store.count()} 条")
        self.dataset_list_label.set_text("\n".join(lines))

    def _toggle_multi_label_training(self):
        """启动多标签训练模式（循环所有标签）"""
        if not self.dataset_store.entries:
            self._update_status("训练集为空，请先添加图形到训练集")
            return
        if self.is_training:
            self._toggle_training()
        self.multi_label_mode = True
        self.train_target = self.bridge.history_length() + int(self.sl_train_steps.value)
        self.btn_train_multi.show(False)
        self.btn_train_multi_stop.show(True)
        self._update_status(f"多标签训练启动！\n"
                          f"目标: {self.sl_train_steps.value} 步\n"
                          f"标签: {', '.join(self.dataset_store.get_labels())}\n"
                          f"条目: {self.dataset_store.count()}")

    def _stop_multi_label_training(self):
        """停止多标签训练"""
        self.multi_label_mode = False
        self.btn_train_multi.show(True)
        self.btn_train_multi_stop.show(False)
        self._update_status(f"多标签训练已停止\n{self.bridge.state_text()}")

    def _generate_multi_label_sample(self) -> Tuple[List[int], str]:
        """从训练集生成多标签样本（移动适配：一个图形 -> 多种变体）"""
        if not self.dataset_store.entries:
            return self._generate_training_sample()
        samples = self.dataset_store.generate_samples(
            1, noise_fn=self._apply_noise, transform_range={"angle": 15, "offset": 6, "scale": 0.2}
        )
        return samples[0] if samples else self._generate_training_sample()

    def _apply_noise(self, intensity):
        """应用噪声"""
        if self.bridge.noise is not None:
            return self.bridge.noise.apply(intensity)
        return intensity

    # ============================================================
    # 渲染
    # ============================================================

    def _draw_preview(self):
        if self.preview_intensity is None:
            return
        gs = self.canvas.grid_size
        px = max(2, self.preview_rect.width // gs)
        off_x = self.preview_rect.x + (self.preview_rect.width - px * gs) // 2
        off_y = self.preview_rect.y + (self.preview_rect.height - px * gs) // 2
        for y in range(gs):
            for x in range(gs):
                v = self.preview_intensity[y * gs + x]
                if v > 0:
                    rect = (off_x + x * px, off_y + y * px, px, px)
                    color = Color(v, v, v)
                    pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, THEME["panel_border"], self.preview_rect, 2)

    def _draw_info_panel(self):
        info_y = self.canvas.rect.bottom + 10
        if info_y < self.window_h - 60:
            info = [
                f"网格: {self.canvas.grid_size}×{self.canvas.grid_size}",
                f"变换: 旋转{self.transform.angle:.0f}° 偏移({self.transform.offset_x},{self.transform.offset_y}) 缩放{self.transform.scale:.1f}x",
                f"标签: {self.label}",
                f"网络: {self.bridge.state_text()}",
            ]
            if self.last_test_result:
                pred, score = self.last_test_result
                info.append(f"上次测试: 预测={pred} 匹配={score}%")
            for i, line in enumerate(info):
                surf = self.font_small.render(line, True, THEME["text_dim"])
                self.screen.blit(surf, (self.margin, info_y + i * 16))

    # ============================================================
    # 主循环
    # ============================================================

    def run(self):
        self._update_transform()
        running = True
        while running:
            dt = self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                # 字符库覆盖层优先处理事件
                if self._handle_charlib_overlay_event(event):
                    continue
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    if event.key == pygame.K_SPACE:
                        self._toggle_training()
                    if event.key == pygame.K_t:
                        self._test_current()
                    if event.key == pygame.K_r:
                        self._random_transform()
                    if event.key == pygame.K_c:
                        self._clear_canvas()
                    if event.key == pygame.K_e:
                        self._toggle_eraser()
                    # Ctrl+Z / Ctrl+Y 撤销重做
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_CTRL:
                        if event.key == pygame.K_z:
                            self._undo()
                        elif event.key == pygame.K_y:
                            self._redo()
                if event.type == pygame.VIDEORESIZE:
                    # 窗口大小变化时重新计算布局
                    self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    self._recalc_layout()
                    self._build_ui()
                    continue

                self.canvas.handle_event(event)
                if event.type == pygame.MOUSEBUTTONUP:
                    self._update_transform()
                    if event.button in (1, 3):
                        self._push_undo()

                for comp in self.ui_components:
                    comp.handle_event(event)

            self._do_training_frame()

            # 渲染
            self.screen.fill(THEME["bg"])
            left_panel = Rect(0, 0, self.left_w + self.margin, self.window_h)
            pygame.draw.rect(self.screen, THEME["panel"], left_panel)
            pygame.draw.line(self.screen, THEME["separator"], (self.left_w + self.margin, 0), (self.left_w + self.margin, self.window_h))

            # 右面板背景
            right_panel = Rect(self.left_w + self.margin, 0, self.right_w + self.margin, self.window_h)
            pygame.draw.rect(self.screen, THEME["panel"], right_panel)

            title = self.font_large.render("函数工厂 — 绘制区", True, THEME["text"])
            self.screen.blit(title, (self.margin, self.margin))
            sub = self.font_small.render("左键绘制/擦除 | 右键擦除 | Ctrl+Z撤销 | Ctrl+Y重做 | E切换橡皮擦 | 空格训练 | ESC退出", True, THEME["text_dim"])
            self.screen.blit(sub, (self.margin, self.margin + 28))

            self.canvas.draw(self.screen)
            self.title_label.draw(self.screen)
            self.preview_label.draw(self.screen)
            self._draw_preview()
            self.label_label.draw(self.screen)
            self.dataset_label.draw(self.screen)
            self.dataset_list_label.draw(self.screen)
            self.status_label.draw(self.screen)

            for comp in self.ui_components:
                comp.draw(self.screen)

            self._draw_info_panel()
            self._draw_charlib_overlay()
            pygame.display.flip()

        pygame.quit()
