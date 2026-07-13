#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gui/dataset_store.py — 自定义训练集存储管理器

独立文件存储用户自定义的图形-标签对，支持：
- 添加/删除/查看条目（多标签共存，不覆盖）
- 保存/加载 JSON 文件
- 生成训练样本（循环所有标签 + 随机变换 + 噪声）
- 被 GUI 和命令行同时导入使用

文件格式:
  {
    "version": "1.0",
    "grid_size": 16,
    "entries": [
      {"label": "A", "intensity": [0,255,...], "description": "...", "created_at": "..."}
    ]
  }
"""
from __future__ import annotations

import json
import os
import datetime
import random
from typing import List, Dict, Tuple, Optional

from gui.utils import apply_transform


class CustomDatasetStore:
    """自定义训练集存储管理器

    每个条目是一个独立的(label, intensity)对，标签不互相覆盖。
    训练时循环使用所有条目，为每个图形应用随机变换和噪声。
    """

    VERSION = "1.0"
    DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "custom_dataset.json")

    def __init__(self, filepath: Optional[str] = None):
        self.entries: List[Dict] = []
        self.grid_size: int = 16
        self.filepath: str = filepath or self.DEFAULT_PATH

    # ============================================================
    # 条目管理
    # ============================================================

    def add(self, label: str, intensity: List[int], description: str = "") -> int:
        """添加一个条目（不覆盖，追加）

        返回新条目的索引。根据 intensity 长度自动推断 grid_size。
        """
        d = len(intensity)
        inferred_gs = int(d ** 0.5)
        if inferred_gs * inferred_gs != d:
            inferred_gs = self.grid_size  #  fallback
        entry = {
            "label": label.upper().strip(),
            "intensity": list(intensity),
            "grid_size": inferred_gs,
            "description": description,
            "created_at": datetime.datetime.now().isoformat(),
        }
        self.entries.append(entry)
        return len(self.entries) - 1

    def remove(self, index: int) -> bool:
        """删除指定索引的条目"""
        if 0 <= index < len(self.entries):
            self.entries.pop(index)
            return True
        return False

    def update(self, index: int, **kwargs) -> bool:
        """更新指定条目的字段"""
        if not (0 <= index < len(self.entries)):
            return False
        for k, v in kwargs.items():
            if k in self.entries[index]:
                self.entries[index][k] = v
        return True

    def get(self, index: int) -> Optional[Dict]:
        """获取指定条目"""
        if 0 <= index < len(self.entries):
            return self.entries[index].copy()
        return None

    def get_all(self) -> List[Dict]:
        """返回所有条目副本"""
        return [e.copy() for e in self.entries]

    def get_labels(self) -> List[str]:
        """返回所有标签列表（去重）"""
        return sorted(set(e["label"] for e in self.entries))

    def get_by_label(self, label: str) -> List[Dict]:
        """获取指定标签的所有条目"""
        label = label.upper().strip()
        return [e.copy() for e in self.entries if e["label"] == label]

    def count(self) -> int:
        return len(self.entries)

    def clear(self):
        """清空所有条目"""
        self.entries.clear()

    # ============================================================
    # 文件持久化
    # ============================================================

    def save(self, filepath: Optional[str] = None) -> str:
        """保存到 JSON 文件，返回保存路径"""
        path = filepath or self.filepath
        data = {
            "version": self.VERSION,
            "grid_size": self.grid_size,
            "entries": self.entries,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    def load(self, filepath: Optional[str] = None) -> bool:
        """从 JSON 文件加载，返回是否成功"""
        path = filepath or self.filepath
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.entries = data.get("entries", [])
            self.grid_size = data.get("grid_size", 16)
            # 验证条目格式
            valid = []
            for e in self.entries:
                if isinstance(e, dict) and "label" in e and "intensity" in e:
                    valid.append(e)
            self.entries = valid
            return True
        except (json.JSONDecodeError, OSError) as e:
            print(f"[CustomDataset] 加载失败: {e}")
            return False

    # ============================================================
    # 训练样本生成（移动适配：一个图形 → 多种变体）
    # ============================================================

    def generate_samples(
        self,
        count: int,
        noise_fn=None,
        transform_range: Optional[Dict] = None,
    ) -> List[Tuple[List[int], str]]:
        """生成训练样本

        循环使用所有条目，每个条目应用随机变换 + 可选噪声。

        Args:
            count: 需要生成的样本数量
            noise_fn: 噪声函数，签名为 noise_fn(intensity) -> intensity
            transform_range: 变换范围字典，如 {"angle": 15, "offset": 3, "scale": 0.2}

        Returns:
            [(intensity, label), ...]
        """
        if not self.entries:
            return []

        tr = transform_range or {"angle": 15, "offset": 6, "scale": 0.2}
        max_angle = tr.get("angle", 15)
        max_offset = tr.get("offset", 6)
        max_scale = tr.get("scale", 0.2)
        gs = self.grid_size

        samples = []
        for i in range(count):
            entry = self.entries[i % len(self.entries)]
            intensity = entry["intensity"][:]
            gs = entry.get("grid_size", self.grid_size)

            # 随机变换（浮点运算，输入层保留精度）
            angle = random.uniform(-max_angle, max_angle)
            offset_x = random.randint(-max_offset, max_offset)
            offset_y = random.randint(-max_offset, max_offset)
            scale = random.uniform(1.0 - max_scale, 1.0 + max_scale)
            intensity = apply_transform(intensity, gs, angle, offset_x, offset_y, scale)

            # 可选噪声
            if noise_fn:
                intensity = noise_fn(intensity)

            samples.append((intensity, entry["label"]))
        return samples

    # ============================================================
    # 批量导出（CSV/JSON 格式，兼容 sgn_input.FileInputSource）
    # ============================================================

    def export_to_csv(self, filepath: str) -> str:
        """导出为 CSV 格式（兼容 sgn_input.FileInputSource）"""
        import csv
        d = self.grid_size * self.grid_size
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = [f"intensity_{i}" for i in range(d)] + ["label"]
            writer.writerow(header)
            for e in self.entries:
                row = e["intensity"] + [e["label"]]
                writer.writerow(row)
        return filepath

    def export_to_json(self, filepath: str) -> str:
        """导出为 JSON 格式（兼容 sgn_input.FileInputSource）"""
        import json
        data = [
            {"intensity": e["intensity"], "label": e["label"], "grid_size": self.grid_size}
            for e in self.entries
        ]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return filepath

    def __repr__(self) -> str:
        labels = self.get_labels()
        return f"CustomDatasetStore(entries={len(self.entries)}, labels={labels}, grid={self.grid_size})"
