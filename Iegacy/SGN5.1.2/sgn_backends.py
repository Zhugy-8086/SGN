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
"""SGN-Lite v5.0 图表后端抽象 - ChartBackend / BackendRegistry

阶段4重构：解耦 matplotlib 硬编码，允许插件替换可视化后端。
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


# ============================================================
# PlotData - 统一数据格式
# ============================================================

@dataclass
class PlotData:
    """所有后端统一接受的标准数据对象"""
    steps: List[int]           # 步数列表
    values: Dict[str, List]    # {"accuracy": [...], "templates": [...]}
    labels: Dict[str, str]     # {"accuracy": "累计准确率(%)"}


# ============================================================
# ChartBackend - 图表后端抽象
# ============================================================

class ChartBackend(ABC):
    """图表后端抽象基类"""

    name: str = ""

    @abstractmethod
    def is_available(self) -> bool:
        """检查后端是否可用（依赖是否已安装）"""
        pass

    @abstractmethod
    def plot_accuracy(self, history: List[Dict], path: Optional[str] = None) -> Optional[str]:
        """绘制累计准确率曲线"""
        pass

    @abstractmethod
    def plot_neurons(self, core, path: Optional[str] = None) -> Optional[str]:
        """绘制神经元状态分布"""
        pass

    @abstractmethod
    def plot_templates(self, history: List[Dict], path: Optional[str] = None) -> Optional[str]:
        """绘制模板增长曲线"""
        pass

    @abstractmethod
    def plot_comprehensive(self, history: List[Dict], core, path: Optional[str] = None) -> Optional[str]:
        """绘制综合面板"""
        pass

    @abstractmethod
    def plot_ascii(self, history: List[Dict]) -> None:
        """ASCII 终端绘图（降级方案）"""
        pass


# ============================================================
# 上下文管理器 - 强制关闭 figure
# ============================================================

@contextmanager
def managed_figure(**kwargs):
    """确保 matplotlib figure 被正确关闭的上下文管理器

    接受 figsize/dpi 等 kwargs，只创建一个 Figure，避免内存浪费。
    """
    import matplotlib.pyplot as plt
    fig = None
    try:
        fig = plt.figure(**kwargs)
        yield fig
    finally:
        if fig is not None:
            plt.close(fig)
        plt.close("all")


# ============================================================
# MatplotlibBackend
# ============================================================

class MatplotlibBackend(ChartBackend):
    """Matplotlib 图表后端"""

    name = "matplotlib"

    def is_available(self) -> bool:
        try:
            import matplotlib
            return True
        except ImportError:
            return False

    def _make_path(self, default_name: str) -> str:
        """生成带时间戳的文件路径，避免覆盖"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.normpath(f"sgn_{default_name}_{ts}.png")

    def plot_accuracy(self, history: List[Dict], path: Optional[str] = None) -> Optional[str]:
        if not self.is_available():
            return None
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = len(history)
        if t == 0:
            return None
        steps = list(range(1, t + 1))
        ver = [1 if x["V"] else 0 for x in history]
        acc = []
        s = 0
        for i, v_step in enumerate(ver):
            s += v_step
            acc.append(s / (i + 1) * 100)

        path = path or self._make_path("accuracy")
        with managed_figure(figsize=(10, 5)) as fig:
            ax = fig.gca()
            ax.plot(steps, acc, "g-", linewidth=0.8, label="Cumulative Accuracy")
            ax.set_xlabel("Step")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("SGN-Lite v5.0 Cumulative Accuracy")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(0, 105)
            fig.savefig(path, dpi=150)
        return path

    def plot_neurons(self, core, path: Optional[str] = None) -> Optional[str]:
        if not self.is_available():
            return None
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 【fix】DiscreteCoordinate → 显示浮点值（仅用于可视化，不进入核心引擎）
        def _base_val(n):
            b = n["base"]
            if hasattr(b, "index") and hasattr(b, "scale"):
                return b.index / b.scale
            return float(b)

        active = [_base_val(n) for n in core.N if not n["L"]]
        locked = [_base_val(n) for n in core.N if n["L"]]
        labels = ["Active", "Locked"]
        counts = [len(active), len(locked)]

        path = path or self._make_path("neurons")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        try:
            colors = ["#2ecc71", "#e74c3c"]
            axes[0].bar(labels, counts, color=colors, edgecolor="black")
            axes[0].set_ylabel("Neuron Count")
            axes[0].set_title("Neuron Active/Locked Count")
            for i, v in enumerate(counts):
                axes[0].text(i, v + 1, str(v), ha="center", va="bottom", fontweight="bold")
            data = [active, locked] if active and locked else ([active] if active else [locked])
            lbls = (["Active", "Locked"] if active and locked else
                    (["Active"] if active else ["Locked"]))
            axes[1].boxplot(data, labels=lbls)
            axes[1].set_ylabel("Base Speed")
            axes[1].set_title("Base Speed Distribution")
            plt.tight_layout()
            fig.savefig(path, dpi=150)
        finally:
            plt.close(fig)
            plt.close("all")
        return path

    def plot_templates(self, history: List[Dict], path: Optional[str] = None) -> Optional[str]:
        if not self.is_available():
            return None
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = len(history)
        if t == 0:
            return None
        steps = list(range(1, t + 1))
        tpl_counts = [info.get("templates", 0) for info in history]
        window = 50
        ver_rates = []
        for i in range(t):
            start = max(0, i - window + 1)
            chunk = history[start:i+1]
            rate = sum(1 for x in chunk if x["V"]) / len(chunk) * 100 if chunk else 0
            ver_rates.append(rate)

        path = path or self._make_path("templates")
        fig, ax1 = plt.subplots(figsize=(10, 5))
        try:
            color1 = "#3498db"
            ax1.set_xlabel("Step")
            ax1.set_ylabel("Template Count", color=color1)
            ax1.plot(steps, tpl_counts, color=color1, linewidth=1.2, label="Templates")
            ax1.tick_params(axis="y", labelcolor=color1)
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            color2 = "#e67e22"
            ax2.set_ylabel("Verify Rate (%)", color=color2)
            ax2.plot(steps, ver_rates, color=color2, linewidth=1.0, linestyle="--", label="Verify Rate")
            ax2.tick_params(axis="y", labelcolor=color2)
            ax2.set_ylim(0, 105)
            plt.title("SGN-Lite v5.0 Templates & Verify Rate")
            fig.tight_layout()
            fig.savefig(path, dpi=150)
        finally:
            plt.close(fig)
            plt.close("all")
        return path

    def plot_comprehensive(self, history: List[Dict], core, path: Optional[str] = None) -> Optional[str]:
        if not self.is_available():
            return None
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = len(history)
        if t == 0:
            return None
        steps = list(range(1, t + 1))
        ver = [1 if x["V"] else 0 for x in history]
        acc = []
        s = 0
        for i, v_step in enumerate(ver):
            s += v_step
            acc.append(s / (i + 1) * 100)

        path = path or self._make_path("comprehensive")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        try:
            axes[0, 0].plot(steps, acc, "g-", linewidth=0.8)
            axes[0, 0].set_title("Cumulative Accuracy")
            axes[0, 0].set_xlabel("Step")
            axes[0, 0].set_ylabel("Accuracy (%)")
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(0, 105)

            tpl_counts = [info.get("templates", 0) for info in history]
            axes[0, 1].plot(steps, tpl_counts, "b-", linewidth=1.0)
            axes[0, 1].set_title("Template Library Growth")
            axes[0, 1].set_xlabel("Step")
            axes[0, 1].set_ylabel("Template Count")
            axes[0, 1].grid(True, alpha=0.3)

            active_hist = [info.get("active", 0) for info in history]
            locked_hist = [info.get("locked", 0) for info in history]
            axes[1, 0].plot(steps, active_hist, "g-", linewidth=0.8, label="Active")
            axes[1, 0].plot(steps, locked_hist, "r-", linewidth=0.8, label="Locked")
            axes[1, 0].set_title("Neuron State Changes")
            axes[1, 0].set_xlabel("Step")
            axes[1, 0].set_ylabel("Count")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            recent = history[-50:]
            r_steps = list(range(t - len(recent) + 1, t + 1))
            r_ver = [1 if x["V"] else 0 for x in recent]
            axes[1, 1].scatter(r_steps, r_ver, c=["g" if v else "r" for v in r_ver], s=20, alpha=0.7)
            axes[1, 1].set_title("Last 50 Steps Verification")
            axes[1, 1].set_xlabel("Step")
            axes[1, 1].set_ylabel("Verified (1=Yes, 0=No)")
            axes[1, 1].set_ylim(-0.2, 1.2)
            axes[1, 1].grid(True, alpha=0.3)

            plt.suptitle("SGN-Lite v5.0 Comprehensive Dashboard", fontsize=14, fontweight="bold")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(path, dpi=150)
        finally:
            plt.close(fig)
            plt.close("all")
        return path

    def plot_ascii(self, history: List[Dict]) -> None:
        """降级到 ASCII"""
        ASCIIChartBackend().plot_ascii(history)


# ============================================================
# ASCIIChartBackend
# ============================================================

class ASCIIChartBackend(ChartBackend):
    """ASCII 终端图表后端（降级方案，不依赖外部库）"""

    name = "ascii"

    def is_available(self) -> bool:
        return True  # 总是可用

    def plot_accuracy(self, history: List[Dict], path: Optional[str] = None) -> Optional[str]:
        self.plot_ascii(history)
        return None

    def plot_neurons(self, core, path: Optional[str] = None) -> Optional[str]:
        print("  [ASCII] Neuron distribution: not implemented, use matplotlib")
        return None

    def plot_templates(self, history: List[Dict], path: Optional[str] = None) -> Optional[str]:
        print("  [ASCII] Template growth: not implemented, use matplotlib")
        return None

    def plot_comprehensive(self, history: List[Dict], core, path: Optional[str] = None) -> Optional[str]:
        self.plot_ascii(history)
        return None

    def plot_ascii(self, history: List[Dict]) -> None:
        from sgn_utils import C
        t = len(history)
        if t == 0:
            print(f"\n  {C.warn('⚠')} 无历史数据")
            return
        v = sum(1 for x in history if x["V"])
        pct = v / t * 100
        print(f"\n  {C.BOLD}ASCII Learning Curve (last 100 steps){C.RST}")
        recent = history[-100:]
        chunk_size = 10
        for i in range(0, len(recent), chunk_size):
            chunk = recent[i:i+chunk_size]
            v_count = sum(1 for x in chunk if x["V"])
            bar = f"{C.GRN}{'█'*v_count}{C.RST}{C.DIM}{'░'*(chunk_size-v_count)}{C.RST}"
            start_step = max(1, t - len(recent) + i + 1)
            end_step = start_step + len(chunk) - 1
            print(f"  Step{start_step:>4}-{end_step:>4}: [{bar}] {v_count:>2}/{len(chunk)}")
        print(f"  Total: {C.GRN}{v}{C.RST}/{t} ({pct:.1f}%)")


# ============================================================
# CSVExportBackend
# ============================================================

class CSVExportBackend(ChartBackend):
    """CSV 导出后端（导出数据供外部编辑器使用）"""

    name = "csv"

    def is_available(self) -> bool:
        return True

    def plot_accuracy(self, history: List[Dict], path: Optional[str] = None) -> Optional[str]:
        import csv
        t = len(history)
        if t == 0:
            return None
        path = path or self._make_path("accuracy.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "cumulative_accuracy"])
            s = 0
            for i, info in enumerate(history, 1):
                s += 1 if info["V"] else 0
                acc = s / i * 100
                writer.writerow([i, f"{acc:.2f}"])
        return path

    def plot_neurons(self, core, path: Optional[str] = None) -> Optional[str]:
        import csv
        path = path or self._make_path("neurons.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["nid", "base_speed", "locked", "encourage_remaining"])
            for i, n in enumerate(core.N):
                # 【v4.3-fix】base 是 DiscreteCoordinate，取 index/scale 或序列化
                base_val = n['base']
                if hasattr(base_val, 'index') and hasattr(base_val, 'scale'):
                    base_str = f"{base_val.index / base_val.scale:.3f}"
                else:
                    base_str = str(base_val)
                writer.writerow([i, base_str, int(n["L"]), n["enc_r"]])
        return path

    def plot_templates(self, history: List[Dict], path: Optional[str] = None) -> Optional[str]:
        import csv
        t = len(history)
        if t == 0:
            return None
        path = path or self._make_path("templates.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "template_count"])
            for i, info in enumerate(history, 1):
                writer.writerow([i, info.get("templates", 0)])
        return path

    def plot_comprehensive(self, history: List[Dict], core, path: Optional[str] = None) -> Optional[str]:
        return self.plot_accuracy(history, path)

    def plot_ascii(self, history: List[Dict]) -> None:
        pass

    def _make_path(self, default_name: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.normpath(f"sgn_{default_name}_{ts}")


# ============================================================
# BackendRegistry - 后端注册表
# ============================================================

class BackendRegistry:
    """后端注册表 - 自动选择可用后端"""

    _backends: List[ChartBackend] = []

    @classmethod
    def register(cls, backend: ChartBackend) -> None:
        cls._backends.append(backend)

    @classmethod
    def auto_select(cls) -> ChartBackend:
        """按优先级选择第一个 available 的后端"""
        for b in cls._backends:
            if b.is_available():
                return b
        return ASCIIChartBackend()

    @classmethod
    def get(cls, name: str) -> Optional[ChartBackend]:
        """按名称获取后端"""
        for b in cls._backends:
            if b.name == name:
                return b
        return None

    @classmethod
    def list_backends(cls) -> List[str]:
        return [b.name for b in cls._backends]

    @classmethod
    def clear(cls) -> None:
        cls._backends.clear()


# 自动注册默认后端（按优先级排序：matplotlib > ascii > csv）
BackendRegistry.register(MatplotlibBackend())
BackendRegistry.register(ASCIIChartBackend())
BackendRegistry.register(CSVExportBackend())
