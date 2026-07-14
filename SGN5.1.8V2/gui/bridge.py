#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gui/bridge.py — SGN 桥接（连接 GUI 与 SGN 核心引擎）"""
from __future__ import annotations

from typing import List, Tuple, Dict, Optional


_SGN_AVAILABLE = False


def _ensure_sgn():
    global _SGN_AVAILABLE
    if _SGN_AVAILABLE:
        return True
    try:
        from engine.core import SGNCore
        from engine.input import DefaultCompositeNoise
        from app.test import _classify
        _SGN_AVAILABLE = True
        return True
    except ImportError as e:
        print(f"[错误] 无法导入 SGN 核心模块: {e}")
        return False


class SGNBridge:
    """连接 GUI 与 SGN 核心引擎

    支持两种初始化模式：
      1. 新建网络：SGNBridge(seed=42)
      2. 复用已有网络：SGNBridge(core=existing_core)
    """

    def __init__(self, seed: int = 42, core=None):
        self.core = core
        self.seed = seed
        self.noise = None
        if core is None:
            self._init()
        else:
            self._init_with_core(core)

    def _init_with_core(self, core):
        """使用外部传入的 core，初始化 noise"""
        from engine.input import DefaultCompositeNoise
        self.noise = DefaultCompositeNoise(0.15)

    def _init(self):
        if not _ensure_sgn():
            return
        from engine.core import SGNCore
        from engine.input import DefaultCompositeNoise
        self.core = SGNCore(seed=self.seed)
        self.noise = DefaultCompositeNoise(0.15)

    def reset(self, seed: int = None):
        """重置网络"""
        if seed is not None:
            self.seed = seed
        self._init()

    def train_step(self, intensity: List[int], label: str) -> Dict:
        """执行单步训练，返回 info 字典"""
        if self.core is None:
            return {}
        return self.core.train(intensity, label)

    def classify(self, intensity: List[int]) -> Tuple[str, int]:
        """对单个样本分类，返回 (预测标签, 匹配度)"""
        if self.core is None:
            return "?", 0
        from app.test import _classify
        return _classify(self.core, intensity)

    def get_state(self) -> Dict:
        if self.core is None:
            return {}
        return self.core.get_state()

    def history_length(self) -> int:
        if self.core is None:
            return 0
        return len(self.core.history)

    def model_save(self, path: str) -> bool:
        if self.core is None:
            return False
        try:
            from app.storage import StorageRegistry
            from engine.config import CONFIG
            backend_name = CONFIG.get("STORAGE_BACKEND", "json")
            backend = StorageRegistry.get(backend_name)
            return backend.save(self.core, path)
        except Exception as e:
            print(f"[bridge] save failed: {e}")
            return False

    def model_load(self, path: str) -> bool:
        if self.core is None:
            return False
        try:
            from app.storage import StorageRegistry
            from engine.config import CONFIG
            backend_name = CONFIG.get("STORAGE_BACKEND", "json")
            backend = StorageRegistry.get(backend_name)
            return backend.load(self.core, path)
        except Exception as e:
            print(f"[bridge] load failed: {e}")
            return False

    def state_text(self) -> str:
        """返回状态文本（用于 GUI 显示）"""
        if self.core is None:
            return "SGN 未加载"
        st = self.get_state()
        if getattr(self.core, "graph_mode", False):
            return (f"神经元: {st.get('neurons', 0)} | "
                    f"图: {st.get('templates', 0)} | "
                    f"步数: {self.history_length()}")
        return (f"神经元: {st.get('neurons', 0)} | "
                f"活跃: {st.get('active', 0)} | "
                f"锁定: {st.get('locked', 0)} | "
                f"模板: {st.get('templates', 0)} | "
                f"步数: {self.history_length()}")
