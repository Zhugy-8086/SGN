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
"""SGN-Lite v5.0 模型持久化模块 - 使用 StorageBackend 抽象

阶段5重构：通过 StorageBackend 接口调用，解耦 JSON 硬编码。
保留 v5.0 兼容 API（save_model, load_model, autosave_check, check_resume）。
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from engine.utils import C
from engine.config import CONFIG
from app.storage import StorageRegistry, IntervalAutosave, DeltaAutosave


# ============================================================
# v5.0 兼容常量
# ============================================================

AUTOSAVE_FILE = "sgn_autosave.json"
AUTOSAVE_BACKUPS = ["sgn_autosave_1.json", "sgn_autosave_2.json", "sgn_autosave_3.json"]
AUTOSAVE_INTERVAL = 50  # 每50步自动保存


# ============================================================
# 自动保存策略实例（单例）
# ============================================================

# 自动保存策略（延迟初始化，避免模块加载时 CONFIG 未就绪）
_autosave_strategy = None

def _init_autosave_strategy():
    """延迟初始化自动保存策略"""
    global _autosave_strategy
    if _autosave_strategy is not None:
        return
    from engine.config import CONFIG
    name = CONFIG.get("AUTOSAVE_STRATEGY", "interval")
    if name == "delta":
        _autosave_strategy = DeltaAutosave(min_interval=AUTOSAVE_INTERVAL)
    else:
        _autosave_strategy = IntervalAutosave(interval=AUTOSAVE_INTERVAL)


# ============================================================
# 核心 API
# ============================================================

def save_model(core, path=None):
    """将引擎状态保存到文件 - v5.0 兼容 API"""
    if path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"sgn_model_{ts}.json"

    backend_name = CONFIG.get("STORAGE_BACKEND", "json")
    backend = StorageRegistry.get(backend_name)
    success = backend.save(core, path)

    if success:
        abs_path = os.path.abspath(path)
        print(f"  {C.ok('✓')} 模型已保存: {C.val(abs_path)}")
        return True
    else:
        print(f"  {C.err('✗')} 保存失败")
        return False


def load_model(core, path):
    """从文件恢复引擎状态 - v5.0 兼容 API"""
    path = os.path.normpath(path)
    if not os.path.exists(path):
        print(f"  {C.err('✗')} 文件不存在: {path}")
        return False

    backend_name = CONFIG.get("STORAGE_BACKEND", "json")
    backend = StorageRegistry.get(backend_name)
    success = backend.load(core, path)

    if success:
        # 版本兼容性检查
        file_version = getattr(core, '_loaded_version', '4.2')
        current_version = "4.2"
        if file_version != current_version:
            print(f"  {C.YEL}⚠ 版本不匹配: 文件={file_version}, 当前={current_version}{C.RST}")

        print(f"  {C.ok('✓')} 模型已加载: {C.val(path)}")
        print(f"  {C.info('ℹ')} {len(core.N)}神经元/{len(core.templates)}模板/{len(core.history)}历史步")
        return True
    else:
        print(f"  {C.err('✗')} 加载失败")
        return False


def autosave_check(core, step):
    """检查是否需要自动保存 - v5.0 兼容 API"""
    global _autosave_strategy
    _init_autosave_strategy()

    if not _autosave_strategy.should_save(step, core):
        return False

    # 轮换备份：3 → 2, 2 → 1, 1 → 当前
    for i in range(len(AUTOSAVE_BACKUPS) - 1, 0, -1):
        if os.path.exists(AUTOSAVE_BACKUPS[i - 1]):
            try:
                os.replace(AUTOSAVE_BACKUPS[i - 1], AUTOSAVE_BACKUPS[i])
            except OSError:
                pass

    # 使用后端保存
    backend_name = CONFIG.get("STORAGE_BACKEND", "json")
    backend = StorageRegistry.get(backend_name)
    success = backend.save(core, AUTOSAVE_FILE)

    if success:
        # 同时保存到备份1
        try:
            shutil.copy2(AUTOSAVE_FILE, AUTOSAVE_BACKUPS[0])
        except Exception:
            pass
        return True
    return False


def check_resume():
    """启动时检查是否有自动保存文件 - v5.0 兼容 API

    Returns:
        str: 找到的文件路径（含备份），用户确认恢复时返回
        None: 未找到文件或用户取消恢复
    """
    candidates = [AUTOSAVE_FILE] + AUTOSAVE_BACKUPS
    found = None
    for f in candidates:
        if os.path.exists(f):
            found = f
            break
    if found:
        print(f"\n  {C.YEL}[发现自动保存]{C.RST} {C.val(found)}")
        mtime = datetime.fromtimestamp(os.path.getmtime(found))
        print(f"  修改时间: {C.val(mtime.strftime('%Y-%m-%d %H:%M:%S'))}")
        try:
            inp = input(f"  是否恢复上次训练? [Y/n]: ").strip().lower()
            if inp not in ("n", "no"):
                return found
        except (EOFError, KeyboardInterrupt):
            return None
    else:
        print(f"\n  {C.DIM}[无自动保存] 未发现 {AUTOSAVE_FILE} 或轮换备份{C.RST}")
    return None


# ============================================================
# v5.0 扩展 API
# ============================================================

def save_model_with_backend(core, path: str, backend_name: str = None) -> bool:
    """使用指定后端保存模型"""
    backend = StorageRegistry.get(backend_name)
    return backend.save(core, path)


def load_model_with_backend(core, path: str, backend_name: str = None) -> bool:
    """使用指定后端加载模型"""
    backend = StorageRegistry.get(backend_name)
    return backend.load(core, path)


def set_autosave_strategy(strategy):
    """设置自动保存策略（插件可调用）"""
    global _autosave_strategy
    _init_autosave_strategy()
    _autosave_strategy = strategy
    # 【fix】走 ConfigRegistry 统一入口，享受校验/同步/钩子/提醒
    from engine.config import ConfigRegistry
    name = "interval" if isinstance(strategy, IntervalAutosave) else "delta"
    ConfigRegistry.set("AUTOSAVE_STRATEGY", name)
