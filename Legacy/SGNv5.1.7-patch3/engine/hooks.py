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
"""SGN-Lite v5.0 事件总线 - HookRegistry

阶段0基础设施：所有后续扩展层的底座。
向外部广播训练生命周期事件，允许插件在不改源码的情况下监听状态。

事件命名约定:
- "sgn:before_step"   - 训练步开始前 (在 core.train 调用前触发)
- "sgn:after_step"    - 训练步完成后 (在 core.train 返回后触发)
- "sgn:on_template_added" - 新模板被添加时
- "sgn:on_neuron_locked"  - 神经元被锁定时 (预留)
- "sgn:on_config_changed" - 配置项变更时 (由 ConfigRegistry 触发)

插件应使用自己的命名空间前缀: "plugin.name:event_name"
"""

from __future__ import annotations

import weakref
import functools
from typing import Callable, Any, Dict, List, Union


# ============================================================
# 弱引用回调包装
# ============================================================

class _WeakCallback:
    """弱引用回调包装器，自动处理方法绑定和函数类型

    对 partial 对象：拆解为 func + 首参弱引用，避免循环引用。
    对绑定方法：弱引用实例。
    对普通函数：弱引用函数本身。
    对 lambda/staticmethod：检测即时 GC，自动转强引用兜底。
    """

    __slots__ = ("_ref", "_func", "_partial_args", "_partial_kwargs", "_strong", "_orig_callback")

    def __init__(self, callback: Callable):
        self._strong = None  # v5.1.7: 强引用兜底（lambda 即时 GC 场景）
        self._orig_callback = None  # v5.1.7: 供 unregister 匹配（partial 场景）

        # v5.1.7: staticmethod/classmethod 对象提取 __func__
        if isinstance(callback, staticmethod):
            callback = callback.__func__
        elif isinstance(callback, classmethod):
            callback = callback.__func__

        if isinstance(callback, functools.partial):
            # 拆解 partial：弱引用 func 和第一个位置参数（如果是对象）
            self._orig_callback = callback  # 供 unregister 匹配
            self._func = callback.func
            if callback.args:
                first_arg = callback.args[0]
                if hasattr(first_arg, "__class__") and not isinstance(first_arg, (str, bytes, int, float, bool, tuple)):
                    self._ref = weakref.ref(first_arg)
                    self._partial_args = callback.args[1:]
                else:
                    self._ref = None
                    self._partial_args = callback.args
            else:
                self._ref = None
                self._partial_args = ()
            self._partial_kwargs = callback.keywords or {}
        elif hasattr(callback, "__self__"):
            # 绑定方法：弱引用实例
            self._ref = weakref.ref(callback.__self__)
            self._func = callback.__func__
            self._partial_args = ()
            self._partial_kwargs = {}
        else:
            # 普通函数：弱引用
            self._ref = weakref.ref(callback)
            self._func = None
            self._partial_args = ()
            self._partial_kwargs = {}
            # v5.1.7: 检测即时 GC（lambda 无外部引用），转强引用兜底
            if self._ref() is None:
                self._strong = callback
                self._ref = None

    def __call__(self, *args, **kwargs) -> bool:
        """调用回调，返回 True 表示成功，False 表示引用已死亡"""
        # v5.1.7: 强引用兜底（lambda 即时 GC 场景）
        if self._strong is not None:
            self._strong(*args, **kwargs)
            return True
        if self._ref is None:
            # partial 无对象参数 或 普通函数死亡 → 直接/已死亡
            if self._func is None:
                return False  # 普通函数已死亡
            # partial 无对象参数，直接调用
            self._func(*self._partial_args, *args, **{**self._partial_kwargs, **kwargs})
            return True
        obj = self._ref()
        if obj is None:
            return False  # 弱引用已死亡
        if self._func is not None:
            # 绑定方法 或 partial（obj 是首参）
            if self._partial_args or self._partial_kwargs:
                # partial：obj 作为第一个参数
                self._func(obj, *self._partial_args, *args, **{**self._partial_kwargs, **kwargs})
            else:
                # 绑定方法
                self._func(obj, *args, **kwargs)
        else:
            # 普通函数
            obj(*args, **kwargs)
        return True


# ============================================================
# HookRegistry
# ============================================================

class HookRegistry:
    """事件注册表 - 所有扩展层的底座

    使用弱引用存储回调，避免插件持有 core 引用导致的循环引用。
    高频事件（每步 emit）中禁止阻塞 IO。
    """

    _callbacks: Dict[str, List[_WeakCallback]] = {}
    _strong_refs: Dict[str, List[Callable]] = {}  # 供需要强引用的场景

    @classmethod
    def register(cls, event: str, callback: Callable, weak: bool = True) -> None:
        """注册事件回调

        Args:
            event: 事件名，建议格式 "namespace:event_name"
            callback: 回调函数/方法
            weak: 是否使用弱引用（默认True，避免循环引用）
        """
        if event not in cls._callbacks:
            cls._callbacks[event] = []
            cls._strong_refs[event] = []

        if weak:
            cls._callbacks[event].append(_WeakCallback(callback))
        else:
            # 强引用场景：插件作者明确需要保持生命周期
            cls._strong_refs[event].append(callback)

    @classmethod
    def unregister(cls, event: str, callback: Callable) -> bool:
        """注销指定回调，返回是否成功找到并移除"""
        found = False
        # 清理弱引用列表中死亡的引用和匹配的回调
        if event in cls._callbacks:
            alive = []
            for wc in cls._callbacks[event]:
                # v5.1.7: 补充 partial/强引用兜底的匹配
                if wc._orig_callback is callback:
                    found = True
                    continue
                if wc._strong is callback:
                    found = True
                    continue
                # 检查是否为同一个回调
                if wc._ref is not None and wc._ref() is callback:
                    found = True
                    continue
                alive.append(wc)
            cls._callbacks[event] = alive

        # 清理强引用列表
        if event in cls._strong_refs:
            before = len(cls._strong_refs[event])
            cls._strong_refs[event] = [c for c in cls._strong_refs[event] if c is not callback]
            if len(cls._strong_refs[event]) < before:
                found = True

        return found

    @classmethod
    def emit(cls, event: str, *args, **kwargs) -> None:
        """触发事件，同步调用所有注册的回调

        每个回调独立包裹 try/except，单个插件异常不会打断主训练流程。
        自动清理已死亡的弱引用。
        """
        callbacks = cls._callbacks.get(event)
        if callbacks:
            alive = []
            for wc in callbacks:
                try:
                    ok = wc(*args, **kwargs)
                    if ok:
                        alive.append(wc)
                    # 若返回 False，引用已死亡，不加入 alive
                except Exception as e:
                    # 钩子异常不得打断主训练流程
                    _log_hook_error(event, e)
            cls._callbacks[event] = alive

        # 强引用回调
        strong_callbacks = cls._strong_refs.get(event)
        if strong_callbacks:
            for cb in strong_callbacks:
                try:
                    cb(*args, **kwargs)
                except Exception as e:
                    _log_hook_error(event, e)

    @classmethod
    def clear(cls, event: Union[str, None] = None) -> None:
        """清理事件注册，用于单测隔离防止状态污染

        Args:
            event: 若为 None，清空所有事件；否则清空指定事件
        """
        if event is None:
            cls._callbacks.clear()
            cls._strong_refs.clear()
        else:
            cls._callbacks.pop(event, None)
            cls._strong_refs.pop(event, None)

    @classmethod
    def count(cls, event: str) -> int:
        """返回指定事件的注册回调数量"""
        weak_count = len(cls._callbacks.get(event, []))
        strong_count = len(cls._strong_refs.get(event, []))
        return weak_count + strong_count

    @classmethod
    def list_events(cls) -> List[str]:
        """返回所有已注册的事件名称列表"""
        events = set(cls._callbacks.keys()) | set(cls._strong_refs.keys())
        return sorted(events)


# ============================================================
# 钩子错误日志
# ============================================================

_hook_error_log: List[Dict[str, Any]] = []
_MAX_HOOK_ERRORS = 100  # 防止内存无限增长


def _log_hook_error(event: str, exc: Exception) -> None:
    """记录钩子执行错误，不抛出异常"""
    global _hook_error_log
    entry = {"event": event, "exc_type": type(exc).__name__, "exc_msg": str(exc)}
    _hook_error_log.append(entry)
    # 截断防止无限增长
    if len(_hook_error_log) > _MAX_HOOK_ERRORS:
        _hook_error_log = _hook_error_log[-_MAX_HOOK_ERRORS:]
    # 静默记录，不干扰主流程


def get_hook_errors() -> List[Dict[str, Any]]:
    """获取最近的钩子错误记录（供调试使用）"""
    return list(_hook_error_log)


def clear_hook_errors() -> None:
    """清空钩子错误记录"""
    global _hook_error_log
    _hook_error_log.clear()
