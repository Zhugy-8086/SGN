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
"""SGN-Lite v5.0 配置管理模块 - ConfigRegistry + 动态UI

阶段1重构：把硬编码的 PARAM_RANGES / DEFAULT_CONFIG / key_map 变成可运行时注册。
所有旧 API (CONFIG, DEFAULT_CONFIG, validate_param, set_config 等) 保持向后兼容。
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type


# ============================================================
# 项目根目录与配置文件路径（绝对路径，不依赖 cwd）
# ============================================================
# engine/config.py 的父目录的父目录即为项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 配置文件统一存放在 config/ 子目录下
CONFIG_DIR = os.path.join(_PROJECT_ROOT, "config")


# ============================================================
# DiscreteCoordinate - 离散坐标（浮点空间整数投影）
# ============================================================

class DiscreteCoordinate:
    """离散坐标：浮点连续空间一次性投影到整数格点，进入后永不还原

    核心哲学：
      - 浮点值进入系统时只做一次性投影，之后永不还原为 float
      - 运算只在同一层级（level）内进行整数算术
      - 跨层级运算必须先通过 coarse_to() / fine_to() 映射到同一层级
      - 对外只暴露 (level, index) 原始整数对，不暴露浮点还原值

    层级（level）定义：
      - level=0 → 整数空间（scale=1），格点间距 1.0
      - level=1 → 0.1 空间（scale=10），格点间距 0.1
      - level=2 → 0.01 空间（scale=100），格点间距 0.01
      - level=3 → 0.001 空间（scale=1000），格点间距 0.001
      - ...无限内缩

    索引（index）：
      - 在该层级下的整数坐标，可为正、负、零
      - 例如 level=2, index=2 → 物理位置 0.02（但系统内部永不还原）

    与彩色识别管线的同构：
      - 彩色识别：RGB 连续强度 → 布尔空间（0/1），运算为集合运算
      - DiscreteCoordinate：浮点连续值 → 整数格点空间，运算为整数算术
      - 两者都是投影系统，只是投影目标不同
    """

    def __init__(self, index: int, level: int = 2):
        """创建离散坐标

        Args:
            index: 该层级下的整数坐标
            level: 层级编号（0=整数空间, 1=0.1空间, 2=0.01空间...）
        """
        self.index = int(index)
        self.level = int(level)
        if self.level < 0:
            raise ValueError(f"level 必须 >= 0，得到 {level}")

    @classmethod
    def from_float(cls, f: float) -> "DiscreteCoordinate":
        """从浮点数一次性投影到离散坐标

        规则：
          - 0.02   → level=2, index=2  （2位小数）
          - 0.002  → level=3, index=2  （3位小数）
          - 0.0002 → level=4, index=2  （4位小数）
          - 2.0    → level=0, index=2  （0位小数）
          - 0.2    → level=1, index=2  （1位小数）

        注意：这是进入系统的唯一入口，投影后永不还原。
        """
        s = str(f)
        if '.' in s:
            decimal_part = s.split('.')[1].rstrip('0')
            digits = len(decimal_part) if decimal_part else 0
        else:
            digits = 0
        level = digits
        scale = 10 ** level
        index = int(round(f * scale))
        return cls(index, level)

    @property
    def scale(self) -> int:
        """当前层级的缩放因子（只读，用于兼容层计算）"""
        return 10 ** self.level

    # ---- 层级映射 ----
    def coarse_to(self, target_level: int) -> "DiscreteCoordinate":
        """粗化到更高层级（level 增大，精度降低）

        例如 level=2, index=2 → coarse_to(3) → level=3, index=20
        """
        if target_level < self.level:
            raise ValueError(f"粗化目标层级 {target_level} 必须 >= 当前层级 {self.level}")
        delta = target_level - self.level
        new_index = self.index * (10 ** delta)
        return DiscreteCoordinate(new_index, target_level)

    def fine_to(self, target_level: int) -> "DiscreteCoordinate":
        """细化到更低层级（level 减小，精度升高）

        例如 level=3, index=20 → fine_to(2) → level=2, index=2
        注意：细化会丢失低位信息（整数除法截断）
        """
        if target_level > self.level:
            raise ValueError(f"细化目标层级 {target_level} 必须 <= 当前层级 {self.level}")
        if target_level == self.level:
            return DiscreteCoordinate(self.index, self.level)
        delta = self.level - target_level
        new_index = self.index // (10 ** delta)
        return DiscreteCoordinate(new_index, target_level)

    def to_level(self, target_level: int) -> "DiscreteCoordinate":
        """映射到任意目标层级（自动判断粗化或细化）"""
        if target_level == self.level:
            return DiscreteCoordinate(self.index, self.level)
        elif target_level > self.level:
            return self.coarse_to(target_level)
        else:
            return self.fine_to(target_level)

    # ---- 同一层级运算（核心） ----
    def _ensure_same_level(self, other: "DiscreteCoordinate") -> tuple:
        """确保两个坐标在同一层级，禁止跨层级直接运算"""
        if not isinstance(other, DiscreteCoordinate):
            raise TypeError(f"运算对象必须是 DiscreteCoordinate，得到 {type(other)}")
        if self.level != other.level:
            raise TypeError(
                f"跨层级运算非法: {self} (level={self.level}) vs {other} (level={other.level}). "
                f"请先调用 .to_level({other.level}) 或 .to_level({self.level}) 统一层级"
            )
        return self.index, other.index, self.level

    def __add__(self, other: "DiscreteCoordinate") -> "DiscreteCoordinate":
        a, b, level = self._ensure_same_level(other)
        return DiscreteCoordinate(a + b, level)

    def __sub__(self, other: "DiscreteCoordinate") -> "DiscreteCoordinate":
        a, b, level = self._ensure_same_level(other)
        return DiscreteCoordinate(a - b, level)

    def __mul__(self, scalar: int) -> "DiscreteCoordinate":
        """与整数标量相乘（缩放）"""
        if not isinstance(scalar, int):
            raise TypeError(f"标量必须是 int，得到 {type(scalar)}")
        return DiscreteCoordinate(self.index * scalar, self.level)

    def __floordiv__(self, scalar: int) -> "DiscreteCoordinate":
        """与整数标量整除"""
        if not isinstance(scalar, int):
            raise TypeError(f"标量必须是 int，得到 {type(scalar)}")
        if scalar == 0:
            raise ZeroDivisionError("整除零")
        return DiscreteCoordinate(self.index // scalar, self.level)

    def __truediv__(self, scalar: int) -> "DiscreteCoordinate":
        """与整数标量真除（结果仍为 DiscreteCoordinate）"""
        if not isinstance(scalar, int):
            raise TypeError(f"标量必须是 int，得到 {type(scalar)}")
        if scalar == 0:
            raise ZeroDivisionError("除零")
        return DiscreteCoordinate(self.index // scalar, self.level)

    # ---- 比较运算（只比较 index，不还原 float） ----
    def _cmp_index(self, other: "DiscreteCoordinate") -> tuple:
        """返回统一层级后的两个 index"""
        if not isinstance(other, DiscreteCoordinate):
            raise TypeError(f"比较对象必须是 DiscreteCoordinate，得到 {type(other)}")
        target_level = max(self.level, other.level)
        a = self.to_level(target_level)
        b = other.to_level(target_level)
        return a.index, b.index

    def __eq__(self, other):
        if isinstance(other, DiscreteCoordinate):
            a, b = self._cmp_index(other)
            return a == b
        return False

    def __lt__(self, other):
        a, b = self._cmp_index(other)
        return a < b

    def __le__(self, other):
        a, b = self._cmp_index(other)
        return a <= b

    def __gt__(self, other):
        a, b = self._cmp_index(other)
        return a > b

    def __ge__(self, other):
        a, b = self._cmp_index(other)
        return a >= b

    def __hash__(self):
        # 【v4.3-fix】__hash__ 必须与 __eq__ 契约一致。
        # __eq__ 跨层级比较（映射到同一层级后比较 index），
        # 因此 __hash__ 也必须让物理等价的坐标具有相同哈希值。
        # 策略：规范化到最小表示层级（去掉末尾的0），然后哈希。
        # 例如 level=3,index=20 → 等价于 level=2,index=2 → 哈希基于 (2, 2)
        idx = self.index
        lvl = self.level
        while lvl > 0 and idx % 10 == 0:
            idx //= 10
            lvl -= 1
        return hash((lvl, idx))

    # ---- 表示 ----
    def __repr__(self):
        return f"DiscreteCoordinate(level={self.level}, index={self.index})"

    def __str__(self):
        # 对外显示原始整数对，不还原浮点
        return f"L{self.level}:I{self.index}"

    # ---- 序列化 ----
    def serialize(self) -> dict:
        return {"level": self.level, "index": self.index}

    @classmethod
    def deserialize(cls, d: dict) -> "DiscreteCoordinate":
        return cls(d.get("index", 0), d.get("level", 2))

    # ---- 兼容性：旧 DiscreteCoordinate 格式转换 ----
    @classmethod
    def from_fixedpoint_legacy(cls, value: int, scale: int) -> "DiscreteCoordinate":
        """从旧 DiscreteCoordinate (value, scale) 转换为 DiscreteCoordinate"""
        # scale = 10^level
        level = 0
        s = scale
        while s > 1:
            if s % 10 != 0:
                # 非 10 的幂次，向上取整到最近的 10 的幂次
                import math
                level = int(math.ceil(math.log10(scale)))
                break
            s //= 10
            level += 1
        return cls(value, level)


# ============================================================
# SGNConstants - 系统级常量
# ============================================================

class SGNConstants:
    """System-level constants - core engine only (input layer constants not included)

    Design principles:
      - Only include constants for core recognition path
      - Input layer (pixel max value, geometric calculations, etc.) keep definitions in respective files
      - All percentage bases unified to 100
    """
    # Percentage base
    PERCENT_BASE = 100

    # Position normalization factor (0~1000 mapped to grid coordinates)
    POSITION_NORM = 1000

    # Hit counter upper limit (uint8)
    MAX_HIT_COUNTER = 255

    # Hit counter increment for OR merge
    HIT_COUNTER_INC_OR = 32

    # Hit counter increment for AND merge
    HIT_COUNTER_INC_AND = 16

    # MinHash prime table
    MINHASH_PRIMES = [31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

    # Vector formula label mapping (v5.1 unified entry)
    VECTOR_LABELS = {
        "line": ["LINE"],
        "circle": ["CIRCLE"],
        "sine": ["SINE"],
        "arch": ["ARCH"],
        "leaf": ["LEAF"],
    }

    # Old name alias mapping (backward compatible)
    VECTOR_ALIASES = {
        "catear": "arch",
    }


# ============================================================
# ConfigItem - 配置项元数据
# ============================================================

@dataclass
class ConfigItem:
    """单个配置项的元数据描述

    Attributes:
        key: 配置键名，如 "MAX_NEURONS"
        default: 默认值（只允许不可变类型）
        val_type: 值类型（int/float/str/DiscreteCoordinate）
        range: (最小值, 最大值) 或 None
        description: 人类可读的描述文本
        category: 分类（"网络架构"/"学习参数"/"噪声参数"/"资源限制"/"系统"）
        requires_rebuild: 修改后是否需要重建 SGNCore
        is_discrete: 是否为离散坐标参数（自动精度推断）
    """
    key: str
    default: Any
    val_type: Type
    range: Optional[Tuple] = None
    description: str = ""
    category: str = "其他"
    requires_rebuild: bool = False
    is_discrete: bool = False


# ============================================================
# ConfigRegistry - 配置注册表
# ============================================================

class ConfigRegistry:
    """配置注册表 - 允许运行时注册新配置项

    所有新 Registry 在模块导入时自动注册默认实现，确保"开箱即用"行为不变。
    重复 key 抛异常中断启动（避免静默覆盖）。
    """

    _schema: Dict[str, ConfigItem] = {}
    _values: Dict[str, Any] = {}
    _config_modified: bool = False
    # 追踪被修改的架构参数（requires_rebuild=True 的键）
    # 用于 _maybe_rebuild_core 精确判断哪些参数需要触发重建
    _arch_modified_keys: set = set()

    @classmethod
    def register(cls, item: ConfigItem) -> None:
        """注册配置项，重复 key 抛 KeyError"""
        if item.key in cls._schema:
            raise KeyError(f"配置项 {item.key} 已存在，插件冲突")
        cls._schema[item.key] = item
        # 深拷贝防御：默认值只允许不可变类型
        cls._values[item.key] = item.default

    @classmethod
    def get(cls, key: str) -> Any:
        """获取配置值"""
        return cls._values.get(key, cls._schema.get(key, ConfigItem(key, None, str)).default)

    @classmethod
    def set(cls, key: str, value: Any) -> Tuple[bool, Optional[str]]:
        """安全设置配置项（带类型转换和范围校验 + 离散坐标自适应 + 双向同步）

        Returns:
            (成功, 消息/警告) 或 (失败, 错误消息)

        【v4.3 离散坐标】
        对于 is_discrete=True 的参数（如 LEARNING_RATE_X100）：
          - 用户输入 0.02 → 自动推断 scale=100, value=2
          - 用户输入 0.002 → 自动推断 scale=1000, value=2
          - 用户输入 2 → 向后兼容包装为 (2, 100)
        核心引擎读取 value 和 scale，不再硬编码 // 100。

        【v4.2 双向同步】
        浮点参数与离散坐标 _X100 参数自动同步：
          - 改 LEARNING_RATE=0.05 → 自动同步 LEARNING_RATE_X100=5
          - 改 LEARNING_RATE_X100=8 → 自动同步 LEARNING_RATE=0.08
        """
        if key not in cls._schema:
            return False, f"未知配置项: {key}"
        item = cls._schema[key]

        # 【v4.3 离散坐标解析】
        if item.val_type == DiscreteCoordinate:
            try:
                if isinstance(value, DiscreteCoordinate):
                    typed = value
                elif isinstance(value, int):
                    # 整数输入：视为 level=0（整数空间）
                    typed = DiscreteCoordinate(value, 0)
                elif isinstance(value, float):
                    typed = DiscreteCoordinate.from_float(value)
                elif isinstance(value, dict) and "index" in value and "level" in value:
                    # 从序列化格式恢复
                    typed = DiscreteCoordinate.deserialize(value)
                else:
                    typed = DiscreteCoordinate.from_float(float(str(value)))
            except (ValueError, TypeError):
                return False, f"类型错误: {key} 需要离散坐标（可输入浮点、整数或 L#:I# 格式）"
            # 范围校验：对浮点还原值校验（仅用于边界检查，不存储）
            if item.range is not None:
                lo, hi = item.range
                float_val = typed.index / typed.scale
                if float_val < lo or float_val > hi:
                    return False, f"{key}=L{typed.level}:I{typed.index} 超出范围 [{lo}, {hi}]"
        elif item.val_type == bool:
            try:
                if isinstance(value, bool):
                    typed = value
                elif isinstance(value, str):
                    typed = value.lower() in ("true", "1", "yes", "y", "on")
                elif isinstance(value, int):
                    typed = bool(value)
                else:
                    typed = bool(value)
            except (ValueError, TypeError):
                return False, f"类型错误: {key} 需要布尔值"
        elif item.val_type == list:
            try:
                if isinstance(value, list):
                    typed = value
                elif isinstance(value, str):
                    import json as _json
                    v = value.strip()
                    if v.startswith('[') and v.endswith(']'):
                        typed = _json.loads(v)
                    else:
                        typed = [x.strip() for x in v.split(",") if x.strip()]
                else:
                    typed = list(value)
            except (ValueError, TypeError):
                return False, f"类型错误: {key} 需要列表"
        else:
            try:
                typed = item.val_type(value)
            except (ValueError, TypeError):
                return False, f"类型错误: {key} 需要 {item.val_type.__name__}"
            if item.range is not None:
                lo, hi = item.range
                if typed < lo or typed > hi:
                    return False, f"{key}={typed} 超出范围 [{lo}, {hi}]"

        old = cls._values.get(key)
        cls._values[key] = typed

        if old != typed:
            cls._config_modified = True
            # 追踪架构参数修改（用于 _maybe_rebuild_core 精确重建检测）
            if item.requires_rebuild:
                cls._arch_modified_keys.add(key)
            # 发射配置变更事件
            try:
                from engine.hooks import HookRegistry
                HookRegistry.emit("sgn:on_config_changed", key=key, old=old, new=typed)
            except ImportError:
                pass
        return True, None

    @classmethod
    def get_schema(cls, key: str) -> Optional[ConfigItem]:
        """获取配置项元数据"""
        return cls._schema.get(key)

    @classmethod
    def is_architecture_param(cls, key: str) -> bool:
        """判断是否为架构参数（修改后需重建网络）"""
        item = cls._schema.get(key)
        return item.requires_rebuild if item else False

    @classmethod
    def get_arch_modified_keys(cls) -> set:
        """返回被修改的架构参数键集合（副本）"""
        return set(cls._arch_modified_keys)

    @classmethod
    def mark_arch_synced(cls) -> None:
        """标记架构参数已同步（重建完成后调用）"""
        cls._arch_modified_keys.clear()

    @classmethod
    def generate_menu(cls) -> Dict[str, Tuple[str, ConfigItem]]:
        """按 category 分组动态生成菜单，自动分配热键

        热键序列: 1-9, 0, a-z（跳过已被控制面板占用的 q/h/r/s/l/m）
        最多支持 30+ 个配置项，避免版本间的冲突覆盖问题。

        Returns:
            OrderedDict{"1": ("MAX_NEURONS", ConfigItem), ...}
        """
        from collections import OrderedDict
        # 按 category 排序分组
        categories_order = ["网络架构", "学习参数", "资源限制", "噪声参数", "系统"]
        grouped = {}
        for item in cls._schema.values():
            cat = item.category
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(item)

        # 热键池：数字 1-9, 0，然后字母 a-z（跳过控制面板已占用的系统命令热键）
        reserved = {"q", "h", "r", "s", "l", "m"}  # 控制面板系统命令
        hotkey_pool = [str(i) for i in range(1, 10)] + ["0"]
        hotkey_pool += [chr(i) for i in range(ord("a"), ord("z") + 1) if chr(i) not in reserved]

        menu = OrderedDict()
        hk_idx = 0
        # 优先按 categories_order 排序
        for cat in categories_order:
            if cat in grouped:
                for item in sorted(grouped[cat], key=lambda x: x.key):
                    if hk_idx < len(hotkey_pool):
                        menu[hotkey_pool[hk_idx]] = (item.key, item)
                        hk_idx += 1
        # 剩余未分类的
        for cat in sorted(grouped.keys()):
            if cat not in categories_order:
                for item in sorted(grouped[cat], key=lambda x: x.key):
                    if hk_idx < len(hotkey_pool):
                        menu[hotkey_pool[hk_idx]] = (item.key, item)
                        hk_idx += 1
        return menu

    @classmethod
    def is_modified(cls) -> bool:
        return cls._config_modified

    @classmethod
    def mark_synced(cls) -> None:
        cls._config_modified = False

    @classmethod
    def reset_all(cls) -> None:
        """恢复所有配置项到默认值"""
        for item in cls._schema.values():
            cls._values[item.key] = item.default
        cls._config_modified = False
        cls._arch_modified_keys.clear()

    @classmethod
    def list_all(cls) -> Dict[str, Any]:
        """返回所有当前配置值的副本"""
        return dict(cls._values)

    @classmethod
    def clear(cls) -> None:
        """清空所有注册项（用于单测隔离）"""
        cls._schema.clear()
        cls._values.clear()
        cls._config_modified = False


# ============================================================
# 预定义配置项注册（自动注册默认实现）
# ============================================================

_ARCH_PARAMS = {"MAX_NEURONS", "TOP_K", "MAX_LOCKOUT", "LAYER_MAX", "D"}

def _register_defaults():
    """在模块导入时自动注册所有默认配置项"""
    defaults = [
        ConfigItem("INPUT_SOURCE_TYPE", "pattern", str, None, "输入源类型(pattern/vector/file)", "高级选项", False),
        ConfigItem("VECTOR_FORMULA", "line", str, None, "矢量公式类型(line/circle/sine/arch/leaf/mixed)", "高级选项", False),
        ConfigItem("VECTOR_GRID", 8, int, (8, 64), "矢量网格大小", "高级选项", False),
        ConfigItem("DATASET_PATH", "", str, None, "数据集文件路径", "高级选项", False),
        ConfigItem("MAX_NEURONS", 256, int, (1, 4096), "神经元数量", "网络架构", True),
        ConfigItem("TOP_K", 6, int, (1, 128), "竞争Top-K", "网络架构", True),
        ConfigItem("MAX_LOCKOUT", 120, int, (1, 1000), "最大锁定阈值", "网络架构", True),
        ConfigItem("LEARNING_RATE", DiscreteCoordinate(2, 2), DiscreteCoordinate, (0.001, 1.0), "学习率", "学习参数", False),
        ConfigItem("WEAKEN_RATE", DiscreteCoordinate(1, 2), DiscreteCoordinate, (0.001, 1.0), "削弱率", "学习参数", False),
        ConfigItem("GAMMA", DiscreteCoordinate(30, 2), DiscreteCoordinate, (0.0001, 2.0), "增益", "学习参数", False),
        ConfigItem("MAX_TEMPLATES", 500, int, (1, 10000), "模板库上限", "资源限制", False),
        ConfigItem("MAX_ITERATIONS", 100000, int, (1, 1000000), "训练总步数", "资源限制", False),
        ConfigItem("SEED", 42, int, (0, 99999), "随机种子", "资源限制", True),
        ConfigItem("FLIP_PROB", 0.1, float, (0.0, 1.0), "噪声翻转概率", "噪声参数", False),
        ConfigItem("NOISE_TYPE", "composite", str, None, "噪声类型(composite/gaussian/salt_pepper/block)", "噪声参数", False),
        ConfigItem("NOISE_SIGMA", 32.0, float, (0.0, 255.0), "高斯噪声标准差", "噪声参数", False),
        ConfigItem("NOISE_BLOCK_SIZE", 2, int, (1, 8), "块遮挡噪声块大小", "噪声参数", False),
        ConfigItem("SPARSE_STEP", 0, int, (0, 1000), "跳步显示间隔(0=关闭)", "界面偏好", False),
        ConfigItem("COMPACT_INTERVAL", 100, int, (10, 10000), "精简模式输出间隔", "界面偏好", False),
        ConfigItem("MODE", "full", str, None, "运行模式(full/compact/blackbox)", "界面偏好", False),
        ConfigItem("LAYER_MAX", 4, int, (1, 8), "最大层数", "网络架构", True),
        ConfigItem("D", 64, int, (8, 1024), "输入维度", "网络架构", True),
        ConfigItem("ENCOURAGE_CNT", 5, int, (0, 20), "鼓励计数", "学习参数", False),
        ConfigItem("ENCOURAGE_BONUS", DiscreteCoordinate(10, 2), DiscreteCoordinate, (0, 2.0), "鼓励奖励", "学习参数", False),
        ConfigItem("SPEED_SAT", DiscreteCoordinate(120, 2), DiscreteCoordinate, (0.001, 5.0), "速度饱和值", "学习参数", False),
        ConfigItem("BASE_INIT", DiscreteCoordinate(50, 2), DiscreteCoordinate, (0.001, 2.0), "初始基础速度", "学习参数", False),
        ConfigItem("HIT_COUNTER_INIT", 128, int, (1, 512), "命中计数器初始值", "学习参数", False),
        ConfigItem("MIN_BASE", DiscreteCoordinate(8, 2), DiscreteCoordinate, (0.001, 1.0), "最小基础速度", "学习参数", False),
        ConfigItem("INTENSITY_DIFF_THRESH", 1, int, (0, 255), "强度差异阈值", "学习参数", False),
        ConfigItem("MIN_MARKED_CNT", 2, int, (1, 16), "最小标记计数", "学习参数", False),
        ConfigItem("OR_THRESH", 85, int, (0, 100), "OR合并阈值(%)", "学习参数", False),
        ConfigItem("AND_THRESH", 80, int, (0, 100), "AND合并阈值(%)", "学习参数", False),
        ConfigItem("AUTO_DELAY_MS", 50, int, (1, 10000), "自动模式延时(ms)", "界面偏好", False),
        ConfigItem("COLOR_OUTPUT", True, bool, None, "启用彩色输出", "界面偏好", False),
        ConfigItem("CHART_BACKEND", "auto", str, None, "图表后端(auto/matplotlib/ascii/csv)", "界面偏好", False),
        ConfigItem("STORAGE_BACKEND", "json", str, None, "存储后端(json/sqlite)", "界面偏好", False),
        ConfigItem("AUTOSAVE_STRATEGY", "interval", str, None, "自动保存策略(interval/delta)", "界面偏好", False),
        ConfigItem("ALLOW_LARGE_GRID_DRAW", False, bool, None, "允许大网格(>8x8)实时绘制", "界面偏好", False),
        ConfigItem("COLOR_SCHEME", "green", str, None, "网格高亮颜色(green/cyan/yellow/white)", "界面偏好", False),
        ConfigItem("ENABLED_METRICS", ["accuracy", "confusion", "noise_robustness"], list, None, "启用的评估指标", "界面偏好", False),
        ConfigItem("TEST_SPLIT", 0.2, float, (0.0, 0.5), "测试集留出比例", "评估", False),
        ConfigItem("VALIDATION_LABELS", [], list, None, "不参与训练的字符列表", "评估", False),
        ConfigItem("NOISE_TEST_TYPE", "composite", str, None, "噪声测试类型(composite/gaussian/salt_pepper/block)", "评估", False),
        ConfigItem("CROSS_VALIDATE_FOLDS", 0, int, (0, 16), "留一字符交叉验证折数(0=关闭)", "评估", False),

        # 双图叠加门控识别架构（jnn.md v4.0）
        ConfigItem("ENABLE_GATE_MATCHING", False, bool, None, "启用门控匹配(双图叠加)", "高级选项", False),
        ConfigItem("HIST_BUFFER_SIZE", 10, int, (1, 100), "历史样本缓冲区大小", "资源限制", False),
        ConfigItem("GATE_HIGH_THRESH", 40, int, (0, 100), "高层匹配阈值(边缘)", "学习参数", False),
        ConfigItem("GATE_LOW_THRESH", 30, int, (0, 100), "低层匹配阈值(像素)", "学习参数", False),
        ConfigItem("PATCH_SIZE", 2, int, (1, 4), "分块边长", "网络架构", False),
        ConfigItem("MAX_ATOMS", 200, int, (10, 10000), "原子字典总容量上限", "资源限制", False),

        # ============================================================
        # v5.0 图模式新增配置
        # ============================================================

        # 主开关
        ConfigItem("ENABLE_GRAPH_MODE", False, bool, None,
                   "启用图模式（多图并行+层级递进）", "高级选项", False),

        # 架构参数
        ConfigItem("STACK_DEPTH", 0, int, (0, 8),
                   "图模式层级深度（0=自动计算）", "网络架构", False),

        ConfigItem("PARALLEL_VIEWS", 3, int, (1, 8),
                   "多图并行数量（数据增强视图数）", "网络架构", False),

        ConfigItem("MAX_NODES_PER_LAYER", 50, int, (1, 200),
                   "每层最大节点数（防爆炸）", "资源限制", False),

        ConfigItem("MAX_TOTAL_NODES", 5000, int, (100, 50000),
                   "图模式总节点数硬上限（熔断）", "资源限制", False),

        # 反馈循环
        ConfigItem("MAX_FEEDBACK_LOOPS", 3, int, (0, 10),
                   "反馈迭代最大次数（0=关闭）", "学习参数", False),

        ConfigItem("FEEDBACK_THRESHOLD", 85, int, (0, 100),
                   "触发反馈的全局匹配度下限", "学习参数", False),

        # 层级下压遗忘
        ConfigItem("DEMOTION_THRESHOLD", 5, int, (1, 50),
                   "节点下压的激活阈值（低于此值下沉）", "学习参数", False),

        ConfigItem("LAYER_COVER_THRESHOLD", 1000, int, (100, 10000),
                   "L0节点被覆盖前的最大存活步数", "学习参数", False),

        # 图匹配
        ConfigItem("GRAPH_SIMILARITY_THRESHOLD", 80, int, (0, 100),
                   "图节点合并相似度阈值(%)", "学习参数", False),

        ConfigItem("GRAPH_LEARNING_RATE", 0.3, float, (0.0, 1.0),
                   "图赫布学习融合率(0.0~1.0, 内部转换为整数百分比)", "学习参数", False),

        # 持久化
        ConfigItem("GRAPH_PERSISTENCE", True, bool, None,
                   "持久化图结构到SQLite", "系统", False),

        ConfigItem("INFERENCE_LAYER", 2, int, (0, 8),
                   "推理时构建到的最高层级", "网络架构", False),

        # ============================================================
        # v5.1 多层神经元架构配置
        # ============================================================

        # 软门控
        ConfigItem("ENABLE_SOFT_GATE", False, bool, None,
                   "启用软门控（替代硬锁 n[L]）", "高级选项", False),
        ConfigItem("GATE_DECAY_RATE", 1, int, (1, 10),
                   "门控衰减步长（每步变化量）", "学习参数", False),
        ConfigItem("GATE_SPECIALIZE_THRESHOLD", 10, int, (3, 50),
                   "专精化触发阈值（连续验证通过次数）", "学习参数", False),

        # 标签专精分片
        ConfigItem("ENABLE_NEURON_SPECIALIZATION", False, bool, None,
                   "启用标签专精分片（路径分割）", "高级选项", False),
        ConfigItem("SPECIALIZATION_SPLIT_THRESHOLD", 20, int, (5, 100),
                   "专精分片触发阈值（连续通过次数）", "学习参数", False),

        # 多层神经元
        ConfigItem("ENABLE_MULTI_LAYER_NEURON", False, bool, None,
                   "启用多层神经元架构", "高级选项", True),
        ConfigItem("NEURON_LAYER_0_COUNT", 128, int, (16, 512),
                   "Layer 0 神经元数（基础特征专家）", "网络架构", True),
        ConfigItem("NEURON_LAYER_1_COUNT", 64, int, (8, 256),
                   "Layer 1 神经元数（概念判断专家）", "网络架构", True),
        ConfigItem("LAYER1_LEARNING_RATE_SCALE", 0.5, float, (0.1, 1.0),
                   "Layer 1 学习率缩放（相对 Layer 0）", "学习参数", False),
        ConfigItem("TOP_K_L1", 4, int, (1, 32),
                   "Layer 1 竞争 Top-K", "网络架构", False),

    ]
    for item in defaults:
        try:
            ConfigRegistry.register(item)
        except KeyError:
            # 已存在时更新元数据（支持热更新范围等属性）
            existing = ConfigRegistry._schema[item.key]
            existing.range = item.range
            existing.description = item.description
            existing.category = item.category
            existing.requires_rebuild = item.requires_rebuild
            existing.is_discrete = item.is_discrete


_register_defaults()


# ============================================================
# VECTOR_GRID → D 自动同步钩子
# ============================================================

def _sync_d_on_vector_grid_change(key, old, new):
    """修改 VECTOR_GRID 时自动同步 D = VECTOR_GRID^2"""
    if key == "VECTOR_GRID" and isinstance(new, int) and new > 0:
        ConfigRegistry._values["D"] = new * new

try:
    from engine.hooks import HookRegistry
    HookRegistry.register("sgn:on_config_changed", _sync_d_on_vector_grid_change, weak=False)
except ImportError:
    pass


# ============================================================
# v5.0 兼容层 - 保持旧 API 不变
# ============================================================

# 运行时配置字典 - 与 ConfigRegistry 双向同步
CONFIG: Dict[str, Any] = ConfigRegistry._values
DEFAULT_CONFIG: Dict[str, Any] = {
    item.key: item.default for item in ConfigRegistry._schema.values()
}
PARAM_RANGES: Dict[str, Tuple] = {
    item.key: item.range for item in ConfigRegistry._schema.values() if item.range is not None
}
# 配置文件路径（绝对路径，指向 config/ 子目录，不依赖运行时 cwd）
CONFIG_FILE = os.path.join(CONFIG_DIR, "sgn_config.json")

# 标记参数是否被修改过（兼容旧变量名）
_config_modified = False


# ============================================================
# _DynamicD - 动态整数代理（必须在 CONFIG 赋值后定义）
# ============================================================

class _DynamicD:
    """动态整数代理 - 总是返回 CONFIG['D'] 的当前值

    解决模块常量 D = 16 与 CONFIG['D'] 修改后不同步的问题。
    用法不变: from engine.config import D 仍然有效，D 在算术上下文中
    自动返回 CONFIG['D'] 的当前值。

    【v4.2 窗口大小识别】
    D 不再由用户手动配置，而是由 SGNCore 在首次 train() 时根据
    len(intensity) 自动推导。CONFIG['D'] 作为向后兼容的回退值。
    当输入源从 8×8 切换到 16×16 时，网络自动重建，无需手动改 D。
    """
    def __int__(self):
        return CONFIG["D"]
    def __index__(self):
        return CONFIG["D"]
    def __eq__(self, other):
        return CONFIG["D"] == other
    def __ne__(self, other):
        return CONFIG["D"] != other
    def __lt__(self, other):
        return CONFIG["D"] < other
    def __le__(self, other):
        return CONFIG["D"] <= other
    def __gt__(self, other):
        return CONFIG["D"] > other
    def __ge__(self, other):
        return CONFIG["D"] >= other
    def __add__(self, other):
        return CONFIG["D"] + other
    def __radd__(self, other):
        return other + CONFIG["D"]
    def __sub__(self, other):
        return CONFIG["D"] - other
    def __rsub__(self, other):
        return other - CONFIG["D"]
    def __mul__(self, other):
        return CONFIG["D"] * other
    def __rmul__(self, other):
        return other * CONFIG["D"]
    def __floordiv__(self, other):
        return CONFIG["D"] // other
    def __rfloordiv__(self, other):
        return other // CONFIG["D"]
    def __truediv__(self, other):
        return CONFIG["D"] / other
    def __rtruediv__(self, other):
        return other / CONFIG["D"]
    def __repr__(self):
        return f"DynamicD({CONFIG['D']})"
    def __str__(self):
        return str(CONFIG["D"])
    def __hash__(self):
        return hash(CONFIG["D"])


# 替换占位符 D = 16
D = _DynamicD()  # 动态获取 CONFIG["D"]，解决同步问题


# ============================================================
# 8×8 标准字符库（0-9、A-Z，共 36 字符）
# 位图来源：经典 8×8 点阵字体，每字符 8 字节，每字节 8 位
# ============================================================

def _bitmap_to_intensity(rows):
    """将 8 个字节（每字节 8 位）转为 64 元素的 0/255 列表"""
    result = []
    for byte in rows:
        for bit in range(7, -1, -1):
            result.append(255 if (byte >> bit) & 1 else 0)
    return result

STANDARD_CHARS_8x8 = {
    '0': _bitmap_to_intensity([0x3C, 0x66, 0x6E, 0x76, 0x66, 0x66, 0x3C, 0x00]),
    '1': _bitmap_to_intensity([0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00]),
    '2': _bitmap_to_intensity([0x3C, 0x66, 0x06, 0x0C, 0x18, 0x30, 0x7E, 0x00]),
    '3': _bitmap_to_intensity([0x3C, 0x66, 0x06, 0x1C, 0x06, 0x66, 0x3C, 0x00]),
    '4': _bitmap_to_intensity([0x0C, 0x1C, 0x3C, 0x6C, 0x7E, 0x0C, 0x0C, 0x00]),
    '5': _bitmap_to_intensity([0x7E, 0x60, 0x7C, 0x06, 0x06, 0x66, 0x3C, 0x00]),
    '6': _bitmap_to_intensity([0x1C, 0x30, 0x60, 0x7C, 0x66, 0x66, 0x3C, 0x00]),
    '7': _bitmap_to_intensity([0x7E, 0x06, 0x0C, 0x18, 0x18, 0x18, 0x18, 0x00]),
    '8': _bitmap_to_intensity([0x3C, 0x66, 0x66, 0x3C, 0x66, 0x66, 0x3C, 0x00]),
    '9': _bitmap_to_intensity([0x3C, 0x66, 0x66, 0x3E, 0x06, 0x0C, 0x38, 0x00]),
    'A': _bitmap_to_intensity([0x18, 0x3C, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x00]),
    'B': _bitmap_to_intensity([0x7C, 0x66, 0x66, 0x7C, 0x66, 0x66, 0x7C, 0x00]),
    'C': _bitmap_to_intensity([0x3C, 0x66, 0x60, 0x60, 0x60, 0x66, 0x3C, 0x00]),
    'D': _bitmap_to_intensity([0x78, 0x6C, 0x66, 0x66, 0x66, 0x6C, 0x78, 0x00]),
    'E': _bitmap_to_intensity([0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x7E, 0x00]),
    'F': _bitmap_to_intensity([0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x00]),
    'G': _bitmap_to_intensity([0x3C, 0x66, 0x60, 0x6E, 0x66, 0x66, 0x3E, 0x00]),
    'H': _bitmap_to_intensity([0x66, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x00]),
    'I': _bitmap_to_intensity([0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00]),
    'J': _bitmap_to_intensity([0x3E, 0x06, 0x06, 0x06, 0x06, 0x66, 0x3C, 0x00]),
    'K': _bitmap_to_intensity([0x66, 0x6C, 0x78, 0x70, 0x78, 0x6C, 0x66, 0x00]),
    'L': _bitmap_to_intensity([0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7E, 0x00]),
    'M': _bitmap_to_intensity([0x63, 0x77, 0x7F, 0x6B, 0x63, 0x63, 0x63, 0x00]),
    'N': _bitmap_to_intensity([0x66, 0x76, 0x7E, 0x7E, 0x6E, 0x66, 0x66, 0x00]),
    'O': _bitmap_to_intensity([0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00]),
    'P': _bitmap_to_intensity([0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60, 0x60, 0x00]),
    'Q': _bitmap_to_intensity([0x3C, 0x66, 0x66, 0x66, 0x6E, 0x3C, 0x06, 0x00]),
    'R': _bitmap_to_intensity([0x7C, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0x66, 0x00]),
    'S': _bitmap_to_intensity([0x3C, 0x66, 0x60, 0x3C, 0x06, 0x66, 0x3C, 0x00]),
    'T': _bitmap_to_intensity([0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00]),
    'U': _bitmap_to_intensity([0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00]),
    'V': _bitmap_to_intensity([0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x00]),
    'W': _bitmap_to_intensity([0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00]),
    'X': _bitmap_to_intensity([0x66, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x66, 0x00]),
    'Y': _bitmap_to_intensity([0x66, 0x66, 0x66, 0x3C, 0x18, 0x18, 0x18, 0x00]),
    'Z': _bitmap_to_intensity([0x7E, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x7E, 0x00]),
}


# ============================================================
# CharRegistry — 字符集插件注册表
# ============================================================
# 设计目标：
#   - 彻底移除 4×4 硬编码，统一为插件化 8×8 字符集
#   - 支持运行时注册新字符集（如手写体、特殊符号集等）
#   - 默认注册 "standard_8x8"（0-9, A-Z，共 36 字符）
#   - 向后兼容：LABELS / PATTERNS 从默认插件派生
# ============================================================

class CharRegistry:
    """字符集插件注册表 — 允许运行时注册和管理字符模板集

    使用方式:
        # 注册新字符集
        CharRegistry.register("my_set", 16, {"X": [...]})

        # 获取默认字符集的标签和模板
        labels = CharRegistry.get_labels()
        patterns = CharRegistry.get_patterns()

        # 获取单个字符的 intensity
        intensity = CharRegistry.get_char("A", grid_size=8)

    向后兼容:
        LABELS 和 PATTERNS 模块级变量从默认插件 "standard_8x8" 派生，
        确保旧代码 `from engine.config import LABELS, PATTERNS` 无需修改。
    """

    # 注册表内部存储: name -> {"grid_size": int, "patterns": Dict[str, List[int]]}
    _registry: Dict[str, dict] = {}
    # 默认字符集名称
    _default_name: str = "standard_8x8"

    @classmethod
    def register(cls, name: str, grid_size: int,
                 patterns: Dict[str, List[int]]) -> None:
        """注册一个字符集插件

        Args:
            name: 字符集名称（唯一标识，重复注册抛 KeyError）
            grid_size: 字符网格边长（如 8 表示 8×8）
            patterns: 字符 -> intensity 列表的映射

        Raises:
            KeyError: 名称已存在
        """
        if name in cls._registry:
            raise KeyError(f"字符集 {name} 已注册，插件冲突")
        # 深拷贝 patterns 防止外部修改影响注册表
        cls._registry[name] = {
            "grid_size": grid_size,
            "patterns": {k: list(v) for k, v in patterns.items()},
        }

    @classmethod
    def unregister(cls, name: str) -> bool:
        """注销字符集插件，返回是否成功"""
        return cls._registry.pop(name, None) is not None

    @classmethod
    def get(cls, name: str = None) -> dict:
        """获取字符集信息 {"grid_size": int, "patterns": dict}"""
        name = name or cls._default_name
        if name not in cls._registry:
            raise KeyError(f"字符集 {name} 未注册")
        return cls._registry[name]

    @classmethod
    def get_patterns(cls, name: str = None) -> Dict[str, List[int]]:
        """获取字符集的 patterns 字典"""
        return cls.get(name)["patterns"]

    @classmethod
    def get_labels(cls, name: str = None) -> List[str]:
        """获取字符集的标签列表（排序后）"""
        return sorted(cls.get_patterns(name).keys())

    @classmethod
    def get_grid_size(cls, name: str = None) -> int:
        """获取字符集的网格大小"""
        return cls.get(name)["grid_size"]

    @classmethod
    def get_char(cls, char: str, grid_size: int = 8,
                 name: str = None) -> Optional[List[int]]:
        """获取指定字符在目标网格大小下的 intensity

        查找顺序:
          1. 若目标 grid_size 与注册集的 grid_size 一致，直接返回副本
          2. 若不一致，使用最近邻缩放适配

        Returns:
            intensity 列表，若字符不存在返回 None
        """
        patterns = cls.get_patterns(name)
        char = char.upper() if len(char) == 1 else char
        if char not in patterns:
            return None
        src_gs = cls.get_grid_size(name)
        base = patterns[char]
        if grid_size == src_gs:
            return list(base)
        # 网格大小不匹配时使用最近邻缩放
        return _scale_intensity(base, src_gs, grid_size)

    @classmethod
    def list_sets(cls) -> List[str]:
        """列出所有已注册的字符集名称"""
        return list(cls._registry.keys())

    @classmethod
    def set_default(cls, name: str) -> None:
        """设置默认字符集名称"""
        if name not in cls._registry:
            raise KeyError(f"字符集 {name} 未注册，无法设为默认")
        cls._default_name = name

    @classmethod
    def clear(cls) -> None:
        """清空所有注册项（用于单测隔离）"""
        cls._registry.clear()
        cls._default_name = "standard_8x8"


def _scale_intensity(intensity: List[int], src_gs: int,
                     dst_gs: int) -> List[int]:
    """最近邻缩放 intensity 列表到目标网格大小

    Args:
        intensity: 源 intensity 列表（长度 = src_gs * src_gs）
        src_gs: 源网格边长
        dst_gs: 目标网格边长

    Returns:
        缩放后的 intensity 列表（长度 = dst_gs * dst_gs）
    """
    result = [0] * (dst_gs * dst_gs)
    ratio = src_gs / dst_gs
    for y in range(dst_gs):
        for x in range(dst_gs):
            sx = int(x * ratio)
            sy = int(y * ratio)
            result[y * dst_gs + x] = intensity[sy * src_gs + sx]
    return result


# ============================================================
# 注册默认字符集：8×8 标准字符（0-9, A-Z，共 36 字符）
# ============================================================
CharRegistry.register("standard_8x8", 8, STANDARD_CHARS_8x8)

# 向后兼容层：LABELS 和 PATTERNS 从默认插件派生
# 旧代码 `from engine.config import LABELS, PATTERNS` 无需修改
LABELS = CharRegistry.get_labels()
PATTERNS = CharRegistry.get_patterns()


def validate_param(key, value):
    """参数范围校验，返回 (是否合法, 提示消息) - v5.0 兼容 API（支持 DiscreteCoordinate）"""
    if key not in PARAM_RANGES and key not in ConfigRegistry._schema:
        return True, None
    ok, msg = ConfigRegistry.set(key, value)
    if not ok:
        return False, msg
    return True, msg


def set_config(key, value):
    """安全设置配置项（带校验）- v5.0 兼容 API（支持 DiscreteCoordinate）"""
    global _config_modified
    ok, msg = ConfigRegistry.set(key, value)
    if not ok:
        return False, msg
    _config_modified = ConfigRegistry.is_modified()
    return True, msg


def is_config_modified():
    """v5.0 兼容 API"""
    return ConfigRegistry.is_modified()


def mark_config_synced():
    """v5.0 兼容 API"""
    global _config_modified
    ConfigRegistry.mark_synced()
    _config_modified = False


def save_config(path=None):
    """保存配置到JSON文件 - v5.0 兼容 API（v4.3-fix：支持 DiscreteCoordinate 序列化）

    Args:
        path: 自定义保存路径，为 None 时使用 CONFIG_FILE（config/sgn_config.json）

    Returns:
        (success: bool, message: str) 元组
    """
    if path is None:
        path = CONFIG_FILE
    try:
        # 确保目标目录存在（config/ 子目录可能不存在）
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 使用 ConfigRegistry 的当前值，递归序列化 DiscreteCoordinate 等对象
        raw = ConfigRegistry.list_all()
        data = {}
        for k, v in raw.items():
            if hasattr(v, "serialize"):
                data[k] = v.serialize()
            else:
                data[k] = v
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True, path
    except Exception as e:
        return False, str(e)
def load_config(path=None):
    """从JSON文件加载配置 - v5.0 兼容 API（v4.3-fix：支持 DiscreteCoordinate 反序列化）

    Args:
        path: 自定义加载路径，为 None 时使用 CONFIG_FILE（config/sgn_config.json）

    Returns:
        (success: bool, message: str) 元组
    """
    global _config_modified
    if path is None:
        path = CONFIG_FILE
    if not os.path.exists(path):
        return False, f"文件不存在: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        unknown_keys = []
        for k, v in loaded.items():
            # 【v4.3-fix】若值为 DiscreteCoordinate 序列化字典，先反序列化
            if isinstance(v, dict) and "index" in v and "level" in v:
                v = DiscreteCoordinate.deserialize(v)
            if k in ConfigRegistry._schema:
                ok, _ = ConfigRegistry.set(k, v)
                if not ok:
                    # 类型/范围不匹配时尝试直接赋值（兼容旧格式）
                    ConfigRegistry._values[k] = v
            else:
                unknown_keys.append(k)
        _config_modified = True
        if unknown_keys:
            return True, f"{path} (警告: 忽略未知键 {unknown_keys})"
        return True, path
    except Exception as e:
        return False, str(e)
def reset_config():
    """恢复默认配置 - v5.0 兼容 API"""
    global _config_modified
    ConfigRegistry.reset_all()
    _config_modified = False


def should_draw_grid(grid_size: int) -> bool:
    """统一判断是否绘制网格：8×8 及以下直接绘制，更大网格需配置开关

    v5.1.5 更新：最小网格从 4×4 提升到 8×8，4×4 硬编码已移除。
    """
    return grid_size <= 8 or CONFIG.get("ALLOW_LARGE_GRID_DRAW", False)
