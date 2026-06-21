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
            # 发射配置变更事件
            try:
                from sgn_hooks import HookRegistry
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
        ConfigItem("VECTOR_FORMULA", "line", str, None, "矢量公式类型(line/circle/sine/catear/mixed)", "高级选项", False),
        ConfigItem("VECTOR_GRID", 8, int, (4, 64), "矢量网格大小", "高级选项", False),
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
        ConfigItem("D", 16, int, (4, 1024), "输入维度", "网络架构", True),
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
        ConfigItem("ALLOW_LARGE_GRID_DRAW", False, bool, None, "允许大网格(>4x4)实时绘制", "界面偏好", False),
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
                   "图赫布学习融合率", "学习参数", False),

        # 持久化
        ConfigItem("GRAPH_PERSISTENCE", True, bool, None,
                   "持久化图结构到SQLite", "系统", False),

        ConfigItem("INFERENCE_LAYER", 2, int, (0, 8),
                   "推理时构建到的最高层级", "网络架构", False),

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
    from sgn_hooks import HookRegistry
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
CONFIG_FILE = "sgn_config.json"

# 标记参数是否被修改过（兼容旧变量名）
_config_modified = False


# ============================================================
# _DynamicD - 动态整数代理（必须在 CONFIG 赋值后定义）
# ============================================================

class _DynamicD:
    """动态整数代理 - 总是返回 CONFIG['D'] 的当前值

    解决模块常量 D = 16 与 CONFIG['D'] 修改后不同步的问题。
    用法不变: from sgn_config import D 仍然有效，D 在算术上下文中
    自动返回 CONFIG['D'] 的当前值。

    【v4.2 窗口大小识别】
    D 不再由用户手动配置，而是由 SGNCore 在首次 train() 时根据
    len(intensity) 自动推导。CONFIG['D'] 作为向后兼容的回退值。
    当输入源从 4×4 切换到 8×8 时，网络自动重建，无需手动改 D。
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


# =========================================================================
# [v4.2 硬件现实与调参指南] 基于实测数据的认知更新
#
# 1. 输入约束
#    本系统输入来自 AI8051U 的 4×4 按键矩阵（P1 口，16 键）。
#    POOLED_DIM = 16 是物理约束，不是算法压缩。
#
# 2. 识别率上限（实测更新）
#    旧认知（v4.1）: 默认 64N/640 步/100 模板 ≈ 62.5%，认为是"硬件极限"。
#    新认知（v4.2）: 256N/3000~10000 步/100~200 模板可达 80~87.5%。
#    结论：之前的 75% 是"小网络短步数"的极限，不是 4×4 像素的极限。
#    真正的瓶颈是：特征分辨率（神经元数）× 收敛时间（步数）× 种子运气。
#
# 3. 神经元数量的"相变阈值"
#    • 64N: 临界悬崖。Top-K=6 时参与率 9.4%，弱者快速被削弱锁定。
#      640 步后全活跃只是假象，再延长就会批量锁定。
#    • 200N: 正反馈循环。参与率 3%，竞争温和，3000 步仍全活跃。
#    • 256N: 稳态液态。参与率 2.3%，几乎不触发锁定，终身可塑。
#    结论：SGN 不是"小网络省资源"，而是"大网络才能活"。
#
# 4. 模板库的"内禀稳态容量"
#    实测：256N/5000 步膨胀到 124 个模板，10000 步收敛回 ~81 个。
#    机制：OR/AND 合并长期运行后起到"自清洁"作用——粗糙模板被压缩。
#    16 字符 × 4×4 像素 × 15% 复合噪声的有效特征子空间，
#    天然稳态约 80±20 个模板。上限 100 刚好够用，500 只是缓冲。
#
# 5. 随机种子的决定性影响
#    create_neuron() 中 rd = random.Random(SEED + nid)。
#    每个神经元的 4 层 16 位初始掩码完全由种子决定。
#    种子 35 和 42 对 0-9A-F 拓扑的覆盖能力可能天差地别。
#    结论：SEED 是核心隐藏变量，不同种子 = 完全不同的网络。
#
# 6. 训练步数不是越多越好
#    • 前 2000 步：快速膨胀期，模板大量入库
#    • 2000~5000 步：缓增期，合并开始生效
#    • 5000~10000 步：收敛期，模板数回稳，识别率提升有限
#    超过收敛点后继续训练，只是在"打磨已有模板"，边际效益递减。
#
# 7. SGN 的本质：噪声实例记忆
#    SGN 不是学"0 的抽象概念"，而是记"这个特定损坏模式像 0"。
#    模板是具体的 16 位掩码，不是分布式权重。
#    因此：对训练噪声过拟合是设计特性；对未见过的高噪声（0.3+）
#    识别率骤降是预期行为——那是不同分布，模板覆盖不到。
#
# 8. 推荐调参策略
#    • 神经元：200~256（64 绝对不够，再往上边际递减）
#    • 模板上限：100~150（稳态 80 个，留余量即可）
#    • 训练步数：3000~5000（1 万步后半段收益有限）
#    • 随机种子：需要网格搜索，35 比 42 明显更好
#    • 学习率/削弱率：若过拟合严重，可降低 LR、提高 WR
#
# 9. Python 版与 C 版的模板库行为差异
#    Python 验证器（本文件）在模板库满时选择"静默丢弃"并打印警告，
#    这是因为 PC 资源无限，Python 版仅作为算法正确性的沙箱验证器，
#    不需要终身运行。而 C 版（MCU 端）实现了完整的 hit_counter 衰减淘汰、
#    唯一模板保护、末位替换机制——因为单片机 RAM 有限，必须实现真正的
#    "遗忘与记忆"才能长期运行。两者差异是工程优先级选择，不是实现遗漏。
#
# 10. "先验机"的定位更新
#    v4.2 证明：无浮点、无反向传播、纯整数赫布学习的 SGN 架构
#    可以在 PC 端达到 80%+ 识别率。当前版本的"窗口太小"问题
#    可以通过增加神经元数量和训练步数大幅缓解，而非不可逾越。
#    算法核心（整数赫布学习 / 竞争机制）保持不变。
#   
#   时间量纲当前实现本质缺什么
# step  计数器
#只是训练轮次，不是物理时间
#响应速度竞争  base + match×γ 
#利用时间差做排序，不是时序编码
# history[]  记录每步结果日志数组不是记忆状态
#黑箱模式  time.sleep() 刻意消除时间干扰反而证明时间只是噪声
#这些都不是时间量纲，只是时间副作用。SGN 目前处理的是静态快照——每一帧输入之间没有因果、没有序列、没有"前一个状态影响后一个判断"。
#   不要把sgn当做一个原型机,目前连原型机都不是基础核心机制没完善只是先验机!!!
# 在传统各种神经网络中具有两种重要变量，空间变量与时间变量这两大类别，SGN目前仅实现其中之一，还远没有达到原型机的程度。
# =========================================================================

# 标签定义
LABELS = list("0123456789ABCDEF")

# =========================================================================
# [v4.2 时间量纲声明] 为什么当前版本不加入时间量纲
#
# 原因：空间变量本身尚未完善、不够稳定，现阶段集中优化空间变量。
#      非遗忘，乃有意推迟。待空间量纲（窗口大小自适应、DiscreteCoordinate
#      精度、矢量输入管道、跨维度一致性）打磨稳定后，再引入时序
#      编码、因果记忆、前状态影响后判断等时间量纲机制。
#
# 当前所有时间相关概念仅为副作用：
#   - step: 训练轮次计数器，不是物理时间
#   - 响应速度竞争: base + match×γ，利用数值排序，不是时序编码
#   - history[]: 每步结果日志数组，不是记忆状态
#   - 黑箱模式 time.sleep(): 刻意消除时间干扰，证明时间只是噪声
#
# 时间量纲若未来引入，必须：
#   1. 不破坏现有空间量纲接口
#   2. 作为新模块（如 sgn_temporal.py）独立实现
#   3. 通过 HookRegistry 接入，不侵入 sgn_core.py
#   4. 保持整数化原则
# =========================================================================


# 字符标准模板 (4×4 点阵，16元素 —— 当前维度锚定)
# 【窗口大小识别】
# 当前 PATTERNS 为 4×4（16元素）。若未来通过 FileInputSource 加载 8×8（64元素）
# 或 16×16（256元素）的外部数据，SGNCore 会在首样本时自动识别 D=len(intensity)，
# 重建神经元掩码位宽，无需修改此字典或新增配置项。
# 噪声总量随像素数等比放大（per-pixel 概率模型）。
PATTERNS = {
    '0': [255,255,255,255, 255,0,0,255, 255,0,0,255, 255,255,255,255],
    '1': [0,0,255,0, 0,255,255,0, 0,0,255,0, 0,255,255,255],
    '2': [255,255,255,255, 0,0,0,255, 255,255,255,0, 255,255,255,255],
    '3': [255,255,255,255, 0,0,0,255, 0,255,255,255, 255,255,255,255],
    '4': [255,0,0,255, 255,0,0,255, 255,255,255,255, 0,0,0,255],
    '5': [255,255,255,255, 255,0,0,0, 255,255,255,255, 0,0,255,255],
    '6': [255,255,255,255, 255,0,0,0, 255,255,255,255, 255,0,0,255],
    '7': [255,255,255,255, 0,0,0,255, 0,0,255,0, 0,255,0,0],
    '8': [255,255,255,255, 255,0,0,255, 255,255,255,255, 255,255,255,255],
    '9': [255,255,255,255, 255,0,0,255, 255,255,255,255, 0,0,255,255],
    'A': [0,255,255,0, 255,0,0,255, 255,255,255,255, 255,0,0,255],
    'B': [255,255,255,0, 255,0,0,255, 255,255,255,0, 255,255,255,0],
    'C': [0,255,255,255, 255,0,0,0, 255,0,0,0, 0,255,255,255],
    'D': [255,255,255,0, 255,0,0,255, 255,0,0,255, 255,255,255,0],
    'E': [255,255,255,255, 255,0,0,0, 255,255,255,0, 255,255,255,255],
    'F': [255,255,255,255, 255,0,0,0, 255,255,255,0, 255,0,0,0],
}


# ============================================================
# 8×8 标准字符库（0-9、A-Z，共 36 字符）
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
    """保存配置到JSON文件 - v5.0 兼容 API"""
    """保存配置到JSON文件 - v5.0 兼容 API（v4.3-fix：支持 DiscreteCoordinate 序列化）"""
    if path is None:
        path = os.path.normpath(CONFIG_FILE)
    try:
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
    """从JSON文件加载配置 - v5.0 兼容 API（兼容旧配置格式）"""
    """从JSON文件加载配置 - v5.0 兼容 API（v4.3-fix：支持 DiscreteCoordinate 反序列化）"""
    global _config_modified
    if path is None:
        path = os.path.normpath(CONFIG_FILE)
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
    """统一判断是否绘制网格：小网格直接绘制，大网格需配置开关"""
    return grid_size <= 4 or CONFIG.get("ALLOW_LARGE_GRID_DRAW", False)
