#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
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
"""MSInt ABI 第二层：存储后端

实现三种存储后端：
  - SlotBackend: 多槽位后端，每个槽位一个 Python int（默认，少量值场景）
  - PackedBackend: 打包后端，所有槽位打包进一个 int（大量小值场景）
  - LazyBackend: 惰性后端，不存储，按需调用计算函数（值可推导场景）

所有后端实现同一套 StorageBackendProtocol 协议。

设计原则：
  - 所有运算用纯整数位移+掩码，不引入浮点
  - 后端不可变（set 方法抛异常）的后端应明确标识

详见: 内部档案
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .protocol import SlotInfo, SlotValidationError


# ============================================================
# 存储后端抽象基类
# ============================================================

class StorageBackend:
    """存储后端抽象基类

    所有存储后端必须继承此类并实现所有抽象方法。

    Class attributes:
        name: 后端名（用于序列化和 describe()）
        immutable: 是否不可变（True 表示 set 方法会抛异常）
    """
    name: str = "base"
    immutable: bool = False

    def __init__(self, slot_table: List[SlotInfo]):
        """初始化后端

        Args:
            slot_table: 槽位定义列表
        """
        self._slot_table = list(slot_table)
        self._slot_indices: Dict[str, int] = {
            s.name: i for i, s in enumerate(self._slot_table)
        }

    # ---- 基本接口 ----
    def get(self, index: int) -> int:
        """读取第 index 个槽位"""
        raise NotImplementedError

    def set(self, index: int, value: int) -> None:
        """写入第 index 个槽位"""
        raise NotImplementedError

    def get_all(self) -> List[int]:
        """读取所有槽位"""
        return [self.get(i) for i in range(len(self._slot_table))]

    def slot_count(self) -> int:
        """槽位数量"""
        return len(self._slot_table)

    @property
    def slot_table(self) -> List[SlotInfo]:
        """槽位定义列表（副本）"""
        return list(self._slot_table)

    def slot_index(self, name: str) -> int:
        """根据槽位名查找索引

        Raises:
            KeyError: 槽位名不存在
        """
        if name not in self._slot_indices:
            raise KeyError(f"槽位 '{name}' 不存在")
        return self._slot_indices[name]

    # ---- 序列化 ----
    def serialize(self) -> Dict[str, Any]:
        """序列化后端状态"""
        raise NotImplementedError

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "StorageBackend":
        """反序列化后端状态"""
        raise NotImplementedError


# ============================================================
# SlotBackend: 多槽位后端
# ============================================================

class SlotBackend(StorageBackend):
    """多槽位后端

    每个槽位存储为一个独立的 Python int。
    适用于：少量动态值，多槽位频繁修改场景。

    内存特点：每槽位占一个 Python int 对象（约 28 字节）。
    """
    name = "slot"
    immutable = False

    def __init__(self, slot_table: List[SlotInfo],
                 values: List[int] = None) -> None:
        """初始化

        Args:
            slot_table: 槽位定义列表
            values: 初始值列表（与 slot_table 等长，缺省全 0）

        Raises:
            ValueError: values 长度不匹配或值超范围
        """
        super().__init__(slot_table)
        if values is None:
            values = [0] * len(slot_table)
        if len(values) != len(slot_table):
            raise ValueError(
                f"values 长度({len(values)}) 与 slot_table 长度({len(slot_table)}) 不匹配"
            )
        # 校验所有值
        for slot, val in zip(slot_table, values):
            slot.validate(val)
        self._values: List[int] = list(values)

    def get(self, index: int) -> int:
        """读取第 index 个槽位"""
        if index < 0 or index >= len(self._values):
            raise IndexError(
                f"槽位索引 {index} 超出范围 [0, {len(self._values)})"
            )
        return self._values[index]

    def set(self, index: int, value: int) -> None:
        """写入第 index 个槽位"""
        if index < 0 or index >= len(self._values):
            raise IndexError(
                f"槽位索引 {index} 超出范围 [0, {len(self._values)})"
            )
        self._slot_table[index].validate(value)
        self._values[index] = value

    def get_all(self) -> List[int]:
        """读取所有槽位（返回副本）"""
        return list(self._values)

    def serialize(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "backend": self.name,
            "values": list(self._values),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any],
                    slot_table: List[SlotInfo]) -> "SlotBackend":
        """从字典反序列化

        Args:
            data: 序列化字典
            slot_table: 槽位定义列表（不在后端状态中，需外部传入）
        """
        if data.get("backend") != cls.name:
            raise ValueError(
                f"后端类型不匹配：期望 {cls.name}，得到 {data.get('backend')}"
            )
        return cls(slot_table=slot_table, values=data["values"])


# ============================================================
# PackedBackend: 打包后端
# ============================================================

class PackedBackend(StorageBackend):
    """打包后端

    所有槽位打包进一个 Python int。
    适用于：大量小值，紧凑存储，槽位不常修改场景。

    内存特点：仅一个 int 对象存储所有槽位。

    限制：所有槽位 bits 之和不能超过 64（Python int 无上限，但建议 ≤64）。
    """
    name = "packed"
    immutable = False

    def __init__(self, slot_table: List[SlotInfo],
                 packed_value: int = 0) -> None:
        """初始化

        Args:
            slot_table: 槽位定义列表
            packed_value: 打包后的整数值（缺省 0）

        Note:
            构造时不校验 packed_value 是否符合槽位范围，
            因为 packed_value 可能是多个合法槽位值的拼接。
            校验在 set 方法中按单个槽位进行。
        """
        super().__init__(slot_table)
        self._total_bits = sum(s.bits for s in slot_table)
        if self._total_bits <= 0:
            raise ValueError("槽位总位数必须 > 0")
        self._packed = int(packed_value)

        # 预计算每个槽位的位偏移
        # 第一个槽位在高位（与 MSInt.from_packed / as_int / _resolve_concat 的
        # 拼接顺序一致：concat(a, b) → a 在高位、b 在低位）
        # 否则 auto_select_backend 在 slot_count > 8 自动选 PackedBackend 时
        # 会对同一 packed_value 解读出相反的槽位顺序，导致静默数据错乱。
        self._offsets: List[int] = []
        offset = self._total_bits
        for slot in slot_table:
            offset -= slot.bits
            self._offsets.append(offset)

    def get(self, index: int) -> int:
        """读取第 index 个槽位"""
        if index < 0 or index >= len(self._slot_table):
            raise IndexError(
                f"槽位索引 {index} 超出范围 [0, {len(self._slot_table)})"
            )
        slot = self._slot_table[index]
        offset = self._offsets[index]
        mask = (1 << slot.bits) - 1
        raw = (self._packed >> offset) & mask
        # 处理符号位
        if slot.signed and (raw & (1 << (slot.bits - 1))):
            raw -= (1 << slot.bits)
        return raw

    def set(self, index: int, value: int) -> None:
        """写入第 index 个槽位"""
        if index < 0 or index >= len(self._slot_table):
            raise IndexError(
                f"槽位索引 {index} 超出范围 [0, {len(self._slot_table)})"
            )
        slot = self._slot_table[index]
        slot.validate(value)
        offset = self._offsets[index]
        mask = (1 << slot.bits) - 1
        # 清除原值，写入新值
        self._packed &= ~(mask << offset)
        # 负数经 Python & 已按补码截断（安全审计 2026-08-16 U1：原三目
        # 两分支相同，化简）
        stored = value & mask
        self._packed |= stored << offset

    def get_all(self) -> List[int]:
        """读取所有槽位"""
        return [self.get(i) for i in range(len(self._slot_table))]

    @property
    def packed_value(self) -> int:
        """底层打包的整数值"""
        return self._packed

    def serialize(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "backend": self.name,
            "packed_value": self._packed,
            "total_bits": self._total_bits,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any],
                    slot_table: List[SlotInfo]) -> "PackedBackend":
        """从字典反序列化"""
        if data.get("backend") != cls.name:
            raise ValueError(
                f"后端类型不匹配：期望 {cls.name}，得到 {data.get('backend')}"
            )
        return cls(slot_table=slot_table, packed_value=data["packed_value"])

    @classmethod
    def from_values(cls, slot_table: List[SlotInfo],
                    values: List[int]) -> "PackedBackend":
        """从值列表创建打包后端

        Args:
            slot_table: 槽位定义列表
            values: 每个槽位的值（与 slot_table 等长）
        """
        if len(values) != len(slot_table):
            raise ValueError(
                f"values 长度({len(values)}) 与 slot_table 长度({len(slot_table)}) 不匹配"
            )
        backend = cls(slot_table=slot_table, packed_value=0)
        for i, val in enumerate(values):
            backend.set(i, val)
        return backend


# ============================================================
# LazyBackend: 惰性后端
# ============================================================

class LazyBackend(StorageBackend):
    """惰性后端

    不存储任何数据，按需调用计算函数获取值。
    适用于：值可从其他数据推导时（如从传感器实时读取、从数据库查询）。

    内存特点：无数据存储，仅持有计算函数引用。

    限制：
      - 不可变（set 方法抛异常）
      - 每次调用 get 都会触发 compute_fn，可能有性能开销
      - compute_fn 返回值长度必须与 slot_table 等长
    """
    name = "lazy"
    immutable = True

    def __init__(self, slot_table: List[SlotInfo],
                 compute_fn: Callable[[], List[int]]) -> None:
        """初始化

        Args:
            slot_table: 槽位定义列表
            compute_fn: 计算函数，返回所有槽位值列表
        """
        super().__init__(slot_table)
        if not callable(compute_fn):
            raise ValueError("compute_fn 必须是可调用对象")
        self._compute_fn = compute_fn
        self._call_count = 0  # 调用计数（用于测试和调试）

    def _compute(self) -> List[int]:
        """调用计算函数并校验返回值"""
        values = self._compute_fn()
        if not isinstance(values, list):
            raise ValueError(
                f"compute_fn 必须返回 list，得到 {type(values).__name__}"
            )
        if len(values) != len(self._slot_table):
            raise ValueError(
                f"compute_fn 返回长度({len(values)}) 与 slot_table 长度"
                f"({len(self._slot_table)}) 不匹配"
            )
        for slot, val in zip(self._slot_table, values):
            slot.validate(val)
        self._call_count += 1
        return values

    def get(self, index: int) -> int:
        """读取第 index 个槽位（触发 compute_fn）"""
        if index < 0 or index >= len(self._slot_table):
            raise IndexError(
                f"槽位索引 {index} 超出范围 [0, {len(self._slot_table)})"
            )
        values = self._compute()
        return values[index]

    def set(self, index: int, value: int) -> None:
        """写入槽位（不可变后端，抛异常）"""
        raise NotImplementedError(
            f"{self.name} 后端不可变，不支持 set 操作"
        )

    def get_all(self) -> List[int]:
        """读取所有槽位（触发一次 compute_fn）"""
        return self._compute()

    @property
    def call_count(self) -> int:
        """计算函数被调用的次数"""
        return self._call_count

    def serialize(self) -> Dict[str, Any]:
        """序列化（惰性后端无法序列化 compute_fn）"""
        return {
            "backend": self.name,
            "values": self._compute(),  # 序列化时计算一次并固化
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any],
                    slot_table: List[SlotInfo]) -> "LazyBackend":
        """从字典反序列化（退化为 SlotBackend 行为）

        反序列化后失去惰性特性，变为固定值列表的惰性后端。
        """
        if data.get("backend") != cls.name:
            raise ValueError(
                f"后端类型不匹配：期望 {cls.name}，得到 {data.get('backend')}"
            )
        values = list(data["values"])
        return cls(slot_table=slot_table, compute_fn=lambda v=values: list(v))


# ============================================================
# PersistentBackend: 持久化后端（文件存储）
# ============================================================

class PersistentBackend(StorageBackend):
    """持久化后端

    基于文件的持久存储，跨进程跨重启。
    适用于：长期记忆、跨会话状态、需要落盘的数据。

    设计：
      - 内存中维护 SlotBackend 作为读写缓存
      - 每次 set() 写穿（write-through）到 JSON 文件
      - 初始化时若文件存在则加载，否则使用空值

    内存特点：与 SlotBackend 相同（每槽位一个 int）。
    性能：每次 set 触发一次文件 IO，大量写入时有性能开销。
    """

    name = "persistent"
    immutable = False

    def __init__(self, slot_table: List[SlotInfo],
                 values: List[int] = None,
                 file_path: str = None) -> None:
        """初始化

        Args:
            slot_table: 槽位定义列表
            values: 初始值列表（缺省全 0；若 file_path 存在则被文件内容覆盖）
            file_path: 持久化文件路径（None = 不落盘，退化为 SlotBackend 行为）

        Raises:
            ValueError: values 长度不匹配或值超范围
            OSError: 文件存在但读取失败
        """
        super().__init__(slot_table)
        self._file_path = file_path
        # 优先从文件加载
        loaded_values: List[int] = []
        if file_path:
            import os
            if os.path.exists(file_path):
                try:
                    import json
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    loaded_values = list(data.get("values", []))
                except (OSError, ValueError, KeyError) as e:
                    raise OSError(
                        f"无法读取持久化文件 {file_path}: {e}"
                    ) from e
        # 决定最终初始值：显式 values > 文件 > 全 0
        if loaded_values:
            init_values = loaded_values
        elif values is not None:
            init_values = list(values)
        else:
            init_values = [0] * len(slot_table)
        # 复用 SlotBackend 的校验和存储逻辑
        self._inner = SlotBackend(slot_table=slot_table, values=init_values)
        # 安全审计 2026-08-16 U2：最近一次落盘失败信息（None = 成功/未写过）。
        # set() 不因落盘失败中断（写穿链路语义保持），调用方可查此属性感知。
        self.last_write_error: Optional[str] = None

    def get(self, index: int) -> int:
        """读取第 index 个槽位"""
        return self._inner.get(index)

    def set(self, index: int, value: int) -> None:
        """写入第 index 个槽位并写穿到文件"""
        self._inner.set(index, value)
        self._flush_to_file()

    def get_all(self) -> List[int]:
        """读取所有槽位"""
        return self._inner.get_all()

    @property
    def file_path(self) -> str:
        """持久化文件路径"""
        return self._file_path

    def _flush_to_file(self) -> None:
        """把当前值写穿到文件"""
        if not self._file_path:
            return
        import json
        import os
        data = {
            "backend": self.name,
            "values": self._inner.get_all(),
        }
        # 原子写：先写临时文件再 rename
        tmp_path = self._file_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp_path, self._file_path)
            self.last_write_error = None  # 安全审计 U2：成功时清除
        except OSError as e:
            # 写失败不抛异常（避免 set 链路中断），但记录到 logging +
            # last_write_error（安全审计 2026-08-16 U2：失败可感知，
            # set() 看似成功但数据未持久化的情况可由调用方查询）
            import logging
            logging.getLogger(__name__).warning(
                "PersistentBackend 写文件失败 %s", self._file_path,
                exc_info=True,
            )
            self.last_write_error = f"{type(e).__name__}: {e}"
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def serialize(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "backend": self.name,
            "values": self._inner.get_all(),
            "file_path": self._file_path,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any],
                    slot_table: List[SlotInfo]) -> "PersistentBackend":
        """从字典反序列化

        注意：反序列化时不重新加载文件（用 data 中的 values），
        避免文件已被修改导致状态不一致。
        """
        if data.get("backend") != cls.name:
            raise ValueError(
                f"后端类型不匹配：期望 {cls.name}，得到 {data.get('backend')}"
            )
        return cls(
            slot_table=slot_table,
            values=data.get("values"),
            file_path=data.get("file_path"),
        )

    @classmethod
    def from_file(cls, slot_table: List[SlotInfo],
                  file_path: str) -> "PersistentBackend":
        """从文件加载（或创建空文件）

        若文件存在则加载其值，否则创建空后端（值为全 0）。
        后续 set() 会自动写穿到该文件。

        Args:
            slot_table: 槽位定义列表
            file_path: 文件路径
        """
        return cls(slot_table=slot_table, file_path=file_path)


# ============================================================
# SparseBackend: 稀疏后端（只存非默认值）
# ============================================================

class SparseBackend(StorageBackend):
    """稀疏后端

    只存储非默认值（默认 0），适用于大部分槽位为 0 的场景。
    适用于：稀疏特征向量、大槽位表但少数非零、稀疏图邻接表。

    内存特点：只持有非零槽位的 dict，O(非零元素数)。
    性能：get/set 都是 O(1) dict 操作。

    限制：
      - 默认值固定为 0（可扩展为构造时指定）
      - set 到默认值会从 dict 中移除该槽位
    """

    name = "sparse"
    immutable = False

    def __init__(self, slot_table: List[SlotInfo],
                 values: List[int] = None,
                 default_value: int = 0) -> None:
        """初始化

        Args:
            slot_table: 槽位定义列表
            values: 初始值列表（缺省全 default_value）
            default_value: 默认值（get 未存储槽位时返回此值，set 到此值会移除槽位）

        Raises:
            ValueError: values 长度不匹配或值超范围
        """
        super().__init__(slot_table)
        self._default_value = default_value
        # 只存非默认值的槽位：index → value
        self._sparse: Dict[int, int] = {}
        if values is None:
            values = [default_value] * len(slot_table)
        if len(values) != len(slot_table):
            raise ValueError(
                f"values 长度({len(values)}) 与 slot_table 长度({len(slot_table)}) 不匹配"
            )
        # 校验所有值并只存非默认值
        for i, (slot, val) in enumerate(zip(slot_table, values)):
            slot.validate(val)
            if val != default_value:
                self._sparse[i] = val

    def get(self, index: int) -> int:
        """读取第 index 个槽位"""
        if index < 0 or index >= len(self._slot_table):
            raise IndexError(
                f"槽位索引 {index} 超出范围 [0, {len(self._slot_table)})"
            )
        return self._sparse.get(index, self._default_value)

    def set(self, index: int, value: int) -> None:
        """写入第 index 个槽位

        若 value 等于 default_value，则从稀疏字典中移除该槽位（节省内存）。
        """
        if index < 0 or index >= len(self._slot_table):
            raise IndexError(
                f"槽位索引 {index} 超出范围 [0, {len(self._slot_table)})"
            )
        self._slot_table[index].validate(value)
        if value == self._default_value:
            # 写入默认值 = 移除槽位
            self._sparse.pop(index, None)
        else:
            self._sparse[index] = value

    def get_all(self) -> List[int]:
        """读取所有槽位"""
        return [self._sparse.get(i, self._default_value)
                for i in range(len(self._slot_table))]

    @property
    def default_value(self) -> int:
        """默认值"""
        return self._default_value

    @property
    def non_zero_count(self) -> int:
        """非默认值槽位数量"""
        return len(self._sparse)

    def serialize(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "backend": self.name,
            "values": self.get_all(),
            "default_value": self._default_value,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any],
                    slot_table: List[SlotInfo]) -> "SparseBackend":
        """从字典反序列化"""
        if data.get("backend") != cls.name:
            raise ValueError(
                f"后端类型不匹配：期望 {cls.name}，得到 {data.get('backend')}"
            )
        return cls(
            slot_table=slot_table,
            values=data.get("values"),
            default_value=int(data.get("default_value", 0)),
        )


# ============================================================
# StreamBackend: 流式后端（带历史窗口）
# ============================================================

class StreamBackend(StorageBackend):
    """流式后端

    每个槽位维护一个固定大小的历史窗口（deque），set() 追加新值，
    get() 返回最新值。适用于传感器数据、连续流、时序信号。

    适用于：
      - 传感器实时读取（保留最近 N 次读数）
      - 滑动窗口统计（平均值、趋势检测）
      - 时序数据缓存

    内存特点：O(slot_count * history_size)。
    性能：get/set 都是 O(1)。

    限制：
      - 不可变标志为 False（支持 set）
      - get 返回最新值；如需历史用 get_history 扩展方法
      - 初始化时窗口为空，第一次 get 未设置槽位返回 default_value
    """

    name = "stream"
    immutable = False

    def __init__(self, slot_table: List[SlotInfo],
                 history_size: int = 10,
                 default_value: int = 0) -> None:
        """初始化

        Args:
            slot_table: 槽位定义列表
            history_size: 每个槽位保留的历史值数量（默认 10）
            default_value: 槽位未设置时 get 返回的默认值

        Raises:
            ValueError: history_size <= 0
        """
        super().__init__(slot_table)
        if history_size <= 0:
            raise ValueError(
                f"history_size 必须是正整数，得到 {history_size}"
            )
        self._history_size = int(history_size)
        self._default_value = default_value
        # 每个槽位一个 deque（ maxlen=history_size）
        from collections import deque
        self._streams: List[Any] = [
            deque(maxlen=history_size) for _ in range(len(slot_table))
        ]

    def get(self, index: int) -> int:
        """读取第 index 个槽位的最新值"""
        if index < 0 or index >= len(self._slot_table):
            raise IndexError(
                f"槽位索引 {index} 超出范围 [0, {len(self._slot_table)})"
            )
        stream = self._streams[index]
        if not stream:
            return self._default_value
        return stream[-1]

    def set(self, index: int, value: int) -> None:
        """追加新值到第 index 个槽位的历史窗口"""
        if index < 0 or index >= len(self._slot_table):
            raise IndexError(
                f"槽位索引 {index} 超出范围 [0, {len(self._slot_table)})"
            )
        self._slot_table[index].validate(value)
        self._streams[index].append(value)

    def get_all(self) -> List[int]:
        """读取所有槽位的最新值"""
        return [self.get(i) for i in range(len(self._slot_table))]

    def get_history(self, index: int, n: int = 0) -> List[int]:
        """读取第 index 个槽位的历史值

        Args:
            index: 槽位索引
            n: 返回最近 n 个值；0 或负数表示返回全部历史

        Returns:
            历史值列表（从旧到新），最多 history_size 个
        """
        if index < 0 or index >= len(self._slot_table):
            raise IndexError(
                f"槽位索引 {index} 超出范围 [0, {len(self._slot_table)})"
            )
        stream = list(self._streams[index])
        if n <= 0:
            return stream
        return stream[-n:]

    @property
    def history_size(self) -> int:
        """每个槽位的历史窗口大小"""
        return self._history_size

    @property
    def default_value(self) -> int:
        """默认值"""
        return self._default_value

    def serialize(self) -> Dict[str, Any]:
        """序列化为字典

        注意：流式后端的完整历史可能很长，序列化时只保留每个槽位的最新值。
        反序列化后失去历史，退化为 SparseBackend-like 行为。
        """
        return {
            "backend": self.name,
            "values": self.get_all(),
            "history_size": self._history_size,
            "default_value": self._default_value,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any],
                    slot_table: List[SlotInfo]) -> "StreamBackend":
        """从字典反序列化

        反序列化后历史窗口为空，但 values 中的值会作为初始最新值写入。
        """
        if data.get("backend") != cls.name:
            raise ValueError(
                f"后端类型不匹配：期望 {cls.name}，得到 {data.get('backend')}"
            )
        backend = cls(
            slot_table=slot_table,
            history_size=int(data.get("history_size", 10)),
            default_value=int(data.get("default_value", 0)),
        )
        # 把序列化时的最新值写入流（成为窗口中唯一值）
        for i, val in enumerate(data.get("values", [])):
            if i < len(slot_table):
                try:
                    backend.set(i, int(val))
                except (ValueError, IndexError):
                    continue
        return backend


# ============================================================
# 后端工厂
# ============================================================

_BACKEND_REGISTRY: Dict[str, type] = {
    SlotBackend.name: SlotBackend,
    PackedBackend.name: PackedBackend,
    LazyBackend.name: LazyBackend,
    PersistentBackend.name: PersistentBackend,
    SparseBackend.name: SparseBackend,
    StreamBackend.name: StreamBackend,
}


def register_backend(name: str, backend_cls: type) -> None:
    """注册自定义存储后端

    Args:
        name: 后端名
        backend_cls: 后端类（必须继承 StorageBackend）
    """
    if not issubclass(backend_cls, StorageBackend):
        raise ValueError(
            f"backend_cls 必须继承 StorageBackend，得到 {backend_cls}"
        )
    _BACKEND_REGISTRY[name] = backend_cls


def get_backend_class(name: str) -> type:
    """根据名获取后端类

    Raises:
        KeyError: 后端名未注册
    """
    if name not in _BACKEND_REGISTRY:
        raise KeyError(
            f"未知后端类型 '{name}'，已注册: {list(_BACKEND_REGISTRY.keys())}"
        )
    return _BACKEND_REGISTRY[name]


def auto_select_backend(slot_count: int) -> str:
    """根据槽位数量自动选择后端

    策略：
      - 槽位数 <= 8：用 SlotBackend（简单直接）
      - 槽位数 > 8：用 PackedBackend（紧凑存储）

    Args:
        slot_count: 槽位数量

    Returns:
        后端名
    """
    if slot_count <= 8:
        return SlotBackend.name
    return PackedBackend.name
