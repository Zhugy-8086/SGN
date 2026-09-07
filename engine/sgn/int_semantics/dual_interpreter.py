#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""DualInterpreter — 双解读器（依赖 MSInt，不再依赖 SplitTree）

v5.3.0 Step 2：从 engine/level.py 迁入 int_semantics 包，
底层由 SplitTree 切换为 MSInt.auto_split。

同一份 int 提供两种解读视角：
  - as_features(raw): 拆分特征向量（通过 MSInt.auto_split）
  - as_identifier(raw): 完整整数标识符（直接返回 raw）

两种解读共享同一份 raw int，互不干扰。
内置 O(1) 哈希索引（index/lookup），用于整数模式快速查找。

与旧版（engine.level.DualInterpreter）的差异：
  - 旧版接受 SplitTree 对象配置拆分；新版接受 split_bits/total_bits 参数
  - 旧版 view_as 会截断叶子值到 leaf_bit_width；新版槽位位宽即输出位宽，无截断 quirk
  - 新版支持任意 split_bits（非 2 的幂也可），真正非对称拆分的基础

设计原则（§6.1）：本模块不 import engine 中的调度/匹配逻辑符号，仅依赖 MSInt ABI。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..msint import MSInt


class DualInterpreter:
    """双解读器：同一份 int 同时给出拆分特征与整数标识两种视角

    设计要点：
      - 不持有神经元状态，只是"解读工具"（无状态）
      - as_features() 依赖 MSInt.auto_split 把 raw 拆成多维特征向量
      - as_identifier() 直接返回 raw（整数模式不改变值）
      - 内置一个 hash_index（int → 候选 ID 列表），用于演示整数模式 O(1) 查找
      - 两种解读共享同一份 raw int，互不干扰

    线程安全（安全审计 2026-08-16 W1）：非线程安全——hash_index/
    remove 等方法读写共享 dict，无锁设计（与 MSInt 同一单线程使用约定）。
    多线程场景需调用方自行加锁或每线程独立实例。

    向后兼容：split_bits=0 时 as_features() 返回单元素 tuple (raw,)，
    等价于不拆分的整体特征。

    Example:
        # 16 位整数拆成 2×8 位特征
        interp = DualInterpreter(split_bits=8, total_bits=16)
        interp.as_features(0x7DFC)     # (0x7D, 0xFC) = (125, 252)
        interp.as_identifier(0x7DFC)   # 32252
        interp.index(0x7DFC, candidate_id=42)
        interp.lookup(0x7DFC)          # [42]

        # 便捷方法：一次调用返回两种视角
        features, identifier = interp.interpret(0x7DFC)
    """

    def __init__(self, split_bits: int = 0, total_bits: int = 64) -> None:
        """初始化双解读器

        Args:
            split_bits: 每个槽位的位宽。
                        0 表示不拆分（as_features 返回 (raw,) 单元素）。
                        >0 时按此位宽等宽拆分 total_bits。
            total_bits: 总位宽（默认 64）。split_bits > 0 时必须能整除 total_bits。

        Raises:
            ValueError: split_bits < 0 或 total_bits <= 0
        """
        if split_bits < 0:
            raise ValueError(f"split_bits 必须 >= 0，得到 {split_bits}")
        if total_bits <= 0:
            raise ValueError(f"total_bits 必须 > 0，得到 {total_bits}")
        if split_bits > 0 and total_bits % split_bits != 0:
            raise ValueError(
                f"total_bits({total_bits}) 必须能被 split_bits({split_bits}) 整除"
            )
        self._split_bits = split_bits
        self._total_bits = total_bits
        # 完整整数 → 候选 ID 列表（整数模式索引）
        self._hash_index: Dict[int, List[int]] = {}

    @property
    def split_bits(self) -> int:
        """每个槽位的位宽（0 表示不拆分）"""
        return self._split_bits

    @property
    def total_bits(self) -> int:
        """总位宽"""
        return self._total_bits

    def as_features(self, raw: int) -> Tuple[int, ...]:
        """拆分模式：把 raw 解读为多维特征向量

        Args:
            raw: 原始整数（必须 >= 0）

        Returns:
            特征向量 tuple。若 split_bits=0，返回 (raw,)。
            若 split_bits>0，返回 N=total_bits/split_bits 个等宽子整数，
            第一个槽位在高位（与 SplitTree 的 DFS 先序一致）。

        Raises:
            ValueError: raw < 0（负数属于干涉层，不参与特征解读）
        """
        if raw < 0:
            raise ValueError(
                f"as_features 只接受非负整数（负数属于干涉层），得到 {raw}"
            )
        if self._split_bits == 0:
            return (raw,)
        # 每次调用创建临时 MSInt（无状态工具，不持有 raw）
        ms = MSInt.auto_split(
            raw, bits=self._split_bits, total_bits=self._total_bits
        )
        n_slots = self._total_bits // self._split_bits
        return tuple(ms.view(f"slot_{i}") for i in range(n_slots))

    def as_identifier(self, raw: int) -> int:
        """整数模式：把 raw 作为全局唯一标识符返回

        不改变 raw 的值，只是赋予"标识符"语义。
        调用方可用此返回值作为 dict key / 文件名 / 哈希索引键。

        Args:
            raw: 原始整数（可正可负，负数也作为标识符合法）

        Returns:
            原始 int 值
        """
        return raw

    def interpret(self, raw: int) -> Tuple[Tuple[int, ...], int]:
        """便捷方法：一次调用返回 (拆分特征向量, 完整标识符)

        Args:
            raw: 原始整数

        Returns:
            (as_features(raw), as_identifier(raw))
        """
        return self.as_features(raw), self.as_identifier(raw)

    def index(self, raw: int, candidate_id: int) -> None:
        """用整数模式建立索引：raw → candidate_id

        Args:
            raw: 整数标识符
            candidate_id: 待索引的候选 ID（如神经元 ID、模板 ID）

        Note:
            允许 raw 重复，同一 raw 可关联多个 candidate_id。
        """
        self._hash_index.setdefault(raw, []).append(candidate_id)

    def lookup(self, raw: int) -> List[int]:
        """整数模式 O(1) 查找：raw → 候选 ID 列表

        Args:
            raw: 整数标识符

        Returns:
            该 raw 对应的候选 ID 列表（空列表表示无候选）
        """
        return list(self._hash_index.get(raw, []))

    def clear_index(self) -> None:
        """清空整数模式索引"""
        self._hash_index.clear()

    def index_size(self) -> int:
        """返回整数模式索引中的 raw 键数量"""
        return len(self._hash_index)
