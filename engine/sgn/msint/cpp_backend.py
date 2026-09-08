# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""SGN MSint C++ 后端 — 统一 API + Python fallback

Stage 3.0.5 Task 5.3: Python fallback 设计

策略：
  - try-import C++ 扩展（sgn.PackedBackend / sgn.MSIntView）
  - 失败时回退到 Python 实现（engine.ms_int.backends.PackedBackend）
  - 回退时打印警告日志（非崩溃）

统一 API:
  CppPackedBackend.from_bits(bits_list, signed_flags, packed)
    → get(index) / set(index, value) / get_all() / get_all_simd()
    → packed_value / slot_count / total_bits

  CppMSIntView.bitsplit(raw, total_bits, target_bits)
  CppMSIntView.concat(values, bits_list)

用法:
    from sgn.msint.cpp_backend import CppPackedBackend, CppMSIntView, USING_CPP

    backend = CppPackedBackend.from_bits([8, 8, 8, 8])
    print(f"C++ 模式: {USING_CPP}")  # True/False
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ============================================================
# try-import C++ 扩展
# ============================================================

_USING_CPP = False
_cpp_PackedBackend = None
_cpp_MSIntView = None
_cpp_SlotSpec = None
_cpp_SplitDot = None
_cpp_MultiScaleView = None
_cpp_PrecisionSelector = None
_cpp_LeveledSplitDot = None

# 安全审计 2026-08-16 L1/S1：原实现顶层 `import sgn`——项目约定
# `import engine.sgn as sgn` 模式下顶层 sgn 模块不存在，import 静默失败
# 回退 Python 实现（性能下降数十倍，仅 warning，难定位）。
# 修复：相对导入（引擎包两种装载方式均正确解析）：
#   - engine.sgn.msint.cpp_backend → from .. = engine.sgn（包属性即 C++ 类型）
#   - sgn.msint.cpp_backend（build/ 直载 .pyd 模式）→ from .. = sgn（.pyd）
# LeveledSplitDot 在旧 .pyd 上可能缺失（包属性为 None），单独 getattr 容错。
try:
    from .. import (
        PackedBackend as _cpp_PackedBackend,
        MSIntView as _cpp_MSIntView,
        SlotSpec as _cpp_SlotSpec,
        SplitDot as _cpp_SplitDot,
        MultiScaleView as _cpp_MultiScaleView,
        PrecisionSelector as _cpp_PrecisionSelector,
    )
    _USING_CPP = True
    try:
        from .. import LeveledSplitDot as _cpp_LeveledSplitDot
    except (ImportError, AttributeError):
        _cpp_LeveledSplitDot = None  # 旧 .pyd 未编译 LeveledSplitDot
    logger.debug("sgn.msint.cpp_backend: 使用 C++ 扩展（PackedBackend / MSIntView / SplitDot / MultiScaleView / PrecisionSelector / LeveledSplitDot）")
except (ImportError, AttributeError) as e:
    logger.warning(
        "sgn.msint.cpp_backend: C++ 扩展不可用，回退到 Python 实现。"
        "原因: %s", e
    )
    _USING_CPP = False


# ============================================================
# Python fallback: PackedBackend 包装
# ============================================================

class _PyPackedBackendWrapper:
    """Python PackedBackend 包装层（接口对齐 C++ PackedBackend）

    将 engine.ms_int.backends.PackedBackend 包装为与 C++ PackedBackend
    相同的 API（from_bits / get / set / get_all / get_all_simd）。
    """

    def __init__(self, bits_list: List[int],
                 signed_flags: Optional[List[bool]] = None,
                 packed: int = 0):
        from ._py.backends import PackedBackend as _PyPackedBackend
        from ._py.protocol import SlotInfo

        self._bits_list = list(bits_list)
        self._signed_flags = list(signed_flags) if signed_flags else [False] * len(bits_list)
        self._total_bits = sum(bits_list)

        # 构造 SlotInfo 列表（name 用占位符）
        slots = []
        for i, (b, s) in enumerate(zip(self._bits_list, self._signed_flags)):
            slots.append(SlotInfo(name=f"slot_{i}", bits=b, signed=s))

        self._backend = _PyPackedBackend(slot_table=slots, packed_value=packed)

    def get(self, index: int) -> int:
        return self._backend.get(index)

    def set(self, index: int, value: int) -> None:
        self._backend.set(index, value)

    def get_all(self) -> List[int]:
        return self._backend.get_all()

    def get_all_simd(self) -> List[int]:
        # Python 模式无 SIMD，等同 get_all
        return self.get_all()

    @property
    def packed_value(self) -> int:
        return self._backend.packed_value

    @property
    def slot_count(self) -> int:
        return len(self._bits_list)

    @property
    def total_bits(self) -> int:
        return self._total_bits

    def serialize(self) -> str:
        import json
        return json.dumps({
            "backend": "packed",
            "packed_value": self.packed_value,
            "total_bits": self._total_bits,
        })

    def __repr__(self) -> str:
        return f"<_PyPackedBackendWrapper slots={self.slot_count} " \
               f"total_bits={self._total_bits} packed=0x{self.packed_value:x}>"


# ============================================================
# 批量 API: batch_get_all
# ============================================================

def _batch_get_all_cpp(bits_list, packed_array, signed_flags=None):
    """C++ 批量读取（numpy 零拷贝）；signed_flags 逐槽符号标志，None=全无符号（A1-1）"""
    import numpy as np
    import sgn as _sgn_mod
    if (isinstance(packed_array, np.ndarray)
            and packed_array.dtype == np.uint64
            and packed_array.flags['C_CONTIGUOUS']):
        arr = packed_array
    else:
        arr = np.ascontiguousarray(packed_array, dtype=np.uint64)
    if signed_flags is None:
        return _sgn_mod.batch_get_all(bits_list, arr)
    return _sgn_mod.batch_get_all(bits_list, arr, list(signed_flags))


def _batch_get_all_py(bits_list, packed_values, signed_flags=None):
    """Python fallback 批量读取（A1-1：支持逐槽符号扩展，与 C++ 口径一致）"""
    import numpy as np
    n_slots = len(bits_list)
    n_values = len(packed_values)
    result = np.zeros((n_values, n_slots), dtype=np.int64)

    # 预计算 offsets 和 masks
    total = sum(bits_list)
    offsets = []
    off = total
    for b in bits_list:
        off -= b
        offsets.append(off)
    masks = [(1 << b) - 1 if b < 64 else ~0 for b in bits_list]

    flags = [False] * n_slots
    if signed_flags is not None:
        for j in range(min(len(signed_flags), n_slots)):
            flags[j] = bool(signed_flags[j])

    for i, pv in enumerate(packed_values):
        for j in range(n_slots):
            raw = (pv >> offsets[j]) & masks[j]
            if flags[j]:
                b = bits_list[j]
                if b == 64:
                    val = raw - (1 << 64) if raw & (1 << 63) else raw
                elif raw & (1 << (b - 1)):
                    val = raw - (1 << b)
                else:
                    val = raw
            else:
                val = raw
            result[i, j] = val
    return result


# 统一 batch API
if _USING_CPP:
    batch_get_all = _batch_get_all_cpp
else:
    batch_get_all = _batch_get_all_py


# ============================================================
# 完整解码流水线: batch_decode_to_float
# ============================================================

def _batch_decode_to_float_cpp(bits_list, packed_array, scale, signed=True):
    """C++ 完整解码（packed → concat → signed → float32 × scale）；A1-1：signed 显式化"""
    import numpy as np
    import sgn as _sgn_mod
    # 跳过 ascontiguousarray 开销：已是 uint64 连续数组时直接使用
    if (isinstance(packed_array, np.ndarray)
            and packed_array.dtype == np.uint64
            and packed_array.flags['C_CONTIGUOUS']):
        arr = packed_array
    else:
        arr = np.ascontiguousarray(packed_array, dtype=np.uint64)
    return _sgn_mod.batch_decode_to_float(bits_list, arr, float(scale), bool(signed))


def _batch_decode_to_float_into_cpp(bits_list, packed_array, scale, output, signed=True):
    """C++ in-place 解码：写入预分配的 output 数组，消除分配开销；A1-1：signed 显式化"""
    import numpy as np
    import sgn as _sgn_mod
    if (isinstance(packed_array, np.ndarray)
            and packed_array.dtype == np.uint64
            and packed_array.flags['C_CONTIGUOUS']):
        arr = packed_array
    else:
        arr = np.ascontiguousarray(packed_array, dtype=np.uint64)
    _sgn_mod.batch_decode_to_float_into(bits_list, arr, float(scale), output,
                                        bool(signed))


def _batch_decode_to_float_py(bits_list, packed_values, scale, signed=True):
    """Python fallback 完整解码（A1-1：signed=False 时为无符号 concat 视角）"""
    import numpy as np
    n_slots = len(bits_list)
    n_values = len(packed_values)
    total = sum(bits_list)

    # 预计算 offsets 和 masks
    offsets = []
    off = total
    for b in bits_list:
        off -= b
        offsets.append(off)
    masks = [(1 << b) - 1 if b < 64 else ~0 for b in bits_list]

    result = np.zeros(n_values, dtype=np.float32)
    for i, pv in enumerate(packed_values):
        # concat: 第一个槽位在高位
        concat_val = 0
        for j in range(n_slots):
            slot_val = (pv >> offsets[j]) & masks[j]
            concat_val = (concat_val << bits_list[j]) | slot_val
        # 符号扩展（A1-1：signed=False 跳过）
        if signed and total < 64:
            sign_bit = 1 << (total - 1)
            if concat_val & sign_bit:
                concat_val |= ~((1 << total) - 1)
        result[i] = float(concat_val) * scale
    return result


def _batch_decode_to_float_into_py(bits_list, packed_values, scale, output,
                                   signed=True):
    """Python fallback in-place 解码"""
    result = _batch_decode_to_float_py(bits_list, packed_values, scale, signed)
    output[:len(result)] = result


# 统一 decode API
if _USING_CPP:
    batch_decode_to_float = _batch_decode_to_float_cpp
    batch_decode_to_float_into = _batch_decode_to_float_into_cpp
else:
    batch_decode_to_float = _batch_decode_to_float_py
    batch_decode_to_float_into = _batch_decode_to_float_into_py


# ============================================================
# Python fallback: MSIntView 实现
# ============================================================

class _PyMSIntView:
    """Python MSIntView 实现（接口对齐 C++ MSIntView）"""

    @staticmethod
    def bitsplit(raw: int, total_bits: int, target_bits: int) -> List[int]:
        """位拆分：把 raw 按 target_bits 拆分，低位在前"""
        if target_bits <= 0:
            raise ValueError("target_bits 必须 > 0")
        if total_bits <= 0:
            raise ValueError("total_bits 必须 > 0")
        if target_bits > total_bits:
            raise ValueError(
                f"target_bits({target_bits}) 不能大于 total_bits({total_bits})"
            )
        # 负数转补码
        if raw < 0:
            raw = raw & ((1 << total_bits) - 1)
        mask = (1 << target_bits) - 1
        result = []
        remaining = raw
        n_parts = (total_bits + target_bits - 1) // target_bits
        for _ in range(n_parts):
            result.append(remaining & mask)
            remaining >>= target_bits
        return result

    @staticmethod
    def bitsplit_index(raw: int, total_bits: int, target_bits: int, idx: int) -> int:
        parts = _PyMSIntView.bitsplit(raw, total_bits, target_bits)
        if idx < 0 or idx >= len(parts):
            raise IndexError(f"bitsplit index {idx} 超出范围 [0, {len(parts)})")
        return parts[idx]

    @staticmethod
    def concat(values: List[int], bits_list: List[int]) -> int:
        """拼接：第一个值在高位，最后一个在低位"""
        if len(values) != len(bits_list):
            raise ValueError(
                f"values 长度({len(values)}) != bits_list 长度({len(bits_list)})"
            )
        result = 0
        for val, bits in zip(values, bits_list):
            mask = (1 << bits) - 1
            result = (result << bits) | (val & mask)
        return result

    @staticmethod
    def concat_signed(values: List[int], bits_list: List[int]) -> int:
        return _PyMSIntView.concat(values, bits_list)


# ============================================================
# Python fallback: SplitDot（前向多精度拆分点积）
# ============================================================

class _PyNibblePrepared:
    """Python NibblePrepared 等价对象（接口对齐 C++ NibblePrepared）。

    4 位摊销路径的预解包缓存：u8 无符号字节（w 侧）、s8 有符号字节（x 侧）、
    Sw/Sx 偏置修正和。由 _PySplitDot.prepare_nibble 构建，供 _PySplitDot.dot_prepared4
    复用（M 输出摊销，热路径无解包）。
    """

    __slots__ = ("n", "K", "u8", "s8", "Sw", "Sx")


class _PySplitDot:
    """Python SplitDot 实现（接口对齐 C++ SplitDot）

    用 Python 任意精度整数做位拆分与点积，作为 C++ 的参考实现，
    用于 H1 数值等价验证。
    """

    @staticmethod
    def split_parts(value: int, total_bits: int, split_bits: int) -> List[int]:
        if total_bits <= 0 or split_bits <= 0:
            raise ValueError("total_bits / split_bits 必须 > 0")
        if total_bits % split_bits != 0:
            raise ValueError("total_bits 必须能被 split_bits 整除")
        n = total_bits // split_bits
        mask = (1 << split_bits) - 1
        parts = []
        for k in range(n):
            if k < n - 1:
                parts.append((value >> (k * split_bits)) & mask)
            else:
                parts.append(value >> ((n - 1) * split_bits))  # 符号扩展
        return parts

    @staticmethod
    def dot_split(w, x, total_bits: int = 32, split_bits: int = 16,
                  trim_high_diag: bool = False) -> List[int]:
        if len(w) != len(x):
            raise ValueError("w 和 x 长度不一致")
        n = total_bits // split_bits
        n_partials = 2 * n - 1
        pw = [_PySplitDot.split_parts(v, total_bits, split_bits) for v in w]
        px = [_PySplitDot.split_parts(v, total_bits, split_bits) for v in x]
        partials = [0] * n_partials
        for a in range(n):
            for b in range(n):
                m = a + b
                if trim_high_diag and m >= n:
                    continue  # 高位对角裁剪（与 C++ 语义一致）
                partials[m] += sum(
                    pw[i][a] * px[i][b] for i in range(len(w))
                )
        return partials

    @staticmethod
    def dot_fused(w, x, total_bits: int = 32, split_bits: int = 16) -> int:
        partials = _PySplitDot.dot_split(w, x, total_bits, split_bits)
        fused = sum(
            partials[m] << (m * split_bits) for m in range(len(partials))
        )
        lo = fused & ((1 << 64) - 1)
        # 与 C++ dot_fused 一致：返回有符号 int64（低 64 位位模式）
        return lo - (1 << 64) if lo >= (1 << 63) else lo

    @staticmethod
    def dot_fused_i32(w, x, split_bits: int = 4) -> int:
        """单融合 int32 截断模式（与 C++ dot_fused_i32 一致）。

        只融合 m<n（移位 < 32）的 partials；高位对角（m>=n）是 2^32 的倍数，
        对低 32 位无贡献 → 内部走 trim_high_diag=True 的裁剪路径。
        要求 32 能被 split_bits 整除。
        """
        total_bits = 32
        if total_bits % split_bits != 0:
            raise ValueError(f"32 必须能被 split_bits({split_bits}) 整除")
        partials = _PySplitDot.dot_split(w, x, total_bits, split_bits,
                                         trim_high_diag=True)
        n = total_bits // split_bits
        acc = 0
        for m in range(n):
            acc += partials[m] << (m * split_bits)
        lo = acc & 0xFFFFFFFF
        # 与 C++ dot_fused_i32 一致：返回有符号 int32（低 32 位位模式）
        return lo - (1 << 32) if lo >= (1 << 31) else lo

    @staticmethod
    def fuse_128(partials, split_bits: int):
        fused = sum(partials[m] << (m * split_bits) for m in range(len(partials)))
        lo = fused & ((1 << 64) - 1)
        hi = (fused >> 64) & ((1 << 64) - 1)
        if hi >= (1 << 63):
            hi -= (1 << 64)
        return (hi, lo)

    @staticmethod
    def prepare_nibble_from_raw(w: List[int], total_bits: int = 32) -> _PyNibblePrepared:
        """Python 等价 prepare_nibble_from_raw（C++ lambda 封装）。

        对长度为 K 的输入列表逐元素拆分 + 打包 + 预解包，产出 NibblePrepared。
        权重离线 / 激活每前向各调用一次，热路径 M 输出复用。
        """
        from typing import List
        split_bits = 4
        n = total_bits // split_bits
        K = len(w)
        bias = 8  # 2^3，与 C++ 偏置修正一致
        # 预分配
        prep = _PyNibblePrepared()
        prep.n = n
        prep.K = K
        prep.u8 = [[0] * K for _ in range(n)]   # w 侧：无符号 nibble → 字节
        prep.s8 = [[0] * K for _ in range(n)]   # x 侧：有符号 nibble → 字节
        prep.Sw = [0] * n
        prep.Sx = [0] * n
        # 逐元素拆分、打包、预解包
        for i in range(K):
            parts = _PySplitDot.split_parts(w[i], total_bits, split_bits)
            for a in range(n):
                v = parts[a]
                # 偏置修正（与 pack_narrow_value C++ 语义一致）：4 位部分恒在
                # [-8, 7]（低位 = value-8 偏置，最高位 = 符号扩展），
                # 无符号字节 u8 = s + 8（0..15），有符号字节 s8 = s（-8..7）。
                # Sw/Sx 均累加 s（C++ NarrowParts 对 w/x 同值，NibblePrepared 拷贝两份）。
                la = (a + 1 < n)  # 低位无符号需偏置
                s = (v - bias) if la else v
                prep.Sw[a] += s
                prep.Sx[a] += s
                prep.u8[a][i] = (s + bias) & 0xFF
                prep.s8[a][i] = s  # 有符号字节：与 C++ int8_t 语义一致，负数保持负值
        return prep

    @staticmethod
    def dot_prepared4(
        prep_w: _PyNibblePrepared, prep_x: _PyNibblePrepared,
        trim_high_diag: bool = False) -> List[int]:
        """Python 等价 narrow_dot_prepared4（4 位预解包热路径）。

        用两个预解包好的 NibblePrepared 做 n² 点积，裁剪语义与 C++ 一致：
        trim_high_diag=True 时跳过高位对角（m=a+c >= n），partials[m>=n] 保持 0。
        返回长度 2n-1，与 full path bit-exact 一致。
        """
        n = prep_w.n
        K = prep_w.K
        assert prep_x.n == n and prep_x.K == K
        bias = 8
        bias2 = bias * bias
        # corr 预计算（偏置修正项）
        partials = [0] * (2 * n - 1)
        for a in range(n):
            la = (a + 1 < n)
            for c in range(n):
                if trim_high_diag and (a + c >= n):
                    continue
                lc = (c + 1 < n)
                # 与 C++ narrow_dot_prepared4 的 corr 公式严格一致：
                #   bias*(la?0:-1)*Sx[c] + (lc?bias*Sw[a]:0) + (la&&lc? K*bias2 : 0)
                # 注意不能用 `la and 0 or -1` 这类写法——Python 中恒为 -1，与 C++ 三目不等价。
                corr = (bias * (0 if la else -1) * prep_x.Sx[c]
                       + (bias * prep_w.Sw[a] if lc else 0)
                       + (K * bias2 if (la and lc) else 0))
                partials[a + c] += corr
        # n² dp 点积（纯预解包字节）
        for a in range(n):
            for c in range(n):
                if trim_high_diag and (a + c >= n):
                    continue
                partials[a + c] += sum(
                    prep_w.u8[a][i] * prep_x.s8[c][i] for i in range(K)
                )
        return partials

    @staticmethod
    def matmul_prepared4(
        prep_ws: List[_PyNibblePrepared], prep_x: _PyNibblePrepared,
        trim_high_diag: bool = False) -> List[List[int]]:
        """Python 等价 matmul_prepared4（批量 M 输出摊销）。

        M 个预解包权重行复用同一预解包激活 x，逐行调用 dot_prepared4
        （C++ 版把 M 循环搬进内核，语义一致）。
        """
        return [
            _PySplitDot.dot_prepared4(pw, prep_x, trim_high_diag)
            for pw in prep_ws
        ]


# ============================================================
# Python fallback: MultiScaleView（1:N 多精度解释）
# ============================================================

class _PyMultiScaleView:
    """Python MultiScaleView 实现（接口对齐 C++ MultiScaleView）"""

    @staticmethod
    def default_levels(total_bits: int):
        if total_bits <= 0:
            raise ValueError("total_bits 必须 > 0")
        levels = []
        b = total_bits // 2
        while b >= 4 and total_bits % b == 0:
            levels.append(b)
            if b == 4:
                break
            b //= 2
        if not levels:
            levels = [16]
        return levels

    @staticmethod
    def interpret_levels(value, total_bits: int, split_bits_list):
        if not split_bits_list:
            raise ValueError("split_bits_list 不能为空")
        return {
            b: _PySplitDot.split_parts(value, total_bits, b)
            for b in split_bits_list
        }

    @staticmethod
    def interpret(value, total_bits: int = 32):
        return _PyMultiScaleView.interpret_levels(
            value, total_bits, _PyMultiScaleView.default_levels(total_bits))

    @staticmethod
    def interpret_batch(values, total_bits: int = 32, split_bits_list=None):
        levels = split_bits_list or _PyMultiScaleView.default_levels(total_bits)
        result = {b: [] for b in levels}
        for v in values:
            per = _PyMultiScaleView.interpret_levels(v, total_bits, levels)
            for b in levels:
                result[b].append(per[b])
        return result

    @staticmethod
    def is_exact(value, total_bits: int, split_bits: int) -> bool:
        parts = _PySplitDot.split_parts(value, total_bits, split_bits)
        rebuilt = sum(p << (k * split_bits) for k, p in enumerate(parts))
        return rebuilt == value


# ============================================================
# Python fallback: PrecisionSelector（Level 逐元素精度选择）
# ============================================================

class _PyPrecisionSelector:
    """Python PrecisionSelector 实现（接口对齐 C++ PrecisionSelector）"""

    def __init__(self, total_bits: int, options, thresholds):
        if total_bits <= 0:
            raise ValueError("total_bits 必须 > 0")
        if not options:
            raise ValueError("options 不能为空")
        for i, b in enumerate(options):
            if b <= 0:
                raise ValueError("每个 split_bits 必须 > 0")
            if total_bits % b != 0:
                raise ValueError(f"total_bits({total_bits}) 必须能被 split_bits({b}) 整除")
            if i > 0 and b >= options[i - 1]:
                raise ValueError("options 必须按从粗到细排列（split_bits 递减）")
        if len(thresholds) != len(options) - 1:
            raise ValueError(
                f"thresholds 长度({len(thresholds)}) 必须等于 options 长度-1({len(options)-1})")
        for i in range(1, len(thresholds)):
            if thresholds[i] <= thresholds[i - 1]:
                raise ValueError("thresholds 必须严格升序")
        self._total_bits = total_bits
        self._options = list(options)
        self._thresholds = list(thresholds)

    @classmethod
    def default_selector(cls):
        return cls(32, [16, 8, 4], [100, 1000])

    def select(self, importance: int) -> int:
        idx = 0
        for k, t in enumerate(self._thresholds):
            if importance >= t:
                idx = k + 1
            else:
                break
        return self._options[idx]

    def interpret(self, value, importance: int):
        return _PySplitDot.split_parts(value, self._total_bits, self.select(importance))

    def interpret_batch(self, values, importances):
        if len(values) != len(importances):
            raise ValueError("values 和 importances 长度不一致")
        return [self.interpret(v, imp) for v, imp in zip(values, importances)]

    @property
    def total_bits(self):
        return self._total_bits

    @property
    def options(self):
        return self._options

    @property
    def thresholds(self):
        return self._thresholds


# ============================================================
# Python fallback: LeveledSplitDot（1:N 多精度解释 + 逐元素精度选择组合）
# ============================================================

class _PyLeveledSplitDot:
    """Python LeveledSplitDot 实现（接口对齐 C++ LeveledSplitDot）

    组合落地：对一批 w/x 元素，每个元素按重要性独立选择拆分粒度，
    异构粒度拆分后按粒度分组做多输出点积，单融合时精确重建原始点积。
    """

    @staticmethod
    def select_levels(importance, total_bits=32, options=None, thresholds=None):
        sel = _PyPrecisionSelector(
            total_bits, options or [16, 8, 4], thresholds or [100, 1000])
        return [sel.select(imp) for imp in importance]

    @staticmethod
    def dot_split_leveled(w, x, importance, total_bits=32, options=None, thresholds=None,
                          trim_high_diag=False):
        if len(w) != len(x) or len(w) != len(importance):
            raise ValueError("w/x/importance 长度不一致")
        levels = _PyLeveledSplitDot.select_levels(importance, total_bits, options, thresholds)
        groups = {}
        for i, b in enumerate(levels):
            groups.setdefault(b, ([], []))
            groups[b][0].append(w[i])
            groups[b][1].append(x[i])
        return {
            b: _PySplitDot.dot_split(wg, xg, total_bits, b, trim_high_diag)
            for b, (wg, xg) in groups.items()
        }

    @staticmethod
    def dot_fused_leveled(w, x, importance, total_bits=32, options=None, thresholds=None):
        groups = _PyLeveledSplitDot.dot_split_leveled(
            w, x, importance, total_bits, options, thresholds)
        total = 0
        for b, partials in groups.items():
            hi, lo = _PySplitDot.fuse_128(partials, b)
            total += (hi << 64) | lo
        lo = total & ((1 << 64) - 1)
        return lo - (1 << 64) if lo >= (1 << 63) else lo

    @staticmethod
    def select_levels_default(importance, total_bits=32):
        """默认参数版本（接口对齐 C++ LeveledSplitDot.select_levels_default）。"""
        return _PyLeveledSplitDot.select_levels(importance, total_bits)

    @staticmethod
    def dot_split_leveled_default(w, x, importance, total_bits=32):
        """默认参数版本（接口对齐 C++ LeveledSplitDot.dot_split_leveled_default）。"""
        return _PyLeveledSplitDot.dot_split_leveled(w, x, importance, total_bits)

    @staticmethod
    def dot_fused_leveled_default(w, x, importance, total_bits=32):
        """默认参数版本（接口对齐 C++ LeveledSplitDot.dot_fused_leveled_default）。"""
        return _PyLeveledSplitDot.dot_fused_leveled(w, x, importance, total_bits)

    @staticmethod
    def select_precision(importance, options=None, thresholds=None):
        """降档决策：按重要性选择精度位数 p（Level 精度调度语义）。

        options: 精度位数列表，从低精度 → 高精度排列（如 {8, 16, 32}），
        thresholds: 升序阈值，长度 = options.size()-1。
        语义：重要性越低 → 精度位数越少（降档越多），高重要度 → 全精度。
        注意：不能复用 select_levels——_PyPrecisionSelector 要求 options 递减
        （粒度语义），而降档的精度位数是递增的。
        """
        opts = options or [8, 16, 32]
        ths = thresholds or [100, 1000]
        if not opts:
            raise ValueError("options 不能为空")
        if len(ths) != len(opts) - 1:
            raise ValueError("thresholds 长度必须等于 options 长度-1")
        levels = []
        for imp in importance:
            idx = 0
            for k, t in enumerate(ths):
                if imp >= t:
                    idx = k + 1
                else:
                    break
            levels.append(opts[idx])
        return levels

    @staticmethod
    def dot_split_leveled_downcast(w, x, importance, total_bits=32,
                                   options=None, thresholds=None):
        """降档摊销多输出：按精度分组，keep_top 后走 4 位摊销路径。

        返回 {p: partials}（p-bit 原始 partials，未缩放；恢复量纲需左移 2(32-p)）。
        """
        if len(w) != len(x) or len(w) != len(importance):
            raise ValueError("w/x/importance 长度不一致")
        levels = _PyLeveledSplitDot.select_precision(importance, options, thresholds)
        groups = {}
        for i, p in enumerate(levels):
            groups.setdefault(p, ([], []))
            groups[p][0].append(w[i])
            groups[p][1].append(x[i])
        result = {}
        for p, (wg, xg) in groups.items():
            s = total_bits - p
            wp = [v >> s for v in wg]
            xp = [v >> s for v in xg]
            pw = _PySplitDot.prepare_nibble_from_raw(wp, p)
            px = _PySplitDot.prepare_nibble_from_raw(xp, p)
            result[p] = _PySplitDot.dot_prepared4(pw, px, False)
        return result

    @staticmethod
    def dot_fused_leveled_downcast(w, x, importance, total_bits=32,
                                   options=None, thresholds=None):
        """降档摊销单融合：跨组恢复量纲 + 累加，返回低 64 位近似结果。

        数值上 ≈ Σ_i w_i·x_i，截断误差 ≈ 2^(1-p)（p 为各元素降档精度）。
        """
        groups = _PyLeveledSplitDot.dot_split_leveled_downcast(
            w, x, importance, total_bits, options, thresholds)
        total = 0
        for p, partials in groups.items():
            hi, lo = _PySplitDot.fuse_128(partials, 4)
            total += ((hi << 64) | lo) << (2 * (total_bits - p))
        lo = total & ((1 << 64) - 1)
        return lo - (1 << 64) if lo >= (1 << 63) else lo

    @staticmethod
    def select_precision_default(importance):
        """降档决策便捷版：默认 {8,16,32}/{100,1000}。"""
        return _PyLeveledSplitDot.select_precision(importance)

    @staticmethod
    def prepare_downcast(w, importance, total_bits=32, options=None, thresholds=None):
        """降档预解包融合：一次传入原始 w，内部分组 + keep_top 截断 + prepare。

        返回 {p: NibblePrepared}（与 C++ prepare_downcast 语义一致）。
        """
        if len(w) != len(importance):
            raise ValueError("w/importance 长度不一致")
        levels = _PyLeveledSplitDot.select_precision(importance, options, thresholds)
        groups = {}
        for i, p in enumerate(levels):
            groups.setdefault(p, []).append(w[i])
        return {
            p: _PySplitDot.prepare_nibble_from_raw([v >> (total_bits - p) for v in g], p)
            for p, g in groups.items()
        }

    @staticmethod
    def dot_split_leveled_downcast_default(w, x, importance, total_bits=32):
        """降档摊销多输出便捷版：默认 {8,16,32}/{100,1000}。"""
        return _PyLeveledSplitDot.dot_split_leveled_downcast(w, x, importance, total_bits)

    @staticmethod
    def dot_fused_leveled_downcast_default(w, x, importance, total_bits=32):
        """降档摊销单融合便捷版：默认 {8,16,32}/{100,1000}。"""
        return _PyLeveledSplitDot.dot_fused_leveled_downcast(w, x, importance, total_bits)


# ============================================================
# 统一 API: CppPackedBackend / CppMSIntView / CppSplitDot
# ============================================================

if _USING_CPP:
    # C++ 模式：直接用 C++ 扩展
    CppPackedBackend = _cpp_PackedBackend
    CppMSIntView = _cpp_MSIntView
    CppSplitDot = _cpp_SplitDot
    CppMultiScaleView = _cpp_MultiScaleView
    CppPrecisionSelector = _cpp_PrecisionSelector
    CppLeveledSplitDot = _cpp_LeveledSplitDot
else:
    # Python fallback 模式
    CppPackedBackend = _PyPackedBackendWrapper
    CppMSIntView = _PyMSIntView
    CppSplitDot = _PySplitDot
    CppMultiScaleView = _PyMultiScaleView
    CppPrecisionSelector = _PyPrecisionSelector
    CppLeveledSplitDot = _PyLeveledSplitDot


# 导出
__all__ = [
    "CppPackedBackend",
    "CppMSIntView",
    "CppSplitDot",
    "CppMultiScaleView",
    "CppPrecisionSelector",
    "CppLeveledSplitDot",
    "batch_get_all",
    "batch_decode_to_float",
    "batch_decode_to_float_into",
    "USING_CPP",
]

# 全局标志：是否使用 C++ 扩展
USING_CPP = _USING_CPP
