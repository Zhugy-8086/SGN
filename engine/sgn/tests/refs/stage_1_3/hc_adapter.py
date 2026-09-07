"""HC8 适配器：纯 Python 实现 HC8 分层坐标 + PyTorch tensor 互转

阶段 1.3（介入深度 3：全整数路径）的核心组件。

为什么用纯 Python 而不是 pysgn：
  - pysgn（C 扩展）需要编译，当前环境未编译
  - 阶段 1.3 的目标是验证 HC 分层坐标的可行性，不是性能
  - 纯 Python 实现可直接移植 hc8.c 的逻辑，无需 C 编译器
  - 后续性能优化阶段可切换到 pysgn

HC8 数据结构（参考 engine/hc/sgn/include/hc/hc8.h）：
  - 6 字节数组 v[0..5]
  - 物理值 = v[0] + v[1]/256 + v[2]/256² + v[3]/256³ + v[4]/256⁴ + v[5]/256⁵
  - 范围 [0, 256)，超度量树深度 6
  - 字节序即字典序，可直接用 tuple 比较

设计：
  - 对称线性量化：scale = max(|w|) / 127，q = round(w / scale) ∈ [-127, 127]
  - 偏移到无符号：q_u = q + 128 ∈ [1, 255]（0 留给零值特判）
  - HC8 只用 v[0] 层存量化值，v[1..5] = 0（阶段 1.3 不分层累加）
  - 后续阶段可用 v[1..5] 存更高精度残差

性能说明：
  - 阶段 1.3 的目标是验证精度，不是性能
  - 纯 Python HC8 比浮点 PyTorch 慢 10-100 倍
  - 每 N 步编码-解码（默认 N=50），避免每步都做 100k+ HC8 构造
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ============================================================
# HC8 常量（参考 hc8.c SGN_HC8_ZERO / SGN_HC8_MAX）
# ============================================================

HC8_LAYERS = 6          # HC8 固定 6 层
HC8_BASE = 256          # 每层基数 256
HC8_MAX_PHYS = 256.0    # 物理值上限（饱和用）


# ============================================================
# HC8 类（纯 Python 实现，对应 C 的 hc8_t）
# ============================================================

class HC8:
    """HC8: 6 层 × 8 位分层坐标

    物理值 = v[0] + v[1]/256 + v[2]/256² + v[3]/256³ + v[4]/256⁴ + v[5]/256⁵
    范围 [0, 256)，超度量树深度 6

    实现参考：
      - engine/hc/sgn/include/hc/hc8.h（数据结构声明）
      - engine/hc/sgn/src/hc8.c（运算实现）
      - engine/hc/sgn/src/hc.c（hc_physical_value / hc_from_double 通用实现）
    """

    __slots__ = ("v",)  # 6 字节数组

    def __init__(self, v: Optional[List[int]] = None):
        """从 6 字节列表构造

        Args:
            v: 长度 6 的列表，每个元素 0~255。None 表示零值。
        """
        if v is None:
            self.v = bytearray(6)
        else:
            if len(v) != 6:
                raise ValueError(f"HC8 需要 6 字节，得到 {len(v)}")
            self.v = bytearray(b & 0xFF for b in v)

    # ---- 工厂方法 ----

    @classmethod
    def from_float(cls, x: float, overflow: str = "saturation") -> "HC8":
        """从浮点数构造（参考 hc8.c hc8_from_double）

        Args:
            x: 浮点值（负数会被截断到 0，>= 256 饱和到 max）
            overflow: "saturation"=饱和（默认），"wrap"=回绕

        Returns:
            HC8 实例
        """
        if overflow not in ("saturation", "wrap"):
            raise ValueError(f"overflow 必须是 saturation/wrap，得到 {overflow}")

        # 处理负数
        if x < 0.0:
            if overflow == "wrap":
                x = x % HC8_BASE
                if x < 0:
                    x += HC8_BASE
            else:
                x = 0.0

        # 处理溢出
        if x >= HC8_MAX_PHYS:
            if overflow == "wrap":
                x = x % HC8_BASE
            else:
                # 饱和到 max（255.999...）
                return cls([255, 255, 255, 255, 255, 255])

        # 逐层提取（参考 hc.c hc_from_double）
        v = bytearray(6)
        scaled = float(x)
        for i in range(6):
            elem = int(scaled)  # 整数部分
            if elem > 255:
                elem = 255
            elif elem < 0:
                elem = 0
            v[i] = elem
            scaled = (scaled - elem) * HC8_BASE

        return cls(list(v))

    @classmethod
    def from_bytes(cls, b: bytes) -> "HC8":
        """从 6 字节序列反序列化"""
        if len(b) != 6:
            raise ValueError(f"HC8 需要 6 字节，得到 {len(b)}")
        return cls(list(b))

    @classmethod
    def zero(cls) -> "HC8":
        """零常量（对应 SGN_HC8_ZERO）"""
        return cls([0, 0, 0, 0, 0, 0])

    @classmethod
    def max_val(cls) -> "HC8":
        """最大值常量（对应 SGN_HC8_MAX）"""
        return cls([255, 255, 255, 255, 255, 255])

    # ---- 转换 ----

    def to_float(self) -> float:
        """转浮点数（参考 hc.c hc_physical_value）

        物理值 = v[0] + v[1]/256 + v[2]/256² + ... + v[5]/256⁵
        """
        result = 0.0
        scale = 1.0
        for i in range(6):
            result += self.v[i] * scale
            scale /= HC8_BASE
        return result

    def to_bytes(self) -> bytes:
        """序列化为 6 字节"""
        return bytes(self.v)

    # ---- 层访问 ----

    def __getitem__(self, idx: int) -> int:
        """层访问（0=整数部分层，1..5=小数层）"""
        return self.v[idx]

    def __setitem__(self, idx: int, value: int) -> None:
        self.v[idx] = value & 0xFF

    def __len__(self) -> int:
        return 6

    # ---- 比较（字典序，参考 hc8.c hc8_less / hc8_equal） ----

    def __eq__(self, other) -> bool:
        if not isinstance(other, HC8):
            return NotImplemented
        return self.v == other.v

    def __ne__(self, other) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __lt__(self, other) -> bool:
        if not isinstance(other, HC8):
            return NotImplemented
        return self.v < other.v

    def __le__(self, other) -> bool:
        if not isinstance(other, HC8):
            return NotImplemented
        return self.v <= other.v

    def __gt__(self, other) -> bool:
        if not isinstance(other, HC8):
            return NotImplemented
        return self.v > other.v

    def __ge__(self, other) -> bool:
        if not isinstance(other, HC8):
            return NotImplemented
        return self.v >= other.v

    def __hash__(self) -> int:
        return hash(bytes(self.v))

    # ---- 加法（饱和，参考 hc8.c hc8_add_sat） ----

    def __add__(self, other: "HC8") -> "HC8":
        """饱和加法：溢出时饱和到 max"""
        if not isinstance(other, HC8):
            return NotImplemented
        result = bytearray(6)
        carry = 0
        for i in range(5, -1, -1):
            s = self.v[i] + other.v[i] + carry
            result[i] = s & 0xFF
            carry = s >> 8
        if carry:
            # 溢出饱和到 max
            return HC8.max_val()
        return HC8(list(result))

    def add_wrap(self, other: "HC8") -> "HC8":
        """回绕加法（参考 hc8.c hc8_add_wrap）"""
        if not isinstance(other, HC8):
            return NotImplemented
        result = bytearray(6)
        carry = 0
        for i in range(5, -1, -1):
            s = self.v[i] + other.v[i] + carry
            result[i] = s & 0xFF
            carry = s >> 8
        return HC8(list(result))

    # ---- 减法（饱和到 0，参考 hc8.c hc8_sub） ----

    def __sub__(self, other: "HC8") -> "HC8":
        """饱和减法：结果 < 0 时饱和到 0"""
        if not isinstance(other, HC8):
            return NotImplemented
        # 借位减法
        result = bytearray(6)
        borrow = 0
        for i in range(5, -1, -1):
            diff = self.v[i] - other.v[i] - borrow
            if diff < 0:
                diff += 256
                borrow = 1
            else:
                borrow = 0
            result[i] = diff
        if borrow:
            # 结果为负，饱和到 0
            return HC8.zero()
        return HC8(list(result))

    # ---- 移位（参考 hc8.c hc8_shift_right） ----

    def shift_right(self, shift: int) -> "HC8":
        """右移 shift 位（除以 256^shift）"""
        if shift < 0:
            raise ValueError(f"shift 必须 >= 0，得到 {shift}")
        if shift >= 6:
            return HC8.zero()
        v = [0] * 6
        for i in range(6 - shift):
            v[i] = self.v[i + shift]
        return HC8(v)

    # ---- 校验和（参考 hc8.c hc8_checksum） ----

    def checksum(self) -> int:
        """计算校验和（权重 [2,3,4,5,6,7] 加权和 mod 256）"""
        weights = [2, 3, 4, 5, 6, 7]
        s = sum(w * self.v[i] for i, w in enumerate(weights))
        return s & 0xFF

    # ---- 表示 ----

    def __repr__(self) -> str:
        return f"HC8({list(self.v)}, phys={self.to_float():.6f})"

    def __str__(self) -> str:
        return f"HC8({list(self.v)})"


# ============================================================
# SHC8 类：有符号 HC8（参考 hc8.h shc8_t）
# ============================================================

class SHC8:
    """有符号 HC8（sign + int_part + 5 层小数）

    用于权重等需要负数的场景。物理值 = (-1)^sign * (int_part + frac.v[0]/256 + ...)

    实现参考 engine/hc/sgn/include/hc/hc8.h shc8_t
    """

    __slots__ = ("sign", "int_part", "frac")

    def __init__(self, sign: int = 0, int_part: int = 0, frac: Optional[HC8] = None):
        self.sign = sign & 0xFF
        self.int_part = int_part & 0xFF
        self.frac = frac if frac is not None else HC8.zero()

    @classmethod
    def from_float(cls, x: float) -> "SHC8":
        """从浮点数构造有符号 HC8"""
        if x == 0.0:
            return cls(sign=0, int_part=0, frac=HC8.zero())
        sign = 1 if x < 0 else 0
        abs_x = abs(x)
        if abs_x >= 256.0:
            abs_x = 255.999999
        int_part = int(abs_x)
        # 小数部分编码到 frac 的 v[0..4]（5 层，因为 int_part 单独存）
        frac = bytearray(6)
        scaled = abs_x - int_part
        for i in range(5):  # 只填 v[0..4]，v[5] 留 0
            scaled *= 256
            elem = int(scaled)
            if elem > 255:
                elem = 255
            frac[i] = elem
            scaled -= elem
        return cls(sign=sign, int_part=int_part, frac=HC8(list(frac)))

    def to_float(self) -> float:
        """转浮点数"""
        val = self.int_part
        scale = 1.0 / 256
        for i in range(5):
            val += self.frac.v[i] * scale
            scale /= 256
        return -val if self.sign else val

    def __repr__(self) -> str:
        return f"SHC8(sign={self.sign}, int={self.int_part}, frac={list(self.frac.v[:5])}, phys={self.to_float():.6f})"


# ============================================================
# HC8 量化方案
# ============================================================

class HC8WeightSchema:
    """HC8 权重量化方案（对称线性量化 + 偏移到无符号）

    流程：
      1. 对称量化：scale = max(|w|) / 127，q = round(w / scale) ∈ [-127, 127]
      2. 偏移到无符号：q_u = q + 128 ∈ [1, 255]（0 留给"未初始化"特判）
      3. HC8 只用 v[0] 层存 q_u，v[1..5] = 0

    反量化：
      1. q_u = hc8.v[0]
      2. q = q_u - 128
      3. w = q * scale

    为什么偏移 128 而不是用 SHC8：
      - HC8 的运算（add/sub/compare）更成熟，pysgn 也主要暴露 HC8
      - 偏移后可直接用 HC8 的整数运算（矩阵乘时统一偏移即可）
      - SHC8 留给未来需要更大动态范围的场景
    """

    def __init__(self, bits: int = 8):
        """Args:
            bits: 量化位宽（必须 8，HC8 的 v[0] 是 8 位）
        """
        if bits != 8:
            raise ValueError(f"HC8WeightSchema 只支持 8-bit，得到 {bits}")
        self.bits = 8
        self.qmin = -127   # 对称量化下界
        self.qmax = 127    # 对称量化上界（避免 -128 导致不对称）
        self.offset = 128  # 偏移到 [1, 255]

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """量化 float tensor 到无符号 int tensor

        Returns:
            (quantized_uint8_tensor, scale)
            quantized_uint8_tensor: 偏移后的无符号值 [1, 255]，int32 类型
            scale: 浮点缩放因子
        """
        t = tensor.detach()
        max_abs = t.abs().max().item()
        if max_abs == 0.0:
            scale = 1.0
        else:
            scale = max_abs / self.qmax
        q = torch.round(t / scale).clamp(self.qmin, self.qmax).to(torch.int32)
        q_u = q + self.offset  # 偏移到 [1, 255]
        return q_u, scale

    def dequantize(self, q_u: torch.Tensor, scale: float) -> torch.Tensor:
        """反量化无符号 int tensor 到 float tensor

        Args:
            q_u: 偏移后的无符号值 [1, 255]
            scale: 量化时的 scale
        """
        q = q_u.to(torch.float32) - self.offset
        return q * scale

    def __repr__(self) -> str:
        return f"HC8WeightSchema(bits=8, offset={self.offset}, qmin={self.qmin}, qmax={self.qmax})"


# ============================================================
# tensor ↔ HC8 互转
# ============================================================

def tensor_to_hc8(
    tensor: torch.Tensor,
    schema: Optional[HC8WeightSchema] = None,
) -> Tuple[List[HC8], float, torch.Size]:
    """float tensor → HC8 列表

    Args:
        tensor: float tensor（任意形状）
        schema: 量化方案（None 表示新建 HC8WeightSchema）

    Returns:
        (hc8_list, scale, shape)
        hc8_list: 扁平化的 HC8 列表（长度 = tensor.numel()）
        scale: 量化 scale
        shape: 原始 tensor 形状
    """
    if schema is None:
        schema = HC8WeightSchema()

    q_u, scale = schema.quantize(tensor)
    flat = q_u.flatten().tolist()

    # 每个 int 值编码为 HC8（只填 v[0]，其余 0）
    hc8_list = []
    for v0 in flat:
        # v0 ∈ [1, 255]，直接存入 v[0]
        hc8_list.append(HC8([int(v0) & 0xFF, 0, 0, 0, 0, 0]))

    return hc8_list, scale, tensor.shape


def hc8_to_tensor(
    hc8_list: List[HC8],
    scale: float,
    shape: torch.Size,
    schema: Optional[HC8WeightSchema] = None,
) -> torch.Tensor:
    """HC8 列表 → float tensor

    Args:
        hc8_list: 扁平化的 HC8 列表
        scale: 量化时的 scale
        shape: 目标 tensor 形状
        schema: 量化方案（None 表示新建 HC8WeightSchema）

    Returns:
        反量化后的 float tensor
    """
    if schema is None:
        schema = HC8WeightSchema()

    # 从 HC8 提取 v[0]（量化值）
    q_u = torch.tensor([h.v[0] for h in hc8_list], dtype=torch.int32)
    q_u = q_u.view(shape)
    return schema.dequantize(q_u, scale)


# ============================================================
# HC8 权重编码器（参考 stage_1_1 的 MSIntWeightEncoder 模式）
# ============================================================

class HC8WeightEncoder:
    """HC8 权重编码器：每 N 步把模型权重编码到 HC8 再解码回来

    设计：
      - 对指定层做"编码-解码"往返：float → HC8 → float
      - 模拟整数前向路径的量化噪声
      - 每 N 步执行一次（默认 N=50），避免每步都做 100k+ HC8 构造
      - 记录量化 MSE、编码时间等统计

    与 stage_1_1 MSIntWeightEncoder 的差异：
      - 用 HC8 替代 MSInt 存储
      - 量化方案是 8-bit 对称 + 偏移（MSInt 是直接存 int）
      - HC8 只用 v[0] 层（后续阶段可扩展到 v[1..5]）
    """

    def __init__(
        self,
        model: nn.Module,
        schema: Optional[HC8WeightSchema] = None,
        encode_interval: int = 50,
    ):
        """Args:
            model: PyTorch 模型
            schema: 量化方案（None 表示新建）
            encode_interval: 编码间隔（步数）
        """
        self.model = model
        self.schema = schema if schema is not None else HC8WeightSchema()
        self.encode_interval = encode_interval

        # 注册的层名 → 层对象
        self._layers: Dict[str, nn.Module] = {}

        # 统计
        self.encode_count = 0
        self.total_encode_time = 0.0
        self.total_quant_mse = 0.0
        self._last_quant_mse = 0.0

    def register_layer(self, name: str) -> None:
        """注册需要编码的层"""
        layer = self.model
        for part in name.split("."):
            layer = getattr(layer, part)
        self._layers[name] = layer

    def maybe_encode_decode(self, step: int) -> bool:
        """如果到了编码间隔，执行一次编码-解码往返

        Returns:
            是否实际执行了编码
        """
        if step % self.encode_interval != 0 or step == 0:
            return False

        t_start = time.time()
        any_encoded = False
        total_sq_error = 0.0
        total_count = 0

        with torch.no_grad():
            for name, layer in self._layers.items():
                if not hasattr(layer, "weight") or layer.weight is None:
                    continue
                w = layer.weight.data
                # 编码：float → HC8
                hc8_list, scale, shape = tensor_to_hc8(w, self.schema)
                # 解码：HC8 → float
                w_deq = hc8_to_tensor(hc8_list, scale, shape, self.schema)
                # 计算量化 MSE
                sq_error = ((w - w_deq) ** 2).sum().item()
                total_sq_error += sq_error
                total_count += w.numel()
                # 写回
                layer.weight.data.copy_(w_deq)
                any_encoded = True

        elapsed = time.time() - t_start
        self.encode_count += 1
        self.total_encode_time += elapsed
        if total_count > 0:
            self._last_quant_mse = total_sq_error / total_count
            self.total_quant_mse += self._last_quant_mse

        return any_encoded

    def get_stats(self) -> dict:
        """获取编码统计"""
        avg_mse = (
            self.total_quant_mse / self.encode_count
            if self.encode_count > 0
            else 0.0
        )
        return {
            "encode_count": self.encode_count,
            "total_encode_time_s": self.total_encode_time,
            "last_quant_mse": self._last_quant_mse,
            "avg_quant_mse": avg_mse,
            "encode_interval": self.encode_interval,
            "registered_layers": list(self._layers.keys()),
        }


# ============================================================
# 自检函数（不运行测试，仅用于 import 时验证基本逻辑）
# ============================================================

def _self_check() -> None:
    """模块加载时的自检（仅打印信息，不抛异常）

    在开发阶段可手动调用以验证基本逻辑正确性。
    生产代码中不调用此函数。
    """
    # 1. 基本构造与物理值
    h_zero = HC8.zero()
    assert h_zero.to_float() == 0.0, f"零值物理值应为 0，得到 {h_zero.to_float()}"

    h_max = HC8.max_val()
    # max 物理值 = 255 + 255/256 + ... ≈ 255.996
    expected_max = 255 + sum(255 / (256 ** i) for i in range(1, 6))
    assert abs(h_max.to_float() - expected_max) < 1e-9, \
        f"max 物理值应为 {expected_max}，得到 {h_max.to_float()}"

    # 2. from_float 往返
    h = HC8.from_float(3.14159)
    assert abs(h.to_float() - 3.14159) < 1e-3, \
        f"from_float(3.14159) 往返误差过大：{h.to_float()}"

    # 3. 加法（饱和）
    a = HC8.from_float(100.0)
    b = HC8.from_float(200.0)
    c = a + b  # 300 → 饱和到 max
    assert c == HC8.max_val(), f"100+200 应饱和到 max，得到 {c}"

    # 4. 减法（饱和到 0）
    d = b - a  # 100
    assert abs(d.to_float() - 100.0) < 1e-3, f"200-100=100，得到 {d.to_float()}"

    e = a - b  # -100 → 饱和到 0
    assert e == HC8.zero(), f"100-200 应饱和到 0，得到 {e}"

    # 5. 比较
    assert HC8.from_float(1.0) < HC8.from_float(2.0)
    assert HC8.from_float(1.0) == HC8.from_float(1.0)

    # 6. tensor 互转
    t = torch.randn(4, 3)
    schema = HC8WeightSchema()
    hc8_list, scale, shape = tensor_to_hc8(t, schema)
    t_back = hc8_to_tensor(hc8_list, scale, shape, schema)
    mse = ((t - t_back) ** 2).mean().item()
    # 8-bit 对称量化 MSE 应该很小（< 1e-3 量级）
    assert mse < 1e-2, f"tensor 互转 MSE 过大：{mse}"

    print(f"[HC8 self-check] OK. tensor roundtrip MSE={mse:.6e}")
    print(f"  HC8(3.14159) = {HC8.from_float(3.14159)}")
    print(f"  HC8(100.0) + HC8(200.0) = {HC8.from_float(100.0) + HC8.from_float(200.0)} (saturated)")
    print(f"  HC8WeightSchema: {schema}")


if __name__ == "__main__":
    _self_check()
