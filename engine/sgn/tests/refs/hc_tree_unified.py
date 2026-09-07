#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HC 树统一框架：UFP 串行 vs 正交并行的 Python 参考实现

方案 C 核心交付物。实现 HC 树的两种数学解读：
  - UFP 串行（残差链）：v[0] 主量化, v[1] 残差... 解码 w ≈ Σ v[i] * scales[i]
  - 正交并行（位段）：c_i = (C >> 8i) & 0xFF 独立提取，解码 C = Σ c_i * 2^(8i)

两种解读共享同一存储（v[N] 字节数组），运算规则不同：
  - 串行 matmul：(depth+1)² 次残差交叉项（适合浅层单步精度）
  - 并行 matmul：N 次位段独立（适合深层跨层精度）

设计原则（遵循用户偏好）：
  - 不修改现有 C 扩展（Python 层实现，未来再迁移）
  - 向后兼容（Level 0 时与当前 HC8 行为一致）
  - 数学严格性（所有运算有形式化定义，见 hc_tree_multiview_unified.md）

关联文档：[内部档案
关联测试：[内部档案

v1.4-rc20 方案 C 实施记录
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


# ============================================================
# 常量
# ============================================================

HC8_OFFSET = 128  # HC8 量化偏移（与 C 扩展一致）
HC8_QMIN = -127
HC8_QMAX = 127


# ============================================================
# 编码模式枚举
# ============================================================

class InterpretationMode:
    """HC 树解读模式"""
    SERIAL = "serial"      # UFP 串行（残差链）
    PARALLEL = "parallel"  # 正交并行（位段）


# ============================================================
# HC 树节点
# ============================================================

@dataclass
class HCTreeNode:
    """HC 树节点（N 字节存储 + 解读元数据）

    存储格式与 C 扩展 HC8 完全兼容：v[N] 字节数组。
    解读模式由 mode 字段决定，不改变存储。

    Attributes:
        v: N 字节数组（uint8），HC8 时 N=6
        scales: UFP 串行解读的 scales 序列（递减），并行模式不用
        mode: 解读模式（"serial" | "parallel"）
    """
    v: np.ndarray  # uint8, shape (N,)
    scales: List[float] = field(default_factory=list)  # 串行模式用
    mode: str = InterpretationMode.SERIAL

    @property
    def n_bytes(self) -> int:
        return len(self.v)


# ============================================================
# 量化/反量化（两种模式共用）
# ============================================================

def quantize_int8(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """量化到 int8（与 HC8 C 扩展一致）

    Returns:
        q: int8 量化值（[-127, 127]）
        scale: 量化 scale = max(|x|) / 127
    """
    x_max = float(np.abs(x).max()) if x.size > 0 else 0.0
    scale = x_max / 127.0 if x_max > 0 else 1.0
    q = np.round(x / scale).astype(np.int32)
    np.clip(q, HC8_QMIN, HC8_QMAX, out=q)
    return q.astype(np.int8), scale


def dequantize_int8(q: np.ndarray, scale: float) -> np.ndarray:
    """反量化 int8 → float"""
    return q.astype(np.float64) * scale


# ============================================================
# UFP 串行解读（残差链）
# ============================================================

def encode_serial(x: np.ndarray, depth: int = 1) -> HCTreeNode:
    """UFP 串行编码（残差链）

    v[0] = quant(x)
    v[1] = quant(x - dequant(v[0]))
    v[2] = quant(x - dequant(v[0]) - dequant(v[1]))
    ...

    Args:
        x: float 数组
        depth: 残差深度（0=仅主量化，1=1层残差，2=2层残差）

    Returns:
        HCTreeNode，v 数组长度 = depth+1，scales 长度 = depth+1
    """
    n_views = depth + 1
    v_list = []
    scales = []
    residual = x.copy()

    for i in range(n_views):
        q, scale = quantize_int8(residual)
        v_list.append((q.astype(np.uint8) + HC8_OFFSET).astype(np.uint8))  # 偏移到 [0, 255]
        scales.append(scale)
        residual = residual - dequantize_int8(q, scale)

    # 拼接为 v[N]（HC8 格式：每个元素 1 字节，这里简化为 uint8 数组）
    # 实际 HC8 是 6 字节，这里用 depth+1 字节（参考实现）
    v = np.stack(v_list, axis=-1)  # shape: (*x.shape, n_views)
    v_flat = v.reshape(-1, n_views)

    return HCTreeNode(v=v_flat, scales=scales, mode=InterpretationMode.SERIAL)


def matmul_serial(
    a_node: HCTreeNode,
    b_node: HCTreeNode,
    m: int, k: int, n: int,
) -> np.ndarray:
    """UFP 串行 matmul（残差交叉项）

    C = Σ_i Σ_j (a_i @ b_j) * (scales_a[i] * scales_b[j])

    复杂度：(depth_a+1) * (depth_b+1) 次 int8×int8 matmul。

    Args:
        a_node: 左侧 HC 树节点，v shape (m*k, n_views_a)
        b_node: 右侧 HC 树节点，v shape (k*n, n_views_b)
        m, k, n: 矩阵维度

    Returns:
        C: float64 数组 (m, n)
    """
    n_views_a = a_node.v.shape[-1]
    n_views_b = b_node.v.shape[-1]

    # 提取 int8 量化值（去偏移）
    a_views = []
    for i in range(n_views_a):
        a_i = a_node.v[:, i].reshape(m, k).astype(np.int32) - HC8_OFFSET
        a_views.append(a_i)

    b_views = []
    for j in range(n_views_b):
        b_j = b_node.v[:, j].reshape(k, n).astype(np.int32) - HC8_OFFSET
        b_views.append(b_j)

    # 残差交叉项累加
    C = np.zeros((m, n), dtype=np.float64)
    for i in range(n_views_a):
        for j in range(n_views_b):
            # int8 × int8 → int32 累加
            c_ij = a_views[i].astype(np.int64) @ b_views[j].astype(np.int64)
            C += c_ij.astype(np.float64) * (a_node.scales[i] * b_node.scales[j])

    return C


def decode_serial(node: HCTreeNode, shape: Optional[Tuple] = None) -> np.ndarray:
    """UFP 串行解码

    w ≈ Σ v[i] * scales[i]
    """
    n_views = node.v.shape[-1]
    result = None
    for i in range(n_views):
        q_i = node.v[:, i].astype(np.int32) - HC8_OFFSET
        contribution = q_i.astype(np.float64) * node.scales[i]
        if result is None:
            result = contribution
        else:
            result += contribution

    if shape is not None:
        result = result.reshape(shape)
    return result


# ============================================================
# 正交并行解读（位段）
# ============================================================
#
# 重要：正交多视角 matmul 的正确用法是分解**累加值 C**（int32/int64），
# 不是分解**原始 int8 输入**。原因：
#   - int8 量化值 q ∈ [-127, 127]，负数在补码下高位字节是 0xFF
#   - 若分解 q 为 uint8 视角，重构时会变成大正数（符号丢失）
#   - 累加值 C 是 int32/int64，分解后重构（模 2^32/2^64）保持正确
#
# 正确流程：
#   首层：x_int8 @ w_int8 → C (int32/int64)
#   后续层：decompose(C) → views; C_next = Σ (views[i] @ w_int8) * 2^(8i)
#
# 参考：orthogonal_multiview_theoretical_test.py

def encode_parallel(x: np.ndarray, n_views: int = 4) -> Tuple[np.ndarray, float]:
    """正交并行编码（量化到 int8，不分解）

    正交多视角 matmul 中，输入和权重都用原始 int8（[-127, 127]），
    不分解为 uint8 视角。视角分解只用于**累加值 C**（见 decompose_parallel）。

    Args:
        x: float 数组
        n_views: 视角数（保留参数，本函数不用，仅为接口一致）

    Returns:
        q: int8 量化值（[-127, 127]）
        scale: 量化 scale
    """
    return quantize_int8(x)


def decompose_parallel(C: np.ndarray, n_views: int = 4) -> List[np.ndarray]:
    """分解 int32/int64 累加值为 n 个 uint8 视角

    c_i = (C >> 8i) & 0xFF

    补码位段提取：负数在补码下位段是无符号的，重构（模 2^(8*n_views)）保持正确。

    Args:
        C: int32/int64 累加值
        n_views: 视角数（4 覆盖 int32，8 覆盖 int64）

    Returns:
        views: list of uint8 数组，长度 n_views
    """
    views = []
    for i in range(n_views):
        shift = 8 * i
        c_i = ((C >> shift) & 0xFF).astype(np.uint8)
        views.append(c_i)
    return views


def reconstruct_from_views(views: List[np.ndarray]) -> np.ndarray:
    """从 uint8 视角重构 int 值

    C = Σ c_i * 2^(8i)（模 2^(8*len(views))）
    """
    C = None
    for i, v in enumerate(views):
        contribution = v.astype(np.int64) * (1 << (8 * i))
        if C is None:
            C = contribution
        else:
            C += contribution
    return C


def matmul_parallel(
    a: np.ndarray,
    b: np.ndarray,
    m: int, k: int, n: int,
    n_views: int = 4,
) -> np.ndarray:
    """正交并行 matmul（首层：标准 int8 × int8 → int64 累加值）

    首层 matmul：C = a_int8 @ b_int8（int64 累加，无视角分解）
    后续层用 matmul_multiview（分解累加值）。

    Args:
        a: int8 数组 (m*k,) 或 (m, k)
        b: int8 数组 (k*n,) 或 (k, n)
        m, k, n: 矩阵维度
        n_views: 视角数（保留参数，首层不用）

    Returns:
        C: int64 数组 (m, n)（整数域累加值，未乘 scale）
    """
    a_2d = a.reshape(m, k).astype(np.int64)
    b_2d = b.reshape(k, n).astype(np.int64)
    return a_2d @ b_2d


def matmul_multiview(
    C_acc: np.ndarray,
    b_int8: np.ndarray,
    m: int, k: int, n: int,
    n_views: int = 4,
) -> np.ndarray:
    """正交并行 matmul（后续层：多视角）

    分解累加值 C_acc 为 n_views 个 uint8 视角，权重保持 int8：
      C_next = Σ_i (c_i @ b_int8) * 2^(8i)

    复杂度：n_views 次 uint8×int8 matmul（无交叉项）。

    Args:
        C_acc: int32/int64 累加值 (m, k)（上一层的输出）
        b_int8: int8 权重 (k*n,) 或 (k, n)
        m, k, n: 矩阵维度
        n_views: 视角数

    Returns:
        C_next: int64 数组 (m, n)（整数域累加值，未乘 scale）
    """
    # 分解累加值为 uint8 视角
    views = decompose_parallel(C_acc, n_views=n_views)

    # 权重保持 int8（有符号）
    b_2d = b_int8.reshape(k, n).astype(np.int64)

    # 多视角 matmul
    C_next = np.zeros((m, n), dtype=np.int64)
    for i in range(n_views):
        c_i = views[i].reshape(m, k).astype(np.int64)
        C_next += (c_i @ b_2d) * (1 << (8 * i))

    return C_next


def relu_parallel(C: np.ndarray) -> np.ndarray:
    """多视角 ReLU（整数域）

    ReLU_mv(C) = C if C >= 0 else 0

    补码表示下，检查符号位即可。负数清零。

    Args:
        C: int32/int64 累加值

    Returns:
        C': ReLU 后的 int 值
    """
    return np.where(C >= 0, C, 0)


def decode_parallel(C: np.ndarray, scale: float) -> np.ndarray:
    """正交并行解码

    y = C * scale（C 是整数累加值，scale 是量化 scale）

    注意：多视角 matmul 后，C 已经是完整整数累加值，只需乘 scale。
    """
    return C.astype(np.float64) * scale


# ============================================================
# 统一接口
# ============================================================

class HCTreeUnified:
    """HC 树统一框架（串行/并行可切换）

    用法：
        # 串行模式（UFP 残差链）
        tree = HCTreeUnified(mode="serial", depth=1)
        a_node = tree.encode(x_float)
        b_node = tree.encode(w_float)
        C = tree.matmul(a_node, b_node, m, k, n)

        # 并行模式（正交位段）
        tree = HCTreeUnified(mode="parallel", n_views=4)
        a_views, a_scale = tree.encode(x_float)
        b_views, b_scale = tree.encode(w_float)
        C = tree.matmul(a_views, b_views, m, k, n)
    """

    def __init__(
        self,
        mode: str = InterpretationMode.SERIAL,
        depth: int = 1,
        n_views: int = 4,
    ):
        """初始化 HC 树统一框架

        Args:
            mode: 解读模式（"serial" | "parallel"）
            depth: 串行模式的 UFP 残差深度
            n_views: 并行模式的视角数
        """
        self.mode = mode
        self.depth = depth
        self.n_views = n_views

    def encode(self, x: np.ndarray):
        """编码 float → HC 树表示

        Returns:
            串行模式：HCTreeNode
            并行模式：(views, scale) tuple
        """
        if self.mode == InterpretationMode.SERIAL:
            return encode_serial(x, depth=self.depth)
        else:
            return encode_parallel(x, n_views=self.n_views)

    def matmul(self, a, b, m: int, k: int, n: int) -> np.ndarray:
        """matmul（根据模式选择算法）

        Returns:
            串行模式：float64 (m, n)（已乘 scale）
            并行模式：int64 (m, n)（整数累加值，未乘 scale）
        """
        if self.mode == InterpretationMode.SERIAL:
            return matmul_serial(a, b, m, k, n)
        else:
            return matmul_parallel(a, b, m, k, n, n_views=self.n_views)

    def relu(self, C: np.ndarray) -> np.ndarray:
        """ReLU（根据模式选择）"""
        if self.mode == InterpretationMode.SERIAL:
            # 串行模式：反量化后 ReLU（当前方案）
            return np.maximum(0, C)
        else:
            # 并行模式：整数域 ReLU
            return relu_parallel(C)

    def decode(self, x, scale: Optional[float] = None, shape: Optional[Tuple] = None) -> np.ndarray:
        """解码 HC 树表示 → float"""
        if self.mode == InterpretationMode.SERIAL:
            return decode_serial(x, shape=shape)
        else:
            assert scale is not None, "并行模式 decode 需要 scale"
            return decode_parallel(x, scale)


# ============================================================
# Level → 解读模式映射
# ============================================================

LEVEL_MODE_MAP = {
    2:  {"mode": InterpretationMode.PARALLEL, "variant": "hc4",      "n_views": 4,  "depth": 0},
    1:  {"mode": InterpretationMode.SERIAL,   "variant": "hc8",      "n_views": 6,  "depth": 2},
    0:  {"mode": InterpretationMode.PARALLEL, "variant": "hc8",      "n_views": 4,  "depth": 0},
    -1: {"mode": InterpretationMode.SERIAL,   "variant": "hc8",      "n_views": 6,  "depth": 1},
    -2: {"mode": InterpretationMode.PARALLEL, "variant": "hc16",     "n_views": 4,  "depth": 0},
}


def create_from_level(level: int) -> HCTreeUnified:
    """根据 Level 创建对应的 HCTreeUnified 实例

    Level  2 → HC4 并行（4 视角 × 4 bit，极致精简）
    Level  1 → HC8 串行（UFP depth=2，单层高精度）
    Level  0 → HC8 并行（4 视角 × 8 bit，跨层精度保持，基准）
    Level -1 → HC8 串行（UFP depth=1，中等精度）
    Level -2 → HC16 并行（4 视角 × 16 bit，深层高精度）
    """
    config = LEVEL_MODE_MAP.get(level)
    if config is None:
        raise ValueError(f"未知 Level: {level}")

    return HCTreeUnified(
        mode=config["mode"],
        depth=config["depth"],
        n_views=config["n_views"],
    )


# ============================================================
# 自检
# ============================================================

def _self_check():
    """自检：统一框架基本功能"""
    print("=" * 60)
    print("hc_tree_unified.py self-check")
    print("=" * 60)

    np.random.seed(42)
    m, k, n = 4, 8, 4

    # 生成随机数据
    x = np.random.randn(m, k).astype(np.float64) * 0.5
    w = np.random.randn(k, n).astype(np.float64) * 0.1

    # 基准：float64 matmul
    y_ref = x @ w
    print(f"\n基准 float64 matmul: shape={y_ref.shape}, range=[{y_ref.min():.4f}, {y_ref.max():.4f}]")

    # 串行模式（depth=1）
    print("\n--- 串行模式（UFP depth=1）---")
    tree_s = HCTreeUnified(mode="serial", depth=1)
    a_node = tree_s.encode(x.flatten())
    b_node = tree_s.encode(w.flatten())
    y_serial = tree_s.matmul(a_node, b_node, m, k, n).reshape(m, n)
    err_serial = np.abs(y_serial - y_ref).max()
    print(f"  串行 matmul: range=[{y_serial.min():.4f}, {y_serial.max():.4f}], max_err={err_serial:.6e}")

    # 并行模式（首层 matmul，标准 int8 × int8）
    print("\n--- 并行模式（首层 int8 matmul）---")
    tree_p = HCTreeUnified(mode="parallel", n_views=4)
    a_int8, a_scale = tree_p.encode(x.flatten())
    b_int8, b_scale = tree_p.encode(w.flatten())
    C_parallel = tree_p.matmul(a_int8, b_int8, m, k, n).reshape(m, n)
    y_parallel = decode_parallel(C_parallel, a_scale * b_scale)
    err_parallel = np.abs(y_parallel - y_ref).max()
    print(f"  并行首层 matmul: range=[{y_parallel.min():.4f}, {y_parallel.max():.4f}], max_err={err_parallel:.6e}")

    # 多视角 matmul（后续层，分解累加值）
    print("\n--- 多视角 matmul（后续层，分解累加值）---")
    # 模拟第 2 层：C_acc 是首层输出 (m, n)=(4,4)，作为第 2 层输入
    # 第 2 层维度：k2 = n（首层输出维度），n2 = 任意
    # 重要：多视角 matmul 要求 C_acc >= 0（ReLU 后），因为 4 视角覆盖 int32，
    # 负数的 int64 高 32 位是 0xFFFFFFFF，4 视角会丢失符号。
    # 在真实流程中，每层 matmul 后都有 ReLU_mv，所以 C_acc >= 0。
    C_acc = relu_parallel(C_parallel)  # ReLU 后 C >= 0
    k2 = n  # 4
    n2 = 6
    w2 = np.random.randn(k2, n2).astype(np.float64) * 0.1
    w2_int8, w2_scale = quantize_int8(w2)
    C_next = matmul_multiview(C_acc, w2_int8, m, k2, n2, n_views=4)
    # 基准：直接用 int64 累加值 matmul（C_acc 非负，4 视角重构正确）
    C_ref = C_acc.astype(np.int64) @ w2_int8.astype(np.int64)
    err_mv = np.abs(C_next - C_ref).max()
    print(f"  多视角 matmul shape: {C_next.shape}, vs int64 基准 max_err={err_mv:.2e}")
    assert err_mv < 1e-10, f"多视角 matmul 应等价 int64 基准, err={err_mv}"
    print("  ✓ 多视角 matmul 正确（4 视角重构与 int64 基准一致，C_acc >= 0）")

    # Level 映射
    print("\n--- Level → 解读模式映射 ---")
    for level in [2, 1, 0, -1, -2]:
        tree = create_from_level(level)
        config = LEVEL_MODE_MAP[level]
        print(f"  Level {level:>2} → {config['variant']:<6} {tree.mode:<8} "
              f"(depth={tree.depth}, n_views={tree.n_views})")

    # 数值一致性检查
    print("\n--- 数值一致性检查 ---")
    # 串行 depth=0 应等价于单次量化 matmul
    tree_s0 = HCTreeUnified(mode="serial", depth=0)
    a0 = tree_s0.encode(x.flatten())
    b0 = tree_s0.encode(w.flatten())
    y_s0 = tree_s0.matmul(a0, b0, m, k, n).reshape(m, n)

    # 并行首层 matmul 应等价于单次量化 matmul
    y_p_first = y_parallel  # 已计算

    # 单次量化 matmul 基准
    q_x, s_x = quantize_int8(x)
    q_w, s_w = quantize_int8(w)
    y_quant = (q_x.astype(np.int64) @ q_w.astype(np.int64)).astype(np.float64) * (s_x * s_w)

    err_s0 = np.abs(y_s0 - y_quant).max()
    err_p_first = np.abs(y_p_first - y_quant).max()
    print(f"  串行 depth=0 vs 单次量化: max_err={err_s0:.2e}")
    print(f"  并行首层 vs 单次量化: max_err={err_p_first:.2e}")
    assert err_s0 < 1e-10, f"串行 depth=0 应等价单次量化, err={err_s0}"
    assert err_p_first < 1e-10, f"并行首层应等价单次量化, err={err_p_first}"
    print("  ✓ 串行 depth=0 和并行首层均等价于单次量化 matmul")

    print("\n=== 自检通过 ===")


if __name__ == "__main__":
    _self_check()
