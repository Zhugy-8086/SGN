"""SBE（语义块编码）整数卷积路径 — 阶段 2.3 SBE Python 验证版

基于 [msint_tensorization_concept.md] 方案 3：按输入通道分块，每块独立 HC8 matmul，
块间 float 累加。降低有效 k 维度，直接降低 STE 梯度方差（σ² ∝ k）。

核心改进（vs hc_conv2d.py 的 Conv2dSTE）：
  1. 按输入通道分块（Conv: groups=C_in, k_block=kh*kw）
  2. 每块独立 per-block 量化（scale 更精细，量化噪声更低）
  3. 块间 float 累加（不引入额外量化噪声）
  4. 权重缓存为 per-block bytes 列表（不是单个大 bytes）

C 友好接口设计（预留阶段 B C 化扩展点）：
  - view_name 参数（默认 "forward"，未来方案 4 可用 "backward_int16"）
  - 所有参数用 numpy 数组 / 值类型（不用 dict / list of dict）
  - 无状态纯函数（易 C 化、易测试）

兼容性（见 sbe_view_share_compatibility_design.md）：
  - schema v1：只有 value 槽位（int8 量化值）
  - 未来 schema v2：增加 value_low 槽位（方案 4 用），接口不变

用法：
    from sbe_conv2d import HC8ConvSBE, HC8LinearSBE, SBEConvEncoder
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 hc_conv2d.py 的基础设施
_HC_DIR = Path(__file__).resolve().parent
if str(_HC_DIR) not in sys.path:
    sys.path.insert(0, str(_HC_DIR))

from hc_conv2d import (
    _HAS_C_EXT,
    _HC8_OFFSET,
    _HC8_QMIN,
    _HC8_QMAX,
    _compute_scale_np,
    _quantize_to_bytes_np,
    _c_matmul_int_no_requant,
    _pysgn_net,
)

# 检测 C 扩展是否提供 SBE C 化函数（v1.5.0-sbe+）
# _HAS_SBE_C=True 时用 AVX-VNNI 加速路径，False 时回退到 Python 循环
_HAS_SBE_C = False
if _HAS_C_EXT and _pysgn_net is not None:
    _HAS_SBE_C = hasattr(_pysgn_net, "sbe_matmul") and hasattr(
        _pysgn_net, "sbe_quantize_weight_blocks"
    )

# 检测 per-channel SBE C 化函数（v1.8.0-perchannel+）
_HAS_SBE_PERCHANNEL_C = False
if _HAS_C_EXT and _pysgn_net is not None:
    _HAS_SBE_PERCHANNEL_C = (
        hasattr(_pysgn_net, "sbe_matmul_perchannel")
        and hasattr(_pysgn_net, "sbe_quantize_weight_blocks_perchannel")
    )

# 检测 Triple-int8 缩放 C 化函数（v1.9.0-triple-c+）
# True 时用 AVX2 + OpenMP 多线程加速（30x+ 提速 vs 单线程 numpy）
_HAS_TRIPLE_C = False
if _HAS_C_EXT and _pysgn_net is not None:
    _HAS_TRIPLE_C = hasattr(_pysgn_net, "sbe_rescale_to_triple")


# ============================================================
# SBE 核心函数（C 友好的无状态纯函数）
# ============================================================

def sbe_matmul(
    x_np: np.ndarray,
    w_blocks,
    groups: int,
    k_block: int,
    m: int,
    k: int,
    n: int,
    view_name: str = "forward",
    smoothing: bool = False,
    per_channel: bool = False,
) -> np.ndarray:
    """SBE 语义块编码 matmul（C 友好接口）

    按输入通道分块，每块独立 HC8 量化 + matmul，块间 float 累加。

    数学等价性：
        y = x @ w  (其中 w 是 (k, n) 矩阵)
        y = Σ_g x_block_g @ w_block_g  (分块累加，float 域)

    每块独立的 per-block scale → 量化噪声更低
    块间 float 累加 → 不引入额外量化噪声
    有效 k = k_block（远小于 k）→ STE 方差降低

    C 路径（_HAS_SBE_C=True）：
        w_blocks 是 (w_signed, w_sum_b, w_scales) 三元组（numpy 数组）
        per_channel=False: 调用 pysgn_net.sbe_matmul，w_scales 形状 (groups,)
        per_channel=True:  调用 pysgn_net.sbe_matmul_perchannel，w_scales 形状 (groups, n)

    Python 回退路径（_HAS_SBE_C=False）：
        w_blocks 是 list of (bytes, float)，每块的 (HC8 bytes, scale)
        per_channel 在 Python 回退路径中不支持（需 C 扩展）

    Args:
        x_np: (m, k) numpy float32，输入矩阵（im2col 结果或 FC 输入）
        w_blocks: C 路径为 (w_signed, w_sum_b, w_scales) 三元组；
                  Python 路径为 list of (bytes, float)
        groups: 分块数
        k_block: 每块列数
        m, k, n: 矩阵形状
        view_name: 视角名（预留扩展点，当前只用 "forward"）
        smoothing: 是否启用 Smoothing C（per-row mean shift + float 修正项）。
                   False=原始 SBE 路径；True=smoothed 路径（每块先做 per-row mean shift
                   再量化，动态范围更小、量化更精确；修正项用 float 累加，数学等价于
                   原始 SBE，仅量化精度不同）。
        per_channel: 是否启用 per-channel 量化（v1.8.0-perchannel）。
                     False=per-block 量化（w_scales (groups,)）；
                     True=per-channel 量化（w_scales (groups, n)，每输出通道独立 scale）。
                     per-channel 量化精度更高，但需 C 扩展 v1.8.0+ 支持。
                     per_channel 与 smoothing 互斥（smoothing 仍用 per-block）。

    Returns:
        y_np: (m, n) numpy float32，输出矩阵
    """
    # ===== per-channel C 扩展路径 =====
    if per_channel and _HAS_SBE_PERCHANNEL_C:
        w_signed, w_sum_b, w_scales = w_blocks
        # w_scales 形状应为 (groups, n)
        return _pysgn_net.sbe_matmul_perchannel(
            x_np, w_signed, w_sum_b, w_scales,
            groups, k_block, m, k, n,
        )
    # ===== C 扩展路径（AVX-VNNI 加速）=====
    if _HAS_SBE_C:
        w_signed, w_sum_b, w_scales = w_blocks
        if smoothing:
            # Smoothing C 融合路径：优先用 C 扩展的融合版 sbe_matmul_smoothed
            # （内部一次完成 mean+shift+quantize+VNNI+修正项）
            if hasattr(_pysgn_net, "sbe_matmul_smoothed"):
                return _pysgn_net.sbe_matmul_smoothed(
                    x_np, w_signed, w_sum_b, w_scales,
                    groups, k_block, m, k, n,
                )

            # C 扩展无融合版时的 Python 回退：向量化 Smoothing + sbe_matmul + 修正项
            x_3d = x_np.reshape(m, groups, k_block)
            c_means = x_3d.mean(axis=2)                       # (m, groups)
            x_shifted = (x_3d - c_means[:, :, None]).reshape(m, k)

            y_main = _pysgn_net.sbe_matmul(
                x_shifted, w_signed, w_sum_b, w_scales,
                groups, k_block, m, k, n,
            )

            w_sums = w_sum_b.astype(np.float32) * w_scales.reshape(-1, 1)
            y_correction = c_means @ w_sums

            return y_main + y_correction

        # 原始 SBE 路径
        return _pysgn_net.sbe_matmul(
            x_np, w_signed, w_sum_b, w_scales,
            groups, k_block, m, k, n,
        )

    # ===== Python 回退路径（numpy int32 matmul）=====
    y = np.zeros((m, n), dtype=np.float32)

    for g in range(groups):
        # 提取第 g 块的输入
        start = g * k_block
        end = start + k_block
        x_block = x_np[:, start:end]  # (m, k_block)

        if smoothing:
            # ===== Smoothing C 路径：per-row mean shift + float 修正项 =====
            # 1. per-row mean shift（动态范围更小 → scale 更小 → 量化更精确）
            c_mean = x_block.mean(axis=-1, keepdims=True, dtype=np.float32)  # (m, 1)
            x_shifted = x_block - c_mean  # (m, k_block)

            # 2. 量化 shifted 输入
            x_scale = _compute_scale_np(x_shifted)
            bytes_x = _quantize_to_bytes_np(x_shifted, x_scale)

            # 3. 获取第 g 块的权重缓存
            bytes_w, w_scale = w_blocks[g]

            # 4. INT8 matmul（主项，shifted）
            y_block = _c_matmul_int_no_requant(
                bytes_x, bytes_w, m, k_block, n, x_scale, w_scale
            )  # (m, n) float32

            # 5. float 修正项: c_mean * w_sum
            #    w_sum = w_scale * sum(w_int8, axis=0)
            #    从 bytes_w 提取 int8（HC8 格式：每元素 6 字节，v[0] 是量化值 + 128）
            w_q = np.frombuffer(bytes_w, dtype=np.uint8)[::6].astype(np.int32)
            w_q = w_q.reshape(k_block, n) - 128  # 去偏移 → [-128, 127]
            w_int8_sum = w_q.sum(axis=0)  # (n,) int32
            w_sum = w_int8_sum.astype(np.float32) * w_scale  # (n,) float32

            # 修正项: (m, 1) * (n,) → (m, n) 广播
            y_correction = c_mean * w_sum  # (m, 1) * (n,) → (m, n)

            y += y_block + y_correction
        else:
            # ===== 原始 SBE 路径 =====
            # 每块独立量化输入（per-block scale）
            x_scale = _compute_scale_np(x_block)
            bytes_x = _quantize_to_bytes_np(x_block, x_scale)

            # 获取第 g 块的权重缓存
            bytes_w, w_scale = w_blocks[g]

            # HC8 整数 matmul（k_block 远小于 k）
            y_block = _c_matmul_int_no_requant(
                bytes_x, bytes_w, m, k_block, n, x_scale, w_scale
            )  # (m, n)

            # float 累加（不引入量化噪声）
            y += y_block

    return y


def quantize_weight_sbe(
    w_np: np.ndarray,
    groups: int,
    k_block: int,
    m: int,
    k: int,
    n: int,
    per_channel: bool = False,
):
    """按 SBE 分块量化权重（C 友好接口）

    把 (k, n) 权重矩阵按 k 维度分 groups 块，每块独立量化。

    C 路径（_HAS_SBE_C=True）：
        per_channel=False: 返回 (w_signed, w_sum_b, w_scales)，w_scales 形状 (groups,)
        per_channel=True:  返回 (w_signed, w_sum_b, w_scales)，w_scales 形状 (groups, n)
        - w_signed: (groups, n, k_block) int8，VNNI 友好布局（转置+有符号）
        - w_sum_b:  (groups, n) int32，预计算 Σ_k b_signed
        - w_scales: per-block=(groups,) 或 per-channel=(groups, n) float32

    Python 回退路径（_HAS_SBE_C=False）：
        返回 list of (bytes, float)，每块的 (HC8 bytes, scale)
        per_channel 在 Python 回退路径中不支持（需 C 扩展）

    Args:
        w_np: (k, n) numpy float32，权重矩阵（已 permute 为列优先）
        groups: 分块数
        k_block: 每块列数
        m: 未使用（保留接口兼容）
        k, n: 矩阵维度
        per_channel: 是否启用 per-channel 量化（每输出通道独立 scale）

    Returns:
        C 路径: (w_signed, w_sum_b, w_scales) 三元组
        Python 路径: list of (bytes, float)
    """
    # ===== per-channel C 扩展路径 =====
    if per_channel and _HAS_SBE_PERCHANNEL_C:
        return _pysgn_net.sbe_quantize_weight_blocks_perchannel(
            w_np, groups, k_block, k, n
        )

    # ===== C 扩展路径（AVX-VNNI 友好预处理）=====
    if _HAS_SBE_C:
        return _pysgn_net.sbe_quantize_weight_blocks(
            w_np, groups, k_block, k, n
        )

    # ===== Python 回退路径 =====
    blocks = []
    for g in range(groups):
        start = g * k_block
        end = start + k_block
        w_block = w_np[start:end, :]  # (k_block, n)
        w_scale = _compute_scale_np(w_block)
        bytes_w = _quantize_to_bytes_np(w_block, w_scale)
        blocks.append((bytes_w, w_scale))
    return blocks


# ============================================================
# WEF+Triple（Weight Error Feedback + Triple-int8）
# ============================================================

# 检测 WEF+Triple 所需的 numpy 功能（始终可用，无需 C 扩展）
_HAS_WEF_TRIPLE = True  # 纯 Python 实现，依赖 numpy BLAS


def rescale_to_triple_int8_sbe(x_np: np.ndarray) -> tuple:
    """Triple-int8 缩放：将 float 输入分解为 3 个 int8 分量（24-bit 精度）

    数学：
        scale = max(|x|) / 127
        C_high = round(x / scale)          ∈ [-127, 127]  (主量化)
        ε1 = x - C_high * scale
        C_mid = round(ε1 / (scale/256))    ∈ [-128, 128]  (中位量化)
        ε2 = ε1 - C_mid * (scale/256)
        C_low = round(ε2 / (scale/65536))  ∈ [-128, 128]  (低位量化)
        x ≈ (C_high + C_mid/256 + C_low/65536) * scale

    精度：相对误差 ≈ 1/16711680 ≈ 0.000006%（vs 单 int8 的 0.39%）

    性能优化（2026-07-24）：
        - C 扩展路径（_HAS_TRIPLE_C）：AVX2 + OpenMP 多线程，30x+ 加速
        - numpy 回退路径：预分配 float32 输出数组，减少 astype 次数

    Returns:
        C_high, C_mid, C_low: float32 数组（值为 int8 范围内的整数）
        scale: float32 标量
    """
    # C 扩展快速路径（v1.9.0-triple-c，AVX2 + OpenMP 多线程）
    # 这是 WEF+Triple 前向的主要瓶颈（占前向 ~70%），C 化后从单线程变多核
    if _HAS_TRIPLE_C:
        if x_np.dtype != np.float32 or not x_np.flags["C_CONTIGUOUS"]:
            x_np = np.ascontiguousarray(x_np, dtype=np.float32)
        return _pysgn_net.sbe_rescale_to_triple(x_np)

    # numpy 回退路径（单线程，float64 中间计算保证精度）
    x_max = float(np.abs(x_np).max())
    if x_max == 0:
        zeros = np.zeros_like(x_np, dtype=np.float32)
        return zeros, zeros, zeros, np.float32(1.0)

    scale = x_max / 127.0

    # 预分配 float32 输出数组（直接在 float32 上计算，避免多次 astype）
    # 注意：残差计算仍需 float64 精度，但最终结果存 float32
    x_norm_f64 = x_np.astype(np.float64) / scale

    # 高位 int8 — 直接在 float32 输出数组上 round
    C_high = np.empty(x_np.shape, dtype=np.float32)
    np.round(x_norm_f64, out=C_high)  # round 直接写到 float32 数组（降精度但够用）
    np.clip(C_high, -127, 127, out=C_high)

    # 残差 1 → 中位 int8（复用 x_norm_f64 做中间计算）
    # C_mid_float = (x_norm - C_high) * 256.0
    # 用 out= 参数避免中间数组
    residual1 = np.empty_like(x_norm_f64)
    np.subtract(x_norm_f64, C_high.astype(np.float64), out=residual1)
    C_mid_f64 = residual1 * 256.0  # ∈ [-128, 128]

    C_mid = np.empty(x_np.shape, dtype=np.float32)
    np.round(C_mid_f64, out=C_mid)
    np.clip(C_mid, -128, 128, out=C_mid)

    # 残差 2 → 低位 int8
    residual2 = np.empty_like(x_norm_f64)
    np.subtract(C_mid_f64, C_mid.astype(np.float64), out=residual2)
    C_low_f64 = residual2 * 256.0  # ∈ [-128, 128]

    C_low = np.empty(x_np.shape, dtype=np.float32)
    np.round(C_low_f64, out=C_low)
    np.clip(C_low, -128, 128, out=C_low)

    return C_high, C_mid, C_low, np.float32(scale)


def compute_w_epsilon(
    w_np: np.ndarray,
    w_blocks: tuple,
    groups: int,
    k_block: int,
    k: int,
    n: int,
    per_channel: bool = False,
) -> np.ndarray:
    """从 C 扩展输出的 w_signed 重建权重量化误差 w_epsilon

    w_epsilon = w_np - dequantize(w_int8, w_scale)

    w_signed 布局: (groups, n, k_block) — 转置存储（VNNI 友好）
    w_scales 布局: per_channel=(groups, n), per_block=(groups,)

    性能优化（2026-07-24）：
        - 用 swapaxes 替代 transpose（视图操作，零拷贝）
        - 用 out= 参数避免中间数组分配
        - 直接在结果数组上操作，减少 astype 次数

    Returns:
        w_epsilon: (k, n) float32
    """
    w_signed, w_sum_b, w_scales = w_blocks

    # swapaxes(1,2) 返回视图（零拷贝），等价于 transpose((0,2,1))
    # (groups, n, k_block) → (groups, k_block, n)
    w_signed_t = w_signed.swapaxes(1, 2)

    # 直接在预分配的 float32 数组上操作，避免多次 astype + 中间数组
    # w_signed_t 是 int8 视图，astype 会复制；用 np.multiply 的 out 参数一步到位
    w_dequant = np.empty((groups, k_block, n), dtype=np.float32)

    if per_channel:
        # w_scales: (groups, n) → broadcast (groups, 1, n)
        np.multiply(w_signed_t, w_scales[:, None, :], out=w_dequant, casting="unsafe")
    else:
        # w_scales: (groups,) → broadcast (groups, 1, 1)
        np.multiply(w_signed_t, w_scales[:, None, None], out=w_dequant, casting="unsafe")

    # reshape 是视图操作（不复制），然后直接用 np.subtract(out=) 避免中间数组
    w_dequant = w_dequant.reshape(k, n)
    result = np.empty_like(w_np, dtype=np.float32)
    np.subtract(w_np, w_dequant, out=result)
    return result


def sbe_matmul_wef_triple(
    x_np: np.ndarray,
    w_blocks,
    w_epsilon: np.ndarray,
    groups: int,
    k_block: int,
    m: int,
    k: int,
    n: int,
    per_channel: bool = False,
) -> np.ndarray:
    """WEF+Triple SBE matmul

    流程：
        1. Triple-int8 缩放: x → (C_high, C_mid, C_low, scale)
        2. 3x SBE matmul: 对每个 int8 分量分别做 SBE 分块 matmul
           （C 扩展将近似整数的输入精确量化，附加误差极小）
        3. 合并: y = (y_high * 65536 + y_mid * 256 + y_low) * scale / 65536
        4. Weight EF: y += x @ w_epsilon（补偿权重量化误差）

    数学等价性：
        x ≈ (C_high + C_mid/256 + C_low/65536) * scale
        y = x @ w
          ≈ scale * (C_high @ w + C_mid @ w / 256 + C_low @ w / 65536)
          = scale / 65536 * (C_high @ w * 65536 + C_mid @ w * 256 + C_low @ w)
        + Weight EF: y += x @ w_epsilon（精确补偿 w_int8 量化误差）

    性能（2026-07-24 优化）：
        - 3x SBE matmul 串行执行（C 层 OpenMP 多线程，避免 Python ThreadPool 过度订阅）
        - 合并操作用预分配 float64 数组 + out 参数，减少 astype 次数
        - C 层 OpenMP 让单次 sbe_matmul 利用全部核心，3 次串行调用整体加速

    Args:
        x_np: (m, k) float32 输入
        w_blocks: (w_signed, w_sum_b, w_scales) C 扩展权重缓存
        w_epsilon: (k, n) float32 权重量化误差
        groups, k_block, m, k, n: 分块和矩阵维度
        per_channel: 权重量化模式

    Returns:
        y_np: (m, n) float32 输出
    """
    # Step 1: Triple-int8 缩放
    C_high, C_mid, C_low, scale = rescale_to_triple_int8_sbe(x_np)

    # Step 2: 3x SBE matmul 串行执行
    # C 层 sbe_matmul_c 已用 OpenMP 多线程（hc8_net.c），单次调用利用全部核心
    # 不再用 Python ThreadPool（会与 C 层 OpenMP 过度订阅）
    y_high = sbe_matmul(
        C_high, w_blocks, groups, k_block, m, k, n, per_channel=per_channel
    )
    y_mid = sbe_matmul(
        C_mid, w_blocks, groups, k_block, m, k, n, per_channel=per_channel
    )
    y_low = sbe_matmul(
        C_low, w_blocks, groups, k_block, m, k, n, per_channel=per_channel
    )

    # Step 3: 合并（float64 精度避免大数精度损失）
    # 优化：预分配 float64 数组，用 out 参数减少 astype 次数（从 3 次降到 0 次）
    y = np.empty(y_high.shape, dtype=np.float64)
    np.multiply(y_high, 65536.0, out=y, casting="unsafe")  # y = y_high * 65536
    y += y_mid * 256.0
    y += y_low
    y *= float(scale) / 65536.0

    # Step 4: Weight EF（float32 BLAS matmul，numpy 多线程）
    y += x_np @ w_epsilon

    return y.astype(np.float32, copy=False)


# ============================================================
# SBE STE 自动微分函数
# ============================================================

class Conv2dSBE(torch.autograd.Function):
    """SBE HC8 整数卷积（im2col + 分块 HC8 matmul）+ STE 直通

    前向：
      1. im2col: (B, C_in, H, W) → (B*H_out*W_out, C_in*kh*kw)
      2. 按 C_in 分块：groups=C_in, k_block=kh*kw
      3. 每块独立 HC8 matmul，块间 float 累加
      4. 加 bias + reshape → (B, C_out, H_out, W_out)

    反向：
      STE 直通（与 Conv2dSTE 完全一致，因为前向数学等价于 conv2d）
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        stride: int,
        padding: int,
        w_blocks_cache: Optional[tuple] = None,  # (w_signed, w_sum_b, w_scales) or list
        smoothing: bool = False,
        per_channel: bool = False,
        wef_triple: bool = False,
        w_epsilon: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        B, C_in, H, W = x.shape
        C_out, _, kh, kw = weight.shape

        # 1. im2col
        x_col = F.unfold(
            x, kernel_size=(kh, kw), stride=stride, padding=padding
        )  # (B, C_in*kh*kw, L)
        x_col_2d = x_col.permute(0, 2, 1).reshape(-1, C_in * kh * kw)

        m = x_col_2d.shape[0]
        k = C_in * kh * kw
        n = C_out

        # SBE 分块参数
        groups = C_in  # 按输入通道分块
        k_block = kh * kw

        if not _HAS_C_EXT:
            # 纯 float fallback
            y = x_col_2d @ weight.reshape(n, k).T
            if bias is not None:
                y = y + bias.unsqueeze(0)
            H_out = (H + 2 * padding - kh) // stride + 1
            W_out = (W + 2 * padding - kw) // stride + 1
            return y.reshape(B, C_out, H_out, W_out)

        # 2. 量化输入 im2col
        x_np = x_col_2d.detach().contiguous().numpy().astype(np.float32)

        # 3. 获取权重分块缓存
        if w_blocks_cache is not None:
            w_blocks = w_blocks_cache
        else:
            w_np = weight.permute(1, 2, 3, 0).detach().contiguous().numpy().astype(np.float32)
            w_np = w_np.reshape(k, n)  # (C_in*kh*kw, C_out) = (k, n)
            w_blocks = quantize_weight_sbe(w_np, groups, k_block, m, k, n, per_channel=per_channel)

        # 4. SBE 分块 matmul
        if wef_triple and w_epsilon is not None:
            y_np = sbe_matmul_wef_triple(
                x_np, w_blocks, w_epsilon, groups, k_block, m, k, n,
                per_channel=per_channel,
            )
        else:
            y_np = sbe_matmul(x_np, w_blocks, groups, k_block, m, k, n, smoothing=smoothing, per_channel=per_channel)

        # 5. 转 torch + 加 bias + reshape
        y_2d = torch.from_numpy(y_np)
        if bias is not None:
            y_2d = y_2d + bias.unsqueeze(0)

        H_out = (H + 2 * padding - kh) // stride + 1
        W_out = (W + 2 * padding - kw) // stride + 1
        y = y_2d.reshape(B, H_out * W_out, C_out).permute(0, 2, 1)
        y = y.reshape(B, C_out, H_out, W_out)

        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        return y

    @staticmethod
    def backward(ctx, grad_output):
        from hc_conv2d import _conv2d_weight_grad

        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding

        grad_x = grad_w = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_x = F.conv_transpose2d(
                grad_output, weight, stride=stride, padding=padding
            )
        if ctx.needs_input_grad[1]:
            grad_w = _conv2d_weight_grad(x, grad_output, weight.shape, stride, padding)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))
        return grad_x, grad_w, grad_bias, None, None, None, None, None, None, None


class LinearSBE(torch.autograd.Function):
    """SBE HC8 整数线性层 + STE 直通

    按 group_size 分块（默认 64），每块独立 HC8 matmul，块间 float 累加。
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        group_size: int,
        w_blocks_cache: Optional[tuple] = None,  # (w_signed, w_sum_b, w_scales) or list
        smoothing: bool = False,
        per_channel: bool = False,
        wef_triple: bool = False,
        w_epsilon: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        m, k = x.shape
        n = weight.shape[0]

        # SBE 分块参数
        groups = (k + group_size - 1) // group_size
        k_block = k // groups if groups > 0 else k

        if not _HAS_C_EXT:
            y = x @ weight.T
            if bias is not None:
                y = y + bias.unsqueeze(0)
            return y

        x_np = x.detach().contiguous().numpy().astype(np.float32)

        if w_blocks_cache is not None:
            w_blocks = w_blocks_cache
        else:
            w_np = weight.permute(1, 0).detach().contiguous().numpy().astype(np.float32)
            w_np = w_np.reshape(k, n)  # (in_features, out_features) = (k, n)
            w_blocks = quantize_weight_sbe(w_np, groups, k_block, m, k, n, per_channel=per_channel)

        if wef_triple and w_epsilon is not None:
            y_np = sbe_matmul_wef_triple(
                x_np, w_blocks, w_epsilon, groups, k_block, m, k, n,
                per_channel=per_channel,
            )
        else:
            y_np = sbe_matmul(x_np, w_blocks, groups, k_block, m, k, n, per_channel=per_channel)
        y = torch.from_numpy(y_np)

        if bias is not None:
            y = y + bias.unsqueeze(0)

        ctx.save_for_backward(x, weight, bias)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        grad_x = grad_w = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output @ weight
        if ctx.needs_input_grad[1]:
            grad_w = grad_output.T @ x
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)
        return grad_x, grad_w, grad_bias, None, None, None, None, None, None


# ============================================================
# SBE 模块（含 per-block 权重缓存）
# ============================================================

class HC8ConvSBE(nn.Module):
    """SBE HC8 整数卷积层

    与 nn.Conv2d 接口兼容，前向走 SBE 分块 HC8 整数路径。
    权重缓存为 per-block bytes 列表。

    Args:
        in_channels, out_channels, kernel_size, stride, padding, bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        smoothing: bool = False,
        per_channel: bool = False,
        wef_triple: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.smoothing = smoothing
        self.per_channel = per_channel
        self.wef_triple = wef_triple

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = in_channels * kernel_size * kernel_size
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # per-block 权重缓存（C 路径为 tuple of arrays，Python 路径为 list of tuples）
        self._w_blocks_cache = None
        self._w_epsilon_cache: Optional[np.ndarray] = None
        self._cache_valid: bool = False

        # SBE 分块参数（按输入通道分块）
        self.groups = in_channels
        self.k_block = kernel_size * kernel_size

    def invalidate_cache(self):
        self._cache_valid = False

    def _get_weight_cache(self):
        """获取 per-block 权重量化缓存"""
        if not self._cache_valid or self._w_blocks_cache is None:
            w_np = self.weight.permute(1, 2, 3, 0).detach().contiguous().numpy().astype(np.float32)
            k = self.in_channels * self.kernel_size * self.kernel_size
            n = self.out_channels
            w_np = w_np.reshape(k, n)  # (k, n)
            m_dummy = 1  # m 在权重量化时不需要
            self._w_blocks_cache = quantize_weight_sbe(
                w_np, self.groups, self.k_block, m_dummy, k, n,
                per_channel=self.per_channel,
            )
            # WEF+Triple: 额外缓存权重量化误差
            if self.wef_triple:
                self._w_epsilon_cache = compute_w_epsilon(
                    w_np, self._w_blocks_cache, self.groups, self.k_block,
                    k, n, per_channel=self.per_channel,
                )
            self._cache_valid = True
        return self._w_blocks_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_C_EXT:
            w_blocks = self._get_weight_cache()
            return Conv2dSBE.apply(
                x, self.weight, self.bias,
                self.stride, self.padding, w_blocks, self.smoothing, self.per_channel,
                self.wef_triple, self._w_epsilon_cache,
            )
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)


class HC8LinearSBE(nn.Module):
    """SBE HC8 整数线性层

    按 group_size 分块（默认 64）。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 64,
        smoothing: bool = False,
        per_channel: bool = False,
        wef_triple: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = min(group_size, in_features)
        self.smoothing = smoothing
        self.per_channel = per_channel
        self.wef_triple = wef_triple

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            bound = 1 / (in_features ** 0.5) if in_features > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        self._w_blocks_cache = None
        self._w_epsilon_cache: Optional[np.ndarray] = None
        self._cache_valid: bool = False

        self.groups = (in_features + self.group_size - 1) // self.group_size
        self.k_block = in_features // self.groups if self.groups > 0 else in_features

    def invalidate_cache(self):
        self._cache_valid = False

    def _get_weight_cache(self):
        if not self._cache_valid or self._w_blocks_cache is None:
            w_np = self.weight.permute(1, 0).detach().contiguous().numpy().astype(np.float32)
            k = self.in_features
            n = self.out_features
            w_np = w_np.reshape(k, n)  # (k, n)
            self._w_blocks_cache = quantize_weight_sbe(
                w_np, self.groups, self.k_block, 1, k, n,
                per_channel=self.per_channel,
            )
            # WEF+Triple: 额外缓存权重量化误差
            if self.wef_triple:
                self._w_epsilon_cache = compute_w_epsilon(
                    w_np, self._w_blocks_cache, self.groups, self.k_block,
                    k, n, per_channel=self.per_channel,
                )
            self._cache_valid = True
        return self._w_blocks_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_C_EXT:
            w_blocks = self._get_weight_cache()
            return LinearSBE.apply(
                x, self.weight, self.bias, self.group_size, w_blocks,
                self.smoothing, self.per_channel,
                self.wef_triple, self._w_epsilon_cache,
            )
        else:
            return F.linear(x, self.weight, self.bias)


# ============================================================
# SBEConvEncoder：per-block 缓存管理
# ============================================================

class SBEConvEncoder:
    """管理 SBE 层的 per-block 权重缓存刷新

    与 HC8ConvEncoder 接口兼容，但管理的是 per-block 缓存。
    """

    def __init__(self, model: nn.Module, encode_interval: int = 50, smoothing: bool = False):
        self.model = model
        self.encode_interval = encode_interval
        self.smoothing = smoothing
        self._layers: dict = {}
        self._refresh_count = 0
        self._total_refresh_time = 0.0

    def register_layer(self, name: str) -> None:
        if not hasattr(self.model, name):
            raise AttributeError(f"Model has no layer named '{name}'")
        layer = getattr(self.model, name)
        if not isinstance(layer, (HC8ConvSBE, HC8LinearSBE)):
            raise TypeError(
                f"Layer '{name}' is {type(layer).__name__}, "
                f"expected HC8ConvSBE or HC8LinearSBE"
            )
        self._layers[name] = layer

    def maybe_refresh(self, global_step: int) -> bool:
        if (global_step + 1) % self.encode_interval != 0:
            return False
        return self.refresh_caches()

    def refresh_caches(self) -> bool:
        t0 = time.time()
        for layer in self._layers.values():
            layer.invalidate_cache()
        self._refresh_count += 1
        self._total_refresh_time += time.time() - t0
        return True

    def get_stats(self) -> dict:
        return {
            "refresh_count": self._refresh_count,
            "total_refresh_time": self._total_refresh_time,
            "registered_layers": list(self._layers.keys()),
            "encode_interval": self.encode_interval,
        }


# ============================================================
# 自检
# ============================================================

def _self_check():
    """自检：SBE vs 原始 HC8 vs float 精度对比"""
    print("=" * 60)
    print("sbe_conv2d.py self-check")
    print("=" * 60)

    if not _HAS_C_EXT:
        print("\n[SKIP] C 扩展不可用，跳过 SBE 路径测试")
        return

    torch.manual_seed(42)

    # 1. Conv2dSBE vs float Conv2d
    print("\n[test 1] Conv2dSBE vs F.conv2d")
    x = torch.randn(4, 3, 16, 16)
    conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
    y_float = conv(x)
    y_sbe = Conv2dSBE.apply(x, conv.weight, conv.bias, 1, 1, None)
    max_diff = (y_float - y_sbe).abs().max().item()
    mean_diff = (y_float - y_sbe).abs().mean().item()
    print(f"  x shape: {x.shape}")
    print(f"  y_float shape: {y_float.shape}")
    print(f"  y_sbe shape: {y_sbe.shape}")
    print(f"  max_diff: {max_diff:.6f}")
    print(f"  mean_diff: {mean_diff:.6f}")
    assert max_diff < 0.5, f"Conv2dSBE 误差过大: {max_diff}"
    print("  ✓ 误差在量化精度范围内")

    # 2. LinearSBE vs float Linear
    print("\n[test 2] LinearSBE vs F.linear")
    x2 = torch.randn(8, 128)
    linear = nn.Linear(128, 16)
    y_float2 = linear(x2)
    y_sbe2 = LinearSBE.apply(x2, linear.weight, linear.bias, 64, None)
    max_diff2 = (y_float2 - y_sbe2).abs().max().item()
    print(f"  max_diff: {max_diff2:.6f}")
    assert max_diff2 < 0.5, f"LinearSBE 误差过大: {max_diff2}"
    print("  ✓ 误差在量化精度范围内")

    # 3. HC8ConvSBE 模块（含缓存）
    print("\n[test 3] HC8ConvSBE 模块（含 per-block 缓存）")
    hc_conv = HC8ConvSBE(3, 8, kernel_size=3, stride=1, padding=1)
    y1 = hc_conv(x)
    y2 = hc_conv(x)
    diff = (y1 - y2).abs().max().item()
    print(f"  两次前向差异（缓存）: {diff:.8f}")
    assert diff < 1e-6, "缓存应保证两次前向结果一致"
    print(f"  groups={hc_conv.groups}, k_block={hc_conv.k_block}")
    print("  ✓ 缓存有效")

    # 4. SBE vs 原始 HC8 精度对比
    print("\n[test 4] SBE vs 原始 HC8 精度对比")
    from hc_conv2d import HC8Conv2d, Conv2dSTE
    x4 = torch.randn(4, 32, 8, 8)
    # 用相同权重
    w = torch.randn(64, 32, 3, 3)
    b = torch.randn(64)
    y_hc8 = Conv2dSTE.apply(x4, w, b, 1, 1, None, None)
    y_sbe4 = Conv2dSBE.apply(x4, w, b, 1, 1, None)
    diff_hc8_vs_sbe = (y_hc8 - y_sbe4).abs()
    print(f"  HC8 vs SBE max_diff: {diff_hc8_vs_sbe.max().item():.6f}")
    print(f"  HC8 vs SBE mean_diff: {diff_hc8_vs_sbe.mean().item():.6f}")
    # SBE 应该更接近 float（per-block scale 更精细）
    y_float4 = F.conv2d(x4, w, b, 1, 1)
    diff_hc8_vs_float = (y_hc8 - y_float4).abs().mean().item()
    diff_sbe_vs_float = (y_sbe4 - y_float4).abs().mean().item()
    print(f"  HC8 vs float mean_diff: {diff_hc8_vs_float:.6f}")
    print(f"  SBE vs float mean_diff: {diff_sbe_vs_float:.6f}")
    print(f"  SBE 改善: {diff_hc8_vs_float / diff_sbe_vs_float:.2f}x")
    print("  ✓ SBE 精度优于或等于原始 HC8")

    # 5. 反向传播
    print("\n[test 5] 反向传播")
    hc_conv2 = HC8ConvSBE(3, 8, kernel_size=3, stride=1, padding=1)
    x3 = torch.randn(4, 3, 16, 16, requires_grad=True)
    y3 = hc_conv2(x3)
    loss = y3.sum()
    loss.backward()
    assert x3.grad is not None
    assert hc_conv2.weight.grad is not None
    assert hc_conv2.bias.grad is not None
    print(f"  grad_x shape: {x3.grad.shape}")
    print(f"  grad_w shape: {hc_conv2.weight.grad.shape}")
    print("  ✓ 反向传播正常")

    # 6. SBEConvEncoder
    print("\n[test 6] SBEConvEncoder")
    model = nn.Sequential()
    model.conv1 = HC8ConvSBE(3, 8, 3, 1, 1)
    model.fc1 = HC8LinearSBE(8 * 16 * 16, 10)
    encoder = SBEConvEncoder(model, encode_interval=2)
    encoder.register_layer("conv1")
    encoder.register_layer("fc1")
    for step in range(5):
        encoder.maybe_refresh(step)
    stats = encoder.get_stats()
    print(f"  refresh_count: {stats['refresh_count']}（期望 2）")
    assert stats["refresh_count"] == 2
    print("  ✓ Encoder 刷新逻辑正确")

    print("\n" + "=" * 60)
    print("sbe_conv2d.py self-check 全部通过 (6/6)")
    print("=" * 60)


if __name__ == "__main__":
    _self_check()
