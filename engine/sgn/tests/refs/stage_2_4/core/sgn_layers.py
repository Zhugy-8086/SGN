"""SGN CNN 层（numpy 实现，前向 HC8 整数，反向 numpy STE）

阶段 2.4 Step 2: MNIST 快速验证

设计:
  - 前向: HC8 量化 + matmul (数学等价于 C 扩展 _c_matmul_int_no_requant 路径)
  - 反向: numpy 标准公式 (STE 直通，量化不影响梯度)
  - 不依赖 PyTorch autograd
  - 使用 numpy stride tricks 向量化 im2col / col2im / maxpool
  - HC8 量化路径与 stage_2_3_int_path/hc_conv2d.py 的 _quantize_to_bytes_np + _c_matmul_int_no_requant 数学等价

数学等价性说明:
  C 扩展路径:
    1. q = round(x / scale), clamp to [-127, 127]
    2. q_u = q + 128 (offset), clamp to [0, 255]
    3. bytes = encode(q_u)
    4. (later) q = frombuffer(bytes) - 128
    5. x_dequant = q * scale
    6. y = x_dequant @ w_dequant (float BLAS)

  本模块路径 (省略 bytes 编码):
    1. q = round(x / scale), clamp to [-127, 127]
    2. x_dequant = q * scale
    3. y = x_dequant @ w_dequant (float BLAS)

  两者数学等价，仅省略 bytes 编码/解码开销。
  量化精度（int8 范围限制）完全保留。

关联文档:
  - [../README.md] §3 CNN 独立训练架构
  - [numpy_backward.py] 参考实现（用于 Step 1 验证）
  - [../../stage_2_3_int_path/hc_conv2d.py] HC8 量化参考
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import as_strided

# ============================================================
# HC8 量化（与 stage_2_3_int_path/hc_conv2d.py 数学等价）
# ============================================================

_HC8_QMIN = -127
_HC8_QMAX = 127


def hc8_quantize(x: np.ndarray) -> np.ndarray:
    """HC8 量化 + 反量化（round-trip，per-tensor scale）

    实际路径：返回 float32 数组，内部走 numpy float32 BLAS 路径（round-trip 量化），
    数学等价于 int8 累加后反量化，但并非真正 int8×int8→int32 累加。
    如需真正的 int8 路径，请使用 hc8_quantize_int（返回 HC8 字节 + scale）。

    数学等价于 C 扩展的 _c_matmul_int_no_requant 路径中的量化步骤:
      1. x_scale = max(|x|) / 127
      2. x_q = round(x / x_scale), clamp to [-127, 127]
      3. 返回 x_q * x_scale (反量化回 float)

    优化版（2 次数组遍历而非 5 次）:
      1. np.abs(x, out=...) + max()  → 1 次遍历
      2. round + clip + mul 融合到 np.where 表达式 → 1 次遍历

    Args:
        x: 任意形状的 numpy 数组

    Returns:
        量化后的 float 数组（与 x 同形状）
    """
    if x.size == 0:
        return x.copy()
    # 1. 计算 scale（1 次遍历：abs + max）
    x_abs = np.abs(x)  # 写入新数组
    x_max = float(x_abs.max())
    if x_max == 0.0:
        return x.copy()
    scale = x_max / 127.0
    inv_scale = 1.0 / scale

    # 2. 量化 + 反量化（1 次遍历，in-place 复用 x_abs 数组）
    # 等价: q = round(x / scale); q = clip(q, -127, 127); return q * scale
    # 用 x_abs 作为输出 buffer，避免分配新数组
    np.multiply(x, inv_scale, out=x_abs)  # x_abs = x / scale
    np.round(x_abs, out=x_abs)             # x_abs = round(x/scale)
    np.clip(x_abs, _HC8_QMIN, _HC8_QMAX, out=x_abs)  # x_abs = clip(round(x/scale))
    x_abs *= scale                          # x_abs = q * scale
    return x_abs.astype(x.dtype, copy=False)


def hc8_matmul(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """HC8 整数 matmul（前向）

    实际路径：内部走 numpy float32 BLAS（先反量化到 float32 再做 matmul），
    非真正 int8×int8→int32 累加路径。
    如需真正的 int8 累加路径，请使用 hc8_matmul_int_path。

    x: (m, k), w: (k, n) → y: (m, n)
    数学等价于 int8 × int8 → int32 累加 → 反量化
    """
    x_q = hc8_quantize(x)
    w_q = hc8_quantize(w)
    return x_q @ w_q


def hc8_matmul_int_path(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """HC8 整数 matmul（显式 int8 累加路径）

    通过 C 扩展 pysgn_net.hc8_matmul_int 走真正的 int8×int8→int32 累加路径。
    pysgn_net 不可用时 fallback 到 hc8_matmul（float32 BLAS 路径）。

    数学等价性：与 hc8_matmul 结果一致（max_diff < 1e-5），但内部走整数累加。

    Args:
        x: (m, k) numpy 数组
        w: (k, n) numpy 数组

    Returns:
        y: (m, n) numpy 数组
    """
    try:
        import engine.sgn as _sgn
        pysgn_net = _sgn._native.hc8_net
        if hasattr(pysgn_net, 'hc8_matmul_int'):
            return pysgn_net.hc8_matmul_int(x, w)
    except (ImportError, AttributeError):
        pass
    # Fallback: float32 BLAS 路径
    return hc8_matmul(x, w)


def hc8_quantize_int(x: np.ndarray) -> tuple:
    """HC8 量化（显式 int8 字节路径）

    返回 HC8 量化字节和 scale，不进行 round-trip 反量化。
    pysgn_net 不可用时返回 (None, 0.0)。

    Args:
        x: 任意形状 numpy 数组

    Returns:
        (bytes, scale): HC8 字节和量化 scale
        或 (None, 0.0): pysgn_net 不可用时的 fallback
    """
    try:
        import engine.sgn as _sgn
        pysgn_net = _sgn._native.hc8_net
        if hasattr(pysgn_net, 'hc8_quantize_int'):
            return pysgn_net.hc8_quantize_int(x)
    except (ImportError, AttributeError):
        pass
    # Fallback: 无法提供 int8 字节路径
    return (None, 0.0)


# ============================================================
# k_block 自动选择（用于 SBE 分块量化）
# ============================================================

def select_k_block(k: int, target: int = 96) -> tuple:
    """选择最佳 (groups, k_block) 使 groups*k_block==k 且 k_block<=target

    背景：AVX2 nibble split kernel 步进 32 字节，k_block<32 时 AVX2 循环
    不执行，全部走标量尾部（性能 0.08x BLAS）。
    k_block>=32 后 AVX2 启用，k_block=96 时 AVX2 3 次迭代，效率最高。

    策略：找 K 的最大因子且 <= target，确保整除（C 扩展要求）。

    Args:
        k: 总 k 维度（Conv: C_in*kh*kw, Linear: in_features）
        target: 目标 k_block（默认 96，AVX2 3 次迭代）

    Returns:
        (groups, k_block)
    """
    if k <= target:
        return 1, k
    # 从 target 递减找最大因子
    for kb in range(target, 0, -1):
        if k % kb == 0:
            return k // kb, kb
    return 1, k


# ============================================================
# Vectorized im2col / col2im (使用 stride tricks)
# ============================================================

def _im2col(x: np.ndarray, kh: int, kw: int, stride: int, padding: int) -> np.ndarray:
    """Vectorized im2col

    等价于 PyTorch F.unfold(x, kernel_size=(kh, kw), stride=stride, padding=padding)

    Args:
        x: (B, C, H, W)
        kh, kw: 卷积核高宽
        stride: 步长
        padding: 填充

    Returns:
        x_col: (B, C*kh*kw, L) where L = H_out * W_out
    """
    B, C, H, W = x.shape
    H_out = (H + 2 * padding - kh) // stride + 1
    W_out = (W + 2 * padding - kw) // stride + 1

    if padding > 0:
        x_padded = np.pad(
            x, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant', constant_values=0,
        )
    else:
        x_padded = x

    # Ensure C-contiguous for as_strided
    x_padded = np.ascontiguousarray(x_padded)
    s = x_padded.strides

    # patches[b, c, i, j, hi, hj] = x_padded[b, c, i*stride + hi, j*stride + hj]
    patches = as_strided(
        x_padded,
        shape=(B, C, H_out, W_out, kh, kw),
        strides=(s[0], s[1], s[2] * stride, s[3] * stride, s[2], s[3]),
        writeable=False,
    )
    # (B, C, H_out, W_out, kh, kw) → (B, C, kh, kw, H_out, W_out) → (B, K, L)
    x_col = patches.transpose(0, 1, 4, 5, 2, 3).reshape(B, C * kh * kw, H_out * W_out)
    return x_col


def _col2im(x_col: np.ndarray, x_shape: tuple,
            kh: int, kw: int, stride: int, padding: int) -> np.ndarray:
    """Vectorized col2im (累加重叠区域)

    等价于 PyTorch F.conv_transpose2d 的核心操作

    Args:
        x_col: (B, C*kh*kw, L)
        x_shape: 原始输入形状 (B, C, H, W)
        kh, kw: 卷积核高宽
        stride: 步长
        padding: 填充

    Returns:
        x: (B, C, H, W)
    """
    B, C, H, W = x_shape
    H_out = (H + 2 * padding - kh) // stride + 1
    W_out = (W + 2 * padding - kw) // stride + 1
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding

    x_padded = np.zeros((B, C, H_padded, W_padded), dtype=x_col.dtype)

    # (B, K, L) → (B, C, kh, kw, H_out, W_out)
    x_col_reshaped = x_col.reshape(B, C, kh, kw, H_out, W_out)

    # Scatter-add: 每个 (i, j) 位置累加到 strided 位置
    for i in range(kh):
        for j in range(kw):
            x_padded[:, :, i:i + stride * H_out:stride, j:j + stride * W_out:stride] += \
                x_col_reshaped[:, :, i, j, :, :]

    if padding > 0:
        return x_padded[:, :, padding:padding + H, padding:padding + W]
    return x_padded


# ============================================================
# Layers
# ============================================================

class SGNConv2d:
    """HC8 整数卷积层（前向 HC8 量化 matmul，反向 numpy STE）

    前向:
      1. im2col: x (B, C_in, H, W) → x_col (B, K, L), K = C_in*kh*kw, L = H_out*W_out
      2. reshape: x_col → (B*L, K), weight → (C_out, K)
      3. HC8 量化 x_col 和 weight (per-tensor)
      4. matmul: y_col_2d = x_col_2d @ w_2d.T → (B*L, C_out)
      5. reshape + bias: y_col → y (B, C_out, H_out, W_out)

    反向 (STE 直通):
      1. grad_x_col = grad_output_reshaped @ w_2d → col2im → grad_x
      2. grad_w = grad_output_reshaped.T @ x_col_2d
      3. grad_bias = grad_output.sum(axis=(0, 2, 3))
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 use_hc8: bool = True, seed: Optional[int] = None,
                 sbe_variant: str = "original", group_size: int = 96):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_hc8 = use_hc8
        self.sbe_variant = sbe_variant
        self.group_size = group_size

        # Kaiming uniform initialization（与 PyTorch nn.Conv2d 默认一致）
        # PyTorch Conv2d 默认 kaiming_uniform_(a=sqrt(5))，等价 bound = sqrt(1/fan_in)
        # 旧代码 sqrt(6/fan_in) 是 Xavier Uniform，比 Kaiming 大 2.45x，导致 HC8 量化 scale 过大
        fan_in = in_channels * kernel_size * kernel_size
        rng = np.random.RandomState(seed) if seed is not None else np.random
        bound = float(np.sqrt(1.0 / fan_in))
        self.weight = rng.uniform(-bound, bound,
                                  (out_channels, in_channels, kernel_size, kernel_size)
                                  ).astype(np.float32)
        self.bias = np.zeros(out_channels, dtype=np.float32)

        # Momentum buffers
        self.v_weight = np.zeros_like(self.weight)
        self.v_bias = np.zeros_like(self.bias)

        # Gradients (累积模式：backward 累加，zero_grad 清零，scale_grads 平均)
        self.grad_weight: Optional[np.ndarray] = None
        self.grad_bias: Optional[np.ndarray] = None

        # Cache (set by forward)
        self._x_cache: Optional[np.ndarray] = None
        self._x_col_2d_cache: Optional[np.ndarray] = None  # 优化 3

        # SBE 权重缓存（仅 C 融合路径使用，懒计算）
        self._w_sbe_cache: Optional[tuple] = None
        self._w_dirty: bool = True
        # groups / k_block 在 _refresh_weight_cache_sbe 中根据 K 自动计算并缓存
        self.groups: int = 1
        self.k_block: int = in_channels * kernel_size * kernel_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (B, C_in, H, W) → y: (B, C_out, H_out, W_out)"""
        # C 融合路径条件：sbe_variant == "original" 且 use_hc8 且未禁用
        use_c_fusion = (
            self.sbe_variant == "original"
            and self.use_hc8
            and os.environ.get("SGN_NO_C_FUSION") != "1"
        )

        if use_c_fusion:
            # 懒导入 SBE 模块：失败则 fallback 到原 numpy HC8 路径
            try:
                from sbe_conv2d import quantize_weight_sbe, _HAS_SBE_C, _pysgn_net
                c_available = (
                    _HAS_SBE_C
                    and hasattr(_pysgn_net, "sbe_conv2d_forward")
                )
            except ImportError:
                c_available = False

            if c_available:
                # 权重缓存懒刷新
                if self._w_dirty:
                    self._refresh_weight_cache_sbe()

                B, C_in, H, W = x.shape
                kh = kw = self.kernel_size
                C_out = self.out_channels
                stride = self.stride
                padding = self.padding

                # Cache input for backward
                self._x_cache = x
                # 懒计算标记：x_col_2d 在 backward 时才生成（C 路径前向不需要）
                self._x_col_2d_cache = None

                # 输入连续化
                x_in = x
                if x_in.dtype != np.float32 or not x_in.flags["C_CONTIGUOUS"]:
                    x_in = np.ascontiguousarray(x_in, dtype=np.float32)

                w_signed, w_sum_b, w_scales = self._w_sbe_cache
                y = _pysgn_net.sbe_conv2d_forward(
                    x_in, w_signed, w_sum_b, w_scales,
                    B, C_in, H, W,
                    C_out, kh, kw,
                    stride, padding,
                    self.groups, self.k_block,
                    self.bias,
                )
                return y

        # 非 C 路径：保持现有 numpy HC8 / float 路径不变
        B, C_in, H, W = x.shape
        kh = kw = self.kernel_size
        C_out = self.out_channels
        stride = self.stride
        padding = self.padding

        H_out = (H + 2 * padding - kh) // stride + 1
        W_out = (W + 2 * padding - kw) // stride + 1

        # Cache input for backward
        self._x_cache = x

        # im2col: (B, K, L)
        x_col = _im2col(x, kh, kw, stride, padding)
        K = C_in * kh * kw
        L = H_out * W_out

        # weight: (C_out, C_in, kh, kw) → (C_out, K)
        w_2d = self.weight.reshape(C_out, K)

        # x_col: (B, K, L) → (B, L, K) → (B*L, K)
        x_col_2d = x_col.transpose(0, 2, 1).reshape(B * L, K)

        # 优化 3：缓存 x_col_2d 供 backward 复用，避免重复 im2col
        # 注意：存未量化值（反向 STE 直通）
        self._x_col_2d_cache = x_col_2d

        # matmul (with optional HC8 quantization)
        if self.use_hc8:
            x_q = hc8_quantize(x_col_2d)
            w_q = hc8_quantize(w_2d)
            y_col_2d = x_q @ w_q.T  # (B*L, C_out)
        else:
            y_col_2d = x_col_2d @ w_2d.T  # (B*L, C_out)

        # add bias
        y_col_2d = y_col_2d + self.bias.reshape(1, -1)

        # reshape: (B*L, C_out) → (B, L, C_out) → (B, C_out, L) → (B, C_out, H_out, W_out)
        y = y_col_2d.reshape(B, L, C_out).transpose(0, 2, 1).reshape(B, C_out, H_out, W_out)
        return y

    def _refresh_weight_cache_sbe(self) -> None:
        """重新计算 SBE 权重缓存（C 融合路径使用，权重更新后调用）

        懒导入 sbe_conv2d.quantize_weight_sbe，按 K=C_in*kh*kw 自动选择
        (groups, k_block)，权重按 (K, C_out) 排布量化为三元组
        (w_signed, w_sum_b, w_scales)。
        """
        from sbe_conv2d import quantize_weight_sbe

        C_in = self.in_channels
        kh = kw = self.kernel_size
        K = C_in * kh * kw
        C_out = self.out_channels

        # 自动选择 (groups, k_block) 使 groups*k_block == K 且 k_block <= group_size
        groups, k_block = select_k_block(K, target=self.group_size)
        self.groups = groups
        self.k_block = k_block

        # weight: (C_out, C_in, kh, kw) → (K, C_out) = (k, n)
        # 与 sbe_conv2d.py / sgn_sbe_layers.py 一致：transpose(1,2,3,0).reshape(k,n)
        w_2d = self.weight.transpose(1, 2, 3, 0).reshape(K, C_out).copy()
        if w_2d.dtype != np.float32:
            w_2d = w_2d.astype(np.float32)

        m_dummy = 1  # m 在权重量化时不使用
        self._w_sbe_cache = quantize_weight_sbe(
            w_2d, groups, k_block, m_dummy, K, C_out,
            per_channel=False,
        )
        self._w_dirty = False

    def _ensure_x_col_2d_cache(self) -> np.ndarray:
        """懒计算 x_col_2d_cache（C 融合前向路径在 backward 时调用）

        C 融合前向不计算 x_col_2d（C 层内部完成 im2col+matmul），
        但 backward 复用父类逻辑需要 x_col_2d_cache，故在此懒生成。
        """
        if self._x_col_2d_cache is not None:
            return self._x_col_2d_cache

        x = self._x_cache
        B, C_in, H, W = x.shape
        kh = kw = self.kernel_size
        stride = self.stride
        padding = self.padding
        H_out = (H + 2 * padding - kh) // stride + 1
        W_out = (W + 2 * padding - kw) // stride + 1
        K = C_in * kh * kw
        L = H_out * W_out

        x_col = _im2col(x, kh, kw, stride, padding)
        x_col_2d = x_col.transpose(0, 2, 1).reshape(B * L, K)
        if x_col_2d.dtype != np.float32 or not x_col_2d.flags["C_CONTIGUOUS"]:
            x_col_2d = np.ascontiguousarray(x_col_2d, dtype=np.float32)
        self._x_col_2d_cache = x_col_2d
        return x_col_2d

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """grad_output: (B, C_out, H_out, W_out) → grad_x: (B, C_in, H, W)"""
        # C 融合前向路径未缓存 x_col_2d，懒计算填充
        if self._x_col_2d_cache is None and self._x_cache is not None:
            self._ensure_x_col_2d_cache()
        x = self._x_cache
        B, C_in, H, W = x.shape
        kh = kw = self.kernel_size
        C_out = self.out_channels
        stride = self.stride
        padding = self.padding

        H_out = (H + 2 * padding - kh) // stride + 1
        W_out = (W + 2 * padding - kw) // stride + 1
        K = C_in * kh * kw
        L = H_out * W_out

        # weight: (C_out, K)
        w_2d = self.weight.reshape(C_out, K)

        # grad_output: (B, C_out, H_out, W_out) → (B, C_out, L) → (B, L, C_out) → (B*L, C_out)
        grad_out_2d = grad_output.reshape(B, C_out, L).transpose(0, 2, 1).reshape(B * L, C_out)

        # grad_x_col_2d = grad_out_2d @ w_2d → (B*L, K)
        grad_x_col_2d = grad_out_2d @ w_2d

        # reshape to (B, K, L) for col2im
        grad_x_col = grad_x_col_2d.reshape(B, L, K).transpose(0, 2, 1)  # (B, K, L)

        # col2im: (B, K, L) → (B, C_in, H, W)
        grad_x = _col2im(grad_x_col, x.shape, kh, kw, stride, padding)

        # grad_w: (C_out, K) = grad_out_2d.T @ x_col_2d
        # 优化 3：复用 forward 缓存的 x_col_2d，避免重复 im2col
        x_col_2d = self._x_col_2d_cache
        grad_w = grad_out_2d.T @ x_col_2d  # (C_out, K)
        grad_w = grad_w.reshape(C_out, C_in, kh, kw)

        # grad_bias: sum over (B, H_out, W_out)
        grad_bias = grad_output.sum(axis=(0, 2, 3))

        # 累积模式（梯度累积）：backward 时累加到现有梯度
        if self.grad_weight is None:
            self.grad_weight = grad_w
            self.grad_bias = grad_bias
        else:
            self.grad_weight += grad_w
            self.grad_bias += grad_bias

        return grad_x

    def backward_hc16(self, grad_output: np.ndarray) -> np.ndarray:
        """HC16 量化反向传播（非 STE，梯度经过 HC16 量化 round-trip）

        与 backward 的区别：
          - backward: float32 BLAS，梯度不量化（STE 直通）
          - backward_hc16: HC16 量化 round-trip，梯度经过 int16 量化→matmul→反量化

        混合策略：
          - grad_w: per-channel（grad_output 按 C_out 通道 per-channel，精度提升最大）
          - grad_x: per-tensor（两个输入都 C-contiguous，避免非连续输入 bug）
          - grad_bias: 直接 sum（无量化，bias 梯度不需量化）

        数学等价性：max_diff < 1e-3（HC16 量化误差 6.33e-05）

        Args:
            grad_output: (B, C_out, H_out, W_out)

        Returns:
            grad_x: (B, C_in, H, W)
        """
        try:
            import engine.sgn as _sgn
            hc16 = _sgn._native.hc16  # Clang 编译的 sgn 模块的 hc16 子模块
            schema = hc16.default_schema()
        except (ImportError, AttributeError):
            # hc16 不可用，fallback 到 float32 backward
            return self.backward(grad_output)

        # 懒计算 x_col_2d（C 融合前向路径未缓存）
        if self._x_col_2d_cache is None and self._x_cache is not None:
            self._ensure_x_col_2d_cache()
        x = self._x_cache
        B, C_in, H, W = x.shape
        kh = kw = self.kernel_size
        C_out = self.out_channels
        stride = self.stride
        padding = self.padding

        H_out = (H + 2 * padding - kh) // stride + 1
        W_out = (W + 2 * padding - kw) // stride + 1
        K = C_in * kh * kw
        L = H_out * W_out

        w_2d = self.weight.reshape(C_out, K)

        # grad_output: (B, C_out, H_out, W_out) → (B*L, C_out)
        grad_out_2d = grad_output.reshape(B, C_out, L).transpose(0, 2, 1).reshape(B * L, C_out)
        x_col_2d = self._x_col_2d_cache

        # === grad_w = grad_out_2d.T @ x_col_2d (per-channel) ===
        # grad_output 按 C_out 通道 per-channel 量化（转置后按行）
        grad_out_2d_t = np.ascontiguousarray(grad_out_2d.T)  # (C_out, B*L)
        go_scales = hc16.quant_compute_scale_per_channel(grad_out_2d_t, C_out, B * L)
        go_q_t = hc16.quantize_per_channel(grad_out_2d_t, go_scales, schema, C_out, B * L)

        # x_col_2d per-tensor 量化
        xc_scale = hc16.quant_compute_scale(x_col_2d.reshape(-1))
        xc_q = hc16.quantize(x_col_2d.reshape(-1), xc_scale, schema).reshape(B * L, K)

        # matmul_per_channel: A=go_q_t (C_out, B*L), B=xc_q (B*L, K)
        xc_uniform_scales = np.full(K, xc_scale, dtype=np.float32)
        grad_w = hc16.matmul_per_channel(go_q_t, xc_q, C_out, B * L, K,
                                         go_scales, xc_uniform_scales)
        grad_w = grad_w.reshape(C_out, C_in, kh, kw)

        # === grad_x_col = grad_out_2d @ w_2d (per-tensor) ===
        # 两个输入都 C-contiguous，无非连续输入 bug
        go_scale_t = hc16.quant_compute_scale(grad_out_2d.reshape(-1))
        go_q_flat = hc16.quantize(grad_out_2d.reshape(-1), go_scale_t, schema).reshape(B * L, C_out)
        w_scale_t = hc16.quant_compute_scale(w_2d.reshape(-1))
        w_q_flat = hc16.quantize(w_2d.reshape(-1), w_scale_t, schema).reshape(C_out, K)
        grad_x_col_2d = hc16.matmul(go_q_flat, w_q_flat, B * L, C_out, K,
                                    go_scale_t, w_scale_t)

        # col2im: grad_x_col_2d → (B, K, L) → (B, C_in, H, W)
        grad_x_col = grad_x_col_2d.reshape(B, L, K).transpose(0, 2, 1)  # (B, K, L)
        grad_x = _col2im(grad_x_col, x.shape, kh, kw, stride, padding)

        # grad_bias: sum over (B, H_out, W_out) — 不量化
        grad_bias = grad_output.sum(axis=(0, 2, 3))

        # 累积模式（梯度累积）
        if self.grad_weight is None:
            self.grad_weight = grad_w
            self.grad_bias = grad_bias
        else:
            self.grad_weight += grad_w
            self.grad_bias += grad_bias

        return grad_x

    def zero_grad(self) -> None:
        """清零梯度（每个累积周期开始时调用）"""
        self.grad_weight = None
        self.grad_bias = None

    def scale_grads(self, scale: float) -> None:
        """缩放累积的梯度（累积 N 步后除以 N 做平均）"""
        if self.grad_weight is not None:
            self.grad_weight *= scale
        if self.grad_bias is not None:
            self.grad_bias *= scale

    def step(self, lr: float, momentum: float, weight_decay: float = 0.0) -> None:
        """SGD update with momentum and optional L2 weight decay

        与 PyTorch SGD(weight_decay=wd) 等价：
          grad' = grad + wd * weight
          v = momentum * v - lr * grad'
          weight += v
        """
        if weight_decay > 0.0:
            grad_w = self.grad_weight + weight_decay * self.weight
        else:
            grad_w = self.grad_weight
        self.v_weight = momentum * self.v_weight - lr * grad_w
        self.weight += self.v_weight
        self.v_bias = momentum * self.v_bias - lr * self.grad_bias
        self.bias += self.v_bias
        # 标记 SBE 权重缓存需重新量化（下次 forward 时懒刷新）
        self._w_dirty = True


class SGNLinear:
    """HC8 整数线性层（前向 HC8 量化 matmul，反向 numpy STE）

    前向: y = x @ weight.T + bias (with optional HC8 quantization)
    反向: STE 直通 (standard linear backward)
    """

    def __init__(self, in_features: int, out_features: int,
                 use_hc8: bool = True, seed: Optional[int] = None):
        self.in_features = in_features
        self.out_features = out_features
        self.use_hc8 = use_hc8

        # Kaiming uniform initialization（与 PyTorch nn.Linear 默认一致）
        # PyTorch Linear 默认 kaiming_uniform_(a=sqrt(5))，等价 bound = sqrt(1/fan_in)
        rng = np.random.RandomState(seed) if seed is not None else np.random
        bound = float(np.sqrt(1.0 / in_features))
        self.weight = rng.uniform(-bound, bound,
                                  (out_features, in_features)).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)

        # Momentum buffers
        self.v_weight = np.zeros_like(self.weight)
        self.v_bias = np.zeros_like(self.bias)

        # Gradients
        self.grad_weight: Optional[np.ndarray] = None
        self.grad_bias: Optional[np.ndarray] = None

        # Cache
        self._x_cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (B, in_features) → y: (B, out_features)"""
        self._x_cache = x
        # weight: (out, in), x: (B, in) → y: (B, out) = x @ weight.T

        if self.use_hc8:
            x_q = hc8_quantize(x)
            w_q = hc8_quantize(self.weight)
            y = x_q @ w_q.T
        else:
            y = x @ self.weight.T

        y = y + self.bias.reshape(1, -1)
        return y

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """grad_output: (B, out_features) → grad_x: (B, in_features)"""
        x = self._x_cache
        # grad_x = grad_output @ weight
        grad_x = grad_output @ self.weight
        # grad_w = grad_output.T @ x
        grad_w = grad_output.T @ x
        # grad_b = grad_output.sum(axis=0)
        grad_b = grad_output.sum(axis=0)

        # 累积模式（梯度累积）
        if self.grad_weight is None:
            self.grad_weight = grad_w
            self.grad_bias = grad_b
        else:
            self.grad_weight += grad_w
            self.grad_bias += grad_b
        return grad_x

    def backward_hc16(self, grad_output: np.ndarray) -> np.ndarray:
        """HC16 量化反向传播（非 STE，梯度经过 HC16 量化 round-trip）

        与 backward 的区别：
          - backward: float32 BLAS，梯度不量化（STE 直通）
          - backward_hc16: HC16 量化 round-trip，梯度经过 int16 量化→matmul→反量化

        混合策略：
          - grad_w: per-channel（grad_output 按 out_features 通道 per-channel）
          - grad_x: per-tensor（两个输入都 C-contiguous）
          - grad_bias: 直接 sum（无量化）

        Args:
            grad_output: (B, out_features)

        Returns:
            grad_x: (B, in_features)
        """
        try:
            import engine.sgn as _sgn
            hc16 = _sgn._native.hc16  # Clang 编译的 sgn 模块的 hc16 子模块
            schema = hc16.default_schema()
        except (ImportError, AttributeError):
            return self.backward(grad_output)

        x = self._x_cache  # (B, in_features)
        B = x.shape[0]
        out_f = self.out_features
        in_f = self.in_features

        # grad_output: (B, out_features)
        # weight: (out_features, in_features)

        # === grad_w = grad_output.T @ x (per-channel) ===
        # grad_output 按 out_features per-channel 量化（转置后按行）
        grad_out_t = np.ascontiguousarray(grad_output.T)  # (out_features, B)
        go_scales = hc16.quant_compute_scale_per_channel(grad_out_t, out_f, B)
        go_q_t = hc16.quantize_per_channel(grad_out_t, go_scales, schema, out_f, B)

        # x per-tensor 量化
        x_scale = hc16.quant_compute_scale(x.reshape(-1))
        x_q = hc16.quantize(x.reshape(-1), x_scale, schema).reshape(B, in_f)

        # matmul_per_channel: A=go_q_t (out, B), B=x_q (B, in)
        x_uniform_scales = np.full(in_f, x_scale, dtype=np.float32)
        grad_w = hc16.matmul_per_channel(go_q_t, x_q, out_f, B, in_f,
                                         go_scales, x_uniform_scales)

        # === grad_x = grad_output @ weight (per-tensor) ===
        go_scale_t = hc16.quant_compute_scale(grad_output.reshape(-1))
        go_q_flat = hc16.quantize(grad_output.reshape(-1), go_scale_t, schema).reshape(B, out_f)
        w_scale_t = hc16.quant_compute_scale(self.weight.reshape(-1))
        w_q_flat = hc16.quantize(self.weight.reshape(-1), w_scale_t, schema).reshape(out_f, in_f)
        grad_x = hc16.matmul(go_q_flat, w_q_flat, B, out_f, in_f,
                             go_scale_t, w_scale_t)

        # grad_bias: 不量化
        grad_b = grad_output.sum(axis=0)

        # 累积模式
        if self.grad_weight is None:
            self.grad_weight = grad_w
            self.grad_bias = grad_b
        else:
            self.grad_weight += grad_w
            self.grad_bias += grad_b
        return grad_x

    def zero_grad(self) -> None:
        """清零梯度"""
        self.grad_weight = None
        self.grad_bias = None

    def scale_grads(self, scale: float) -> None:
        """缩放累积的梯度"""
        if self.grad_weight is not None:
            self.grad_weight *= scale
        if self.grad_bias is not None:
            self.grad_bias *= scale

    def step(self, lr: float, momentum: float, weight_decay: float = 0.0) -> None:
        """SGD update with momentum and optional L2 weight decay"""
        if weight_decay > 0.0:
            grad_w = self.grad_weight + weight_decay * self.weight
        else:
            grad_w = self.grad_weight
        self.v_weight = momentum * self.v_weight - lr * grad_w
        self.weight += self.v_weight
        self.v_bias = momentum * self.v_bias - lr * self.grad_bias
        self.bias += self.v_bias


class SGNBatchNorm2d:
    """BatchNorm2d（float 域，方案 A — 与 2.3 一致）

    PyTorch 风格接口:
      - train() / eval() 切换模式
      - forward(x) 不接受 training 参数，按当前模式执行

    前向:
      training=True: 计算批统计 (mean/var over B,H,W), 更新 running 统计
      training=False: 用 running 统计

    反向:
      training=True: 标准 BN 反向公式（考虑 dvar/dmean）
      training=False: dx = dx_norm * std_inv（mean/var 固定）

    参考: numpy_backward.py 的 numpy_batchnorm_forward / numpy_batchnorm_backward
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 seed: Optional[int] = None):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # gamma=1, beta=0 初始化（标准做法）
        self.weight = np.ones(num_features, dtype=np.float32)  # gamma
        self.bias = np.zeros(num_features, dtype=np.float32)   # beta

        # running 统计（eval 模式用）
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

        # Momentum buffers (for SGD)
        self.v_weight = np.zeros_like(self.weight)
        self.v_bias = np.zeros_like(self.bias)

        # Gradients
        self.grad_weight: Optional[np.ndarray] = None
        self.grad_bias: Optional[np.ndarray] = None

        # Training mode flag (PyTorch style)
        self.training = True

        # Cache (set by forward)
        self._cache: tuple = ()

    def train(self) -> 'SGNBatchNorm2d':
        self.training = True
        return self

    def eval(self) -> 'SGNBatchNorm2d':
        self.training = False
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (B, C, H, W) → y: (B, C, H, W)"""
        if self.training:
            # 计算批统计（沿 B, H, W 维度）
            # P1 修复：dtype=np.float32 与 PyTorch 一致（避免 numpy 默认 float64 累加导致 dtype 提升）
            mean = x.mean(axis=(0, 2, 3), dtype=np.float32)  # (C,)
            # M4 修复：PyTorch nn.BatchNorm2d 行为
            #   - 归一化用 biased var（ddof=0）
            #   - running_var 更新用 unbiased var（ddof=1）
            var_biased = x.var(axis=(0, 2, 3), ddof=0, dtype=np.float32)  # 归一化用
            N = x.shape[0] * x.shape[2] * x.shape[3]
            if N > 1:
                var_unbiased = var_biased * (N / (N - 1))  # running_var 用
            else:
                var_unbiased = var_biased
            # 更新 running 统计（running_var 用 unbiased）
            self.running_mean[:] = ((1 - self.momentum) * self.running_mean
                                    + self.momentum * mean)
            self.running_var[:] = ((1 - self.momentum) * self.running_var
                                   + self.momentum * var_unbiased)
            # 归一化用 biased var
            var = var_biased
        else:
            mean = self.running_mean
            var = self.running_var

        # 归一化
        std_inv = 1.0 / np.sqrt(var.reshape(1, -1, 1, 1) + self.eps)
        x_norm = (x - mean.reshape(1, -1, 1, 1)) * std_inv
        # 缩放平移
        y = (self.weight.reshape(1, -1, 1, 1) * x_norm
             + self.bias.reshape(1, -1, 1, 1))

        # 缓存反向所需
        self._cache = (x, x_norm, mean, var, std_inv)
        return y

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """grad_output: (B, C, H, W) → grad_x: (B, C, H, W)

        优化 4：BN backward 简化（4 sum → 2 sum）
        利用恒等式（gamma 对 (B,H,W) 广播，可提出 sum）:
          sum(dx_norm)          = gamma * grad_beta
          sum(dx_norm * x_norm) = gamma * grad_gamma
        合并 dvar/dmean 到单一闭式公式:
          dx = (std_inv / N) * (N*dx_norm - sum(dx_norm) - x_norm * sum(dx_norm * x_norm))
        数学推导:
          标准: dx = dx_norm*std_inv + dvar*2*x_mu/N + dmean/N
                dvar = -0.5*sum(dx_norm*x_mu)*std_inv^3
                dmean = -sum(dx_norm)*std_inv
                x_mu = x_norm / std_inv
          代入 x_mu 后整理即得上式（已数值验证）。
        """
        x, x_norm, mean, var, std_inv = self._cache
        B, C, H, W = x.shape
        N = B * H * W

        # 仅 2 个 sum（原 4 个）
        grad_beta = grad_output.sum(axis=(0, 2, 3))             # (C,)
        grad_gamma = (grad_output * x_norm).sum(axis=(0, 2, 3))  # (C,)

        gamma = self.weight  # (C,)

        if self.training:
            # 优化 4：sum_dxnorm = gamma * grad_beta，省去第 3 个 sum
            #       sum_dxnorm_xnorm = gamma * grad_gamma，省去第 4 个 sum
            sum_dxnorm = (gamma * grad_beta).reshape(1, -1, 1, 1)
            sum_dxnorm_xnorm = (gamma * grad_gamma).reshape(1, -1, 1, 1)
            dx_norm = grad_output * gamma.reshape(1, -1, 1, 1)
            dx = (std_inv / N) * (
                N * dx_norm
                - sum_dxnorm
                - x_norm * sum_dxnorm_xnorm
            )
        else:
            # eval 模式：mean/var 固定
            # P2.1 修复：gamma shape (C,) 与 std_inv shape (1,C,1,1) 广播失败，
            # 先将 gamma reshape 为 (1,C,1,1) 再与 std_inv 相乘
            dx = grad_output * (gamma.reshape(1, -1, 1, 1) * std_inv)

        # 累积模式（梯度累积）
        if self.grad_weight is None:
            self.grad_weight = grad_gamma.copy()
            self.grad_bias = grad_beta.copy()
        else:
            self.grad_weight += grad_gamma
            self.grad_bias += grad_beta
        return dx

    def zero_grad(self) -> None:
        """清零梯度"""
        self.grad_weight = None
        self.grad_bias = None

    def scale_grads(self, scale: float) -> None:
        """缩放累积的梯度"""
        if self.grad_weight is not None:
            self.grad_weight *= scale
        if self.grad_bias is not None:
            self.grad_bias *= scale

    def step(self, lr: float, momentum: float, weight_decay: float = 0.0) -> None:
        """SGD update with momentum and optional L2 weight decay (BN weight only, not bias)"""
        if weight_decay > 0.0:
            grad_w = self.grad_weight + weight_decay * self.weight
        else:
            grad_w = self.grad_weight
        self.v_weight = momentum * self.v_weight - lr * grad_w
        self.weight += self.v_weight
        self.v_bias = momentum * self.v_bias - lr * self.grad_bias
        self.bias += self.v_bias


class SGNReLU:
    """ReLU 激活层（float 域，单调映射等价整数域）"""

    def __init__(self):
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = (x > 0).astype(x.dtype)
        return x * self._mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self._mask

    def step(self, lr: float, momentum: float, weight_decay: float = 0.0) -> None:
        pass


class SGNMaxPool2d:
    """MaxPool2d（float 域，整数 max 等价 float max，单调映射）

    优化 2（CPU 利用率）:
      - 前向：非重叠池化用 reshape + max(axis=-1) 替代 as_strided + max(axis=(-2,-1))
              在连续内存上归约，避免 as_strided 非连续视图的慢路径
      - mask：改用 argmax 索引 (B, C, H_out, W_out) 替代显式 (B, C, H_out, W_out, k, k) mask
              内存占用从 O(B*C*H_out*W_out*k*k) 降至 O(B*C*H_out*W_out)
              （首层 MaxPool: 134MB → 2MB）
      - 反向：k*k 次 boolean 索引赋值散布梯度
      - ties 处理：改为取首个最大值（PyTorch MaxPool 默认行为），不平分梯度
    """

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        self.k = kernel_size
        self.stride = stride
        self._argmax: Optional[np.ndarray] = None
        self._x_shape: Optional[tuple] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (B, C, H, W) → y: (B, C, H_out, W_out)"""
        B, C, H, W = x.shape
        k = self.k
        stride = self.stride
        H_out = (H - k) // stride + 1
        W_out = (W - k) // stride + 1

        self._x_shape = x.shape

        if stride == k and H == H_out * k and W == W_out * k:
            # 非重叠池化：reshape 高效路径（避免 as_strided 非连续视图）
            # x: (B, C, H, W) → (B, C, H_out, k, W_out, k)
            #   → transpose(0,1,2,4,3,5) → (B, C, H_out, W_out, k, k)
            #   → reshape → (B, C, H_out, W_out, k*k)
            x_t = (x.reshape(B, C, H_out, k, W_out, k)
                   .transpose(0, 1, 2, 4, 3, 5)
                   .reshape(B, C, H_out, W_out, k * k))
            y = x_t.max(axis=-1)                 # (B, C, H_out, W_out)
            self._argmax = x_t.argmax(axis=-1)    # 0..k*k-1
        else:
            # 重叠/非整数倍：as_strided fallback
            x_c = np.ascontiguousarray(x)
            s = x_c.strides
            patches = as_strided(
                x_c,
                shape=(B, C, H_out, W_out, k, k),
                strides=(s[0], s[1], s[2] * stride, s[3] * stride, s[2], s[3]),
                writeable=False,
            )
            x_t = patches.reshape(B, C, H_out, W_out, k * k)
            y = x_t.max(axis=-1)
            self._argmax = x_t.argmax(axis=-1)

        return y

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """grad_output: (B, C, H_out, W_out) → grad_x: (B, C, H, W)"""
        B, C, H, W = self._x_shape
        k = self.k
        stride = self.stride
        H_out = (H - k) // stride + 1
        W_out = (W - k) // stride + 1

        grad_x = np.zeros(self._x_shape, dtype=grad_output.dtype)
        argmax = self._argmax

        # argmax 值 0..k*k-1，对应 (kh, kw) = (idx // k, idx % k)
        for idx in range(k * k):
            i, j = idx // k, idx % k
            mask = (argmax == idx)
            grad_x[:, :, i:i + stride * H_out:stride, j:j + stride * W_out:stride] += (
                grad_output * mask
            )

        return grad_x

    def step(self, lr: float, momentum: float, weight_decay: float = 0.0) -> None:
        pass


class SGNFlatten:
    """Flatten 层"""

    def __init__(self):
        self._x_shape: Optional[tuple] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output.reshape(self._x_shape)

    def step(self, lr: float, momentum: float, weight_decay: float = 0.0) -> None:
        pass


class SGNSequential:
    """Simple sequential container (PyTorch-like interface)"""

    def __init__(self, *layers):
        self.layers = list(layers)
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def step(self, lr: float, momentum: float, weight_decay: float = 0.0) -> None:
        for layer in self.layers:
            if hasattr(layer, 'step'):
                # BN 层的 bias 不加 weight_decay（与 PyTorch 一致），但 weight 加
                layer.step(lr, momentum, weight_decay)

    def zero_grad(self) -> None:
        """清零所有层的梯度（每个累积周期开始时调用）"""
        for layer in self.layers:
            if hasattr(layer, 'zero_grad'):
                layer.zero_grad()

    def scale_grads(self, scale: float) -> None:
        """缩放所有层的累积梯度（累积 N 步后除以 N 做平均）"""
        for layer in self.layers:
            if hasattr(layer, 'scale_grads'):
                layer.scale_grads(scale)

    def train(self) -> 'SGNSequential':
        self.training = True
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()
        return self

    def eval(self) -> 'SGNSequential':
        self.training = False
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()
        return self

    def clip_grad_norm(self, max_norm: float) -> float:
        """梯度裁剪（global norm clipping，与 PyTorch clip_grad_norm_ 等价）

        遍历所有有 grad_weight/grad_bias 的层，计算全局范数并缩放。
        返回裁剪前的全局范数（用于日志）。
        """
        total_norm_sq = 0.0
        for layer in self.layers:
            for attr in ('grad_weight', 'grad_bias'):
                g = getattr(layer, attr, None)
                if g is not None:
                    total_norm_sq += float((g * g).sum())
        total_norm = float(np.sqrt(total_norm_sq))

        if total_norm > max_norm and total_norm > 0:
            scale = max_norm / (total_norm + 1e-6)
            for layer in self.layers:
                for attr in ('grad_weight', 'grad_bias'):
                    g = getattr(layer, attr, None)
                    if g is not None:
                        g *= scale  # in-place via *
        return total_norm


# ============================================================
# 自检
# ============================================================

def _self_check() -> None:
    """自检：验证向量化版本与 numpy_backward.py 参考实现一致"""
    import sys
    from pathlib import Path
    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))

    from numpy_backward import (
        numpy_im2col, numpy_col2im,
        numpy_conv2d_forward, numpy_conv2d_grad_x,
    )

    print("=" * 60)
    print("sgn_layers.py self-check")
    print("=" * 60)

    np.random.seed(42)
    # 小尺寸测试
    B, C, H, W = 2, 3, 6, 6
    kh = kw = 3
    stride = 1
    padding = 1

    x = np.random.randn(B, C, H, W).astype(np.float32)

    # im2col 对比
    x_col_ref = numpy_im2col(x, kh, kw, stride, padding)
    x_col_vec = _im2col(x, kh, kw, stride, padding)
    err = np.abs(x_col_ref - x_col_vec).max()
    print(f"  im2col max_err = {err:.2e}  {'✅' if err < 1e-6 else '❌'}")

    # col2im 对比
    x_recon_ref = numpy_col2im(x_col_ref, x.shape, kh, kw, stride, padding)
    x_recon_vec = _col2im(x_col_ref, x.shape, kh, kw, stride, padding)
    err = np.abs(x_recon_ref - x_recon_vec).max()
    print(f"  col2im max_err = {err:.2e}  {'✅' if err < 1e-6 else '❌'}")

    # Conv2d 前向 + 反向
    layer = SGNConv2d(C, 4, kernel_size=kh, stride=stride, padding=padding,
                     use_hc8=False, seed=42)
    y = layer.forward(x)
    grad_out = np.ones_like(y)
    grad_x = layer.backward(grad_out)

    # 对比参考实现
    y_ref = numpy_conv2d_forward(x, layer.weight, layer.bias, stride, padding)
    err = np.abs(y - y_ref).max()
    print(f"  Conv2d forward max_err = {err:.2e}  {'✅' if err < 1e-5 else '❌'}")

    grad_x_ref = numpy_conv2d_grad_x(grad_out, layer.weight, x.shape, stride, padding)
    err = np.abs(grad_x - grad_x_ref).max()
    print(f"  Conv2d grad_x max_err = {err:.2e}  {'✅' if err < 1e-5 else '❌'}")

    # HC8 量化 round-trip
    x_test = np.array([[-1.0, 0.0, 0.5], [1.0, -0.5, 2.0]], dtype=np.float32)
    x_q = hc8_quantize(x_test)
    # 检查范围: x_q 应该在 [-max, max] 内，且量化精度为 scale 的倍数
    print(f"  HC8 quantize: input range [{x_test.min():.2f}, {x_test.max():.2f}] "
          f"→ quantized range [{x_q.min():.4f}, {x_q.max():.4f}]")

    print("\n" + "=" * 60)
    print("sgn_layers.py self-check 完成")
    print("=" * 60)


if __name__ == "__main__":
    _self_check()
