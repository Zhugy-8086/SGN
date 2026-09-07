"""HC8 整数卷积路径（阶段 2.3 核心组件）

设计原则（基于阶段 1.4 经验 + 阶段 2.3 分析意见）：
  1. C 扩展零改动：im2col 后卷积 = 2D 矩阵乘，复用 _linear_ste_direct_c 的 C 路径
  2. STE 反向用 PyTorch autograd：前向 HC8 matmul，反向 float 公式（不走 im2col 反向）
  3. 权重量化缓存：encode_interval 步内复用量化后的权重 bytes，不每步量化
  4. 只支持 MaxPool（baseline 无 AvgPool）：整数 max 等价 float max（单调映射）
  5. BN 方案 A（保留 float）：BN 不参与 HC8 量化
  6. C matmul 输出会重新量化到 HC8（额外 ~0.4% 误差/层），监控累积

核心组件：
  - Conv2dSTE: torch.autograd.Function，前向 im2col + HC8 matmul，反向 STE 直通
  - LinearSTE: torch.autograd.Function，前向 HC8 matmul，反向 STE 直通
  - HC8Conv2d: nn.Module 包装 Conv2dSTE，含权重量化缓存
  - HC8Linear: nn.Module 包装 LinearSTE，含权重量化缓存
  - HC8ConvEncoder: 管理权重编码-解码循环（参考 MSIntConvEncoder 设计模式）

用法：
    from hc_conv2d import HC8Conv2d, HC8Linear, HC8ConvEncoder
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# C 扩展检测（复用阶段 1.4 的 pysgn_net）
# ============================================================

# C 扩展经统一 sgn 模块导入（engine.sgn._native.hc8_net）——旧独立扩展
# 目录已并入 hc/ext/，无路径注入需求（A3 审计 2026-09-07）

_HAS_C_EXT = False
_pysgn_net = None
_C_SCHEMA = None
if not os.environ.get("SGN_NO_C_EXT"):
    try:
        import engine.sgn as _sgn
        _pysgn_net = _sgn._native.hc8_net
        _C_SCHEMA = _pysgn_net.default_schema()
        _HAS_C_EXT = True
    except (ImportError, AttributeError):
        pass


# ============================================================
# HC8 量化辅助函数（内联，不创建 HC8 对象）
# ============================================================

# HC8 schema 常量（与 C 扩展默认 schema 一致：qmin=-127, qmax=127, offset=128）
_HC8_OFFSET = 128
_HC8_QMIN = -127
_HC8_QMAX = 127


def _quantize_to_bytes(x_flat: list, scale: float) -> bytes:
    """量化 flat float list 到 HC8 bytes（C 扩展）

    Args:
        x_flat: float list
        scale: 量化 scale = max(|x|) / 127

    Returns:
        HC8 bytes（每个元素 6 字节，v[0] 存量化值，v[1..5]=0）
    """
    return _pysgn_net.quantize(x_flat, scale, _C_SCHEMA)


def _compute_scale(x_flat: list) -> float:
    """计算量化 scale = max(|x|) / 127"""
    x_max = max(abs(v) for v in x_flat) if x_flat else 0.0
    return x_max / 127 if x_max > 0 else 1.0


def _dequantize_bytes(bytes_y: bytes, y_scale: float) -> list:
    """反量化 HC8 bytes 到 float list（C 扩展）"""
    return _pysgn_net.dequantize(bytes_y, y_scale, _C_SCHEMA)


def _compute_scale_np(x_np: np.ndarray) -> float:
    """计算量化 scale = max(|x|) / 127（numpy 版本，避免 Python 迭代）"""
    x_max = float(np.abs(x_np).max()) if x_np.size > 0 else 0.0
    return x_max / 127.0 if x_max > 0 else 1.0


def _quantize_to_bytes_np(x_np: np.ndarray, scale: float) -> bytes:
    """numpy 直接量化为 HC8 bytes（跳过 C 扩展 quantize + tolist）

    HC8 格式: 每个元素 6 字节, v[0]=量化值+128, v[1..5]=0
    量化语义与 C 扩展 hc8_quantize 完全一致:
      1. q = round(x / scale)
      2. clamp q 到 [-127, 127]
      3. q_u = q + 128, clamp 到 [0, 255]
      4. v[0] = q_u, v[1..5] = 0

    性能提升来源:
      - 跳过 torch tensor → list 转换（tolist() 极慢）
      - 跳过 list → C vector 转换（pybind11 开销）
      - numpy 向量化运算 + 直接构造 bytes

    Args:
        x_np: numpy float 数组（任意形状，内部 flatten）
        scale: 量化 scale

    Returns:
        HC8 bytes（长度 = x_np.size * 6）
    """
    if scale == 0.0:
        scale = 1.0
    inv_scale = 1.0 / scale

    # 量化: q = round(x * inv_scale), clamp 到 [-127, 127]
    q = np.round(x_np.ravel() * inv_scale).astype(np.int32)
    np.clip(q, _HC8_QMIN, _HC8_QMAX, out=q)

    # 偏移到无符号: q_u = q + 128, clamp 到 [0, 255]
    q_u = (q + _HC8_OFFSET).astype(np.int32)
    np.clip(q_u, 0, 255, out=q_u)
    q_u = q_u.astype(np.uint8)

    # 构造 6 字节格式: [q_u, 0, 0, 0, 0, 0] 连续存储
    n = q_u.size
    bytes_arr = np.zeros(n * 6, dtype=np.uint8)
    bytes_arr[::6] = q_u
    return bytes_arr.tobytes()


def _c_matmul(
    bytes_a: bytes, bytes_b: bytes,
    m: int, k: int, n: int,
    a_scale: float, b_scale: float,
) -> Tuple[list, float]:
    """HC8 矩阵乘（C 扩展，含输出重新量化）

    Returns:
        (y_flat, y_scale): float list（长度 m*n）和输出 scale
    """
    bytes_y, y_scale = _pysgn_net.matmul(
        bytes_a, bytes_b, m, k, n, a_scale, b_scale, _C_SCHEMA
    )
    # 反量化（C matmul 输出重新量化到 HC8，需要反量化为 float）
    y_flat = _dequantize_bytes(bytes_y, y_scale)
    return y_flat, y_scale


def _c_matmul_int_no_requant(
    bytes_a: bytes, bytes_b: bytes,
    m: int, k: int, n: int,
    a_scale: float, b_scale: float,
) -> np.ndarray:
    """HC8 整数矩阵乘（不重新量化输出）

    保留完整 HC8 整数路径（int8×int8→int32 累加），但跳过输出重新量化。
    避免重新量化的非线性 scale 变化破坏 STE 梯度。

    流程:
      1. 从 HC8 bytes 提取 int8 量化值（每 6 字节取 v[0]）
      2. 反量化到 float32（乘以 scale）
      3. numpy float32 BLAS 矩阵乘（多线程加速）
      4. 输出 float32 结果（不重新量化）

    数学等价性: c = (a_q @ b_q) * a_scale * b_scale
                      = (a_q * a_scale) @ (b_q * b_scale)  (分配律)
    故先反量化再 BLAS matmul 严格等价于先 int32 matmul 再反量化。

    性能: float32 走 BLAS（多线程），vs 旧 int32 单线程快 12-88x
    （见 hc8_train_perf_diagnosis_and_roadmap.md §7.0.4）。

    Returns:
        y_2d: numpy float32 数组 (m, n)（直接返回 numpy，避免 tolist 开销）
    """
    # 提取 int8 量化值（每个 HC8 6 字节，v[0] 是量化值 with offset 128）
    a_q = np.frombuffer(bytes_a, dtype=np.uint8)[::6].astype(np.float32)
    a_q = (a_q.reshape(m, k) - 128.0) * a_scale  # 反量化到 float32 [-128,127]*scale
    b_q = np.frombuffer(bytes_b, dtype=np.uint8)[::6].astype(np.float32)
    b_q = (b_q.reshape(k, n) - 128.0) * b_scale  # 反量化到 float32

    # float32 BLAS 矩阵乘（多线程，数学等价于 int32 累加后反量化）
    c_float = a_q @ b_q  # (m, n) float32

    return c_float  # (m, n) numpy array


# ============================================================
# STE 自动微分函数
# ============================================================

class Conv2dSTE(torch.autograd.Function):
    """HC8 整数卷积（im2col + HC8 matmul）+ STE 直通

    前向：
      1. im2col: (B, C_in, H, W) → (B*H_out*W_out, C_in*kh*kw)
      2. 量化 im2col 矩阵和权重（展平）
      3. HC8 matmul: (B*H_out*W_out, C_in*kh*kw) @ (C_in*kh*kw, C_out)
      4. 反量化 + 加 bias + reshape → (B, C_out, H_out, W_out)

    反向：
      STE 直通：grad_output → grad_input（走 PyTorch Conv2d 反向公式）
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,           # (B, C_in, H, W)
        weight: torch.Tensor,      # (C_out, C_in, kh, kw)
        bias: Optional[torch.Tensor],  # (C_out,)
        stride: int,
        padding: int,
        w_bytes_cache: Optional[bytes] = None,  # 权重量化缓存
        w_scale_cache: Optional[float] = None,
    ) -> torch.Tensor:
        B, C_in, H, W = x.shape
        C_out, _, kh, kw = weight.shape

        # 1. im2col: (B, C_in*kh*kw, H_out*W_out)
        x_col = F.unfold(
            x, kernel_size=(kh, kw), stride=stride, padding=padding
        )  # (B, C_in*kh*kw, L) where L = H_out*W_out

        # 转置为 (B*L, C_in*kh*kw)
        x_col_2d = x_col.permute(0, 2, 1).reshape(-1, C_in * kh * kw)
        m = x_col_2d.shape[0]  # B * H_out * W_out
        k = C_in * kh * kw
        n = C_out

        if not _HAS_C_EXT:
            # 纯 Python fallback（无 C 扩展时）
            y = x_col_2d @ weight.reshape(n, k).T  # (m, n)
            if bias is not None:
                y = y + bias.unsqueeze(0)
            y = y.reshape(B, -1, n).permute(0, 2, 1)  # (B, C_out, L)
            H_out = (H + 2 * padding - kh) // stride + 1
            W_out = (W + 2 * padding - kw) // stride + 1
            return y.reshape(B, C_out, H_out, W_out)

        # 2. 量化输入 im2col 矩阵（numpy 路径，避免 tolist 开销）
        x_np = x_col_2d.detach().contiguous().numpy()  # (m, k)
        x_scale = _compute_scale_np(x_np)
        bytes_x = _quantize_to_bytes_np(x_np, x_scale)

        # 3. 权重量化（使用缓存或重新量化）
        if w_bytes_cache is not None and w_scale_cache is not None:
            bytes_w = w_bytes_cache
            w_scale = w_scale_cache
        else:
            # weight: (C_out, C_in, kh, kw) → (k, n) 列优先 = permute(1,2,3,0).flatten()
            w_np = weight.permute(1, 2, 3, 0).detach().contiguous().numpy()  # (k, n)
            w_scale = _compute_scale_np(w_np)
            bytes_w = _quantize_to_bytes_np(w_np, w_scale)

        # 4. HC8 整数 matmul（int8×int8→int32 累加，不重新量化输出）
        y_np = _c_matmul_int_no_requant(bytes_x, bytes_w, m, k, n, x_scale, w_scale)  # (m, n)

        # 5. 转 torch + 加 bias
        y_2d = torch.from_numpy(y_np)
        if bias is not None:
            y_2d = y_2d + bias.unsqueeze(0)

        # (m, n) → (B, L, n) → (B, n, L) → (B, C_out, H_out, W_out)
        H_out = (H + 2 * padding - kh) // stride + 1
        W_out = (W + 2 * padding - kw) // stride + 1
        y = y_2d.reshape(B, H_out * W_out, C_out).permute(0, 2, 1)
        y = y.reshape(B, C_out, H_out, W_out)

        # 保存反向所需上下文
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding

        # STE 直通：用 PyTorch float Conv2d 反向公式计算梯度
        # 前向等价于 conv2d(x, weight, bias, stride, padding)
        # 所以反向直接用 conv2d 的梯度公式
        grad_x = grad_w = grad_bias = None

        if ctx.needs_input_grad[0]:
            # grad_x = conv2d_backward(grad_output, weight)
            grad_x = F.conv_transpose2d(
                grad_output, weight, stride=stride, padding=padding
            )
        if ctx.needs_input_grad[1]:
            # grad_w = conv2d_weight_backward(x, grad_output)
            grad_w = _conv2d_weight_grad(x, grad_output, weight.shape, stride, padding)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        return grad_x, grad_w, grad_bias, None, None, None, None


def _conv2d_weight_grad(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    weight_shape: torch.Size,
    stride: int,
    padding: int,
) -> torch.Tensor:
    """计算卷积权重的梯度（复用 PyTorch unfold 实现）

    grad_w[c_out, c_in, kh, kw] = sum_{b, i, j} grad_output[b, c_out, i, j] * x_col[b, c_in*kh*kw, i*j]
    """
    C_out, C_in, kh, kw = weight_shape
    # x_col: (B, C_in*kh*kw, L)
    x_col = F.unfold(x, kernel_size=(kh, kw), stride=stride, padding=padding)
    # grad_output: (B, C_out, H_out, W_out) → (B, C_out, L)
    B, _, H_out, W_out = grad_output.shape
    L = H_out * W_out
    grad_out_2d = grad_output.reshape(B, C_out, L)

    # x_col: (B, K, L) where K = C_in*kh*kw, L = H_out*W_out
    # grad_out_2d: (B, C_out, L)
    # grad_w[c_out, k] = sum_b sum_l grad_out_2d[b, c_out, l] * x_col[b, k, l]
    # einsum: b=batch, i=C_out, l=L, k=K → 结果 (i, k) = (C_out, K)
    grad_w = torch.einsum("bil,bkl->ik", grad_out_2d, x_col)
    return grad_w.reshape(C_out, C_in, kh, kw)


class LinearSTE(torch.autograd.Function):
    """HC8 整数线性层 + STE 直通

    前向：im2col 不需要，直接 2D matmul
    反向：STE 直通（PyTorch Linear 反向公式）
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,           # (B, in_features)
        weight: torch.Tensor,      # (out_features, in_features)
        bias: Optional[torch.Tensor],
        w_bytes_cache: Optional[bytes] = None,
        w_scale_cache: Optional[float] = None,
    ) -> torch.Tensor:
        m, k = x.shape
        n = weight.shape[0]

        if not _HAS_C_EXT:
            y = x @ weight.T
            if bias is not None:
                y = y + bias.unsqueeze(0)
            return y

        # 量化输入（numpy 路径，避免 tolist 开销）
        x_np = x.detach().contiguous().numpy()  # (m, k)
        x_scale = _compute_scale_np(x_np)
        bytes_x = _quantize_to_bytes_np(x_np, x_scale)

        # 权重量化（缓存或重新量化）
        if w_bytes_cache is not None and w_scale_cache is not None:
            bytes_w = w_bytes_cache
            w_scale = w_scale_cache
        else:
            # weight: (n, k)，matmul 需要 B = W^T (k, n)，按列优先展开
            w_np = weight.permute(1, 0).detach().contiguous().numpy()  # (k, n) 列优先
            w_scale = _compute_scale_np(w_np)
            bytes_w = _quantize_to_bytes_np(w_np, w_scale)

        # HC8 整数 matmul（不重新量化输出，返回 numpy array）
        y_np = _c_matmul_int_no_requant(bytes_x, bytes_w, m, k, n, x_scale, w_scale)  # (m, n)
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

        return grad_x, grad_w, grad_bias, None, None


# ============================================================
# HC8 模块（含权重量化缓存）
# ============================================================

class HC8Conv2d(nn.Module):
    """HC8 整数卷积层

    与 nn.Conv2d 接口兼容，但前向走 HC8 整数路径（im2col + HC8 matmul）。
    权重量化缓存：每 encode_interval 步重新量化，中间步骤复用缓存。

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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 标准 Conv2d 参数（用于初始化和 PyTorch 反向）
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # 初始化（与 nn.Conv2d 一致）
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = in_channels * kernel_size * kernel_size
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # 权重量化缓存
        self._w_bytes_cache: Optional[bytes] = None
        self._w_scale_cache: Optional[float] = None
        self._cache_valid: bool = False

    def invalidate_cache(self):
        """使权重量化缓存失效（下次前向时重新量化）"""
        self._cache_valid = False

    def _get_weight_cache(self) -> Tuple[bytes, float]:
        """获取权重量化缓存，如失效则重新量化"""
        if not self._cache_valid or self._w_bytes_cache is None:
            # weight: (C_out, C_in, kh, kw) → (k, n) 列优先展平 = (C_in*kh*kw, C_out)
            w_np = self.weight.permute(1, 2, 3, 0).detach().contiguous().numpy()
            self._w_scale_cache = _compute_scale_np(w_np)
            self._w_bytes_cache = _quantize_to_bytes_np(w_np, self._w_scale_cache)
            self._cache_valid = True
        return self._w_bytes_cache, self._w_scale_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_C_EXT:
            w_bytes, w_scale = self._get_weight_cache()
            return Conv2dSTE.apply(
                x, self.weight, self.bias,
                self.stride, self.padding,
                w_bytes, w_scale,
            )
        else:
            # 无 C 扩展 fallback：直接 float Conv2d
            return F.conv2d(
                x, self.weight, self.bias, self.stride, self.padding
            )


class HC8Linear(nn.Module):
    """HC8 整数线性层

    与 nn.Linear 接口兼容，前向走 HC8 整数路径。
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            bound = 1 / (in_features ** 0.5) if in_features > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        self._w_bytes_cache: Optional[bytes] = None
        self._w_scale_cache: Optional[float] = None
        self._cache_valid: bool = False

    def invalidate_cache(self):
        self._cache_valid = False

    def _get_weight_cache(self) -> Tuple[bytes, float]:
        if not self._cache_valid or self._w_bytes_cache is None:
            # weight: (out, in) → (k, n) 列优先 = (in, out)
            w_np = self.weight.permute(1, 0).detach().contiguous().numpy()
            self._w_scale_cache = _compute_scale_np(w_np)
            self._w_bytes_cache = _quantize_to_bytes_np(w_np, self._w_scale_cache)
            self._cache_valid = True
        return self._w_bytes_cache, self._w_scale_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_C_EXT:
            w_bytes, w_scale = self._get_weight_cache()
            return LinearSTE.apply(x, self.weight, self.bias, w_bytes, w_scale)
        else:
            return F.linear(x, self.weight, self.bias)


# ============================================================
# HC8ConvEncoder：权重编码-解码循环管理
# ============================================================

class HC8ConvEncoder:
    """管理 HC8 卷积层的权重量化缓存刷新

    设计参考 MSIntConvEncoder（阶段 2.1），但：
      - 存储用 HC8 bytes（不是 MSInt SlotBackend）
      - 前向走 HC8 matmul（不是解码到 float）
      - 缓存刷新 = invalidate_cache（下次前向重新量化）

    用法：
        encoder = HC8ConvEncoder(model, encode_interval=50)
        encoder.register_layer("conv1")
        # 训练循环中每 encode_interval 步：
        encoder.refresh_caches()  # 使所有注册层的缓存失效
    """

    def __init__(self, model: nn.Module, encode_interval: int = 50):
        self.model = model
        self.encode_interval = encode_interval
        self._layers: dict[str, HC8Conv2d | HC8Linear] = {}
        self._refresh_count = 0
        self._total_refresh_time = 0.0

    def register_layer(self, name: str) -> None:
        """注册需要管理缓存刷新的层

        Args:
            name: 层名（model 的属性名，如 "conv1", "fc1"）
        """
        if not hasattr(self.model, name):
            raise AttributeError(f"Model has no layer named '{name}'")
        layer = getattr(self.model, name)
        if not isinstance(layer, (HC8Conv2d, HC8Linear)):
            raise TypeError(
                f"Layer '{name}' is {type(layer).__name__}, "
                f"expected HC8Conv2d or HC8Linear"
            )
        self._layers[name] = layer

    def maybe_refresh(self, global_step: int) -> bool:
        """每 encode_interval 步刷新所有注册层的缓存

        Args:
            global_step: 当前全局步数

        Returns:
            是否执行了刷新
        """
        if (global_step + 1) % self.encode_interval != 0:
            return False
        return self.refresh_caches()

    def refresh_caches(self) -> bool:
        """使所有注册层的权重量化缓存失效"""
        import time
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
    """自检：HC8 卷积 vs float 卷积精度对比"""
    print("=" * 60)
    print("hc_conv2d.py self-check")
    print("=" * 60)

    if not _HAS_C_EXT:
        print("\n[SKIP] C 扩展不可用，跳过 HC8 路径测试")
        print("  （需要编译 pysgn_net，设置 SGN_NO_C_EXT=1 也会跳过）")
        return

    torch.manual_seed(42)

    # 1. Conv2dSTE vs float Conv2d
    print("\n[test 1] Conv2dSTE vs F.conv2d")
    x = torch.randn(4, 3, 16, 16)
    conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
    y_float = conv(x)
    y_hc8 = Conv2dSTE.apply(
        x, conv.weight, conv.bias, 1, 1, None, None
    )
    max_diff = (y_float - y_hc8).abs().max().item()
    mean_diff = (y_float - y_hc8).abs().mean().item()
    print(f"  x shape: {x.shape}")
    print(f"  y_float shape: {y_float.shape}")
    print(f"  y_hc8 shape: {y_hc8.shape}")
    print(f"  max_diff: {max_diff:.6f}")
    print(f"  mean_diff: {mean_diff:.6f}")
    assert max_diff < 0.5, f"Conv2dSTE 误差过大: {max_diff}"
    print("  ✓ 误差在量化精度范围内")

    # 2. LinearSTE vs float Linear
    print("\n[test 2] LinearSTE vs F.linear")
    x2 = torch.randn(8, 32)
    linear = nn.Linear(32, 16)
    y_float2 = linear(x2)
    y_hc8_2 = LinearSTE.apply(x2, linear.weight, linear.bias, None, None)
    max_diff2 = (y_float2 - y_hc8_2).abs().max().item()
    print(f"  max_diff: {max_diff2:.6f}")
    assert max_diff2 < 0.5, f"LinearSTE 误差过大: {max_diff2}"
    print("  ✓ 误差在量化精度范围内")

    # 3. HC8Conv2d 模块（含缓存）
    print("\n[test 3] HC8Conv2d 模块（含缓存）")
    hc_conv = HC8Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
    y1 = hc_conv(x)
    y2 = hc_conv(x)  # 第二次应使用缓存
    diff = (y1 - y2).abs().max().item()
    print(f"  两次前向差异（缓存）: {diff:.8f}")
    assert diff < 1e-6, "缓存应保证两次前向结果一致"
    print("  ✓ 缓存有效")

    # 4. 反向传播
    print("\n[test 4] 反向传播")
    hc_conv2 = HC8Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
    x3 = torch.randn(4, 3, 16, 16, requires_grad=True)
    y3 = hc_conv2(x3)
    loss = y3.sum()
    loss.backward()
    assert x3.grad is not None, "输入梯度未计算"
    assert hc_conv2.weight.grad is not None, "权重梯度未计算"
    assert hc_conv2.bias.grad is not None, "bias 梯度未计算"
    print(f"  grad_x shape: {x3.grad.shape}")
    print(f"  grad_w shape: {hc_conv2.weight.grad.shape}")
    print(f"  grad_b shape: {hc_conv2.bias.grad.shape}")
    print("  ✓ 反向传播正常")

    # 5. HC8ConvEncoder
    print("\n[test 5] HC8ConvEncoder")
    model = nn.Sequential()
    model.conv1 = HC8Conv2d(3, 8, 3, 1, 1)
    model.fc1 = HC8Linear(8 * 16 * 16, 10)
    encoder = HC8ConvEncoder(model, encode_interval=2)
    encoder.register_layer("conv1")
    encoder.register_layer("fc1")
    # 模拟 5 步训练
    for step in range(5):
        encoder.maybe_refresh(step)
    stats = encoder.get_stats()
    print(f"  refresh_count: {stats['refresh_count']}（期望 2，step 1 和 3）")
    assert stats["refresh_count"] == 2, f"刷新次数不对: {stats['refresh_count']}"
    print("  ✓ Encoder 刷新逻辑正确")

    print("\n" + "=" * 60)
    print("hc_conv2d.py self-check 全部通过 (5/5)")
    print("=" * 60)


if __name__ == "__main__":
    _self_check()
