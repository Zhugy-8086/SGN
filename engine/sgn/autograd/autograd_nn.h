// autograd_nn.h - autograd-aware 神经网络算子（Phase 5）
//
// Stage 3.2 Phase 5：为 linear/relu/conv2d/maxpool/bn 添加 autograd-aware 版本。
//
// 设计（与 matmul 一致的模式）：
//   1. 始终执行前向计算
//   2. 若 Tape 正在记录 且 任一输入 requires_grad，则 record 到 tape
//   3. backward_fn 捕获前向所需的输入（浅拷贝，共享 storage）
//   4. 输出 requires_grad = 任一输入 requires_grad
//
// Context（BNContext/Conv2DContext/MaxPoolContext）的处理：
//   - 这些 context 包含 backward 需要的中间结果（如 BN 的 mean/rstd/x_normalized）
//   - autograd-aware 版本在 stack 上创建临时 context，捕获到 backward_fn lambda
//   - 调用方无需手动管理 context
//
// 反向传播策略选择（Stage 3.x）：
//   - 通过 StrategyContext::set(BackwardStrategy) 设置全局策略
//   - 默认 FLOAT32（纯 float32，与 Phase 5 行为一致）
//   - STE 模式：前向对激活做量化+反量化，反向 float32 直通
//   - GEF 模式：前向 float32，反向梯度做 Q16 网格确定性舍入量化
//   - SR  模式：前向 float32，反向梯度做 Q16 网格伯努利随机舍入量化
//   - 其他策略（HC16/EF_SGD/MSINT/LEVEL_AMP）为桩，待实现
//     （HC16 为历史命名，见 backward_strategy.h 顶部术语说明）

#pragma once

#include "tensor.h"
#include "ops_nn.h"
#include "backward_strategy.h"
#include "autograd.h"

namespace sgn_autograd {

// ============================================================================
// autograd-aware 算子（策略感知版本）
// ============================================================================
// 这些算子根据当前的 StrategyContext::get() 选择 forward/backward 实现：
//   FLOAT32: 使用标准 float32 算子（与 Phase 5 相同）
//   STE:     使用 STE 前向（量化+反量化），backward 与 FLOAT32 相同

// linear: Y = X @ W^T + b
Tensor linear(const Tensor& X, const Tensor& W, const Tensor& b);

// relu: Y = max(0, X)
Tensor relu(const Tensor& X);
Tensor sigmoid(const Tensor& X);
Tensor tanh(const Tensor& X);
Tensor gelu(const Tensor& X);
Tensor silu(const Tensor& X);
Tensor layernorm(const Tensor& X, const Tensor& gamma, const Tensor& beta, float eps);

// add: Y = A + B（ResNet 残差连接；backward dA=dY, dB=dY，与策略无关）
Tensor add(const Tensor& A, const Tensor& B);
Tensor mul(const Tensor& A, const Tensor& B);

// bn (训练模式): 归一化 + 仿射
//   dim: 0=沿 batch 维归一化（BatchNorm1d），1=沿 spatial 维（BatchNorm2d reshape 后）
//   running_mean/var 会被原地修改（与 PyTorch 一致）
Tensor bn_train(
    const Tensor& X, const Tensor& gamma, const Tensor& beta,
    Tensor& running_mean, Tensor& running_var,
    float momentum, float eps, int dim
);

// batchnorm2d: 4D 输入 (B, C, H, W) 的 BN，内部 reshape 到 (C, B*H*W) 调 bn_train(dim=1)
Tensor batchnorm2d(
    const Tensor& X, const Tensor& gamma, const Tensor& beta,
    Tensor& running_mean, Tensor& running_var,
    float momentum, float eps
);

// conv2d: Y = im2col(X) @ W_col + b
Tensor conv2d(
    const Tensor& X, const Tensor& W, const Tensor& b,
    int stride, int padding
);

// maxpool2d: 沿 (kernel, kernel) 窗口取最大值
Tensor maxpool2d(const Tensor& X, int kernel, int stride);

// avgpool2d: 沿 (kernel, kernel) 窗口取均值（GAP，ResNet 分类头；backward dX=dY/k²）
Tensor avgpool2d(const Tensor& X, int kernel, int stride);

// ============================================================================
// 算子融合：conv2d + relu（autograd-aware）
// ============================================================================
// 将 conv2d forward + relu forward 合并为一个 tape 记录。
// backward 时调用 conv2d_relu_backward，在 col2im 前对 dY 应用 relu mask，
// 省去一次独立的 relu_backward 调用。

// conv2d_relu: Y = relu(conv2d(X, W, b))
Tensor conv2d_relu(
    const Tensor& X, const Tensor& W, const Tensor& b,
    int stride, int padding
);

// ============================================================================
// 策略设置辅助函数
// ============================================================================

// 设置全局反向传播策略
inline void set_backward_strategy(BackwardStrategy s) {
    StrategyContext::set(s);
}

// 获取当前策略
inline BackwardStrategy get_backward_strategy() {
    return StrategyContext::get();
}

// 设置前向 STE 量化配置（仅影响 linear/conv2d 的 *_forward_ste 前向量化）
inline void set_ste_quant_config(int bits, float clip_sigma = 4.0f) {
    StrategyContext::quant_config().bits = bits;
    StrategyContext::quant_config().clip_sigma = clip_sigma;
}

// 设置反向梯度量化配置（仅影响 GEF/SR/A1 的 Tape::backward 梯度量化，
// 独立于前向 STE 配置；链路核查 2026-08-21 P0-2 修复：此前与 STE 共用同一配置，
// 前向切位宽会同步覆写反向位宽）
inline void set_quant_config(int bits, float clip_sigma = 4.0f) {
    StrategyContext::bwd_quant_config().bits = bits;
    StrategyContext::bwd_quant_config().clip_sigma = clip_sigma;
}

// set_sr_seed 定义在 backward_strategy.h 中（已被 autograd_nn.h 包含）

}  // namespace sgn_autograd
