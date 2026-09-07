// ops_norm.h - 归一化族算子接口（模块化：归一化族独立成文件）
//
// 基础设施缺口分析（主框架基础设施缺口分析_2026_09_07.md）批次 4：
// LayerNorm（前置 Transformer/KV-cache 方向，路线档扩展目标 2）。
//
// 范围（本批最小形态）：2D 输入 (B, C)，沿末维 C 归一化（逐样本）。
// 与 BatchNorm（沿通道跨 batch 统计）语义不同——二者互补非替代。
//
// 契约与 ops_nn.h 一致：contiguous 校验、f32、OpenMP 并行（沿行并行，
// 行内归约确定性顺序——维持跨线程确定性纪律，不并行行内归约）。
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 zhugy-8086
#pragma once

#include "tensor.h"

#include <cstddef>

namespace sgn_autograd {

struct LayerNormContext {
    Tensor x_hat;    // (B, C) 归一化中间值（backward 的 dγ/dX 需要）
    Tensor rstd;     // (B,)   1/sqrt(var + eps)
};

// 前向：Y = (X - mean_c(X)) / sqrt(var_c(X) + eps) * gamma + beta
//   X (B,C) contiguous；gamma/beta (C,)；eps 常见 1e-5
Tensor layernorm_forward(const Tensor& X, const Tensor& gamma,
                         const Tensor& beta, float eps,
                         LayerNormContext& ctx);

// 反向：返回 {dX (B,C), dgamma (C), dbeta (C)}
//   dX_c = rstd_b·(dyγ_c − mean_c(dyγ) − x̂_c·mean_c(dyγ·x̂))，dyγ = dY·γ
std::vector<Tensor> layernorm_backward(const Tensor& dY, const Tensor& gamma,
                                       const LayerNormContext& ctx);

}  // namespace sgn_autograd
