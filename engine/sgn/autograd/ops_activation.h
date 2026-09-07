// ops_activation.h - 激活族算子接口（模块化：激活族独立成文件）
//
// 基础设施缺口分析（主框架基础设施缺口分析_2026_09_07.md）批次 2：
// 激活族从 ops_nn.{h,cpp} 独立成族（sigmoid/tanh 迁入 + gelu/silu 新增），
// 后续激活一律落位本文件，避免 NN 族文件膨胀。
//
// 每个激活：forward（值）+ backward（dX，捕获输出 Y 免重算激活值）。
// 契约与 ops_nn.h 一致：裸指针语义、contiguous 校验、OpenMP 并行、f32。
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 zhugy-8086
#pragma once

#include "tensor.h"

#include <cstddef>

namespace sgn_autograd {

// sigmoid: Y = 1/(1+exp(-X))；dX = dY·Y·(1−Y)
Tensor sigmoid_forward(const Tensor& X);
Tensor sigmoid_backward(const Tensor& Y, const Tensor& dY);

// tanh: Y = tanh(X)；dX = dY·(1−Y²)
Tensor tanh_forward(const Tensor& X);
Tensor tanh_backward(const Tensor& Y, const Tensor& dY);

// gelu（精确 erf 式）：Y = 0.5·X·(1+erf(X/√2))；
// dX = dY·(Φ(X) + X·φ(X))，Φ = 0.5(1+erf)，φ = exp(−X²/2)/√(2π)
Tensor gelu_forward(const Tensor& X);
Tensor gelu_backward(const Tensor& X, const Tensor& dY);

// silu（Swish）：Y = X·sigmoid(X)；dX = dY·sig(X)·(1 + X·(1−sig(X)))
Tensor silu_forward(const Tensor& X);
Tensor silu_backward(const Tensor& X, const Tensor& dY);

}  // namespace sgn_autograd
