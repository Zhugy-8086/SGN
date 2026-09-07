// ops_activation.cpp - 激活族算子实现（模块化：激活族独立成文件）
//
// 模块化说明（基础设施缺口分析批次 2）：sigmoid/tanh 自 ops_nn.cpp 迁入，
// gelu/silu 新增——激活族全部落位本文件，NN 族（linear/conv/pool/bn）留
// ops_nn.cpp。内核契约（contiguous 校验 + OpenMP guided）与 ops_nn 一致。
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 zhugy-8086

#include "ops_activation.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sgn_autograd {

namespace {

inline void check_contiguous_1arg(const Tensor& X, const char* name) {
    if (!X.is_contiguous()) {
        throw std::invalid_argument(std::string(name) + ": X must be contiguous");
    }
}

inline void check_contiguous_2arg(const Tensor& A, const Tensor& B, const char* name) {
    if (!A.is_contiguous() || !B.is_contiguous()) {
        throw std::invalid_argument(std::string(name) + ": inputs must be contiguous");
    }
    if (A.shape() != B.shape()) {
        throw std::invalid_argument(std::string(name) + ": shape mismatch");
    }
}

}  // namespace

// ---- sigmoid ----
Tensor sigmoid_forward(const Tensor& X) {
    check_contiguous_1arg(X, "sigmoid_forward");
    Tensor result(X.shape());
    const size_t n = X.numel();
    const float* x = X.data();
    float* y = result.data();
    #pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < n; ++i) {
        y[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
    return result;
}

Tensor sigmoid_backward(const Tensor& Y, const Tensor& dY) {
    check_contiguous_2arg(Y, dY, "sigmoid_backward");
    Tensor dX(Y.shape());
    const size_t n = Y.numel();
    const float* y = Y.data();
    const float* dy = dY.data();
    float* dx = dX.data();
    #pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < n; ++i) {
        dx[i] = dy[i] * y[i] * (1.0f - y[i]);
    }
    return dX;
}

// ---- tanh ----
Tensor tanh_forward(const Tensor& X) {
    check_contiguous_1arg(X, "tanh_forward");
    Tensor result(X.shape());
    const size_t n = X.numel();
    const float* x = X.data();
    float* y = result.data();
    #pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < n; ++i) {
        y[i] = std::tanh(x[i]);
    }
    return result;
}

Tensor tanh_backward(const Tensor& Y, const Tensor& dY) {
    check_contiguous_2arg(Y, dY, "tanh_backward");
    Tensor dX(Y.shape());
    const size_t n = Y.numel();
    const float* y = Y.data();
    const float* dy = dY.data();
    float* dx = dX.data();
    #pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < n; ++i) {
        dx[i] = dy[i] * (1.0f - y[i] * y[i]);
    }
    return dX;
}

// ---- gelu（精确 erf 式）----
Tensor gelu_forward(const Tensor& X) {
    check_contiguous_1arg(X, "gelu_forward");
    Tensor result(X.shape());
    const size_t n = X.numel();
    const float* x = X.data();
    float* y = result.data();
    const float k = 0.70710678118654752440f;   // 1/√2
    #pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < n; ++i) {
        y[i] = 0.5f * x[i] * (1.0f + std::erf(x[i] * k));
    }
    return result;
}

Tensor gelu_backward(const Tensor& X, const Tensor& dY) {
    check_contiguous_2arg(X, dY, "gelu_backward");
    Tensor dX(X.shape());
    const size_t n = X.numel();
    const float* x = X.data();
    const float* dy = dY.data();
    float* dx = dX.data();
    const float k = 0.70710678118654752440f;
    const float c = 0.39894228040143267794f;   // 1/√(2π)
    #pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < n; ++i) {
        const float xi = x[i];
        const float phi = 0.5f * (1.0f + std::erf(xi * k));   // Φ(x)
        const float pdf = c * std::exp(-0.5f * xi * xi);      // φ(x)
        dx[i] = dy[i] * (phi + xi * pdf);
    }
    return dX;
}

// ---- silu（Swish）----
Tensor silu_forward(const Tensor& X) {
    check_contiguous_1arg(X, "silu_forward");
    Tensor result(X.shape());
    const size_t n = X.numel();
    const float* x = X.data();
    float* y = result.data();
    #pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < n; ++i) {
        y[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
    return result;
}

Tensor silu_backward(const Tensor& X, const Tensor& dY) {
    check_contiguous_2arg(X, dY, "silu_backward");
    Tensor dX(X.shape());
    const size_t n = X.numel();
    const float* x = X.data();
    const float* dy = dY.data();
    float* dx = dX.data();
    #pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < n; ++i) {
        const float sig = 1.0f / (1.0f + std::exp(-x[i]));
        dx[i] = dy[i] * sig * (1.0f + x[i] * (1.0f - sig));
    }
    return dX;
}

}  // namespace sgn_autograd
