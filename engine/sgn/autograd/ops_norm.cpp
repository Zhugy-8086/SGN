// ops_norm.cpp - 归一化族算子实现（模块化：归一化族独立成文件）
//
// LayerNorm（2D，沿末维归一化）：基础设施缺口分析批次 4。
// 行内归约保持确定性顺序（串行归约、行间 OpenMP）——跨线程确定性纪律。
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 zhugy-8086

#include "ops_norm.h"

#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sgn_autograd {

namespace {

inline void check_ln_inputs(const Tensor& X, const Tensor& gamma,
                            const Tensor& beta) {
    if (!X.is_contiguous() || !gamma.is_contiguous() || !beta.is_contiguous()) {
        throw std::invalid_argument("layernorm: inputs must be contiguous");
    }
    if (X.ndim() != 2) {
        throw std::invalid_argument("layernorm: X must be 2D (B, C)");
    }
    const int64_t C = X.shape()[1];
    if (gamma.shape().size() != 1 || gamma.shape()[0] != C ||
        beta.shape().size() != 1 || beta.shape()[0] != C) {
        throw std::invalid_argument("layernorm: gamma/beta must be 1D of size C");
    }
}

}  // namespace

Tensor layernorm_forward(const Tensor& X, const Tensor& gamma,
                         const Tensor& beta, float eps,
                         LayerNormContext& ctx) {
    check_ln_inputs(X, gamma, beta);
    const int64_t B = X.shape()[0];
    const int64_t C = X.shape()[1];

    Tensor Y(X.shape());
    Tensor mean({B});
    Tensor rstd({B});
    ctx.x_hat = Tensor(X.shape());
    ctx.rstd = Tensor({B});

    const float* x = X.data();
    const float* g = gamma.data();
    const float* bt = beta.data();
    float* y = Y.data();
    float* xh = ctx.x_hat.data();
    float* rs = ctx.rstd.data();
    float* mu = mean.data();

    #pragma omp parallel for schedule(guided)
    for (int64_t b = 0; b < B; ++b) {
        const float* xr = x + b * C;
        // 行内归约：确定性顺序（串行两遍——mean 与 var）
        float sum = 0.0f;
        for (int64_t c = 0; c < C; ++c) sum += xr[c];
        float mu_b = sum / static_cast<float>(C);
        float var = 0.0f;
        for (int64_t c = 0; c < C; ++c) {
            const float d = xr[c] - mu_b;
            var += d * d;
        }
        var /= static_cast<float>(C);
        const float r = 1.0f / std::sqrt(var + eps);
        mu[b] = mu_b;
        rs[b] = r;
        for (int64_t c = 0; c < C; ++c) {
            const float xh_c = (xr[c] - mu_b) * r;
            xh[b * C + c] = xh_c;
            y[b * C + c] = xh_c * g[c] + bt[c];
        }
    }
    return Y;
}

std::vector<Tensor> layernorm_backward(const Tensor& dY, const Tensor& gamma,
                                       const LayerNormContext& ctx) {
    check_ln_inputs(dY, gamma, gamma);
    if (dY.shape() != ctx.x_hat.shape()) {
        throw std::invalid_argument("layernorm_backward: dY shape mismatch");
    }
    const int64_t B = dY.shape()[0];
    const int64_t C = dY.shape()[1];

    Tensor dX(dY.shape());
    Tensor dgamma({C});
    Tensor dbeta({C});

    const float* dy = dY.data();
    const float* g = gamma.data();
    const float* xh = ctx.x_hat.data();
    const float* rs = ctx.rstd.data();
    float* dx = dX.data();
    float* dg = dgamma.data();
    float* db = dbeta.data();

    #pragma omp parallel for schedule(guided)
    for (int64_t b = 0; b < B; ++b) {
        const float* dyr = dy + b * C;
        const float* xhr = xh + b * C;
        float* dxr = dx + b * C;
        // dyγ = dY·gamma（逐元素）；行内归约串行（确定性）
        float s1 = 0.0f, s2 = 0.0f;
        for (int64_t c = 0; c < C; ++c) {
            const float dyg = dyr[c] * g[c];
            s1 += dyg;
            s2 += dyg * xhr[c];
        }
        const float m1 = s1 / static_cast<float>(C);
        const float m2 = s2 / static_cast<float>(C);
        const float r = rs[b];
        for (int64_t c = 0; c < C; ++c) {
            const float dyg = dyr[c] * g[c];
            dxr[c] = r * (dyg - m1 - xhr[c] * m2);
        }
    }
    // dgamma/dbeta：跨行求和（行间 OpenMP 累加需确定性——按行串行累加，
    // 与 D0"小任务 fork/join 反噬"结论一致，C 通常小）
    for (int64_t b = 0; b < B; ++b) {
        const float* dyr = dy + b * C;
        const float* xhr = xh + b * C;
        for (int64_t c = 0; c < C; ++c) {
            dg[c] += dyr[c] * xhr[c];
            db[c] += dyr[c];
        }
    }
    return {dX, dgamma, dbeta};
}

}  // namespace sgn_autograd
