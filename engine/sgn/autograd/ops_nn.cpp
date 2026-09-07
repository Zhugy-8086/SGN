// ops_nn.cpp - 神经网络算子实现（Phase 3）
//
// Stage 3.2 Phase 3：适配 6 层 CNN+BN 的算子实现。
//
// 实现说明：
//   - linear/relu/bn/maxpool：朴素 C++ 实现，float32
//   - conv2d：im2col + matmul，朴素实现（col2im backward 复用 sgn.col2im_add）
//   - 所有算子要求输入 contiguous，否则抛异常
//   - BN 统一按 2D 处理，BatchNorm2d 由调用方 reshape

#include "ops_nn.h"
#include "ops.h"
#include "dispatch/registry.h"
#include "mkern/simd/simd_api.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace sgn_autograd {

// 融合算子阈值：spatial < 此值时回退到分离算子
constexpr int64_t FUSION_SPATIAL_THRESHOLD = 64;

// ============================================================================
// BufferPool 实现
//
// 注意：P1 已将 Storage 接入全局可插拔分配器（allocator.h），启用 PoolAllocator
// 时 Storage 的释放自动归还 free-list。BufferPool 的 Tensor 对象缓存与此重叠——
// 因为 conv2d 临时矩阵（dY_col/dx_col/dY_conv）的 Tensor 构造开销远小于 matmul
// 计算量，双缓存无额外收益。故 BufferPool 简化为空壳，acquire 创建新 Tensor
// （Storage 经 PoolAllocator 复用内存），release 置空参数使 Storage 自动归还池。
// 保持 API 签名不变以最小化调用方改动。
// ============================================================================
Tensor BufferPool::acquire(const std::vector<int64_t>& shape) {
    return Tensor(shape);
}

void BufferPool::release(Tensor&& t) {
    // Tensor 析构时 Storage 的 shared_ptr 释放经 sgn_deallocate_floats 归还
    // PoolAllocator（若已启用）。置空参数使其 Storage 立即释放。
    t = Tensor();
}

void BufferPool::clear() {
    // 无缓存状态，clear 为空
}

// ============================================================================
// linear forward: Y = X @ W^T + b
// ============================================================================
//   X: (m, in), W: (out, in), b: (out,)
//   Y[i,j] = sum_l X[i,l] * W[j,l] + b[j]
Tensor linear_forward(const Tensor& X, const Tensor& W, const Tensor& b) {
    if (X.ndim() != 2 || W.ndim() != 2) {
        throw std::invalid_argument("linear_forward: X and W must be 2D");
    }
    if (!X.is_contiguous() || !W.is_contiguous() || !b.is_contiguous()) {
        throw std::invalid_argument("linear_forward: inputs must be contiguous");
    }

    int64_t m = X.shape()[0];
    int64_t out_features = W.shape()[0];
    int64_t in_w = W.shape()[1];
    int64_t in_features = X.shape()[1];

    if (in_features != in_w) {
        throw std::invalid_argument("linear_forward: X.columns must match W.columns");
    }
    if (static_cast<int64_t>(b.numel()) != out_features) {
        throw std::invalid_argument("linear_forward: b.size must match W.rows");
    }

    // Y = X @ W^T + b: 复用 AVX2 向量化的 matmul_forward
    // W^T: (in_features, out_features)
    Tensor Y = matmul_forward(X, transpose_2d(W));
    const float* b_ptr = b.data();
    float* y_ptr = Y.data();
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < out_features; ++j) {
            y_ptr[i * out_features + j] += b_ptr[j];
        }
    }
    return Y;
}

// ============================================================================
// linear backward:
//   dX = dY @ W       (m×out @ out×in = m×in)  → dX[i,l] = sum_j dY[i,j] * W[j,l]
//   dW = dY^T @ X     (out×m @ m×in = out×in)  → dW[j,l] = sum_i dY[i,j] * X[i,l]
//   db = sum_rows(dY) (out,)
// ============================================================================
std::tuple<Tensor, Tensor, Tensor> linear_backward(
    const Tensor& X, const Tensor& W, const Tensor& /*b*/, const Tensor& dY
) {
    if (X.ndim() != 2 || W.ndim() != 2 || dY.ndim() != 2) {
        throw std::invalid_argument("linear_backward: inputs must be 2D");
    }
    if (!X.is_contiguous() || !W.is_contiguous() || !dY.is_contiguous()) {
        throw std::invalid_argument("linear_backward: inputs must be contiguous");
    }

    int64_t m = X.shape()[0];
    int64_t out_features = W.shape()[0];

    if (dY.shape()[0] != m || dY.shape()[1] != out_features) {
        throw std::invalid_argument("linear_backward: dY shape mismatch");
    }

    const float* dy_ptr = dY.data();

    // --- dX = dY @ W: 复用 AVX2 matmul_forward ---
    Tensor dX = matmul_forward(dY, W);

    // --- dW = dY^T @ X: 复用 AVX2 matmul_forward ---
    Tensor dW = matmul_forward(transpose_2d(dY), X);

    // --- db = sum_rows(dY) ---
    // 沿 batch 维求和，j 循环并行化（非连续访问，不适合 SIMD）
    Tensor db({out_features});
    float* db_ptr = db.data();
    #pragma omp parallel for schedule(guided)
    for (int64_t j = 0; j < out_features; ++j) {
        float sum = 0.0f;
        for (int64_t i = 0; i < m; ++i) {
            sum += dy_ptr[i * out_features + j];
        }
        db_ptr[j] = sum;
    }

    return {dX, dW, db};
}

// ============================================================================
// relu forward: Y = max(0, X)
// ============================================================================
Tensor relu_forward(const Tensor& X) {
    if (!X.is_contiguous()) {
        throw std::invalid_argument("relu_forward: X must be contiguous");
    }
    // 构造独立 Tensor（不共享 storage，因为要写自己的值）
    Tensor result(X.shape());
    size_t n = X.numel();
    const float* x_ptr = X.data();
    float* y_ptr = result.data();
    #pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < n; ++i) {
        y_ptr[i] = x_ptr[i] > 0.0f ? x_ptr[i] : 0.0f;
    }
    return result;
}

// ============================================================================
// relu backward: dX = dY * (X > 0 ? 1 : 0)
// ============================================================================
Tensor relu_backward(const Tensor& X, const Tensor& dY) {
    if (X.shape() != dY.shape()) {
        throw std::invalid_argument("relu_backward: X and dY shape mismatch");
    }
    if (!X.is_contiguous() || !dY.is_contiguous()) {
        throw std::invalid_argument("relu_backward: inputs must be contiguous");
    }
    Tensor dX(X.shape());
    size_t n = X.numel();
    const float* x_ptr = X.data();
    const float* dy_ptr = dY.data();
    float* dx_ptr = dX.data();
    #pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < n; ++i) {
        dx_ptr[i] = x_ptr[i] > 0.0f ? dy_ptr[i] : 0.0f;
    }
    return dX;
}

// ============================================================================
// add forward: Y = A + B  （逐元素；ResNet 残差连接，A 与 B 同形状，或 B 为标量）
// ============================================================================
Tensor add_forward(const Tensor& A, const Tensor& B) {
    if (!A.is_contiguous() || !B.is_contiguous()) {
        throw std::invalid_argument("add_forward: inputs must be contiguous");
    }
    const bool b_scalar = (B.numel() == 1);
    if (!b_scalar && A.shape() != B.shape()) {
        throw std::invalid_argument("add_forward: A and B shape mismatch (需同形状或 B 为标量)");
    }
    Tensor result(A.shape());
    size_t n = A.numel();
    const float* a_ptr = A.data();
    const float* b_ptr = B.data();
    float* y_ptr = result.data();
    if (b_scalar) {
        const float b0 = b_ptr[0];
        #pragma omp parallel for schedule(guided)
        for (size_t i = 0; i < n; ++i) {
            y_ptr[i] = a_ptr[i] + b0;
        }
    } else {
        #pragma omp parallel for schedule(guided)
        for (size_t i = 0; i < n; ++i) {
            y_ptr[i] = a_ptr[i] + b_ptr[i];
        }
    }
    return result;
}

Tensor mul_forward(const Tensor& A, const Tensor& B) {
    if (!A.is_contiguous() || !B.is_contiguous()) {
        throw std::invalid_argument("mul_forward: inputs must be contiguous");
    }
    if (A.shape() != B.shape()) {
        throw std::invalid_argument("mul_forward: A and B shape mismatch");
    }
    Tensor result(A.shape());
    size_t n = A.numel();
    const float* a = A.data();
    const float* b = B.data();
    float* y = result.data();
    #pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < n; ++i) {
        y[i] = a[i] * b[i];
    }
    return result;
}

// ============================================================================
// batch_norm forward (训练模式)
// ============================================================================
// 实现 2D 通用版本，dim 指定归一化维度：
//   dim=0: 沿行归一化每列（BatchNorm1d，X=(batch, features)）
//   dim=1: 沿列归一化每行（BatchNorm2d reshape 后，X=(N*H*W, C)）
//
// 数学：
//   mean = X.mean(dim)  // 沿 dim 求均值
//   var = X.var(dim)    // 沿 dim 求方差
//   x_norm = (X - mean) / sqrt(var + eps)
//   Y = gamma * x_norm + beta
//   running_mean = (1-momentum)*running_mean + momentum*mean
//   running_var = (1-momentum)*running_var + momentum*var
Tensor bn_forward_train(
    const Tensor& X, const Tensor& gamma, const Tensor& beta,
    Tensor& running_mean, Tensor& running_var,
    float momentum, float eps, int dim,
    BNContext& ctx
) {
    if (X.ndim() != 2) {
        throw std::invalid_argument("bn_forward_train: X must be 2D");
    }
    if (!X.is_contiguous()) {
        throw std::invalid_argument("bn_forward_train: X must be contiguous");
    }

    int64_t m = X.shape()[0];
    int64_t n = X.shape()[1];

    // 根据 dim 确定归一化方向和特征维度
    // dim=0: 归一化每列（沿 batch 维），特征数 = n，N = m
    // dim=1: 归一化每行（沿 spatial 维），特征数 = m，N = n
    int64_t feat_dim = (dim == 0) ? n : m;
    int64_t N = (dim == 0) ? m : n;

    // 空维度防护：N=0 时 mean/var 的 sum/N 与 1/sqrt(var+eps) 产生 inf/NaN
    // 污染整个计算图（安全审计 2026-08-16 A1-1）
    if (N <= 0) {
        throw std::invalid_argument(
            "bn_forward_train: normalization dimension is empty (N=0)");
    }

    if (static_cast<int64_t>(gamma.numel()) != feat_dim || static_cast<int64_t>(beta.numel()) != feat_dim) {
        throw std::invalid_argument("bn_forward_train: gamma/beta size mismatch");
    }
    if (static_cast<int64_t>(running_mean.numel()) != feat_dim || static_cast<int64_t>(running_var.numel()) != feat_dim) {
        throw std::invalid_argument("bn_forward_train: running_mean/var size mismatch");
    }

    const float* x_ptr = X.data();
    const float* gamma_ptr = gamma.data();
    const float* beta_ptr = beta.data();
    float* rm_ptr = running_mean.data();
    float* rv_ptr = running_var.data();

    // 计算均值
    ctx.mean = Tensor({feat_dim});
    float* mean_ptr = ctx.mean.data();
    // 归约内核迁出到 sgn::simd::sum_f32（AVX2 8 路 + 水平归约 + 标量尾，见 simd 原语层）
    #pragma omp parallel for schedule(guided)
    for (int64_t f = 0; f < feat_dim; ++f) {
        const float* col = (dim == 0) ? (x_ptr + f) : (x_ptr + f * n);
        const int64_t stride = (dim == 0) ? n : 1;
        mean_ptr[f] = sgn::simd::sum_f32(col, N, stride) / static_cast<float>(N);
    }

    // 计算方差 + rstd
    ctx.rstd = Tensor({feat_dim});
    float* rstd_ptr = ctx.rstd.data();
    // 归约内核迁出到 sgn::simd::sum_sq_dev_f32（AVX2 8 路 + 水平归约 + 标量尾，见 simd 原语层）
    #pragma omp parallel for schedule(guided)
    for (int64_t f = 0; f < feat_dim; ++f) {
        float mu = mean_ptr[f];
        const float* col = (dim == 0) ? (x_ptr + f) : (x_ptr + f * n);
        const int64_t stride = (dim == 0) ? n : 1;
        float sum_sq = sgn::simd::sum_sq_dev_f32(col, N, stride, mu);
        float var = sum_sq / static_cast<float>(N);
        rstd_ptr[f] = 1.0f / std::sqrt(var + eps);
        rm_ptr[f] = (1.0f - momentum) * rm_ptr[f] + momentum * mu;
        rv_ptr[f] = (1.0f - momentum) * rv_ptr[f] + momentum * var;
    }

    // 归一化 + 仿射
    ctx.x_normalized = Tensor({m, n});
    Tensor Y({m, n});
    float* xn_ptr = ctx.x_normalized.data();
    float* y_ptr = Y.data();
    #pragma omp parallel for schedule(guided)
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            int64_t f = (dim == 0) ? j : i;
            float xn = (x_ptr[i * n + j] - mean_ptr[f]) * rstd_ptr[f];
            xn_ptr[i * n + j] = xn;
            y_ptr[i * n + j] = gamma_ptr[f] * xn + beta_ptr[f];
        }
    }
    return Y;
}

// ============================================================================
// bn backward (训练模式)
// ============================================================================
//   dgamma = sum(dY * x_normalized)  沿归一化维度
//   dbeta = sum(dY)                  沿归一化维度
//   dX = (1/N) * rstd * (N*dY - sum(dY) - x_norm * sum(dY*x_norm))
std::tuple<Tensor, Tensor, Tensor> bn_backward_train(
    const BNContext& ctx, const Tensor& gamma, const Tensor& dY, int dim
) {
    if (dY.shape() != ctx.x_normalized.shape()) {
        throw std::invalid_argument("bn_backward_train: dY shape mismatch");
    }
    int64_t m = dY.shape()[0];
    int64_t n = dY.shape()[1];
    int64_t feat_dim = (dim == 0) ? n : m;
    int64_t N = (dim == 0) ? m : n;

    // 空维度防护：N=0 时 dX 公式的 1/N 产生 inf/NaN（审计 A1-1）
    if (N <= 0) {
        throw std::invalid_argument(
            "bn_backward_train: normalization dimension is empty (N=0)");
    }

    const float* dy_ptr = dY.data();
    const float* xn_ptr = ctx.x_normalized.data();
    const float* rstd_ptr = ctx.rstd.data();
    const float* gamma_ptr = gamma.data();

    // 单遍扫描：计算 sum_dy 和 sum_dy_xn 数组
    // 同时用于 dgamma、dbeta 和 dX，避免重复遍历 dY 和 x_normalized
    // 归约内核迁出到 sgn::simd::sum_sumprod_f32（单遍双累加，AVX2 + 标量回退，见 simd 原语层）
    std::vector<float> sum_dy_arr(feat_dim), sum_dy_xn_arr(feat_dim);
    #pragma omp parallel for schedule(guided)
    for (int64_t f = 0; f < feat_dim; ++f) {
        const float* dy_col = (dim == 0) ? (dy_ptr + f) : (dy_ptr + f * n);
        const float* xn_col = (dim == 0) ? (xn_ptr + f) : (xn_ptr + f * n);
        const int64_t stride = (dim == 0) ? n : 1;
        sgn::simd::sum_sumprod_f32(dy_col, xn_col, N, stride,
                              &sum_dy_arr[f], &sum_dy_xn_arr[f]);
    }

    // 从数组计算 dgamma 和 dbeta
    // dgamma = sum(dY * x_normalized)（注意：不含 gamma 因子。Y=gamma·Xhat+beta，
    // ∂Y/∂gamma=Xhat=x_normalized，故 dgamma = Σ(dY·x_normalized)。
    // 修复 2026-08-22：此前误乘 gamma_ptr，污染 gamma 绝对学习轨迹）
    Tensor dgamma({feat_dim});
    Tensor dbeta({feat_dim});
    float* dg_ptr = dgamma.data();
    float* db_ptr = dbeta.data();
    for (int64_t f = 0; f < feat_dim; ++f) {
        db_ptr[f] = sum_dy_arr[f];
        dg_ptr[f] = sum_dy_xn_arr[f];
    }

    // 计算 dX
    // dX[i,j] = (1/N) * rstd[f] * gamma[f] * (N*dy - sum_dy - xn * sum_dy_xn)
    //        = (rstd[f] * gamma[f] / N) * (N*dy - sum_dy - xn * sum_dy_xn)
    Tensor dX({m, n});
    float* dx_ptr = dX.data();
    #pragma omp parallel for schedule(guided)
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            int64_t f = (dim == 0) ? j : i;
            float scale = rstd_ptr[f] * gamma_ptr[f] / N;
            dx_ptr[i * n + j] = scale * (N * dy_ptr[i * n + j]
                                         - sum_dy_arr[f]
                                         - xn_ptr[i * n + j] * sum_dy_xn_arr[f]);
        }
    }

    return {dX, dgamma, dbeta};
}

// ============================================================================
// bn_forward_train_4d / bn_backward_train_4d：batchnorm2d 直接按 (B,C,H,W) 布局计算
// ============================================================================
// 目的：消除旧 batchnorm2d 的 permute+contiguous 深拷贝（每 fwd 2 次、每 bwd 2 次）。
// 数值：与旧路径「reshape 到 (C,B*H*W) → bn_forward_train(dim=1) → reshape 回」逐位一致——
//       每通道的 mean/var 归约按 (b,h,w) 全局顺序、同一 sum256 累加器、同一水平归约，
//       浮点求和次序完全相同（H*W 为 8 的倍数时 8 宽 chunk 恒落在同一 batch 内、连续读取）。
// 阶段 4（2026-08-31）：本文件 AVX2/FMA intrinsic 依赖文件级 ISA 选项
// （CMake set_source_files_properties），与现状全局 -mavx2 编译等价；
// 带 #pragma omp 的内核无法用 target() 属性（lld 崩溃，见调查 §9）。
static inline float bn_hadd256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

Tensor bn_forward_train_4d(
    const Tensor& X, const Tensor& gamma, const Tensor& beta,
    Tensor& running_mean, Tensor& running_var,
    float momentum, float eps, BNContext& ctx
) {
    if (X.ndim() != 4) {
        throw std::invalid_argument("bn_forward_train_4d: X must be 4D");
    }
    if (!X.is_contiguous()) {
        throw std::invalid_argument("bn_forward_train_4d: X must be contiguous");
    }
    int64_t B = X.shape()[0], C = X.shape()[1], H = X.shape()[2], W = X.shape()[3];
    int64_t HW = H * W;
    int64_t N = B * HW;  // 每通道归约元素数 = batch * 空间
    if (N <= 0) {
        throw std::invalid_argument("bn_forward_train_4d: normalization dimension is empty (N=0)");
    }
    if (static_cast<int64_t>(gamma.numel()) != C ||
        static_cast<int64_t>(beta.numel()) != C ||
        static_cast<int64_t>(running_mean.numel()) != C ||
        static_cast<int64_t>(running_var.numel()) != C) {
        throw std::invalid_argument("bn_forward_train_4d: gamma/beta/rm/rv size mismatch");
    }

    const float* x_ptr = X.data();
    const float* gamma_ptr = gamma.data();
    const float* beta_ptr = beta.data();
    float* rm_ptr = running_mean.data();
    float* rv_ptr = running_var.data();
    const int64_t CHW = C * HW;
    const int64_t n8 = N - (N % 8);

    ctx.mean = Tensor({C});
    ctx.rstd = Tensor({C});
    float* mean_ptr = ctx.mean.data();
    float* rstd_ptr = ctx.rstd.data();

    // ---- pass 1: mean ----
    #pragma omp parallel for schedule(guided)
    for (int64_t c = 0; c < C; ++c) {
        __m256 sum256 = _mm256_setzero_ps();
        for (int64_t g = 0; g < n8; g += 8) {
            int64_t b = g / HW;
            int64_t hw = g - b * HW;
            __m256 v;
            if (hw + 8 <= HW) {
                v = _mm256_loadu_ps(x_ptr + b * CHW + c * HW + hw);
            } else {
                float tmp[8];
                for (int t = 0; t < 8; ++t) {
                    int64_t gg = g + t, bb = gg / HW, hh = gg - bb * HW;
                    tmp[t] = x_ptr[bb * CHW + c * HW + hh];
                }
                v = _mm256_loadu_ps(tmp);
            }
            sum256 = _mm256_add_ps(sum256, v);
        }
        float sum = bn_hadd256(sum256);
        for (int64_t g = n8; g < N; ++g) {
            int64_t b = g / HW, hw = g - b * HW;
            sum += x_ptr[b * CHW + c * HW + hw];
        }
        mean_ptr[c] = sum / N;
    }

    // ---- pass 2: var + rstd + running stats ----
    #pragma omp parallel for schedule(guided)
    for (int64_t c = 0; c < C; ++c) {
        float mu = mean_ptr[c];
        __m256 mu256 = _mm256_set1_ps(mu);
        __m256 sum_sq256 = _mm256_setzero_ps();
        for (int64_t g = 0; g < n8; g += 8) {
            int64_t b = g / HW;
            int64_t hw = g - b * HW;
            __m256 x;
            if (hw + 8 <= HW) {
                x = _mm256_loadu_ps(x_ptr + b * CHW + c * HW + hw);
            } else {
                float tmp[8];
                for (int t = 0; t < 8; ++t) {
                    int64_t gg = g + t, bb = gg / HW, hh = gg - bb * HW;
                    tmp[t] = x_ptr[bb * CHW + c * HW + hh];
                }
                x = _mm256_loadu_ps(tmp);
            }
            __m256 d = _mm256_sub_ps(x, mu256);
            sum_sq256 = _mm256_add_ps(sum_sq256, _mm256_mul_ps(d, d));
        }
        float sum_sq = bn_hadd256(sum_sq256);
        for (int64_t g = n8; g < N; ++g) {
            int64_t b = g / HW, hw = g - b * HW;
            float d = x_ptr[b * CHW + c * HW + hw] - mu;
            sum_sq += d * d;
        }
        float var = sum_sq / N;
        rstd_ptr[c] = 1.0f / std::sqrt(var + eps);
        rm_ptr[c] = (1.0f - momentum) * rm_ptr[c] + momentum * mu;
        rv_ptr[c] = (1.0f - momentum) * rv_ptr[c] + momentum * var;
    }

    // ---- 归一化 + 仿射（逐元素独立，顺序不影响数值）----
    ctx.x_normalized = Tensor({B, C, H, W});
    Tensor Y({B, C, H, W});
    float* xn_ptr = ctx.x_normalized.data();
    float* y_ptr = Y.data();
    #pragma omp parallel for schedule(guided)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t c = 0; c < C; ++c) {
            float mu = mean_ptr[c], rstd = rstd_ptr[c];
            float g = gamma_ptr[c], bt = beta_ptr[c];
            const float* base = x_ptr + b * CHW + c * HW;
            float* xnb = xn_ptr + b * CHW + c * HW;
            float* yb = y_ptr + b * CHW + c * HW;
            for (int64_t j = 0; j < HW; ++j) {
                float xn = (base[j] - mu) * rstd;
                xnb[j] = xn;
                yb[j] = g * xn + bt;
            }
        }
    }
    return Y;
}

// bn_backward_train_4d：与 bn_forward_train_4d 配对（dim=1 语义：每通道沿 B*H*W 归约）
std::tuple<Tensor, Tensor, Tensor> bn_backward_train_4d(
    const BNContext& ctx, const Tensor& gamma, const Tensor& dY
) {
    if (dY.ndim() != 4) {
        throw std::invalid_argument("bn_backward_train_4d: dY must be 4D");
    }
    if (!dY.is_contiguous()) {
        throw std::invalid_argument("bn_backward_train_4d: dY must be contiguous");
    }
    int64_t B = dY.shape()[0], C = dY.shape()[1], H = dY.shape()[2], W = dY.shape()[3];
    int64_t HW = H * W;
    int64_t N = B * HW;
    if (N <= 0) {
        throw std::invalid_argument("bn_backward_train_4d: normalization dimension is empty (N=0)");
    }
    const float* dy_ptr = dY.data();
    const float* xn_ptr = ctx.x_normalized.data();
    const float* rstd_ptr = ctx.rstd.data();
    const float* gamma_ptr = gamma.data();
    const int64_t CHW = C * HW;
    const int64_t n8 = N - (N % 8);

    // sum_dy / sum_dy_xn（归约次序与旧 2D 路径逐位一致）
    std::vector<float> sum_dy_arr(C), sum_dy_xn_arr(C);
    #pragma omp parallel for schedule(guided)
    for (int64_t c = 0; c < C; ++c) {
        __m256 sd256 = _mm256_setzero_ps();
        __m256 sdx256 = _mm256_setzero_ps();
        for (int64_t g = 0; g < n8; g += 8) {
            int64_t b = g / HW;
            int64_t hw = g - b * HW;
            __m256 dy, xn;
            if (hw + 8 <= HW) {
                dy = _mm256_loadu_ps(dy_ptr + b * CHW + c * HW + hw);
                xn = _mm256_loadu_ps(xn_ptr + b * CHW + c * HW + hw);
            } else {
                float dtmp[8], xtmp[8];
                for (int t = 0; t < 8; ++t) {
                    int64_t gg = g + t, bb = gg / HW, hh = gg - bb * HW;
                    int64_t off = bb * CHW + c * HW + hh;
                    dtmp[t] = dy_ptr[off];
                    xtmp[t] = xn_ptr[off];
                }
                dy = _mm256_loadu_ps(dtmp);
                xn = _mm256_loadu_ps(xtmp);
            }
            sd256 = _mm256_add_ps(sd256, dy);
            sdx256 = _mm256_add_ps(sdx256, _mm256_mul_ps(dy, xn));
        }
        float sd = bn_hadd256(sd256);
        float sdx = bn_hadd256(sdx256);
        for (int64_t g = n8; g < N; ++g) {
            int64_t b = g / HW, hw = g - b * HW;
            int64_t off = b * CHW + c * HW + hw;
            sd += dy_ptr[off];
            sdx += dy_ptr[off] * xn_ptr[off];
        }
        sum_dy_arr[c] = sd;
        sum_dy_xn_arr[c] = sdx;
    }

    Tensor dgamma({C}), dbeta({C});
    float* dg_ptr = dgamma.data();
    float* db_ptr = dbeta.data();
    // dgamma = sum(dY * x_normalized)（不含 gamma 因子；修复 2026-08-22，见 2D 版注释）
    for (int64_t c = 0; c < C; ++c) {
        db_ptr[c] = sum_dy_arr[c];
        dg_ptr[c] = sum_dy_xn_arr[c];
    }

    Tensor dX({B, C, H, W});
    float* dx_ptr = dX.data();
    #pragma omp parallel for schedule(guided)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t c = 0; c < C; ++c) {
            float scale = rstd_ptr[c] * gamma_ptr[c] / N;
            float sd = sum_dy_arr[c], sdx = sum_dy_xn_arr[c];
            const float* dbase = dy_ptr + b * CHW + c * HW;
            const float* xbase = xn_ptr + b * CHW + c * HW;
            float* dxb = dx_ptr + b * CHW + c * HW;
            for (int64_t j = 0; j < HW; ++j) {
                dxb[j] = scale * (N * dbase[j] - sd - xbase[j] * sdx);
            }
        }
    }
    return {dX, dgamma, dbeta};
}

// ============================================================================
// conv2d forward: im2col + matmul
// ============================================================================
// 简化实现：朴素 im2col，然后 matmul
//
// X: (batch, in_c, in_h, in_w)
// W: (out_c, in_c, kh, kw)
// b: (out_c,)
// Y: (batch, out_c, out_h, out_w)
//
// im2col 布局：
//   x_col: (in_c*kh*kw, out_h*out_w) per batch，合并 batch 后 (in_c*kh*kw, batch*out_h*out_w)
//   W_col: (out_c, in_c*kh*kw)  — 直接 reshape W
//   Y_col = W_col @ x_col: (out_c, batch*out_h*out_w)
//   再 reshape 为 (batch, out_c, out_h, out_w)
Tensor conv2d_forward(
    const Tensor& X, const Tensor& W, const Tensor& b,
    int stride, int padding,
    Conv2DContext& ctx
) {
    if (X.ndim() != 4 || W.ndim() != 4) {
        throw std::invalid_argument("conv2d_forward: X and W must be 4D");
    }
    if (!X.is_contiguous() || !W.is_contiguous() || !b.is_contiguous()) {
        throw std::invalid_argument("conv2d_forward: inputs must be contiguous");
    }

    ctx.batch = X.shape()[0];
    ctx.in_channels = X.shape()[1];
    ctx.in_h = X.shape()[2];
    ctx.in_w = X.shape()[3];
    ctx.out_channels = W.shape()[0];
    ctx.kh = W.shape()[2];
    ctx.kw = W.shape()[3];
    ctx.stride = stride;
    ctx.padding = padding;

    ctx.out_h = (ctx.in_h + 2 * padding - ctx.kh) / stride + 1;
    ctx.out_w = (ctx.in_w + 2 * padding - ctx.kw) / stride + 1;

    int64_t in_c_kh_kw = ctx.in_channels * ctx.kh * ctx.kw;
    int64_t spatial = ctx.out_h * ctx.out_w;

    // x_col: (in_c*kh*kw, batch*spatial)
    ctx.x_col = Tensor({in_c_kh_kw, ctx.batch * spatial});
    float* xcol_ptr = ctx.x_col.data();
    const float* x_ptr = X.data();

    // im2col（P1-E：行条带法端口，零填充缓冲 + 连续读写，纯拷贝）
    const Conv2dKernelSet& cks_fwd = conv2d_registry();
    cks_fwd.im2col(x_ptr, xcol_ptr, ctx.batch, ctx.in_channels,
                   ctx.in_h, ctx.in_w, ctx.out_h, ctx.out_w,
                   ctx.kh, ctx.kw, stride, padding);

    // W_col: reshape W (out_c, in_c, kh, kw) → (out_c, in_c*kh*kw)
    // W 已经是行优先，直接 reshape
    Tensor W_col = W.reshape({ctx.out_channels, in_c_kh_kw});

    // Y_col = W_col @ x_col: (out_c, batch*spatial)
    Tensor Y_col = matmul_forward(W_col, ctx.x_col);

    // 加 bias + 转置 Y_col→Y 合并为一次遍历（减少一次内存遍历）
    // Y_col 布局: (out_c, batch*spatial), Y 布局: (batch, out_c, out_h, out_w)
    const float* b_ptr = b.data();
    float* ycol_ptr = Y_col.data();
    Tensor Y({ctx.batch, ctx.out_channels, ctx.out_h, ctx.out_w});
    float* y_ptr = Y.data();
    for (int64_t b_idx = 0; b_idx < ctx.batch; ++b_idx) {
        for (int64_t out_c = 0; out_c < ctx.out_channels; ++out_c) {
            float bias = b_ptr[out_c];
            for (int64_t oc = 0; oc < spatial; ++oc) {
                y_ptr[(b_idx * ctx.out_channels + out_c) * spatial + oc] =
                    ycol_ptr[out_c * (ctx.batch * spatial) + b_idx * spatial + oc] + bias;
            }
        }
    }

    return Y;
}

// ============================================================================
// 专用 dW GEMM（裸指针端口版）：dW_col = dY_col @ x_col^T
// ============================================================================
//   dY_col: (out_c, bs), x_col: (in_c*kh*kw, bs), bs = batch*spatial
//   C[i,j] = sum_l A[i,l] * B[j,l], i∈out_c, j∈k, l∈bs
//
// 裸指针版供 dispatch/registry.cpp 通过 Conv2dKernelSet::dw 注册，
// 契约：输出缓冲 C 由调用方分配（out_c × k 行主序），内核只写不拥有。

// 标量参照（bit-exact 锚点）
void dw_scalar(const float* A, const float* B, float* C,
               int64_t out_c, int64_t bs, int64_t k) {
    for (int64_t i = 0; i < out_c; ++i) {
        for (int64_t j = 0; j < k; ++j) {
            float sum = 0.0f;
            for (int64_t l = 0; l < bs; ++l) {
                sum += A[i * bs + l] * B[j * bs + l];
            }
            C[i * k + j] = sum;
        }
    }
}

// AVX2 版（从 dW_from_dYcol_xcol 提取，I_BLOCK=4 寄存器分块 + 8 路归约）
// 数值档位：kRounding（8 路并行累加归约序与标量不同）
void dw_avx2(const float* dy, const float* xc, float* c,
             int64_t out_c, int64_t bs, int64_t k) {
    constexpr int64_t I_BLOCK = 4;
    const int64_t i_blocked = (out_c / I_BLOCK) * I_BLOCK;
    #pragma omp parallel for schedule(guided) if(out_c * k * bs >= (1 << 18))
    for (int64_t i0 = 0; i0 < i_blocked; i0 += I_BLOCK) {
        const float* a0 = dy + (i0 + 0) * bs;
        const float* a1 = dy + (i0 + 1) * bs;
        const float* a2 = dy + (i0 + 2) * bs;
        const float* a3 = dy + (i0 + 3) * bs;
        for (int64_t j = 0; j < k; ++j) {
            const float* bj = xc + j * bs;
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();
            int64_t l = 0;
            for (; l + 8 <= bs; l += 8) {
                __m256 bv = _mm256_loadu_ps(bj + l);
                c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + l), bv, c0);
                c1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + l), bv, c1);
                c2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + l), bv, c2);
                c3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + l), bv, c3);
            }
            float s0 = bn_hadd256(c0);
            float s1 = bn_hadd256(c1);
            float s2 = bn_hadd256(c2);
            float s3 = bn_hadd256(c3);
            for (; l < bs; ++l) {
                float bv = bj[l];
                s0 += a0[l] * bv;
                s1 += a1[l] * bv;
                s2 += a2[l] * bv;
                s3 += a3[l] * bv;
            }
            c[(i0 + 0) * k + j] = s0;
            c[(i0 + 1) * k + j] = s1;
            c[(i0 + 2) * k + j] = s2;
            c[(i0 + 3) * k + j] = s3;
        }
    }
    // 尾部 i 行（out_c % 4）
    for (int64_t i = i_blocked; i < out_c; ++i) {
        const float* ai = dy + i * bs;
        for (int64_t j = 0; j < k; ++j) {
            const float* bj = xc + j * bs;
            __m256 csum = _mm256_setzero_ps();
            int64_t l = 0;
            for (; l + 8 <= bs; l += 8) {
                csum = _mm256_fmadd_ps(_mm256_loadu_ps(ai + l),
                                       _mm256_loadu_ps(bj + l), csum);
            }
            float s = bn_hadd256(csum);
            for (; l < bs; ++l) s += ai[l] * bj[l];
            c[i * k + j] = s;
        }
    }
}

// ============================================================================
// 专用 dx_col GEMM（裸指针端口版，P0-A：j-tile 重排）
// ============================================================================
//   W_col: (out_c, r), dY_col: (out_c, bs), dx_col: (r, bs)
//   C[r,c] = sum_l W_col[l,r] * dY_col[l,c] = sum_l A[l*r + r] * B[l*bs + c]
//
// P0-A 方案（调研文档 §4）：现役 transpose_a 按 C 行块并行、每块重读 B(dY_col
// 2MB) → 总流量大。本内核外层按 j（bs 方向）分块（J_TILE≈512）：B 列段驻 L2
// 只读一遍；A(W_col) 3.4KB 恒驻 L1；内层仍 broadcast+loadu+FMA。
//
// 逐位一致性：单元素 C[r,c] 的归约序 = l 0..out_c 串行 FMA（与现役
// matmul_fwd_transpose_a 的 m-tile 串行累加顺序一致）→ 逐位一致。
// 并行仅跨 i 块（写集独立），不在归约维并行（§6.4）。
// ============================================================================
void dx_avx2(const float* A, const float* B, float* C,
             int64_t r, int64_t out_c, int64_t bs) {
    constexpr int64_t J_TILE = 512;
    constexpr int64_t I_BLOCK = 4;

    for (int64_t j0 = 0; j0 < bs; j0 += J_TILE) {
        const int64_t j_end = (j0 + J_TILE < bs) ? j0 + J_TILE : bs;
        const int64_t i_blocked = (r / I_BLOCK) * I_BLOCK;
        // 单 OpenMP region：i 块独立写 C，l 全序串行累加 → 逐位一致
        #pragma omp parallel for schedule(guided) if(r * out_c * (j_end - j0) >= (1 << 18))
        for (int64_t i0 = 0; i0 < i_blocked; i0 += I_BLOCK) {
            int64_t j = j0;
            for (; j + 8 <= j_end; j += 8) {
                __m256 c0 = _mm256_setzero_ps();
                __m256 c1 = _mm256_setzero_ps();
                __m256 c2 = _mm256_setzero_ps();
                __m256 c3 = _mm256_setzero_ps();
                for (int64_t l = 0; l < out_c; ++l) {
                    __m256 b = _mm256_loadu_ps(B + l * bs + j);
                    __m256 a0 = _mm256_set1_ps(A[l * r + (i0 + 0)]);
                    __m256 a1 = _mm256_set1_ps(A[l * r + (i0 + 1)]);
                    __m256 a2 = _mm256_set1_ps(A[l * r + (i0 + 2)]);
                    __m256 a3 = _mm256_set1_ps(A[l * r + (i0 + 3)]);
                    c0 = _mm256_fmadd_ps(a0, b, c0);
                    c1 = _mm256_fmadd_ps(a1, b, c1);
                    c2 = _mm256_fmadd_ps(a2, b, c2);
                    c3 = _mm256_fmadd_ps(a3, b, c3);
                }
                _mm256_storeu_ps(C + (i0 + 0) * bs + j, c0);
                _mm256_storeu_ps(C + (i0 + 1) * bs + j, c1);
                _mm256_storeu_ps(C + (i0 + 2) * bs + j, c2);
                _mm256_storeu_ps(C + (i0 + 3) * bs + j, c3);
            }
            // 尾部 j 列（标量）
            for (; j < j_end; ++j) {
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                for (int64_t l = 0; l < out_c; ++l) {
                    float bv = B[l * bs + j];
                    s0 += A[l * r + (i0 + 0)] * bv;
                    s1 += A[l * r + (i0 + 1)] * bv;
                    s2 += A[l * r + (i0 + 2)] * bv;
                    s3 += A[l * r + (i0 + 3)] * bv;
                }
                C[(i0 + 0) * bs + j] = s0;
                C[(i0 + 1) * bs + j] = s1;
                C[(i0 + 2) * bs + j] = s2;
                C[(i0 + 3) * bs + j] = s3;
            }
        }
        // 尾部 i 行（r % 4）
        for (int64_t i = i_blocked; i < r; ++i) {
            int64_t j = j0;
            for (; j + 8 <= j_end; j += 8) {
                __m256 csum = _mm256_setzero_ps();
                for (int64_t l = 0; l < out_c; ++l) {
                    __m256 b = _mm256_loadu_ps(B + l * bs + j);
                    csum = _mm256_fmadd_ps(_mm256_set1_ps(A[l * r + i]), b, csum);
                }
                _mm256_storeu_ps(C + i * bs + j, csum);
            }
            for (; j < j_end; ++j) {
                float s = 0.0f;
                for (int64_t l = 0; l < out_c; ++l) {
                    s += A[l * r + i] * B[l * bs + j];
                }
                C[i * bs + j] = s;
            }
        }
    }
}

// dx_col 标量参照（bit-exact 锚点）
void dx_scalar(const float* A, const float* B, float* C,
               int64_t r, int64_t out_c, int64_t bs) {
    for (int64_t i = 0; i < r; ++i) {
        for (int64_t j = 0; j < bs; ++j) {
            float s = 0.0f;
            for (int64_t l = 0; l < out_c; ++l) {
                s += A[l * r + i] * B[l * bs + j];
            }
            C[i * bs + j] = s;
        }
    }
}

// ============================================================================
// db 行和（裸指针端口版）：db[oc] = sum_l dY_col[oc, l]（反向步骤 ③）
// ============================================================================
//   dY_col: (out_c, bs)，db: (out_c,)。纯归约 → 输出由调用方分配，内核只写。
// 逐位一致性：db_scalar 串行累加（bit-exact 锚点）；db_avx2 8 路并行累加 →
//   归约序不同 → kRounding（数值档位由 Conv2dKernelSet::num_level 声明）。

// 标量参照（bit-exact 锚点）
void db_scalar(const float* dY_col, float* db, int64_t out_c, int64_t bs) {
    for (int64_t oc = 0; oc < out_c; ++oc) {
        const float* row = dY_col + oc * bs;
        float sum = 0.0f;
        for (int64_t l = 0; l < bs; ++l) {
            sum += row[l];
        }
        db[oc] = sum;
    }
}

// AVX2 版（8 宽 loadu/add + 横向归约 + 标量尾 → sgn::simd::sum_f32，见 simd 原语层）
// 数值档位：kRounding（8 路并行累加归约序与标量不同）
void db_avx2(const float* dY_col, float* db, int64_t out_c, int64_t bs) {
    #pragma omp parallel for schedule(guided) if(out_c * bs >= (1 << 18))
    for (int64_t oc = 0; oc < out_c; ++oc) {
        const float* row = dY_col + oc * bs;
        db[oc] = sgn::simd::sum_f32(row, bs, 1);
    }
}

// ============================================================================
// col2im 行条带法（裸指针端口版，P0-B）
// ============================================================================
//   dx_col: (r, bs) 行主序，r = in_c*kh*kw，bs = batch*spatial
//   dX_pad: (batch, in_c, padded_h, padded_w)，累加语义（调用前必须清零，契约 3）
//   row = (ic*kh + ki)*kw + kj；ph = oh*stride + ki；pw = ow*stride + kj
//
// P0-B 方案（调研文档 §4）：旧版对每个输出位置 gather 8 行同列（跨步读）+ 逐元素
// div/mod。本内核换主循环 (b, ic, oh, ki, kj, ow)：内层 ow 时源 dx_col 行段连续读、
// 目标 dX_pad 行内连续写（stride=1 时）；div/mod 上提到循环外。
//
// 逐位一致性（关键）：对固定目标元素 (b,ic,ph,pw)，命中的 (oh,ow,ki,kj) 组合中
//   ki 唯一、kj 唯一（由 oh/ow 决定）。旧 gather 版按 oc=(oh,ow) 外层升序累加，
//   即 oh 升序、ow 升序。本内核外层 oh 升序、内层 kj【降序】→ 命中的 ow 升序，
//   与旧版完全一致 → 逐位一致。（若 kj 升序则会得到 ow 降序 → 非逐位一致，勿改）
// 并行仅按 b 划分（§6.4：col2im 写集跨 b 无重叠）。
// ============================================================================
void col2im_avx2(const float* dx_col, float* dX_pad,
                 int64_t batch, int64_t in_c,
                 int64_t out_h, int64_t out_w,
                 int64_t bs, int64_t kh, int64_t kw,
                 int64_t stride, int64_t /*pad*/,
                 int64_t padded_h, int64_t padded_w) {
    const int64_t spatial = out_h * out_w;
    // 并行仅按 b 划分（col2im 写集跨 b 独立）
    #pragma omp parallel for schedule(guided) if(batch * in_c * out_h * kh * kw * out_w >= (1 << 18))
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t ic = 0; ic < in_c; ++ic) {
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ki = 0; ki < kh; ++ki) {
                    const int64_t ph = oh * stride + ki;  // 恒 [0, padded_h)
                    for (int64_t kj = kw - 1; kj >= 0; --kj) {  // 降序 → 命中 ow 升序
                        const int64_t row = (ic * kh + ki) * kw + kj;
                        const float* src = dx_col + row * bs + b * spatial + oh * out_w;
                        float* dst = dX_pad + ((b * in_c + ic) * padded_h + ph) * padded_w + kj;
                        if (stride == 1) {
                            int64_t ow = 0;
                            for (; ow + 8 <= out_w; ow += 8) {
                                __m256 v = _mm256_loadu_ps(src + ow);
                                __m256 d = _mm256_loadu_ps(dst + ow);
                                _mm256_storeu_ps(dst + ow, _mm256_add_ps(d, v));
                            }
                            for (; ow < out_w; ++ow) dst[ow] += src[ow];
                        } else {
                            for (int64_t ow = 0; ow < out_w; ++ow) {
                                dst[ow * stride] += src[ow];
                            }
                        }
                    }
                }
            }
        }
    }
}

// col2im 标量参照（bit-exact 锚点；与 avx2 相同的累加次序）
void col2im_scalar(const float* dx_col, float* dX_pad,
                   int64_t batch, int64_t in_c,
                   int64_t out_h, int64_t out_w,
                   int64_t bs, int64_t kh, int64_t kw,
                   int64_t stride, int64_t /*pad*/,
                   int64_t padded_h, int64_t padded_w) {
    const int64_t spatial = out_h * out_w;
    #pragma omp parallel for schedule(guided) if(batch * in_c * out_h * kh * kw * out_w >= (1 << 18))
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t ic = 0; ic < in_c; ++ic) {
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ki = 0; ki < kh; ++ki) {
                    const int64_t ph = oh * stride + ki;
                    for (int64_t kj = kw - 1; kj >= 0; --kj) {
                        const int64_t row = (ic * kh + ki) * kw + kj;
                        const float* src = dx_col + row * bs + b * spatial + oh * out_w;
                        float* dst = dX_pad + ((b * in_c + ic) * padded_h + ph) * padded_w + kj;
                        for (int64_t ow = 0; ow < out_w; ++ow) {
                            dst[ow * stride] += src[ow];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// im2col（前向，裸指针端口版，P1-E）：行条带法，与 P0-B col2im 完全对称
// ============================================================================
//   X: (batch, in_c, in_h, in_w) → x_col: (r×bs) 行主序，越界填 0
//   r = in_c*kh*kw；bs = batch*spatial；spatial = out_h*out_w
//   x_col[row, b*spatial + oh*out_w + ow] = X[b, ic, oh*stride - pad + ki,
//                                              ow*stride - pad + kj]
//   row = (ic*kh + ki)*kw + kj
//
// P1-E 方案（调研文档 §4，与 P0-B 对称）：主循环 (b, ic, ki, kj, oh)，内层 ow
//   连续写 x_col（row 固定，b*spatial + oh*out_w + ow 连续）；源 X 走零填充
//   缓冲（pad 映射 ph=oh*stride+ki、pw=ow*stride+kj 恒界内，完全无裁剪判断）。
//   stride=1 时源行段也连续 → loadu/storeu 8 宽。
//
// 逐位一致性：纯拷贝无归约，任意次序都逐位一致（与旧 gather 版一致）。
// 并行仅按 b 划分（每 b 各自建 pad 缓冲，写集跨 b 独立）。
// ============================================================================
void im2col_avx2(const float* X, float* x_col,
                 int64_t batch, int64_t in_c,
                 int64_t in_h, int64_t in_w,
                 int64_t out_h, int64_t out_w,
                 int64_t kh, int64_t kw, int64_t stride, int64_t pad) {
    const int64_t spatial = out_h * out_w;
    const int64_t bs = batch * spatial;
    const int64_t padded_h = in_h + 2 * pad;
    const int64_t padded_w = in_w + 2 * pad;
    const int64_t xch = in_h * in_w;              // 单通道 X 行主序大小
    // 并行仅按 b 划分（每 b 独立建 pad 缓冲，写集跨 b 独立）
    #pragma omp parallel for schedule(guided) if(batch * in_c * spatial >= (1 << 18))
    for (int64_t b = 0; b < batch; ++b) {
        // 每 b 一份 pad 缓冲（跨 ic 复用；线程私有 → 无 false sharing）
        std::vector<float> xp((size_t)padded_h * padded_w, 0.0f);
        for (int64_t ic = 0; ic < in_c; ++ic) {
            // 零填充缓冲（中心拷贝 X 通道 + 边缘 0），9 tap 全纯行拷
            const float* xin = X + (b * in_c + ic) * xch;
            std::fill(xp.begin(), xp.end(), 0.0f);
            for (int64_t ih = 0; ih < in_h; ++ih) {
                const float* src = xin + ih * in_w;
                float* dst = xp.data() + (ih + pad) * padded_w + pad;
                std::memcpy(dst, src, (size_t)in_w * sizeof(float));
            }
            for (int64_t ki = 0; ki < kh; ++ki) {
                for (int64_t kj = 0; kj < kw; ++kj) {
                    const int64_t row = (ic * kh + ki) * kw + kj;
                    float* dst_row = x_col + row * bs + b * spatial;
                    for (int64_t oh = 0; oh < out_h; ++oh) {
                        const int64_t ph = oh * stride + ki;   // 恒 [0, padded_h)
                        const float* src = xp.data() + ph * padded_w + kj;
                        float* dst = dst_row + oh * out_w;
                        if (stride == 1) {
                            int64_t ow = 0;
                            for (; ow + 8 <= out_w; ow += 8) {
                                _mm256_storeu_ps(dst + ow, _mm256_loadu_ps(src + ow));
                            }
                            for (; ow < out_w; ++ow) dst[ow] = src[ow];
                        } else {
                            for (int64_t ow = 0; ow < out_w; ++ow) {
                                dst[ow] = src[ow * stride];
                            }
                        }
                    }
                }
            }
        }
    }
}

// im2col 标量参照（bit-exact 锚点；与 avx2 相同的 pad 缓冲结构，无向量化）
void im2col_scalar(const float* X, float* x_col,
                   int64_t batch, int64_t in_c,
                   int64_t in_h, int64_t in_w,
                   int64_t out_h, int64_t out_w,
                   int64_t kh, int64_t kw, int64_t stride, int64_t pad) {
    const int64_t spatial = out_h * out_w;
    const int64_t bs = batch * spatial;
    const int64_t padded_h = in_h + 2 * pad;
    const int64_t padded_w = in_w + 2 * pad;
    const int64_t xch = in_h * in_w;
    #pragma omp parallel for schedule(guided) if(batch * in_c * spatial >= (1 << 18))
    for (int64_t b = 0; b < batch; ++b) {
        std::vector<float> xp((size_t)padded_h * padded_w, 0.0f);
        for (int64_t ic = 0; ic < in_c; ++ic) {
            const float* xin = X + (b * in_c + ic) * xch;
            std::fill(xp.begin(), xp.end(), 0.0f);
            for (int64_t ih = 0; ih < in_h; ++ih) {
                const float* src = xin + ih * in_w;
                float* dst = xp.data() + (ih + pad) * padded_w + pad;
                std::memcpy(dst, src, (size_t)in_w * sizeof(float));
            }
            for (int64_t ki = 0; ki < kh; ++ki) {
                for (int64_t kj = 0; kj < kw; ++kj) {
                    const int64_t row = (ic * kh + ki) * kw + kj;
                    float* dst_row = x_col + row * bs + b * spatial;
                    for (int64_t oh = 0; oh < out_h; ++oh) {
                        const int64_t ph = oh * stride + ki;
                        const float* src = xp.data() + ph * padded_w + kj;
                        float* dst = dst_row + oh * out_w;
                        for (int64_t ow = 0; ow < out_w; ++ow) {
                            dst[ow] = src[ow * stride];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// dY→dY_col 转置（裸指针端口版，P1-C）：纯搬运
// ============================================================================
//   dY: (batch, out_c, spatial) 行主序 → dY_col: (out_c, batch*spatial)
//   dY_col[oc, b*spatial + s] = dY[(b*out_c + oc)*spatial + s]
//   即把 batch 与 out_c 两个外层维度互换；每个 (b, oc) 行段在源与目标
//   都是 spatial 连续的 → 转置退化为"逐段搬运"（纯复制，无重排计算）。
//
// P1-C 方案（调研文档 §4）：64×64 tile 分块（b×oc），内层 spatial 8 宽
//   loadu/storeu 连续搬；tile 使每个 OpenMP 线程的写区更局部（dY_col 中
//   oc 连续行段），减少 fork/join 次数。纯搬运 → 与标量逐位一致。
// ============================================================================
void dycol_avx2(const float* dY, float* dY_col,
                int64_t batch, int64_t out_c, int64_t spatial) {
    const int64_t b_spatial = batch * spatial;   // dY_col 每 oc 行宽
    constexpr int64_t TILE = 64;
    // 64×64 tile（b×oc）；写区跨 (b,oc) 独立
    #pragma omp parallel for schedule(guided) collapse(2) \
        if(batch * out_c * spatial >= (1 << 18))
    for (int64_t b0 = 0; b0 < batch; b0 += TILE) {
        const int64_t b_end = (b0 + TILE < batch) ? b0 + TILE : batch;
        for (int64_t oc0 = 0; oc0 < out_c; oc0 += TILE) {
            const int64_t oc_end = (oc0 + TILE < out_c) ? oc0 + TILE : out_c;
            for (int64_t b = b0; b < b_end; ++b) {
                for (int64_t oc = oc0; oc < oc_end; ++oc) {
                    const float* src = dY + (b * out_c + oc) * spatial;
                    float* dst = dY_col + oc * b_spatial + b * spatial;
                    int64_t s = 0;
                    for (; s + 8 <= spatial; s += 8) {
                        _mm256_storeu_ps(dst + s, _mm256_loadu_ps(src + s));
                    }
                    for (; s < spatial; ++s) dst[s] = src[s];
                }
            }
        }
    }
}

// dY→dY_col 标量参照（bit-exact 锚点）
void dycol_scalar(const float* dY, float* dY_col,
                  int64_t batch, int64_t out_c, int64_t spatial) {
    const int64_t b_spatial = batch * spatial;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t oc = 0; oc < out_c; ++oc) {
            const float* src = dY + (b * out_c + oc) * spatial;
            float* dst = dY_col + oc * b_spatial + b * spatial;
            for (int64_t s = 0; s < spatial; ++s) dst[s] = src[s];
        }
    }
}

// ============================================================================
// 专用 dW GEMM（Tensor 包装，保留向后兼容）：dW_col = dY_col @ x_col^T
// ============================================================================
//   dY_col: (out_c, bs), x_col: (in_c*kh*kw, bs), bs = batch*spatial
//   dW[i,j] = sum_l dY_col[i,l] * x_col[j,l]，i∈out_c, j∈in_c*kh*kw, l∈bs
//
// 节奏 1（2026-08-16）：实现已迁入 Conv2dKernelSet::dw（dw_avx2 / dw_scalar），
// 本函数保留为 Tensor 包装，内部经 conv2d_registry() 调用 dW 端口——
// 算子层只认接口，具体内核由注册表按 CPU 能力/环境变量选择。
//
// 数值档位：avx2 内核 kRounding（8 路并行累加归约序与标量不同）；scalar bit-exact。
// ============================================================================
static Tensor dW_from_dYcol_xcol(const Tensor& dY_col, const Tensor& x_col) {
    const int64_t m = dY_col.shape()[0];  // out_c
    const int64_t bs = dY_col.shape()[1]; // batch*spatial
    const int64_t n = x_col.shape()[0];   // in_c*kh*kw
    Tensor C({m, n});
    const Conv2dKernelSet& ks = conv2d_registry();
    ks.dw(dY_col.data(), x_col.data(), C.data(), m, bs, n);
    return C;
}

// ============================================================================
// conv2d backward
// ============================================================================
//   dY: (batch, out_c, out_h, out_w)
//   dW: (out_c, in_c, kh, kw)
//   db: (out_c,)
//   dX: (batch, in_c, in_h, in_w) — 需要 col2im
//
// 步骤：
//   1. dY → dY_col: (out_c, batch*spatial)
//   2. dW_col = dY_col @ x_col^T: (out_c, in_c*kh*kw) → reshape dW
//   3. db = sum over (batch, spatial) per out_c
//   4. dx_col = W_col^T @ dY_col: (in_c*kh*kw, batch*spatial) → col2im → dX
std::tuple<Tensor, Tensor, Tensor> conv2d_backward(
    const Conv2DContext& ctx, const Tensor& /*X*/, const Tensor& W,
    const Tensor& dY
) {
    int64_t in_c_kh_kw = ctx.in_channels * ctx.kh * ctx.kw;
    int64_t spatial = ctx.out_h * ctx.out_w;
    int64_t batch_spatial = ctx.batch * spatial;

    // 1. dY (batch, out_c, out_h, out_w) → dY_col (out_c, batch*spatial)
    //    节奏 3 P1-C：改经 conv2d_registry() 调 dycol 端口（64×64 tile + 8 宽
    //    向量化搬运，纯复制 → 与标量逐位一致）。
    Tensor dY_col({ctx.out_channels, batch_spatial});
    const Conv2dKernelSet& cks = conv2d_registry();
    cks.dycol(dY.data(), dY_col.data(), ctx.batch, ctx.out_channels, spatial);
    const float* dycol_ptr = dY_col.data();

    // 2. dW_col = dY_col @ x_col^T: (out_c, in_c*kh*kw)
    //    专用 dW 内核（2026-08-16）：单 OpenMP region + 连续读，见 dW_from_dYcol_xcol。
    //    实测 conv1 dW：旧 gather 型 transpose_b 6-7ms → 本内核 ~0.5ms。
    Tensor W_col = W.reshape({ctx.out_channels, in_c_kh_kw});
    Tensor dW_col = dW_from_dYcol_xcol(dY_col, ctx.x_col);
    // reshape dW_col → dW (out_c, in_c, kh, kw)
    Tensor dW = dW_col.reshape({ctx.out_channels, ctx.in_channels, ctx.kh, ctx.kw});

    // 3. db = sum over (batch, spatial) per out_c
    //    通道化：经 conv2d_registry() 调 db 端口（标量参照 / AVX2 双实现）
    Tensor db({ctx.out_channels});
    cks.db(dycol_ptr, db.data(), ctx.out_channels, batch_spatial);

    // 4. dx_col = W_col^T @ dY_col: (in_c*kh*kw, batch*spatial)
    //    W_col: (out_c, in_c*kh_kw), dY_col: (out_c, batch*spatial)
    //    节奏 3 P0-A：改经 conv2d_registry() 调 dx 端口（j-tile 重排，B 单遍读），
    //    归约序 l 串行 → 与旧 gather 型 transpose_a 逐位一致。
    Tensor dx_col({in_c_kh_kw, batch_spatial});
    cks.dx(W_col.data(), dY_col.data(), dx_col.data(),
           in_c_kh_kw, ctx.out_channels, batch_spatial);
    const float* dxcol_ptr = dx_col.data();

    // col2im: dx_col → dX (累加，因为有重叠的 receptive field)
    // 节奏 3 P0-B：改经 conv2d_registry() 调 col2im 端口（行条带法）——
    // 内层 ow 连续读写，div/mod 上提；对固定目标元素累加序（oh/ow 升序）与
    // 旧 gather 版逐位一致。
    Tensor dX({ctx.batch, ctx.in_channels, ctx.in_h, ctx.in_w});
    float* dx_ptr = dX.data();
    const int64_t pad = ctx.padding;
    const int64_t padded_h = ctx.in_h + 2 * pad;
    const int64_t padded_w = ctx.in_w + 2 * pad;
    // pad==0 时缓冲即 dX，直接就地累加，无需额外分配与拷贝
    Tensor dX_pad;
    float* buf_ptr;
    if (pad != 0) {
        dX_pad = Tensor({ctx.batch, ctx.in_channels, padded_h, padded_w});  // 零初始化
        buf_ptr = dX_pad.data();
    } else {
        buf_ptr = dx_ptr;
    }
    cks.col2im(dxcol_ptr, buf_ptr, ctx.batch, ctx.in_channels,
               ctx.out_h, ctx.out_w, batch_spatial,
               ctx.kh, ctx.kw, ctx.stride, pad, padded_h, padded_w);
    // 拷贝有效区域回 dX（仅 pad!=0；纯 memcpy，不改变任何值）
    if (pad != 0) {
        #pragma omp parallel for schedule(guided)
        for (int64_t b_idx = 0; b_idx < ctx.batch; ++b_idx) {
            for (int64_t ic = 0; ic < ctx.in_channels; ++ic) {
                const float* src =
                    buf_ptr + ((b_idx * ctx.in_channels + ic) * padded_h + pad) * padded_w + pad;
                float* dst = dx_ptr + ((b_idx * ctx.in_channels + ic) * ctx.in_h) * ctx.in_w;
                for (int64_t h = 0; h < ctx.in_h; ++h) {
                    std::memcpy(dst + h * ctx.in_w, src + h * padded_w, ctx.in_w * sizeof(float));
                }
            }
        }
    }

    BufferPool::release(std::move(dY_col));

    return {dX, dW, db};
}

// ============================================================================
// maxpool2d forward
// ============================================================================
// X: (batch, channels, in_h, in_w)
// kernel, stride: 窗口参数
// Y: (batch, channels, out_h, out_w)
// argmax: 记录每个输出位置在输入中的线性索引（用于 backward 路由）
Tensor maxpool2d_forward(
    const Tensor& X, int kernel, int stride,
    MaxPoolContext& ctx
) {
    if (X.ndim() != 4) {
        throw std::invalid_argument("maxpool2d_forward: X must be 4D");
    }
    if (!X.is_contiguous()) {
        throw std::invalid_argument("maxpool2d_forward: X must be contiguous");
    }

    ctx.batch = X.shape()[0];
    ctx.channels = X.shape()[1];
    ctx.in_h = X.shape()[2];
    ctx.in_w = X.shape()[3];
    ctx.kernel = kernel;
    ctx.stride = stride;
    ctx.out_h = (ctx.in_h - kernel) / stride + 1;
    ctx.out_w = (ctx.in_w - kernel) / stride + 1;

    Tensor Y({ctx.batch, ctx.channels, ctx.out_h, ctx.out_w});
    // 数据竞争修复（2026-08-30）：argmax 改用 int64 vector 存储
    ctx.argmax = std::vector<int64_t>(ctx.batch * ctx.channels * ctx.out_h * ctx.out_w, 0);
    float* y_ptr = Y.data();
    int64_t* am_ptr = ctx.argmax.data();
    const float* x_ptr = X.data();

    #pragma omp parallel for schedule(guided) collapse(2)
    for (int64_t b = 0; b < ctx.batch; ++b) {
        for (int64_t c = 0; c < ctx.channels; ++c) {
            for (int64_t oh = 0; oh < ctx.out_h; ++oh) {
                for (int64_t ow = 0; ow < ctx.out_w; ++ow) {
                    int64_t out_idx = ((b * ctx.channels + c) * ctx.out_h + oh) * ctx.out_w + ow;
                    float max_val = -std::numeric_limits<float>::infinity();
                    int64_t max_lin_idx = 0;
                    for (int64_t ki = 0; ki < kernel; ++ki) {
                        for (int64_t kj = 0; kj < kernel; ++kj) {
                            int64_t ih = oh * stride + ki;
                            int64_t iw = ow * stride + kj;
                            if (ih < ctx.in_h && iw < ctx.in_w) {
                                int64_t in_idx = ((b * ctx.channels + c) * ctx.in_h + ih) * ctx.in_w + iw;
                                float val = x_ptr[in_idx];
                                if (val > max_val) {
                                    max_val = val;
                                    max_lin_idx = in_idx;
                                }
                            }
                        }
                    }
                    y_ptr[out_idx] = max_val;
                    am_ptr[out_idx] = max_lin_idx;
                }
            }
        }
    }
    return Y;
}

// ============================================================================
// maxpool2d backward: 梯度路由
// ============================================================================
// dX 全零，只在 argmax 记录的位置加上 dY 的值
Tensor maxpool2d_backward(const MaxPoolContext& ctx, const Tensor& dY) {
    Tensor dX({ctx.batch, ctx.channels, ctx.in_h, ctx.in_w});
    const float* dy_ptr = dY.data();
    // argmax 为 int64（2026-08-30 数据竞争/精度修复，见 maxpool2d_forward）
    const int64_t* am_ptr = ctx.argmax.data();
    float* dx_ptr = dX.data();

    int64_t out_numel = ctx.batch * ctx.channels * ctx.out_h * ctx.out_w;
    // 数据竞争修复（2026-08-30）：当 stride < kernel 时，多个输出位置可能映射到同一输入位置，
    // OpenMP 并行写入会导致 += 竞争。使用 atomic 操作保护。
    #pragma omp parallel for schedule(guided)
    for (int64_t i = 0; i < out_numel; ++i) {
        int64_t in_idx = static_cast<int64_t>(am_ptr[i]);
        #pragma omp atomic
        dx_ptr[in_idx] += dy_ptr[i];
    }
    return dX;
}

// ============================================================================
// avgpool2d forward: Y = 窗口均值（GAP）
// ============================================================================
// X: (batch, channels, in_h, in_w)，kernel/stride 同 maxpool。
// Y: (batch, channels, out_h, out_w)。backward 需形状信息 → AvgPoolContext。
Tensor avgpool2d_forward(
    const Tensor& X, int kernel, int stride,
    AvgPoolContext& ctx
) {
    if (X.ndim() != 4) {
        throw std::invalid_argument("avgpool2d_forward: X must be 4D");
    }
    if (!X.is_contiguous()) {
        throw std::invalid_argument("avgpool2d_forward: X must be contiguous");
    }

    ctx.batch = X.shape()[0];
    ctx.channels = X.shape()[1];
    ctx.in_h = X.shape()[2];
    ctx.in_w = X.shape()[3];
    ctx.kernel = kernel;
    ctx.stride = stride;
    ctx.out_h = (ctx.in_h - kernel) / stride + 1;
    ctx.out_w = (ctx.in_w - kernel) / stride + 1;

    Tensor Y({ctx.batch, ctx.channels, ctx.out_h, ctx.out_w});
    const float* x_ptr = X.data();
    float* y_ptr = Y.data();
    const float inv = 1.0f / (float)(kernel * kernel);

    #pragma omp parallel for schedule(guided) collapse(2)
    for (int64_t b = 0; b < ctx.batch; ++b) {
        for (int64_t c = 0; c < ctx.channels; ++c) {
            for (int64_t oh = 0; oh < ctx.out_h; ++oh) {
                for (int64_t ow = 0; ow < ctx.out_w; ++ow) {
                    float acc = 0.0f;
                    for (int64_t ki = 0; ki < kernel; ++ki) {
                        for (int64_t kj = 0; kj < kernel; ++kj) {
                            int64_t ih = oh * stride + ki;
                            int64_t iw = ow * stride + kj;
                            if (ih < ctx.in_h && iw < ctx.in_w) {
                                int64_t in_idx = ((b * ctx.channels + c) * ctx.in_h + ih) * ctx.in_w + iw;
                                acc += x_ptr[in_idx];
                            }
                        }
                    }
                    int64_t out_idx = ((b * ctx.channels + c) * ctx.out_h + oh) * ctx.out_w + ow;
                    y_ptr[out_idx] = acc * inv;
                }
            }
        }
    }
    return Y;
}

// ============================================================================
// avgpool2d backward: dX 在窗口内均分 dY
// ============================================================================
// dX 全零，窗口内每个位置 += dY / k²（平均梯度均分回窗口）。
Tensor avgpool2d_backward(const AvgPoolContext& ctx, const Tensor& dY) {
    Tensor dX({ctx.batch, ctx.channels, ctx.in_h, ctx.in_w});
    const float* dy_ptr = dY.data();
    float* dx_ptr = dX.data();
    const float inv = 1.0f / (float)(ctx.kernel * ctx.kernel);

    #pragma omp parallel for schedule(guided) collapse(2)
    for (int64_t b = 0; b < ctx.batch; ++b) {
        for (int64_t c = 0; c < ctx.channels; ++c) {
            for (int64_t oh = 0; oh < ctx.out_h; ++oh) {
                for (int64_t ow = 0; ow < ctx.out_w; ++ow) {
                    int64_t out_idx = ((b * ctx.channels + c) * ctx.out_h + oh) * ctx.out_w + ow;
                    const float g = dy_ptr[out_idx] * inv;
                    for (int64_t ki = 0; ki < ctx.kernel; ++ki) {
                        for (int64_t kj = 0; kj < ctx.kernel; ++kj) {
                            int64_t ih = oh * ctx.stride + ki;
                            int64_t iw = ow * ctx.stride + kj;
                            if (ih < ctx.in_h && iw < ctx.in_w) {
                                int64_t in_idx = ((b * ctx.channels + c) * ctx.in_h + ih) * ctx.in_w + iw;
                                dx_ptr[in_idx] += g;
                            }
                        }
                    }
                }
            }
        }
    }
    return dX;
}

// ============================================================================
// 线性 STE: Y = deq(Q(X @ W^T)) + b
// ============================================================================
// 与 linear_forward 的区别：在 matmul 结果上做量化+反量化。
// backward 与 linear_backward 完全相同（STE 直通）。
Tensor linear_forward_ste(const Tensor& X, const Tensor& W, const Tensor& b,
                          const QuantConfig& qcfg) {
    // 先做标准 float32 linear_forward
    Tensor Y_fp = linear_forward(X, W, b);

    // 对 Y_fp 做量化+反量化（STE 前向模拟量化推理）
    Tensor Y({Y_fp.shape()[0], Y_fp.shape()[1]});
    float scale = compute_scale(Y_fp.data(), Y_fp.numel(), qcfg.bits);
    quantize_dequantize(Y_fp.data(), Y.data(), Y_fp.numel(), scale, qcfg.bits, qcfg.clip_sigma);
    return Y;
}

// ============================================================================
// conv2d STE: Y = deq(Q(im2col(X) @ W_col)) + b
// ============================================================================
Tensor conv2d_forward_ste(
    const Tensor& X, const Tensor& W, const Tensor& b,
    int stride, int padding,
    Conv2DContext& ctx,
    const QuantConfig& qcfg
) {
    // 先做标准 float32 conv2d_forward
    Tensor Y = conv2d_forward(X, W, b, stride, padding, ctx);

    // 对 Y 做量化+反量化（STE 前向模拟量化推理）
    size_t n = Y.numel();
    float scale = compute_scale(Y.data(), n, qcfg.bits);
    Tensor Y_ste({ctx.batch, ctx.out_channels, ctx.out_h, ctx.out_w});
    quantize_dequantize(Y.data(), Y_ste.data(), n, scale, qcfg.bits, qcfg.clip_sigma);
    return Y_ste;
}

// ============================================================================
// conv2d + relu 融合 forward
// ============================================================================
// 步骤：
//   1. conv2d_forward → Y_conv（(B, C_out, H_out, W_out)）
//   2. relu: Y_relu = max(0, Y_conv)，同时记录 mask
// 返回 Y_relu，ctx 中保存 conv2d context 和 relu mask
Tensor conv2d_relu_forward(
    const Tensor& X, const Tensor& W, const Tensor& b,
    int stride, int padding,
    Conv2DReluContext& ctx
) {
    // 计算输出 spatial 尺寸
    (void)X;  // batch/in_c 在该变体中未使用（L1 清理 2026-09-07）
    int64_t in_h = X.shape()[2];
    int64_t in_w = X.shape()[3];
    int64_t kh = W.shape()[2];
    int64_t kw = W.shape()[3];
    int64_t out_h = (in_h + 2 * padding - kh) / stride + 1;
    int64_t out_w = (in_w + 2 * padding - kw) / stride + 1;
    int64_t spatial = out_h * out_w;

    // 阈值门控：小特征图回退到分离算子
    if (spatial < FUSION_SPATIAL_THRESHOLD) {
        // 小特征图：使用分离的 conv2d forward，但保留 relu mask 用于融合 backward
        Tensor Y_conv = conv2d_forward(X, W, b, stride, padding, ctx.conv_ctx);
        // 仍然记录 relu mask（开销很小）
        size_t n = Y_conv.numel();
        ctx.relu_mask = Tensor(Y_conv.shape());
        Tensor Y_relu(Y_conv.shape());
        const float* yc_ptr = Y_conv.data();
        float* mask_ptr = ctx.relu_mask.data();
        float* yr_ptr = Y_relu.data();
        #pragma omp parallel for schedule(guided)
        for (size_t i = 0; i < n; ++i) {
            float val = yc_ptr[i];
            bool active = val > 0.0f;
            mask_ptr[i] = active ? 1.0f : 0.0f;
            yr_ptr[i] = active ? val : 0.0f;
        }
        return Y_relu;
    }

    // 1. conv2d forward
    Tensor Y_conv = conv2d_forward(X, W, b, stride, padding, ctx.conv_ctx);

    // 2. relu forward + mask
    size_t n = Y_conv.numel();
    ctx.relu_mask = Tensor(Y_conv.shape());
    Tensor Y_relu(Y_conv.shape());
    const float* yc_ptr = Y_conv.data();
    float* mask_ptr = ctx.relu_mask.data();
    float* yr_ptr = Y_relu.data();

    #pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < n; ++i) {
        float val = yc_ptr[i];
        bool active = val > 0.0f;
        mask_ptr[i] = active ? 1.0f : 0.0f;
        yr_ptr[i] = active ? val : 0.0f;
    }
    return Y_relu;
}

// ============================================================================
// conv2d + relu 融合 backward
// ============================================================================
// 在 col2im 步骤中同时应用 relu 梯度 mask。
// 等价于：先做 conv2d_backward 得到 dX_conv，再做 relu_backward: dX = dX_conv * mask
// 但省去了 dX_conv 的写回-读取往返。
//
// 数学：
//   dX_conv 的每个元素通过 col2im 累加得到：dx_ptr[idx] += dxcol_ptr[row * BS + col_idx]
//   融合后：dx_ptr[idx] += dxcol_ptr[row * BS + col_idx] * relu_mask[idx]
// 注意：relu_mask 是前向 relu 时 Y_conv > 0 的 mask，shape 与 dY 相同 (B, C_out, H_out, W_out)
//       但 dX 的 shape 是 (B, C_in, H, W)，mask 需要对应到 dX 的每个位置。
//       实际上，relu 的 mask 是逐元素应用在 dX_conv 上的，即先做完 col2im 得到 dX_conv，
//       再乘以 mask。但 dX_conv 的 shape 与 X 相同，mask 的 shape 与 Y_conv 相同，两者不同。
//
// 正确做法：先做标准 conv2d_backward 得到 dX_conv，然后对 dX_conv 做 relu_backward。
// 因为 relu 的输入是 conv2d 的输出 Y_conv，不是 X。
// 所以本融合不能直接修改 col2im 步骤，而是：
//   1. conv2d_backward → dX_conv, dW, db
//   2. 对 dY（即 relu 的 grad_output）先做 relu_backward 得到 dY_conv
//   3. 用 dY_conv 做 conv2d_backward
//
// 即：融合方向是"先 relu backward，再 conv2d backward"，而非"先 conv2d backward，再 relu backward"。
// 这是因为链式法则：∂L/∂X = ∂L/∂Y_relu * ∂Y_relu/∂Y_conv * ∂Y_conv/∂X
//                      = (dY * mask) * ∂Y_conv/∂X
// 所以只需在 conv2d_backward 之前对 dY 应用 relu mask。
std::tuple<Tensor, Tensor, Tensor> conv2d_relu_backward(
    const Conv2DReluContext& ctx, const Tensor& /*X*/, const Tensor& W,
    const Tensor& dY
) {
    // 先对 dY 应用 relu 梯度 mask：dY_conv = dY * mask
    // 这样就等于先做了 relu backward，再做 conv2d backward
    int64_t spatial = ctx.conv_ctx.out_h * ctx.conv_ctx.out_w;
    int64_t batch_spatial = ctx.conv_ctx.batch * spatial;

    Tensor dY_conv = BufferPool::acquire({ctx.conv_ctx.out_channels, batch_spatial});
    const float* dy_ptr = dY.data();
    const float* mask_ptr = ctx.relu_mask.data();
    float* dyc_ptr = dY_conv.data();

    #pragma omp parallel for schedule(guided) collapse(2)
    for (int64_t b_idx = 0; b_idx < ctx.conv_ctx.batch; ++b_idx) {
        for (int64_t out_c = 0; out_c < ctx.conv_ctx.out_channels; ++out_c) {
            for (int64_t oc = 0; oc < spatial; ++oc) {
                size_t src_idx = (b_idx * ctx.conv_ctx.out_channels + out_c) * spatial + oc;
                dyc_ptr[out_c * batch_spatial + b_idx * spatial + oc] =
                    dy_ptr[src_idx] * mask_ptr[src_idx];
            }
        }
    }

    // 然后做标准 conv2d_backward（但使用 dY_conv 替代 dY）
    // 复用 conv2d_backward 的核心逻辑，但跳过 dY→dY_col 步骤（已在上面完成）
    int64_t in_c_kh_kw = ctx.conv_ctx.in_channels * ctx.conv_ctx.kh * ctx.conv_ctx.kw;

    // dW
    Tensor W_col = W.reshape({ctx.conv_ctx.out_channels, in_c_kh_kw});
    Tensor dW_col = dW_from_dYcol_xcol(dY_conv, ctx.conv_ctx.x_col);
    Tensor dW = dW_col.reshape({ctx.conv_ctx.out_channels, ctx.conv_ctx.in_channels,
                                 ctx.conv_ctx.kh, ctx.conv_ctx.kw});

    // db = sum over (batch, spatial) per out_c（通道化：经 conv2d_registry() 调 db 端口）
    const Conv2dKernelSet& cks = conv2d_registry();
    Tensor db({ctx.conv_ctx.out_channels});
    cks.db(dyc_ptr, db.data(), ctx.conv_ctx.out_channels, batch_spatial);

    // dx_col = W_col^T @ dY_conv（节奏 3 P0-A：经 conv2d_registry() 调 dx 端口）
    Tensor dx_col({in_c_kh_kw, batch_spatial});
    cks.dx(W_col.data(), dY_conv.data(), dx_col.data(),
           in_c_kh_kw, ctx.conv_ctx.out_channels, batch_spatial);
    const float* dxcol_ptr = dx_col.data();

    // dX: col2im
    // 节奏 3 P0-B：改经 conv2d_registry() 调 col2im 端口（行条带法）——
    // 内层 ow 连续读写，div/mod 上提；对固定目标元素累加序与旧 gather 版逐位一致。
    Tensor dX({ctx.conv_ctx.batch, ctx.conv_ctx.in_channels, ctx.conv_ctx.in_h, ctx.conv_ctx.in_w});
    float* dx_ptr = dX.data();
    const int64_t pad = ctx.conv_ctx.padding;
    const int64_t padded_h = ctx.conv_ctx.in_h + 2 * pad;
    const int64_t padded_w = ctx.conv_ctx.in_w + 2 * pad;
    Tensor dX_pad;
    float* buf_ptr;
    if (pad != 0) {
        dX_pad = Tensor({ctx.conv_ctx.batch, ctx.conv_ctx.in_channels, padded_h, padded_w});
        buf_ptr = dX_pad.data();
    } else {
        buf_ptr = dx_ptr;
    }
    cks.col2im(dxcol_ptr, buf_ptr, ctx.conv_ctx.batch, ctx.conv_ctx.in_channels,
               ctx.conv_ctx.out_h, ctx.conv_ctx.out_w, batch_spatial,
               ctx.conv_ctx.kh, ctx.conv_ctx.kw, ctx.conv_ctx.stride, pad,
               padded_h, padded_w);
    // 拷贝有效区域回 dX（仅 pad!=0）
    if (pad != 0) {
        #pragma omp parallel for schedule(guided)
        for (int64_t b_idx = 0; b_idx < ctx.conv_ctx.batch; ++b_idx) {
            for (int64_t ic = 0; ic < ctx.conv_ctx.in_channels; ++ic) {
                const float* src =
                    buf_ptr + ((b_idx * ctx.conv_ctx.in_channels + ic) * padded_h + pad) * padded_w + pad;
                float* dst = dx_ptr + ((b_idx * ctx.conv_ctx.in_channels + ic) * ctx.conv_ctx.in_h) * ctx.conv_ctx.in_w;
                for (int64_t h = 0; h < ctx.conv_ctx.in_h; ++h) {
                    std::memcpy(dst + h * ctx.conv_ctx.in_w, src + h * padded_w,
                                ctx.conv_ctx.in_w * sizeof(float));
                }
            }
        }
    }

    BufferPool::release(std::move(dY_conv));
    BufferPool::release(std::move(dx_col));

    return {dX, dW, db};
}

}  // namespace sgn_autograd
