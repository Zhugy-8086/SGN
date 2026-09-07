// pair_grad_carrier.cpp - int8 对梯度载体实现（见 pair_grad_carrier.h 头注）
#include "pair_grad_carrier.h"

#include <cmath>
#include <stdexcept>

#include "mkern/simd/simd_api.h"
// SR 共享内核：SR 循环体唯一实现点在 autograd/sr_kernel.cpp（2026-09-02
// 审查 F5 提取），本文件与 backward_strategy.h 的 sr_quantize_grad 共享同一
// 内核——同输入同种子下 pair 化的 q 与现有 SR float 路径逐位一致由结构保证。
#include "../autograd/backward_strategy.h"

namespace sgn_msint {

// 共享 SR 内核（autograd/sr_kernel.cpp 唯一实现点，见 sr_quantize_to_pair 注释）
using sgn_autograd::sr_quantize_q;

void sr_quantize_to_pair(const float* g, size_t n, float clip_sigma,
                         PairGradCarrier& out) {
    // 结构（2026-09-02 审查 F5 二期）：SR 循环体已提取为共享内核
    // sgn_autograd::sr_quantize_q（autograd/sr_kernel.cpp 唯一实现点），
    // 本函数为薄壳——与 backward_strategy.h sr_quantize_grad 共享同一内核，
    // RNG 序列对齐由结构保证（原"两处手写循环改动不同步"风险已消除）。
    if (n == 0) {
        out = PairGradCarrier{};
        return;
    }
    // per-tensor scale（与 sr_quantize_grad / compute_scale 一致，Q16 = 32767）
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float a = std::fabs(g[i]);
        if (a > max_abs) max_abs = a;
    }
    out.n = n;
    out.h.resize(n);
    out.l.resize(n);
    if (max_abs < 1e-12f) {
        // 全零（零邻域）：q=0 → h=128, l=0。与 float 路径 sr_quantize_grad
        // early-return（保持原值 ~1e-14）为已声明语义分叉（见头注），RNG 均 0 消耗。
        out.scale = 1.0f;
        for (size_t i = 0; i < n; ++i) { out.h[i] = 128; out.l[i] = 0; }
        return;
    }
    const float max_val = 32767.0f;
    const float scale = max_abs / max_val;
    out.scale = scale;

    // 共享内核出 q（同 SRNG 同舍入序列，每元素恰 1 个随机数）
    std::vector<int32_t> q(n);
    sgn_autograd::sr_quantize_q(g, n, scale, clip_sigma, max_val, q.data());

    // pair 编码（U 方案）：q ∈ [-32767, 32767]，
    // h_s = q >> 8（算术右移）∈ [-128, 127]，h_u8 = h_s + 128 ∈ [0, 255]
    for (size_t i = 0; i < n; ++i) {
        const int32_t h_s = q[i] >> 8;
        out.h[i] = static_cast<uint8_t>(h_s + 128);
        out.l[i] = static_cast<uint8_t>(q[i] & 0xFF);
    }
}

void decode_pair_fine_f32(const PairGradCarrier& p, float* out) {
    for (size_t i = 0; i < p.n; ++i) {
        out[i] = static_cast<float>(pair_decode_q(p, i)) * p.scale;
    }
}

void decode_pair_coarse_f32(const PairGradCarrier& p, float* out) {
    for (size_t i = 0; i < p.n; ++i) {
        const int32_t h_s = static_cast<int32_t>(p.h[i]) - 128;
        out[i] = static_cast<float>(256 * h_s) * p.scale;
    }
}

int64_t pair_dot_fine(const PairGradCarrier& p, const int8_t* w8) {
    if (p.n == 0) return 0;
    // Sw = Σ w8（int64，摊销一次；dot8 的 i8 侧直接用 w8）
    int64_t sw = 0;
    for (size_t i = 0; i < p.n; ++i) sw += w8[i];
    // §七 U 方案 fine 恒等式：Σq·w = 256·dot8(h_u8,w8) + dot8(l_u8,w8) − 32768·Sw
    const int64_t d_h = sgn::simd::dot8(p.h.data(), w8, p.n);
    const int64_t d_l = sgn::simd::dot8(p.l.data(), w8, p.n);
    return 256 * d_h + d_l - 32768 * sw;
}

int64_t pair_dot_coarse(const PairGradCarrier& p, const int8_t* w8) {
    if (p.n == 0) return 0;
    int64_t sw = 0;
    for (size_t i = 0; i < p.n; ++i) sw += w8[i];
    // §七 coarse 恒等式：Σq_c·w = 256·(dot8(h_u8,w8) − 128·Sw)
    const int64_t d_h = sgn::simd::dot8(p.h.data(), w8, p.n);
    return 256 * (d_h - 128 * sw);
}

}  // namespace sgn_msint
