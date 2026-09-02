// avx2_reduce.cpp - AVX2 float 归约/累加原语
//
// 背景：fixes_相关修复/simd指令集加速文件拆分计划_2026_08_29.md + 综合执行计划
//   （simd_dot8_avx2中间路径立项_2026_09_02.md §二.5 拆分）。本文件为 avx2.cpp
//   按原语域拆分的 float 归约域；dot 域在 avx2_dot.cpp、decode 域在 avx2_decode.cpp。
//   逐字迁移自 bn_forward_train / bn_backward_train / accumulate_grad，
//   浮点累加顺序与迁移前完全一致（kRounding，见 simd_api.h 契约说明）。
//
// 编译：CMake 对本文件加 -mavx2 -mfma（sum 系用 _mm256_fmadd 可选，本实现用 add/mul）。
// 仅 __AVX2__ 下编译；非 AVX2 平台由 simd/scalar.cpp 提供标量锚点。

#include "simd/simd_api.h"

#if defined(__AVX2__)
#include <immintrin.h>

namespace sgn::simd {
namespace {
// AVX2 水平归约（8×float → float），与 ops_nn.cpp 的 bn_hadd256 同序。
inline float hsum256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
} // namespace

float sum_f32_avx2(const float* p, int64_t n, int64_t stride) {
    __m256 sum256 = _mm256_setzero_ps();
    int64_t i = 0;
    if (stride == 1) {
        for (; i + 7 < n; i += 8) {
            sum256 = _mm256_add_ps(sum256, _mm256_loadu_ps(p + i));
        }
    } else {
        for (; i + 7 < n; i += 8) {
            __m256 v = _mm256_set_ps(
                p[(i + 7) * stride], p[(i + 6) * stride], p[(i + 5) * stride],
                p[(i + 4) * stride], p[(i + 3) * stride], p[(i + 2) * stride],
                p[(i + 1) * stride], p[(i + 0) * stride]);
            sum256 = _mm256_add_ps(sum256, v);
        }
    }
    float sum = hsum256(sum256);
    for (; i < n; ++i) sum += p[i * stride];
    return sum;
}

float sum_sq_dev_f32_avx2(const float* p, int64_t n, int64_t stride, float mu) {
    __m256 mu256 = _mm256_set1_ps(mu);
    __m256 sum_sq256 = _mm256_setzero_ps();
    int64_t i = 0;
    if (stride == 1) {
        for (; i + 7 < n; i += 8) {
            __m256 x = _mm256_loadu_ps(p + i);
            __m256 d = _mm256_sub_ps(x, mu256);
            sum_sq256 = _mm256_add_ps(sum_sq256, _mm256_mul_ps(d, d));
        }
    } else {
        for (; i + 7 < n; i += 8) {
            __m256 x = _mm256_set_ps(
                p[(i + 7) * stride], p[(i + 6) * stride], p[(i + 5) * stride],
                p[(i + 4) * stride], p[(i + 3) * stride], p[(i + 2) * stride],
                p[(i + 1) * stride], p[(i + 0) * stride]);
            __m256 d = _mm256_sub_ps(x, mu256);
            sum_sq256 = _mm256_add_ps(sum_sq256, _mm256_mul_ps(d, d));
        }
    }
    float sum_sq = hsum256(sum_sq256);
    for (; i < n; ++i) {
        float d = p[i * stride] - mu;
        sum_sq += d * d;
    }
    return sum_sq;
}

void sum_sumprod_f32_avx2(const float* a, const float* b, int64_t n, int64_t stride,
                     float* out_sum, float* out_sumprod) {
    __m256 sd256 = _mm256_setzero_ps();
    __m256 sdp256 = _mm256_setzero_ps();
    int64_t i = 0;
    if (stride == 1) {
        for (; i + 7 < n; i += 8) {
            __m256 av = _mm256_loadu_ps(a + i);
            __m256 bv = _mm256_loadu_ps(b + i);
            sd256 = _mm256_add_ps(sd256, av);
            sdp256 = _mm256_add_ps(sdp256, _mm256_mul_ps(av, bv));
        }
    } else {
        for (; i + 7 < n; i += 8) {
            __m256 av = _mm256_set_ps(
                a[(i + 7) * stride], a[(i + 6) * stride], a[(i + 5) * stride],
                a[(i + 4) * stride], a[(i + 3) * stride], a[(i + 2) * stride],
                a[(i + 1) * stride], a[(i + 0) * stride]);
            __m256 bv = _mm256_set_ps(
                b[(i + 7) * stride], b[(i + 6) * stride], b[(i + 5) * stride],
                b[(i + 4) * stride], b[(i + 3) * stride], b[(i + 2) * stride],
                b[(i + 1) * stride], b[(i + 0) * stride]);
            sd256 = _mm256_add_ps(sd256, av);
            sdp256 = _mm256_add_ps(sdp256, _mm256_mul_ps(av, bv));
        }
    }
    float sd = hsum256(sd256);
    float sdp = hsum256(sdp256);
    for (; i < n; ++i) {
        sd += a[i * stride];
        sdp += a[i * stride] * b[i * stride];
    }
    *out_sum = sd;
    *out_sumprod = sdp;
}

void accum_f32_avx2(float* dst, const float* src, int64_t n) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 s = _mm256_loadu_ps(&src[i]);
        __m256 d = _mm256_loadu_ps(&dst[i]);
        _mm256_storeu_ps(&dst[i], _mm256_add_ps(d, s));
    }
    for (; i < n; ++i) {
        dst[i] += src[i];
    }
}

} // namespace sgn::simd
#endif