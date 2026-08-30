// avx2.cpp - AVX2/SSSE3 原语实现（__AVX2__）
//
// 背景：fixes_相关修复/simd指令集加速文件拆分计划_2026_08_29.md
//   - Step 1：sgn::simd::dot16 由 msint/split_dot.cpp 的 dot16_avx2 / dot16_body 迁移。
//   - Step 2：sgn::simd::decode_i16_f32 由 msint/packed_backend_bindings.cpp 的
//             decode_to_float_core 快速路径 1（SSSE3 + 原始两条子路径）迁移。
//
// 注意：当前构建全局 -mavx2（CMakeLists.txt），本文件仅在 __AVX2__ 下编译。
// 非 AVX2 平台（ARM/RISC-V）由 simd/scalar.cpp 提供标量锚点，无重复符号。

#include "simd/simd_api.h"

#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>

namespace sgn::simd {
namespace {
// AVX2 窄精度点积：int16[K] × int16[K] → int64 精确累加。
// 每 16 元素一次迭代：先符号扩展到 int32，再用 _mm256_mul_epi32
// （signed 32×32→64，vpmuldq）分偶/奇 lane 得到 8 个 int64 乘积，累加到 4×int64。
// 注意：不能用 _mm256_madd_epi16——它会将相邻 2 个 int32 乘积相加，
// 当相邻元素均为 -32768（对应低位无符号部分 u=0 的边界值）时，pair 和为 2^31，
// 恰好溢出 int32、丢失 2^32，破坏 bit-exact。
// 整数加法可交换/结合，累加顺序不影响结果，与标量路径 bit-exact 一致。
// 16 宽点积主体（主循环与零填充尾部共用；整数加法可交换/结合，累加顺序不影响结果，
// 与标量路径 bit-exact 一致）。
inline __m256i dot16_body(__m256i acc, const int16_t* a, const int16_t* b) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));  // 16 int16
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
    // 16 int16 → 低/高各 8 个 int32
    __m256i va_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(va));
    __m256i va_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(va, 1));
    __m256i vb_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vb));
    __m256i vb_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vb, 1));
    // 偶 lane（0,2,4,6）+ 奇 lane（1,3,5,7）的 int64 乘积
    acc = _mm256_add_epi64(acc, _mm256_mul_epi32(va_lo, vb_lo));
    acc = _mm256_add_epi64(acc, _mm256_mul_epi32(_mm256_srli_epi64(va_lo, 32),
                                                 _mm256_srli_epi64(vb_lo, 32)));
    acc = _mm256_add_epi64(acc, _mm256_mul_epi32(va_hi, vb_hi));
    acc = _mm256_add_epi64(acc, _mm256_mul_epi32(_mm256_srli_epi64(va_hi, 32),
                                                 _mm256_srli_epi64(vb_hi, 32)));
    return acc;
}
} // namespace

int64_t dot16_avx2(const int16_t* a, const int16_t* b, size_t K) {
    __m256i acc = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 16 <= K; i += 16) {
        acc = dot16_body(acc, a + i, b + i);
    }
    if (i < K) {
        // K 尾部不足 16：零填充做一次 16 宽 SIMD（bit-exact——填充 0 的乘积为 0，
        // 不改变累加；元素先拷入本地缓冲，无越界读）。
        alignas(32) int16_t pa[16] = {0}, pb[16] = {0};
        const size_t t = K - i;
        std::memcpy(pa, a + i, t * sizeof(int16_t));
        std::memcpy(pb, b + i, t * sizeof(int16_t));
        acc = dot16_body(acc, pa, pb);
    }
    int64_t v[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(v), acc);
    return v[0] + v[1] + v[2] + v[3];
}

void decode_i16_f32_avx2(const uint64_t* pv_ptr, int n_values, float scale, float* res_ptr) {
    // 源：msint/packed_backend_bindings.cpp decode_to_float_core 快速路径 1
    //（backward_int16 8+8 schema：每个 uint64 低 16 位 = 1 个 int16，高 48 位无效）。
    // 两条子路径 + 标量尾部，与迁移前逻辑逐字一致（float 乘 scale 用 mul 不融合）。
    __m256 scale_vec = _mm256_set1_ps(scale);
    int i = 0;
#if defined(__SSSE3__)
    // SSSE3 路径：PSHUFB + PALIGNR + PMOVSXWD（低延迟字节级操作）
    // 每 lane 有 2 个 uint64，各含 2 字节有效数据（d0@[0,1], d1@[8,9]）：
    //   lane0 → 输出 [12,13,14,15] = [d0,d1]，lane1 → 输出 [0,1,2,3] = [d2,d3]
    const __m256i pack_mask = _mm256_setr_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 8, 9,
         0,  1,  8,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1,-1,-1
    );
    for (; i + 7 < n_values; i += 8) {
        __m256i v0 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(pv_ptr + i));
        __m256i v1 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(pv_ptr + i + 4));

        // PSHUFB 打包：将稀疏的 int16 值压缩到 lane 边界
        __m256i v0_packed = _mm256_shuffle_epi8(v0, pack_mask);
        __m256i v1_packed = _mm256_shuffle_epi8(v1, pack_mask);

        // 提取 128-bit lane，PALIGNR 拼接跨 lane 数据
        __m128i v0_lo = _mm256_extracti128_si256(v0_packed, 0);
        __m128i v0_hi = _mm256_extracti128_si256(v0_packed, 1);
        __m128i v1_lo = _mm256_extracti128_si256(v1_packed, 0);
        __m128i v1_hi = _mm256_extracti128_si256(v1_packed, 1);

        // PALIGNR(v_hi, v_lo, 12): (v_lo || v_hi) >> 12 → [d0, d1, d2, d3, 0x00*12]
        __m128i r0 = _mm_alignr_epi8(v0_hi, v0_lo, 12);
        __m128i r1 = _mm_alignr_epi8(v1_hi, v1_lo, 12);

        // PMOVSXWD 符号扩展：4 个 int16 → 4 个 int32
        __m256i sx0 = _mm256_cvtepi16_epi32(r0);
        __m256i sx1 = _mm256_cvtepi16_epi32(r1);

        // 合并：[d0..d7] int32 → float32 → × scale
        __m256i combined = _mm256_permute2x128_si256(sx0, sx1, 0x20);
        __m256 floats = _mm256_cvtepi32_ps(combined);
        __m256 result_vec = _mm256_mul_ps(floats, scale_vec);
        _mm256_storeu_ps(&res_ptr[i], result_vec);
    }
#else
    // 原始路径：permutevar8x32 + permute2f128 + slli/srai（通用 32 位级操作，
    // 在 __AVX2__ 无 __SSSE3__ 的极端组合下作为回退，语义等价）
    __m256i permute_idx = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
    for (; i + 7 < n_values; i += 8) {
        __m256i v0 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(pv_ptr + i));
        __m256i v1 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(pv_ptr + i + 4));
        // 提取有效 32 位元素（偶数索引）
        __m256i p0 = _mm256_permutevar8x32_epi32(v0, permute_idx);
        __m256i p1 = _mm256_permutevar8x32_epi32(v1, permute_idx);
        // 合并低 128 位 → 256 位 [d0..d7]
        __m256i combined = _mm256_permute2f128_si256(p0, p1, 0x20);
        // 符号扩展 int16 → int32（slli 移出高位，srai 符号扩展 bit15）
        __m256i shifted = _mm256_slli_epi32(combined, 16);
        __m256i sign_extended = _mm256_srai_epi32(shifted, 16);
        // int32 → float32 → × scale
        __m256 floats = _mm256_cvtepi32_ps(sign_extended);
        __m256 result_vec = _mm256_mul_ps(floats, scale_vec);
        _mm256_storeu_ps(&res_ptr[i], result_vec);
    }
#endif
    // 标量尾部（两条子路径共用）
    for (; i < n_values; ++i) {
        int16_t val16 = static_cast<int16_t>(pv_ptr[i] & 0xFFFF);
        res_ptr[i] = static_cast<float>(val16) * scale;
    }
}

// ---- float 归约/累加原语（Step 3；逐字迁移自 bn_forward_train / bn_backward_train /
// accumulate_grad，浮点累加顺序与迁移前完全一致）----

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
