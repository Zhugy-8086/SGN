// avx512.cpp - AVX-512 原语实现（服务器加速，P1）
//
// 背景：simd服务器加速计划_2026_08_30.md P1。
// 由 simd/x86/avx2.cpp 的 256 位版本升级到 512 位，compute-bound 原语：
//   - dot16          : _mm512_mul_epi32（32 int16/迭代，16×int64 累加器），4 累加器展开
//   - sum_f32 系      : 16 路浮点累加 + 归约（stride=1 连续 / stride>1 构建）
//   - accum_f32      : 16 路加法
// decode_i16_f32 为带宽型（非 compute-bound），512 位打包（permutexvar_epi8）逻辑复杂、
// 收益有限，本轮不实现——dispatch 层让 decode 落到 AVX2 版。
//
// 编译：本文件由 CMake 单独加 -mavx512f -mavx512bw（set_source_files_properties），
// 符号常驻编译；运行时经 __builtin_cpu_supports("avx512f") 检测选入。
//
// 数值契约：
//   - dot16 bit-exact（整数加法可交换/结合，多累加器不影响）
//   - sum 系为 kRounding（浮点累加顺序与 256 位版不同：512 位 16 路 vs 256 位 8 路，
//     与标量锚点亦非逐位一致——属预期浮点归约差异，同 avx2 的 kRounding 档位）

#include "mkern/simd/simd_api.h"

#include <cstring>

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>

namespace sgn::simd {
namespace {
// 32 宽 int16 点积主体：符号扩展到 int32，vpmuldq 偶/奇 lane 得 16×int64 累加。
// 禁 _mm512_madd_epi16：相邻元素均 -32768 时 pair 和 2^31 溢出 int32（同 256 位结论）。
inline __m512i dot32_body(__m512i acc, const int16_t* a, const int16_t* b) {
    // 32 int16 → 低/高各 16 个 int32（_mm512_cvtepi16_epi32 每次取 16 个 int16）
    __m256i va_lo16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));      // 低 16
    __m256i va_hi16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 16)); // 高 16
    __m256i vb_lo16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
    __m256i vb_hi16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 16));
    __m512i va_lo = _mm512_cvtepi16_epi32(va_lo16);
    __m512i va_hi = _mm512_cvtepi16_epi32(va_hi16);
    __m512i vb_lo = _mm512_cvtepi16_epi32(vb_lo16);
    __m512i vb_hi = _mm512_cvtepi16_epi32(vb_hi16);
    // 偶 lane（0,2,...,14）+ 奇 lane（1,3,...,15）的 int64 乘积（每半各 8 对）
    acc = _mm512_add_epi64(acc, _mm512_mul_epi32(va_lo, vb_lo));
    acc = _mm512_add_epi64(acc, _mm512_mul_epi32(_mm512_srli_epi64(va_lo, 32),
                                                 _mm512_srli_epi64(vb_lo, 32)));
    acc = _mm512_add_epi64(acc, _mm512_mul_epi32(va_hi, vb_hi));
    acc = _mm512_add_epi64(acc, _mm512_mul_epi32(_mm512_srli_epi64(va_hi, 32),
                                                 _mm512_srli_epi64(vb_hi, 32)));
    return acc;
}

// madd 快速路径（2026-08-31，simd服务器加速计划 §8：512 位 dot16 同范式恢复）。
// 用 _mm512_madd_epi16（1 指令/32 元素，相邻两对乘积相加为 int32）替代 vpmuldq。
// 溢出边界（§7 数据实验）：仅当批内存在 -32768 时可能溢出；批内无 -32768 则
// pair 和 ≤ 2×2³⁰ < 2³¹，永不溢出。故先检测 a、b 两侧是否含 -32768：
//   - 无 → madd 全速（1 指令/32 元素，理论 ~3-5× vpmuldq）
//   - 有 → 该批回退 vpmuldq（dot32_body，保持 bit-exact）
// 整数加法可交换/结合 → 混用不破坏逐位结果（只要 madd 中间不溢出，检测保证）。
inline __m512i dot32_madd(__m512i acc64, const int16_t* a, const int16_t* b) {
    __m512i va = _mm512_loadu_si512(a);  // 32 int16
    __m512i vb = _mm512_loadu_si512(b);
    // 检测两侧是否含 -32768（int16 位模式 0x8000；mask 比较，需 AVX512BW）
    const __m512i kMin = _mm512_set1_epi16(static_cast<int16_t>(-32768));
    __mmask32 m = _mm512_cmpeq_epi16_mask(va, kMin);
    m |= _mm512_cmpeq_epi16_mask(vb, kMin);
    // mask 非零 → 存在 -32768 → 回退 vpmuldq
    if (m != 0) {
        return dot32_body(acc64, a, b);
    }
    // madd：16×int32（相邻对乘积和，检测保证不溢出）→ 符号扩展为 16×int64 → 累加
    __m512i m32 = _mm512_madd_epi16(va, vb);
    __m512i lo64 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(m32));
    __m512i hi64 = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(m32, 1));
    return _mm512_add_epi64(acc64, _mm512_add_epi64(lo64, hi64));
}

// 16 个 int64 → int64 归约（两半相加）。
inline int64_t reduce16_i64(__m512i acc) {
    __m256i lo = _mm512_castsi512_si256(acc);
    __m256i hi = _mm512_extracti64x4_epi64(acc, 1);
    __m256i sum = _mm256_add_epi64(lo, hi);
    int64_t v[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(v), sum);
    return v[0] + v[1] + v[2] + v[3];
}
} // namespace

int64_t dot16_avx512(const int16_t* a, const int16_t* b, size_t K) {
    // madd 快速路径（2026-08-31，§8 同范式恢复）：安全批 madd 全速 + 危险批 vpmuldq
    // 回退；4 累加器展开。整数加法可交换/结合，与标量锚点 bit-exact 一致。
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    __m512i acc2 = _mm512_setzero_si512();
    __m512i acc3 = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 128 <= K; i += 128) {
        acc0 = dot32_madd(acc0, a + i,       b + i);
        acc1 = dot32_madd(acc1, a + i + 32,  b + i + 32);
        acc2 = dot32_madd(acc2, a + i + 64,  b + i + 64);
        acc3 = dot32_madd(acc3, a + i + 96,  b + i + 96);
    }
    for (; i + 32 <= K; i += 32) {
        acc0 = dot32_madd(acc0, a + i, b + i);
    }
    if (i < K) {
        // K 尾部不足 32：零填充做一次满宽 512 位（bit-exact——填充 0 乘积为 0，无越界读）
        alignas(64) int16_t pa[32] = {0}, pb[32] = {0};
        const size_t t = K - i;
        std::memcpy(pa, a + i, t * sizeof(int16_t));
        std::memcpy(pb, b + i, t * sizeof(int16_t));
        acc0 = dot32_madd(acc0, pa, pb);
    }
    __m512i acc = _mm512_add_epi64(_mm512_add_epi64(acc0, acc1),
                                   _mm512_add_epi64(acc2, acc3));
    return reduce16_i64(acc);
}

// ---- float 归约/累加（R1 可复现化：改为 8 路规范语义，与 scalar/avx2 逐位一致）----
// 原 16 路 + _mm512_reduce_add_ps 归约顺序与标量/avx2 不同（kRounding 跨后端漂移）；
// R1 统一为【固定 8 路 + 固定归约树 + 固定尾】（见 simd_api.h 契约），故本文件 float
// 归约用 _mm256 宽度处理（8 路连续分块），hsum256 归约树 = 标量/avx2 完全一致 → kBitExact。
// 注：sum 系为带宽/归约型，8 路 vs 16 路对峰值效率影响可忽略，复现性优先。

namespace {
// AVX2 水平归约（8×float → float），与 avx2_reduce.cpp 的 hsum256 同序 = 规范固定树。
inline float hsum256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
} // namespace

float sum_f32_avx512(const float* p, int64_t n, int64_t stride) {
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
    float sum = hsum256(sum256);  // 固定归约树
    for (; i < n; ++i) sum += p[i * stride];  // 固定标量尾
    return sum;
}

float sum_sq_dev_f32_avx512(const float* p, int64_t n, int64_t stride, float mu) {
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

void sum_sumprod_f32_avx512(const float* a, const float* b, int64_t n, int64_t stride,
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

void accum_f32_avx512(float* dst, const float* src, int64_t n) {
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 s = _mm512_loadu_ps(&src[i]);
        __m512 d = _mm512_loadu_ps(&dst[i]);
        _mm512_storeu_ps(&dst[i], _mm512_add_ps(d, s));
    }
    for (; i < n; ++i) {
        dst[i] += src[i];
    }
}

} // namespace sgn::simd
#endif
