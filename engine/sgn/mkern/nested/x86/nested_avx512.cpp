// nested_avx512.cpp - mkern/nested AVX512 后端（F+DQ，quant/dequant 8 宽）
//
// 背景：nested_quant设计档_2026_09_05.md §五.3/4。与 avx2 版同构（位等同性论证
// 见 nested_avx2.cpp 头注释），全部换原生指令：
//   - _mm512_mullo_epi64（DQ）：64 位乘法原生；
//   - _mm512_srai_epi64（F）：64 位算术移位原生；
//   - _mm512_cvtepu64_pd / _mm512_cvtepi64_pd（DQ）：正确舍入整型转换，
//     与标量 (double) 逐位一致（53 位 U 精确；任意 int64 q 正确舍入）；
//   - _mm512_cvttpd_epi64（DQ）：饱和 lane 的 indefinite 值被掩码混合覆盖。
// 值重建同序：(double)q → ·2^m（指数平移，精确）→ ·u（一次舍入）= 标量 ldexp·u。
//
// 尾部：本地标量循环用全局索引（计数器式 RNG，与向量体/锚点逐位一致）；
// dequant 尾直接消费标量锚点（无索引依赖）。
//
// 编译：CMake 对本文件加 -mavx512f -mavx512dq（per-file ISA 纪律）；
// 运行时 dispatch 按 CPUID（AVX512F+DQ）选入。

#include "mkern/nested/nested_api.h"

#include <cmath>

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#include <immintrin.h>

namespace sgn::mkern::nested {

namespace {

// 量化尾部（全局索引；与 scalar.cpp contract_uniform 同式）
inline void quant_tail(int64_t* code, const float* h, int64_t cnt, int64_t i0,
                       double ud, uint64_t seed) {
    for (int64_t k = 0; k < cnt; ++k) {
        const double x = static_cast<double>(h[k]) / ud;
        int64_t I;
        if (x >= 2147483647.0)      I = 2147483647;
        else if (x <= -2147483648.0) I = -2147483648;
        else {
            const double fl = std::floor(x);
            I = static_cast<int64_t>(fl);
            uint64_t z = seed + 0x9E3779B97F4A7C15ULL *
                         static_cast<uint64_t>(i0 + k);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            z ^= z >> 31;
            if (static_cast<double>(z >> 11) * 0x1.0p-53 < x - fl) ++I;
        }
        code[k] = I;
    }
}

} // namespace

void nested_quant_i32_avx512(int64_t* code, const float* h, int64_t n,
                             float u, uint64_t seed) {
    const double ud = static_cast<double>(u);
    const __m512d vd = _mm512_set1_pd(ud);
    const __m512d sat_hi = _mm512_set1_pd(2147483647.0);
    const __m512d sat_lo = _mm512_set1_pd(-2147483648.0);
    const __m512i golden = _mm512_set1_epi64(static_cast<int64_t>(
        0x9E3779B97F4A7C15ULL));
    const __m512i c1 = _mm512_set1_epi64(static_cast<int64_t>(
        0xBF58476D1CE4E5B9ULL));
    const __m512i c2 = _mm512_set1_epi64(static_cast<int64_t>(
        0x94D049BB133111EBULL));
    const __m512i idx03 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    const __m512i seedv = _mm512_set1_epi64(static_cast<int64_t>(seed));
    const __m512i one = _mm512_set1_epi64(1);
    const __m512i imax = _mm512_set1_epi64(2147483647);
    const __m512i imin = _mm512_set1_epi64(-2147483647 - 1);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m512d x = _mm512_div_pd(
            _mm512_cvtps_pd(_mm256_loadu_ps(h + i)), vd);
        const __mmask8 m_hi = _mm512_cmp_pd_mask(x, sat_hi, _CMP_GE_OQ);
        const __mmask8 m_lo = _mm512_cmp_pd_mask(x, sat_lo, _CMP_LE_OQ);
        const __m512d xf = _mm512_floor_pd(x);
        const __m512d frac = _mm512_sub_pd(x, xf);
        // RNG：z = seed + golden·(i+k)，三轮终混（与规格 a 位等同）
        __m512i z = _mm512_add_epi64(
            seedv, _mm512_mullo_epi64(golden, _mm512_add_epi64(idx03,
                                                               _mm512_set1_epi64(i))));
        z = _mm512_xor_si512(z, _mm512_srli_epi64(z, 30));
        z = _mm512_mullo_epi64(z, c1);
        z = _mm512_xor_si512(z, _mm512_srli_epi64(z, 27));
        z = _mm512_mullo_epi64(z, c2);
        z = _mm512_xor_si512(z, _mm512_srli_epi64(z, 31));
        const __m512d Ud = _mm512_mul_pd(_mm512_cvtepu64_pd(
                                             _mm512_srli_epi64(z, 11)),
                                         _mm512_set1_pd(0x1.0p-53));
        const __mmask8 inc = _mm512_cmp_pd_mask(Ud, frac, _CMP_LT_OQ);
        __m512i I = _mm512_cvttpd_epi64(xf);
        I = _mm512_mask_add_epi64(I, inc, I, one);
        I = _mm512_mask_mov_epi64(I, m_hi, imax);
        I = _mm512_mask_mov_epi64(I, m_lo, imin);
        _mm512_storeu_si512(reinterpret_cast<void*>(code + i), I);
    }
    quant_tail(code + i, h + i, n - i, i, ud, seed);
}

void nested_dequant_avx512(float* out, const int64_t* code, int64_t n,
                           float u, int level) {
    if (level != 4 && level != 8 && level != 16 && level != 32) return;
    const int m = 32 - level;
    const double ud = static_cast<double>(u);
    const __m512d vud = _mm512_set1_pd(ud);
    const __m512d vpow2m = _mm512_set1_pd(std::ldexp(1.0, m));
    const __m512i one = _mm512_set1_epi64(1);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m512i I = _mm512_loadu_si512(
            reinterpret_cast<const void*>(code + i));
        __m512i q;
        if (m == 0) {
            q = I;                                             // level 32 恒等
        } else {
            q = _mm512_srai_epi64(I, m);                       // floor(I/2^m)
            const __m512i r = _mm512_sub_epi64(I, _mm512_slli_epi64(q, m));
            const __m512i half = _mm512_set1_epi64(int64_t(1) << (m - 1));
            const __mmask8 gt = _mm512_cmpgt_epi64_mask(r, half);
            const __mmask8 eq = _mm512_cmpeq_epi64_mask(r, half);
            const __mmask8 odd = _mm512_test_epi64_mask(q, one);
            const __mmask8 adj = static_cast<__mmask8>(gt | (eq & odd));
            q = _mm512_mask_add_epi64(q, adj, q, one);         // half-to-even
        }
        // ldexp((double)q, m)·u：cvtqq2pd 正确舍入 → 指数平移 → ·u
        const __m512d val = _mm512_mul_pd(
            _mm512_mul_pd(_mm512_cvtepi64_pd(q), vpow2m), vud);
        _mm256_storeu_ps(out + i, _mm512_cvtpd_ps(val));
    }
    nested_dequant_scalar(out + i, code + i, n - i, u, level);
}

} // namespace sgn::mkern::nested

#endif  // __AVX512F__ && __AVX512DQ__
