// nested_avx2.cpp - mkern/nested AVX2 后端（quant 4×f64 / dequant 4×int64）
//
// 背景：nested_quant设计档_2026_09_05.md §五.3/4。bit-exact 契约见 nested_api.h
// 头注释（RNG/SR 判式/饱和/tie/值域）——本文件是规格的第二实现，boundary 测试
// 与标量锚点三方对拍（N1 跨后端逐位一致）。
//
// 位等同性要点（每个非常规步骤的证明）：
//   - mullo64：a·b mod 2^64 = lo32(a_lo·b_lo) + (lo32(a_lo·b_hi + a_hi·b_lo))·2^32
//     （二项展开 mod 2^64；_mm256_mul_epu32 只取每 64 位 lane 的低 32 位；
//     wrapping 加法即模运算，正确）。
//   - srai_epi64：AVX2 无 64 位算术移位——逻辑移位 or 符号填充
//     fill = (lane<0 ? −1 : 0) << (64−s)，s ≥ 1 时 64−s ≤ 63 无移位计数回绕。
//   - U = (double)(z>>11)·2^-53（53 位）：hi21·2^32 + lo32 拆分转换，两操作数
//     及和均 < 2^53，IEEE 加法精确 → 与标量 (double)(z>>11) 逐位一致。
//   - (double)q（q 任意 int64）：hi32·2^32（精确）+ lo32（精确）→ IEEE 加法
//     返回真和的最近舍入 = (double)q 的正确舍入 → 与标量转换逐位一致。
//   - 值重建 ldexp((double)q, m)·u：SIMD 侧 (double)q · 2^m（指数域平移，精确）
//     · u（一次舍入）——与标量 ldexp·u 完全同序同舍入。
//   - 除法/floor/比较：IEEE 正确舍入，与标量同输入位型同结果。
//
// 尾部：量化体是计数器式 RNG（U_i = f(seed, i)），尾部本地标量循环用全局索引
// i——与向量体、与标量锚点天然逐位一致（无需 RNG 状态衔接）。
//
// 编译：CMake 对本文件加 -mavx2（per-file ISA 纪律）。

#include "mkern/nested/nested_api.h"

#include <cmath>

#if defined(__AVX2__)
#include <immintrin.h>

namespace sgn::mkern::nested {

namespace {

// ---- 64 位乘法/移位合成（AVX2 缺原生 mullo64 / srai64）----

inline __m256i mullo_epi64_avx2(__m256i a, __m256i b) {
    const __m256i a_hi = _mm256_shuffle_epi32(a, 0xB1);   // 32 位半交换
    const __m256i b_hi = _mm256_shuffle_epi32(b, 0xB1);
    const __m256i ll = _mm256_mul_epu32(a, b);            // lo32·lo32（64 位）
    const __m256i x = _mm256_add_epi64(_mm256_mul_epu32(a, b_hi),
                                       _mm256_mul_epu32(a_hi, b));
    const __m256i x_lo = _mm256_and_si256(x, _mm256_set1_epi64x(0xFFFFFFFFLL));
    return _mm256_add_epi64(ll, _mm256_slli_epi64(x_lo, 32));
}

inline __m256i srai_epi64_avx2(__m256i v, int s) {       // 1 ≤ s ≤ 63
    const __m256i neg = _mm256_srli_epi64(v, 63);         // 0 或 1
    const __m256i sm = _mm256_sub_epi64(_mm256_setzero_si256(), neg);  // 0 或 −1
    const __m256i fill = _mm256_slli_epi64(sm, 64 - s);   // 高 s 位填符号
    return _mm256_or_si256(_mm256_srli_epi64(v, s), fill);
}

// 把每 64 位 lane 的低 32 位打包成 epi32x4（位型视图）
inline __m128i pack_lo32(__m256i v) {
    return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(
        v, _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6)));
}

// (double)lo32（无符号 32 位）：int32 位型视图减 2^31 转 signed 域转换，再 +2^31
inline __m256d cvt_lo32u_pd(__m256i lo) {
    const __m128i s = _mm_sub_epi32(pack_lo32(lo),
                                   _mm_set1_epi32(-2147483647 - 1));
    return _mm256_add_pd(_mm256_cvtepi32_pd(s), _mm256_set1_pd(2147483648.0));
}

// (double)q（q 任意 int64，正确舍入 = 标量转换逐位一致）
inline __m256d cvt_epu64_pair_pd(__m256i q) {
    const __m256d vhi = _mm256_mul_pd(
        _mm256_cvtepi32_pd(pack_lo32(srai_epi64_avx2(q, 32))),
        _mm256_set1_pd(4294967296.0));
    return _mm256_add_pd(vhi, cvt_lo32u_pd(q));
}

// 规格 a：元素 i0..i0+3 的 U ∈ [0,1)（与 scalar.cpp contract_uniform 位等同）
inline __m256d contract_uniform4_avx2(uint64_t seed, int64_t i0) {
    const __m256i idx = _mm256_add_epi64(_mm256_setr_epi64x(0, 1, 2, 3),
                                         _mm256_set1_epi64x(i0));
    __m256i z = _mm256_add_epi64(
        _mm256_set1_epi64x(static_cast<int64_t>(seed)),
        mullo_epi64_avx2(_mm256_set1_epi64x(static_cast<int64_t>(
                             0x9E3779B97F4A7C15ULL)),
                         idx));
    z = _mm256_xor_si256(z, _mm256_srli_epi64(z, 30));
    z = mullo_epi64_avx2(z, _mm256_set1_epi64x(static_cast<int64_t>(
                              0xBF58476D1CE4E5B9ULL)));
    z = _mm256_xor_si256(z, _mm256_srli_epi64(z, 27));
    z = mullo_epi64_avx2(z, _mm256_set1_epi64x(static_cast<int64_t>(
                              0x94D049BB133111EBULL)));
    z = _mm256_xor_si256(z, _mm256_srli_epi64(z, 31));
    const __m256i zz = _mm256_srli_epi64(z, 11);          // 53 位
    const __m256d vhi = _mm256_mul_pd(
        _mm256_cvtepi32_pd(pack_lo32(_mm256_srli_epi64(zz, 32))),
        _mm256_set1_pd(4294967296.0));
    return _mm256_mul_pd(_mm256_add_pd(vhi, cvt_lo32u_pd(zz)),
                         _mm256_set1_pd(0x1.0p-53));
}

// 量化尾部（全局索引——计数器式 RNG 使向量体/尾部/锚点天然逐位一致）
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

void nested_quant_i32_avx2(int64_t* code, const float* h, int64_t n,
                           float u, uint64_t seed) {
    const double ud = static_cast<double>(u);
    const __m256d vd = _mm256_set1_pd(ud);
    const __m256d sat_hi = _mm256_set1_pd(2147483647.0);
    const __m256d sat_lo = _mm256_set1_pd(-2147483648.0);
    const __m256i one = _mm256_set1_epi64x(1);
    const __m256i imax = _mm256_set1_epi64x(2147483647);
    const __m256i imin = _mm256_set1_epi64x(-2147483647 - 1);
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d x = _mm256_div_pd(_mm256_cvtps_pd(_mm_loadu_ps(h + i)), vd);
        // 饱和掩码（规格 c，先于 SR——与标量分支同序）
        const __m256d m_hi = _mm256_cmp_pd(x, sat_hi, _CMP_GE_OQ);
        const __m256d m_lo = _mm256_cmp_pd(x, sat_lo, _CMP_LE_OQ);
        const __m256d xf = _mm256_floor_pd(x);
        const __m256d frac = _mm256_sub_pd(x, xf);
        // Bernoulli：I = floor(x) + [U < frac]
        const __m256d Ud = contract_uniform4_avx2(seed, i);
        const __m256i inc = _mm256_and_si256(
            _mm256_castpd_si256(_mm256_cmp_pd(Ud, frac, _CMP_LT_OQ)), one);
        __m256i I = _mm256_cvtepi32_epi64(_mm256_cvttpd_epi32(xf));
        I = _mm256_add_epi64(I, inc);
        // 饱和混合（cvt 对超域 lane 给 indefinite 值，被混合覆盖；域内已证不溢）
        I = _mm256_blendv_epi8(I, imax, _mm256_castpd_si256(m_hi));
        I = _mm256_blendv_epi8(I, imin, _mm256_castpd_si256(m_lo));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(code + i), I);
    }
    quant_tail(code + i, h + i, n - i, i, ud, seed);
}

void nested_dequant_avx2(float* out, const int64_t* code, int64_t n,
                         float u, int level) {
    if (level != 4 && level != 8 && level != 16 && level != 32) return;
    const int m = 32 - level;
    const double ud = static_cast<double>(u);
    const __m256d vud = _mm256_set1_pd(ud);
    const __m256d vpow2m = _mm256_set1_pd(std::ldexp(1.0, m));  // 精确 2^m
    const __m256i one = _mm256_set1_epi64x(1);
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256i I = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(code + i));
        __m256i q;
        if (m == 0) {
            q = I;                                             // level 32 恒等
        } else {
            q = srai_epi64_avx2(I, m);                         // floor(I/2^m)
            const __m256i r = _mm256_sub_epi64(I, _mm256_slli_epi64(q, m));
            const __m256i half = _mm256_set1_epi64x(int64_t(1) << (m - 1));
            const __m256i adj = _mm256_or_si256(
                _mm256_cmpgt_epi64(r, half),
                _mm256_and_si256(_mm256_cmpeq_epi64(r, half),
                                 _mm256_and_si256(q, one)));   // half-to-even
            q = _mm256_add_epi64(q, _mm256_and_si256(adj, one));
        }
        // ldexp((double)q, m)·u：拆分转换（正确舍入）→ 指数平移（精确）→ ·u
        const __m256d val = _mm256_mul_pd(
            _mm256_mul_pd(cvt_epu64_pair_pd(q), vpow2m), vud);
        _mm_storeu_ps(out + i, _mm256_cvtpd_ps(val));
    }
    nested_dequant_scalar(out + i, code + i, n - i, u, level);
}

} // namespace sgn::mkern::nested

#endif  // __AVX2__
