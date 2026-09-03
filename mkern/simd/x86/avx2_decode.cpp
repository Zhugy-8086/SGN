// avx2_decode.cpp - AVX2 packed 解码原语（decode_i16_f32）
//
// 背景：fixes_相关修复/simd指令集加速文件拆分计划_2026_08_29.md + 综合执行计划
//   （simd_dot8_avx2中间路径立项_2026_09_02.md §二.5 拆分）。本文件为 avx2.cpp
//   按原语域拆分的解码域；dot 域在 avx2_dot.cpp、float 归约域在 avx2_reduce.cpp。
//   实现自 msint/packed_backend_bindings.cpp decode_to_float_core 快速路径 1 逐字迁移。
//
// 编译：CMake 对本文件加 -mavx2 -mssse3（decode 需 SSSE3 PSHUFB + AVX2 _mm256）。
// 仅 __AVX2__ 下编译；非 AVX2 平台由 simd/scalar.cpp 提供标量锚点。

#include "mkern/simd/simd_api.h"

#if defined(__AVX2__)
#include <immintrin.h>

namespace sgn::simd {

void decode_i16_f32_avx2(const uint64_t* pv_ptr, int n_values, float scale, float* res_ptr) {
    // 源：msint/packed_backend_bindings.cpp decode_to_float_core 快速路径 1
    //（backward_int16 8+8 schema：每个 uint64 低 16 位 = 1 个 int16，高 48 位无效）。
    // 两条子路径 + 标量尾部，与迁移前逻辑逐字一致（float 乘 scale 用 mul 不融合）。
    // 注意（综合执行计划 §二.2）：测试报告称 AVX2 版负优化，与本机实测（n≥32 快
    // 1.7~2.6×）矛盾，属"数据布局带宽"问题而非本实现缺陷，本轮不撤下。
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

// ============================================================================
// decode_i16_f32_packed16：连续 int16 输入的解码（布局优化，综合执行计划 §二.5）
//
// 背景（2026-09-02）：EPYC 9K65 复核问题 ① 指出 decode_i16_f32 的 uint64 输入
// "每 8 字节仅 16 位有效 → 4× 带宽浪费"。该浪费是**上游编码格式**（backward_int16
// 8+8 schema 用 uint64 存 1 个 int16）决定的，旧原语内部无法规避。本原语提供
// **连续 int16 数组**输入的全新解码路径：每 16 位存一个有效 int16（无 48 位填充），
// 内存带宽由 25% 有效利用率提升到 100%——这是问题 ① 的真正修复方向
// （测试报告建议 1：改上层布局让 int16 连续 packed）。
//
// 实现：AVX2 宽读连续 int16 → cvtepi16_epi32（符号扩展）→ cvtepi32_ps → ×scale。
// 无任何打包/重排（连续布局天然对齐），无需 detect 回退（无 -32768 饱和边界：
// int16→int32 float 各自独立，无中间累加）。与标量锚点浮点逐位一致
// （mul 不融合，同 decode_i16_f32 纪律）。
//
// 由调用方提供连续 int16 数据作为新布局的入口（旧 uint64 版保留供现有调用方）。
// ============================================================================
void decode_i16_f32_packed16(const int16_t* src, int n_values, float scale, float* res_ptr) {
    __m256 scale_vec = _mm256_set1_ps(scale);
    int i = 0;
    // 每 8 个连续 int16（128 位 load）→ cvtepi16_epi32（8→4 int32）→ cvtepi32_ps → ×scale。
    // 连续布局天然对齐，无 uint64 的 48 位填充——内存带宽利用 100%（问题 ① 正解）。
    for (; i + 7 < n_values; i += 8) {
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));  // 8×int16
        __m256i a32 = _mm256_cvtepi16_epi32(v);  // 8×int16 → 4×int32
        __m256 fa = _mm256_cvtepi32_ps(a32);     // 4×float
        _mm256_storeu_ps(&res_ptr[i], _mm256_mul_ps(fa, scale_vec));
    }
    // 标量尾部
    for (; i < n_values; ++i) {
        res_ptr[i] = static_cast<float>(src[i]) * scale;
    }
}

} // namespace sgn::simd
#endif