// ssse3.cpp - SSSE3/AVX2 原语实现（__SSSE3__）
//
// 背景：fixes_相关修复/simd指令集加速文件拆分计划_2026_08_29.md
// Step 2：sgn::simd::reverse_bytes8 / batch_reverse_u8 由 msint/packed_backend.cpp 的
// get_all_simd_8bit_unsigned_ / batch_get_all SSSE3 分支迁移。
//
// 注意：当前构建全局 -mavx2（CMakeLists.txt），本文件仅在 __SSSE3__ 下编译。
//   - reverse_bytes8 仅需 128 位 PSHUFB（真 SSSE3）；
//   - batch_reverse_u8 用 _mm256_*（实际需 AVX2，与迁移前 packed_backend.cpp 一致）。
// 非 SSSE3/非 x86 平台由 simd/scalar.cpp 提供标量锚点，无重复符号。

#include "simd/simd_api.h"

#if defined(__SSSE3__)
#include <immintrin.h>

namespace sgn::simd {

void reverse_bytes8_ssse3(uint64_t packed, int n, int64_t* out) {
    // 128 位 PSHUFB：v 低 64 位 = packed_（小端字节序），反转前 n 字节。
    // shuffle_mask[i] = n-1-i → reversed[i] = v[n-1-i] = packed 字节 n-1-i。
    __m128i v = _mm_set_epi64x(0, static_cast<long long>(packed));
    alignas(16) uint8_t shuffle_mask[16] = {};
    for (int i = 0; i < n; ++i) {
        shuffle_mask[i] = static_cast<uint8_t>(n - 1 - i);
    }
    __m128i mask = _mm_load_si128(reinterpret_cast<const __m128i*>(shuffle_mask));
    __m128i reversed = _mm_shuffle_epi8(v, mask);

    alignas(16) uint8_t buf[16];
    _mm_store_si128(reinterpret_cast<__m128i*>(buf), reversed);
    for (int i = 0; i < n; ++i) {
        out[i] = static_cast<int64_t>(buf[i]);
    }
}

void batch_reverse_u8_ssse3(const uint64_t* packed_values, int n_values, int n_slots,
                            int64_t* result) {
    // AVX2 PSHUFB：每 4 个 uint64 一组，per-lane 反转 8 字节。
    // 前提（调用方保证）：n_slots ∈ [1,8] 且全 8-bit 等宽。
    // 语义（与标量锚点一致）：slot[j] = pv 字节 (n_slots-1-j)；反转后 rev[i]=pv[7-i]，
    // 故 slot[j] = rev[8-n_slots+j] —— n_slots<8 时需偏移 (8-n_slots) 取"后 n_slots 字节"。
    // 修复（2026-08-29 审查）：原 packed_backend.cpp SSSE3 分支此处直接取 rev 前 n_slots
    // 字节（注释声称 PALIGNR 滑窗但未实现），n_slots∈[4,7] 时取错槽位、与标量尾部矛盾；
    // 迁移时被带入，由 simd_boundary_test 边界扫描抓出。n_slots=8 时偏移为 0，行为不变。
    const __m256i rev_mask = _mm256_setr_epi8(
        7, 6, 5, 4, 3, 2, 1, 0,  15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0,  15, 14, 13, 12, 11, 10, 9, 8
    );
    const int off = 8 - n_slots;  // n_slots<8 时的滑窗偏移（n_slots=8 → 0）

    int i = 0;
    const int n_vec = (n_values / 4) * 4;
    for (; i < n_vec; i += 4) {
        __m256i v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(&packed_values[i]));
        __m256i rev = _mm256_shuffle_epi8(v, rev_mask);

        alignas(32) uint8_t buf[32];
        _mm256_store_si256(reinterpret_cast<__m256i*>(buf), rev);
        for (int vi = 0; vi < 4; ++vi) {
            for (int j = 0; j < n_slots; ++j) {
                result[static_cast<size_t>(i + vi) * n_slots + j] =
                    static_cast<int64_t>(buf[vi * 8 + off + j]);
            }
        }
    }

    // 标量尾部
    for (; i < n_values; ++i) {
        uint64_t pv = packed_values[i];
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&pv);
        for (int j = 0; j < n_slots; ++j) {
            result[static_cast<size_t>(i) * n_slots + j] =
                static_cast<int64_t>(bytes[n_slots - 1 - j]);
        }
    }
}

} // namespace sgn::simd
#endif
