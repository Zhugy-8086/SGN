// avxvnni.cpp - AVX-VNNI 原语实现（__AVXVNNI__）
//
// 背景：fixes_相关修复/simd指令集加速文件拆分计划_2026_08_29.md
// Step 1：sgn::simd::dot8 / dot4 由 msint/split_dot.cpp 的 dot8_vnni / dot8_body /
// dot4_dpbusd 迁移（dot4 复用 dot8_body，与迁移前同一 TU 内联关系一致）。
//
// 注意：当前构建全局 -mavxvnni（CMakeLists.txt），本文件仅在 __AVXVNNI__ 下编译。
// 非 AVX-VNNI 平台由 simd/scalar.cpp 提供标量锚点，无重复符号。

#include "mkern/simd/simd_api.h"

#include <cstring>

#if defined(__AVXVNNI__)
#include <immintrin.h>

namespace sgn::simd {
namespace {
// AVX-VNNI 窄精度点积：uint8[K] × int8[K] → int64 精确累加。
// 每 32 元素一次迭代：_mm256_dpbusd_epi32（vpdpbusd，unsigned×signed → int32 累加，
// 32 MAC/指令），无饱和、32 位精确累加，最后扩到 int64 归约。
// 注意：不能用 _mm256_maddubs_epi16——它把相邻两个 byte 乘积相加成 int16，
// 当两者均为满幅（255*128）时 pair 和 = 65280 > int16 上限 32767，饱和破坏 bit-exact
// （与 hc8_net.c 既有结论一致）。dpbusd 直接累加到 int32，无此问题。
// 32 宽点积主体（8 位/4 位共用；主循环与零填充尾部复用）。
inline __m256i dot8_body(__m256i acc, const uint8_t* a, const int8_t* b) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));  // 32 uint8
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));  // 32 int8
    return _mm256_dpbusd_epi32(acc, va, vb);  // 8×int32 += 每 4 个 byte 乘积（精确）
}
} // namespace

int64_t dot8_vnni(const uint8_t* a, const int8_t* b, size_t K) {
    // 4 累加器展开（P1，服务器加速）：每轮 4×32 = 128 元素，隐藏 dpbusd 延迟、
    // 填满执行端口。整数加法可交换/结合，与标量锚点 bit-exact 一致。
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 128 <= K; i += 128) {
        acc0 = dot8_body(acc0, a + i,       b + i);
        acc1 = dot8_body(acc1, a + i + 32,  b + i + 32);
        acc2 = dot8_body(acc2, a + i + 64,  b + i + 64);
        acc3 = dot8_body(acc3, a + i + 96,  b + i + 96);
    }
    // 剩余 32 的整数倍：单累加器
    for (; i + 32 <= K; i += 32) {
        acc0 = dot8_body(acc0, a + i, b + i);
    }
    if (i < K) {
        // K 尾部不足 32：零填充做一次 32 宽 SIMD（bit-exact——填充 0 的乘积为 0，
        // 不改变累加；元素先拷入本地缓冲，无越界读）。
        alignas(32) uint8_t pa[32] = {0};
        alignas(32) int8_t pb[32] = {0};
        const size_t t = K - i;
        std::memcpy(pa, a + i, t);
        std::memcpy(pb, b + i, t);
        acc0 = dot8_body(acc0, pa, pb);
    }
    // 合并 4 累加器（整数加法可结合，顺序无关 → bit-exact）
    __m256i acc = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1),
                                   _mm256_add_epi32(acc2, acc3));
    int32_t v[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(v), acc);
    int64_t sum = 0;
    for (int t = 0; t < 8; ++t) {
        sum += static_cast<int64_t>(v[t]);
    }
    return sum;
}

int64_t dot4_vnni(const uint8_t* u8, const int8_t* s8, size_t K) {
    // 4 位预解包后的点积：uint8[K] × int8[K] → int64（纯 dpbusd，32 MAC/指令，
    // 无解包指令；语义等价于逐元素 u8×s8 求和，bit-exact）。4 累加器展开同 dot8。
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 128 <= K; i += 128) {
        acc0 = dot8_body(acc0, u8 + i,       s8 + i);
        acc1 = dot8_body(acc1, u8 + i + 32,  s8 + i + 32);
        acc2 = dot8_body(acc2, u8 + i + 64,  s8 + i + 64);
        acc3 = dot8_body(acc3, u8 + i + 96,  s8 + i + 96);
    }
    for (; i + 32 <= K; i += 32) {
        acc0 = dot8_body(acc0, u8 + i, s8 + i);
    }
    if (i < K) {
        // K 尾部不足 32：零填充做一次 32 宽 SIMD（bit-exact——填充 0 的乘积为 0，
        // 不改变累加；元素先拷入本地缓冲，无越界读）。
        alignas(32) uint8_t pa[32] = {0};
        alignas(32) int8_t pb[32] = {0};
        const size_t t = K - i;
        std::memcpy(pa, u8 + i, t);
        std::memcpy(pb, s8 + i, t);
        acc0 = dot8_body(acc0, pa, pb);
    }
    __m256i acc = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1),
                                   _mm256_add_epi32(acc2, acc3));
    int32_t v[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(v), acc);
    int64_t sum = 0;
    for (int t = 0; t < 8; ++t) {
        sum += static_cast<int64_t>(v[t]);
    }
    return sum;
}

// ============================================================================
// dot4_packed_vnni：4 位打包点积（R2，mkern微内核层实施计划 §3.3）
// 直接消费 nibble 打包布局（读 K/2 字节），内核内 SIMD 解包 + vpdpbusd 点积。
// 解包后 a ∈ [0,15] 无符号、b 符号扩展 -8..7 → vpdpbusd 精确累加（bit-exact）。
// ============================================================================
namespace {
// 32 字节打包 → 低/高 nibble 各 32 字节（同 avx2_dot.cpp unpack4，AVX2 指令集内）
inline void unpack4_vnni(const __m256i va, const __m256i vb,
                         __m256i& a_lo, __m256i& a_hi,
                         __m256i& b_lo, __m256i& b_hi) {
    const __m256i kLo = _mm256_set1_epi8(0x0F);
    a_lo = _mm256_and_si256(va, kLo);
    a_hi = _mm256_and_si256(_mm256_srli_epi16(va, 4), kLo);
    const __m256i kSxt = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7,  -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7,  -8, -7, -6, -5, -4, -3, -2, -1);
    __m256i b_raw_lo = _mm256_and_si256(vb, kLo);
    __m256i b_raw_hi = _mm256_and_si256(_mm256_srli_epi16(vb, 4), kLo);
    // vpshufb(表, 索引)：第一个参数是被查表数据，第二个是索引（每字节低 4 位）
    b_lo = _mm256_shuffle_epi8(kSxt, b_raw_lo);
    b_hi = _mm256_shuffle_epi8(kSxt, b_raw_hi);
}

// 32 元素 u8×s8 点积累加（vpdpbusd 精确，int32 lane）
inline __m256i dot4_packed_vnni_accum(__m256i acc, const __m256i a, const __m256i b) {
    return _mm256_dpbusd_epi32(acc, a, b);
}
} // namespace

int64_t dot4_packed_vnni(const uint8_t* a_packed, const int8_t* b_packed, size_t K) {
    // 4 累加器展开（R2 修订 2026-09-03）：原 2 链被 vpdpbusd 延迟（~4-5 周期）串行化，
    // 中段 K 实测仅解包路径的 0.41-0.64×；4 链后 K=16384 收敛至 ~0.9×、K=65536 反超
    // 解包路径 ~1.27×（带宽减半红利在 L1 外兑现，见实施计划 §五.2 修订记录）。
    // 每 128 元素批 = 2 个 64 元素块（各 32 字节打包），lo/hi 分散到 4 条独立链。
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    size_t i = 0;  // 元素索引
    for (; i + 128 <= K; i += 128) {
        __m256i va = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(a_packed + (i >> 1)));
        __m256i vb = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(b_packed + (i >> 1)));
        __m256i a_lo, a_hi, b_lo, b_hi;
        unpack4_vnni(va, vb, a_lo, a_hi, b_lo, b_hi);
        acc0 = dot4_packed_vnni_accum(acc0, a_lo, b_lo);
        acc1 = dot4_packed_vnni_accum(acc1, a_hi, b_hi);
        va = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(a_packed + ((i + 64) >> 1)));
        vb = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(b_packed + ((i + 64) >> 1)));
        unpack4_vnni(va, vb, a_lo, a_hi, b_lo, b_hi);
        acc2 = dot4_packed_vnni_accum(acc2, a_lo, b_lo);
        acc3 = dot4_packed_vnni_accum(acc3, a_hi, b_hi);
    }
    // 剩余 64 元素的整数倍：回落 acc0/acc1 双链
    for (; i + 64 <= K; i += 64) {
        __m256i va = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(a_packed + (i >> 1)));
        __m256i vb = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(b_packed + (i >> 1)));
        __m256i a_lo, a_hi, b_lo, b_hi;
        unpack4_vnni(va, vb, a_lo, a_hi, b_lo, b_hi);
        acc0 = dot4_packed_vnni_accum(acc0, a_lo, b_lo);
        acc1 = dot4_packed_vnni_accum(acc1, a_hi, b_hi);
    }
    // 标量尾部（剩余 <64 元素）
    int64_t tail = 0;
    for (; i < K; ++i) {
        const size_t j = i >> 1;
        const int sh = 4 * (static_cast<int>(i) & 1);
        const int a_i = static_cast<int>((a_packed[j] >> sh) & 0x0F);
        const int b_i = static_cast<int>(static_cast<int8_t>(
            static_cast<uint8_t>((b_packed[j] >> sh) & 0x0F) << 4) >> 4);
        tail += static_cast<int64_t>(a_i) * static_cast<int64_t>(b_i);
    }
    // 合并 4 累加器（整数加法可结合，顺序无关 → bit-exact）
    __m256i acc = _mm256_add_epi32(_mm256_add_epi32(acc0, acc2),
                                   _mm256_add_epi32(acc1, acc3));
    int32_t v[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(v), acc);
    int64_t sum = tail;
    for (int t = 0; t < 8; ++t) sum += static_cast<int64_t>(v[t]);
    return sum;
}

} // namespace sgn::simd
#endif
