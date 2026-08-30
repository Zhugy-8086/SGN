// avxvnni.cpp - AVX-VNNI 原语实现（__AVXVNNI__）
//
// 背景：fixes_相关修复/simd指令集加速文件拆分计划_2026_08_29.md
// Step 1：sgn::simd::dot8 / dot4 由 msint/split_dot.cpp 的 dot8_vnni / dot8_body /
// dot4_dpbusd 迁移（dot4 复用 dot8_body，与迁移前同一 TU 内联关系一致）。
//
// 注意：当前构建全局 -mavxvnni（CMakeLists.txt），本文件仅在 __AVXVNNI__ 下编译。
// 非 AVX-VNNI 平台由 simd/scalar.cpp 提供标量锚点，无重复符号。

#include "simd/simd_api.h"

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
} // namespace sgn::simd
#endif
