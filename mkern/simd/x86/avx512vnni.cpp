// avx512vnni.cpp - AVX-512 VNNI 原语实现（服务器加速，P1）
//
// 背景：simd服务器加速计划_2026_08_30.md P1。
// dot8/dot4 由 simd/x86/avxvnni.cpp 的 256 位版本升级到 512 位：
//   - _mm512_dpbusd_epi32（AVX512VNNI，64 MAC/指令，16×int32 累加器）
//   - 4 累加器展开（每轮 4×64 = 256 元素，填满执行端口）
//   - 零填充尾部（bit-exact 范式沿用 avx2/avxvnni）
//
// 编译：本文件由 CMake 单独加 -mavx512f -mavx512vnni -mavx512bw
// （set_source_files_properties，见 CMakeLists.txt），故 __AVX512VNNI__ 常驻定义、
// 符号常驻编译（本机无 AVX-512 也可链接）。运行时经 simd_dispatch.cpp 的
// __builtin_cpu_supports("avx512vnni") 检测，仅目标 CPU 支持时选入。
// 多累加器展开：整数加法可交换/结合，与标量锚点 bit-exact 一致。
//
// 数值语义（与 256 位 avxvnni.cpp 一致）：
//   - dot8 : uint8[K] × int8[K] → int64 精确点积
//   - dot4 : 4 位预解包后的 u8[K] × s8[K] → int64（复用 dot8 主体，与迁移前同一载体）
//   - 禁 _mm512_maddubs_epi16：255*128 pair 和饱和破坏 bit-exact（同 256 位结论）

#include "mkern/simd/simd_api.h"

#include <cstring>

#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
#include <immintrin.h>

namespace sgn::simd {
namespace {
// 64 宽点积主体（8 位/4 位共用；主循环与零填充尾部复用）。
inline __m512i dot64_body(__m512i acc, const uint8_t* a, const int8_t* b) {
    __m512i va = _mm512_loadu_si512(a);  // 64 uint8
    __m512i vb = _mm512_loadu_si512(b);  // 64 int8
    return _mm512_dpbusd_epi32(acc, va, vb);  // 64 MAC/指令，16×int32 精确累加
}

// 16 个 int32 → int64 归约（先扩展为两半 int64，再相加）。
inline int64_t reduce16_i32(__m512i acc) {
    // 低/高各 8 个 int32 → int64（_mm512_cvtepi32_epi64 返回 __m512i = 8×int64）
    __m512i lo64 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(acc));
    __m512i hi64 = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(acc, 1));
    __m512i sum64 = _mm512_add_epi64(lo64, hi64);
    return _mm512_reduce_add_epi64(sum64);
}
} // namespace

int64_t dot8_avx512vnni(const uint8_t* a, const int8_t* b, size_t K) {
    // 4 累加器展开：每轮 4×64 = 256 元素，隐藏 dpbusd 延迟、填满执行端口。
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    __m512i acc2 = _mm512_setzero_si512();
    __m512i acc3 = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 256 <= K; i += 256) {
        acc0 = dot64_body(acc0, a + i,       b + i);
        acc1 = dot64_body(acc1, a + i + 64,  b + i + 64);
        acc2 = dot64_body(acc2, a + i + 128, b + i + 128);
        acc3 = dot64_body(acc3, a + i + 192, b + i + 192);
    }
    // 剩余 64 的整数倍：单累加器
    for (; i + 64 <= K; i += 64) {
        acc0 = dot64_body(acc0, a + i, b + i);
    }
    // 尾部不足 64：零填充做一次满宽 512 位（bit-exact——填充 0 乘积为 0，无越界读）
    if (i < K) {
        alignas(64) uint8_t pa[64] = {0};
        alignas(64) int8_t  pb[64] = {0};
        const size_t t = K - i;
        std::memcpy(pa, a + i, t);
        std::memcpy(pb, b + i, t);
        acc0 = dot64_body(acc0, pa, pb);
    }
    // 合并 4 累加器（整数加法可结合，顺序无关 → bit-exact）
    __m512i acc = _mm512_add_epi32(_mm512_add_epi32(acc0, acc1),
                                   _mm512_add_epi32(acc2, acc3));
    return reduce16_i32(acc);
}

int64_t dot4_avx512vnni(const uint8_t* u8, const int8_t* s8, size_t K) {
    // 4 位预解包后的点积：与 dot8 同载体（dpbusd 直接 u8×s8 精确累加），
    // 语义等价于逐元素 u8×s8 求和，bit-exact。输入已由调用方预解包为满宽字节。
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    __m512i acc2 = _mm512_setzero_si512();
    __m512i acc3 = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 256 <= K; i += 256) {
        acc0 = dot64_body(acc0, u8 + i,       s8 + i);
        acc1 = dot64_body(acc1, u8 + i + 64,  s8 + i + 64);
        acc2 = dot64_body(acc2, u8 + i + 128, s8 + i + 128);
        acc3 = dot64_body(acc3, u8 + i + 192, s8 + i + 192);
    }
    for (; i + 64 <= K; i += 64) {
        acc0 = dot64_body(acc0, u8 + i, s8 + i);
    }
    if (i < K) {
        alignas(64) uint8_t pa[64] = {0};
        alignas(64) int8_t  pb[64] = {0};
        const size_t t = K - i;
        std::memcpy(pa, u8 + i, t);
        std::memcpy(pb, s8 + i, t);
        acc0 = dot64_body(acc0, pa, pb);
    }
    __m512i acc = _mm512_add_epi32(_mm512_add_epi32(acc0, acc1),
                                   _mm512_add_epi32(acc2, acc3));
    return reduce16_i32(acc);
}

// ============================================================================
// dot4_packed_avx512vnni：4 位打包点积（R2，mkern微内核层实施计划 §3.3）
// 512 位解包 + vpdpbusd。直接消费 nibble 打包布局（读 K/2 字节）。
// 解包后 a ∈ [0,15] 无符号、b 符号扩展 -8..7 → _mm512_dpbusd_epi32 精确累加（bit-exact）。
// 本机（Arrow Lake）无 AVX-512 无法真跑，逻辑同 256 位 vnni 版，待远程 EPYC 验证。
// ============================================================================
namespace {
// 64 字节打包 → 低/高 nibble 各 64 字节
inline void unpack4_avx512vnni(const __m512i va, const __m512i vb,
                               __m512i& a_lo, __m512i& a_hi,
                               __m512i& b_lo, __m512i& b_hi) {
    const __m512i kLo = _mm512_set1_epi8(0x0F);
    a_lo = _mm512_and_si512(va, kLo);
    a_hi = _mm512_and_si512(_mm512_srli_epi16(va, 4), kLo);
    // b 有符号 nibble 符号扩展表（AVX512 无 _mm512_setr_epi8，用常量数组 load）
    alignas(64) static const int8_t kSxtData[64] = {
        0, 1, 2, 3, 4, 5, 6, 7,  -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7,  -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7,  -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7,  -8, -7, -6, -5, -4, -3, -2, -1};
    const __m512i kSxt = _mm512_loadu_si512(kSxtData);
    __m512i b_raw_lo = _mm512_and_si512(vb, kLo);
    __m512i b_raw_hi = _mm512_and_si512(_mm512_srli_epi16(vb, 4), kLo);
    // vpshufb(表, 索引)：第一个参数是被查表数据，第二个是索引（每字节低 4 位）
    b_lo = _mm512_shuffle_epi8(kSxt, b_raw_lo);
    b_hi = _mm512_shuffle_epi8(kSxt, b_raw_hi);
}

inline __m512i dot4_packed_avx512vnni_accum(__m512i acc, const __m512i a, const __m512i b) {
    return _mm512_dpbusd_epi32(acc, a, b);
}
} // namespace

int64_t dot4_packed_avx512vnni(const uint8_t* a_packed, const int8_t* b_packed, size_t K) {
    // 4 累加器展开（R2 修订 2026-09-03，同 256 位 vnni 版）：原 2 链被 vpdpbusd 延迟
    // 串行化；每 256 元素批 = 2 个 128 元素块（各 64 字节打包），lo/hi 分散到 4 条独立链。
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    __m512i acc2 = _mm512_setzero_si512();
    __m512i acc3 = _mm512_setzero_si512();
    size_t i = 0;  // 元素索引
    for (; i + 256 <= K; i += 256) {
        __m512i va = _mm512_loadu_si512(a_packed + (i >> 1));
        __m512i vb = _mm512_loadu_si512(b_packed + (i >> 1));
        __m512i a_lo, a_hi, b_lo, b_hi;
        unpack4_avx512vnni(va, vb, a_lo, a_hi, b_lo, b_hi);
        acc0 = dot4_packed_avx512vnni_accum(acc0, a_lo, b_lo);
        acc1 = dot4_packed_avx512vnni_accum(acc1, a_hi, b_hi);
        va = _mm512_loadu_si512(a_packed + ((i + 128) >> 1));
        vb = _mm512_loadu_si512(b_packed + ((i + 128) >> 1));
        unpack4_avx512vnni(va, vb, a_lo, a_hi, b_lo, b_hi);
        acc2 = dot4_packed_avx512vnni_accum(acc2, a_lo, b_lo);
        acc3 = dot4_packed_avx512vnni_accum(acc3, a_hi, b_hi);
    }
    // 剩余 128 元素的整数倍：回落 acc0/acc1 双链
    for (; i + 128 <= K; i += 128) {
        __m512i va = _mm512_loadu_si512(a_packed + (i >> 1));
        __m512i vb = _mm512_loadu_si512(b_packed + (i >> 1));
        __m512i a_lo, a_hi, b_lo, b_hi;
        unpack4_avx512vnni(va, vb, a_lo, a_hi, b_lo, b_hi);
        acc0 = dot4_packed_avx512vnni_accum(acc0, a_lo, b_lo);
        acc1 = dot4_packed_avx512vnni_accum(acc1, a_hi, b_hi);
    }
    // 标量尾部（剩余 <128 元素）
    int64_t tail = 0;
    for (; i < K; ++i) {
        const size_t j = i >> 1;
        const int sh = 4 * (static_cast<int>(i) & 1);
        const int a_i = static_cast<int>((a_packed[j] >> sh) & 0x0F);
        const int b_i = static_cast<int>(static_cast<int8_t>(
            static_cast<uint8_t>((b_packed[j] >> sh) & 0x0F) << 4) >> 4);
        tail += static_cast<int64_t>(a_i) * static_cast<int64_t>(b_i);
    }
    __m512i acc = _mm512_add_epi32(_mm512_add_epi32(acc0, acc2),
                                   _mm512_add_epi32(acc1, acc3));
    int64_t sum = tail + reduce16_i32(acc);
    return sum;
}

} // namespace sgn::simd
#endif
