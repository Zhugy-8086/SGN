// avx2_dot.cpp - AVX2 整型点积原语（dot16 + dot8/dot4 AVX2 中间档）
//
// 背景：fixes_相关修复/simd指令集加速文件拆分计划_2026_08_29.md + 综合执行计划
//   （simd_dot8_avx2中间路径立项_2026_09_02.md）。本文件为 avx2.cpp 按原语域拆分的
//   点积域：dot16（迁移）+ dot8/dot4 AVX2 中间档（新增，见立项 §一）。
//   decode（avx2_decode.cpp）与 float 归约（avx2_reduce.cpp）另文件，见拆分说明。
//
// 编译：CMake 对本文件加 -mavx2（per-file，与 avx2.cpp 拆分前一致）。
// 仅 __AVX2__ 下编译；非 AVX2 平台由 simd/scalar.cpp 提供标量锚点。

#include "mkern/simd/simd_api.h"

#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>

namespace sgn::simd {

// ============================================================================
// dot16：int16[K] × int16[K] → int64 精确点积（madd 快速路径 + vpmuldq 回退）
// ============================================================================
namespace {
// AVX2 窄精度点积：int16[K] × int16[K] → int64 精确累加。
// 每 16 元素一次迭代：先符号扩展到 int32，再用 _mm256_mul_epi32
// （signed 32×32→64，vpmuldq）分偶/奇 lane 得到 8 个 int64 乘积，累加到 4×int64。
// 注意：不能用 _mm256_madd_epi16——它会将相邻 2 个 int32 乘积相加，
// 当相邻元素均为 -32768（对应低位无符号部分 u=0 的边界值）时，pair 和为 2^31，
// 恰好溢出 int32、丢失 2^32，破坏 bit-exact。
// 整数加法可交换/结合，累加顺序不影响结果，与标量路径 bit-exact 一致。
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

// madd 快速路径（2026-08-31 数据实验定案，见 simd服务器加速计划 §7）：
// 用 _mm256_madd_epi16（1 指令/16 元素，相邻两对乘积相加为 int32）替代 vpmuldq。
// 溢出边界（实验确认）：仅当批内存在 -32768（int16 唯一满幅值）时，同一对两乘积
// 可能同号达 ±2³⁰ 使 pair 和越界 int32；若批内无 -32768，则 pair 和 ≤ 2×2³⁰ < 2³¹，
// 永不溢出。故本函数先检测 a、b 两侧是否含 -32768：
//   - 无 → madd 全速（1 指令/16 元素，理论 ~3-5× vpmuldq）
//   - 有 → 该批回退 vpmuldq（dot16_body，保持 bit-exact）
// 整数加法可交换/结合：混用 madd（int32 中间，立即扩 int64）与 vpmuldq（直接 int64）
// 不改变最终逐位结果（只要 madd 中间不溢出——检测保证）。返回 int64 累加结果。
inline __m256i dot16_madd16(__m256i acc64, const int16_t* a, const int16_t* b) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
    // 检测两侧是否含 -32768（int16 位模式 0x8000）
    const __m256i kMin = _mm256_set1_epi16(static_cast<int16_t>(-32768));
    __m256i m = _mm256_or_si256(_mm256_cmpeq_epi16(va, kMin),
                                _mm256_cmpeq_epi16(vb, kMin));
    // movemask 归约：非零 → 存在 -32768 → 回退 vpmuldq
    if (_mm256_movemask_epi8(m) != 0) {
        return dot16_body(acc64, a, b);
    }
    // madd：8×int32（相邻对乘积和，检测保证不溢出）→ 符号扩展为 8×int64 → 累加
    __m256i m32 = _mm256_madd_epi16(va, vb);
    __m256i lo64 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(m32));
    __m256i hi64 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(m32, 1));
    return _mm256_add_epi64(acc64, _mm256_add_epi64(lo64, hi64));
}
} // namespace

int64_t dot16_avx2(const int16_t* a, const int16_t* b, size_t K) {
    // madd 快速路径（2026-08-31 数据实验定案）：安全批 madd 全速 + 危险批 vpmuldq
    // 回退；4 累加器展开。整数加法可交换/结合，与标量锚点 bit-exact 一致。
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 64 <= K; i += 64) {
        acc0 = dot16_madd16(acc0, a + i,       b + i);
        acc1 = dot16_madd16(acc1, a + i + 16,  b + i + 16);
        acc2 = dot16_madd16(acc2, a + i + 32,  b + i + 32);
        acc3 = dot16_madd16(acc3, a + i + 48,  b + i + 48);
    }
    // 剩余 16 的整数倍：单累加器
    for (; i + 16 <= K; i += 16) {
        acc0 = dot16_madd16(acc0, a + i, b + i);
    }
    if (i < K) {
        // K 尾部不足 16：零填充做一次 16 宽 SIMD（bit-exact——填充 0 的乘积为 0，
        // 不改变累加；元素先拷入本地缓冲，无越界读）。
        alignas(32) int16_t pa[16] = {0}, pb[16] = {0};
        const size_t t = K - i;
        std::memcpy(pa, a + i, t * sizeof(int16_t));
        std::memcpy(pb, b + i, t * sizeof(int16_t));
        acc0 = dot16_madd16(acc0, pa, pb);
    }
    // 合并 4 累加器（整数加法可结合，顺序无关 → bit-exact）
    __m256i acc = _mm256_add_epi64(_mm256_add_epi64(acc0, acc1),
                                   _mm256_add_epi64(acc2, acc3));
    int64_t v[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(v), acc);
    return v[0] + v[1] + v[2] + v[3];
}

// ============================================================================
// dot8 / dot4 AVX2 中间档（新增，2026-09-02 综合执行计划 §一）
// uint8[K] × int8[K] → int64 精确点积（4 位预解包后 dot4 复用同主体）
// ============================================================================
namespace {
// 32 元素一批，双路径累加到两个 int32 累加器（lo/hi，各 8×int32）：
//   - 安全批（a 全 bit7==0，即 a ≤ 127）：_mm256_maddubs_epi16 全速。
//     vpmaddubsw 为 u8×s8→i16 饱和，相邻对和 ≤ 2×(127×127=16129) = 32258 < 32768，
//     永不饱和 → bit-exact，且 16 对/指令密度最高（无符号余数路径 a∈[0,15] 恒安全）。
//   - 危险批（a 含 ≥128）：cvtepu8/cvtepi8 → 16 位 → _mm256_madd_epi16 精确。
//     i16×i16→i32 中间，乘积 ≤ 255×127 在 i32 内无溢出 → bit-exact。
// 判定用 _mm256_movemask_epi8(va)==0（检 a 高位），单指令低成本。
inline void dot8_avx2_accum32(__m256i& lo, __m256i& hi,
                              const uint8_t* a, const int8_t* b) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));  // 32 uint8
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));  // 32 int8
    if (_mm256_movemask_epi8(va) == 0) {
        // 安全批：全 maddubs 快速路径（a ≤ 127，对和 < 32768 永不确定饱和）
        __m256i m16 = _mm256_maddubs_epi16(va, vb);  // 16×int16 对和（安全精确）
        lo = _mm256_add_epi32(lo, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(m16)));
        hi = _mm256_add_epi32(hi, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(m16, 1)));
    } else {
        // 危险批（a 含 ≥128）：精确路径，逐对 i16 乘积 → madd_epi16 → i32
        __m256i al = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(va));  // 16 u16
        __m256i ah = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(va, 1));
        __m256i bl = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));  // 16 i16
        __m256i bh = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
        lo = _mm256_add_epi32(lo, _mm256_madd_epi16(al, bl));  // i16×i16→i32 精确
        hi = _mm256_add_epi32(hi, _mm256_madd_epi16(ah, bh));
    }
}
} // namespace

int64_t dot8_avx2(const uint8_t* a, const int8_t* b, size_t K) {
    // 中间档：2 组 int32 累加器（lo/hi 各 8×int32），每批 32 元素。整数加法可交换/
    // 结合，与标量锚点 bit-exact 一致（浮点不适用，整型适用）。
    // 溢出边界（对齐 avxvnni.cpp dot8_vnni 的 i32 语义）：每 lane 累计对和
    // ≤ 32258×2 危险批 / 逐对 ≤ 130560，K=65536 时 2048 批 × 130560 ≈ 2.67e8 < 2^31 安全。
    __m256i lo0 = _mm256_setzero_si256(), hi0 = _mm256_setzero_si256();
    __m256i lo1 = _mm256_setzero_si256(), hi1 = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 64 <= K; i += 64) {
        dot8_avx2_accum32(lo0, hi0, a + i,       b + i);
        dot8_avx2_accum32(lo1, hi1, a + i + 32,  b + i + 32);
    }
    for (; i + 32 <= K; i += 32) {
        dot8_avx2_accum32(lo0, hi0, a + i, b + i);
    }
    if (i < K) {
        // 尾部：零填充做一次 32 宽 SIMD（bit-exact——填充 0 乘积为 0，无越界读）
        alignas(32) uint8_t pa[32] = {0};
        alignas(32) int8_t  pb[32] = {0};
        const size_t t = K - i;
        std::memcpy(pa, a + i, t);
        std::memcpy(pb, b + i, t);
        dot8_avx2_accum32(lo0, hi0, pa, pb);
    }
    __m256i lo = _mm256_add_epi32(lo0, lo1);
    __m256i hi = _mm256_add_epi32(hi0, hi1);
    int32_t vl[8], vh[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(vl), lo);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(vh), hi);
    // 归约：8×lo + 8×hi = 16 个 int32 → int64
    int64_t sum = 0;
    for (int t = 0; t < 8; ++t) sum += static_cast<int64_t>(vl[t]) + static_cast<int64_t>(vh[t]);
    return sum;
}

int64_t dot4_avx2(const uint8_t* u8, const int8_t* s8, size_t K) {
    // 4 位预解包后的点积：输入已由调用方 unpack_nibble_u/s 扩展为满宽字节，
    // 走 dot8 同款双路径主体（与 vnni/avx512vnni 的 dot4=dot8 同载体纪律一致）。
    // a = 低位 u8（unpack 后 ∈ [0,15] 或 [0,255]，前者恒 bit7=0 → 全 maddubs 全速）
    return dot8_avx2(u8, s8, K);
}

// ============================================================================
// dot4_packed_avx2：4 位打包点积（R2，mkern微内核层实施计划 §3.3）
// 直接消费 nibble 打包布局（读 K/2 字节），内核内 SIMD 解包 + vpmaddubsw 点积。
// 解包后 a ∈ [0,15]（bit7=0）→ vpmaddubsw 对和 ≤ 240 < 32768 恒不饱和，纯全速无检测。
// ============================================================================
namespace {
// 32 字节打包 → 低/高 nibble 各 32 字节（a 无符号 u8 / b 有符号 int8 符号扩展）
inline void unpack4_avx2(const __m256i va, const __m256i vb,
                         __m256i& a_lo, __m256i& a_hi,
                         __m256i& b_lo, __m256i& b_hi) {
    const __m256i kLo = _mm256_set1_epi8(0x0F);
    // a 无符号 nibble：AND 提取低 4 位；srli_epi16(4) 把高 nibble 移入低 4 位
    a_lo = _mm256_and_si256(va, kLo);
    a_hi = _mm256_and_si256(_mm256_srli_epi16(va, 4), kLo);
    // b 有符号 nibble：先 AND 清 bit7（vpshufb 对 bit7=1 输入输出 0，须先清），
    // 再用符号扩展表 vpshufb 查表（index = nibble 0-15 → int8 -8..7）
    const __m256i kSxt = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7,  -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7,  -8, -7, -6, -5, -4, -3, -2, -1);
    __m256i b_raw_lo = _mm256_and_si256(vb, kLo);
    __m256i b_raw_hi = _mm256_and_si256(_mm256_srli_epi16(vb, 4), kLo);
    // vpshufb(表, 索引)：第一个参数是被查表数据，第二个是索引（每字节低 4 位）
    b_lo = _mm256_shuffle_epi8(kSxt, b_raw_lo);
    b_hi = _mm256_shuffle_epi8(kSxt, b_raw_hi);
}

// 32 元素 u8×s8 点积累加（a∈[0,15] → vpmaddubsw 恒安全，对和 ≤ 240 < 32768）
inline void dot4_packed_accum32(__m256i& lo, __m256i& hi,
                                const __m256i a, const __m256i b) {
    __m256i m16 = _mm256_maddubs_epi16(a, b);  // 16×int16（无饱和）
    lo = _mm256_add_epi32(lo, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(m16)));
    hi = _mm256_add_epi32(hi, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(m16, 1)));
}
} // namespace

int64_t dot4_packed_avx2(const uint8_t* a_packed, const int8_t* b_packed, size_t K) {
    __m256i lo0 = _mm256_setzero_si256(), hi0 = _mm256_setzero_si256();
    __m256i lo1 = _mm256_setzero_si256(), hi1 = _mm256_setzero_si256();
    size_t i = 0;  // 元素索引
    // 每 64 元素批 = 32 字节打包（解包出 2×32 u8 + 2×32 int8 → 4 组 32 宽点积）
    for (; i + 64 <= K; i += 64) {
        __m256i va = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(a_packed + (i >> 1)));
        __m256i vb = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(b_packed + (i >> 1)));
        __m256i a_lo, a_hi, b_lo, b_hi;
        unpack4_avx2(va, vb, a_lo, a_hi, b_lo, b_hi);
        dot4_packed_accum32(lo0, hi0, a_lo, b_lo);  // 低 nibble（元素 0,2,4...）
        dot4_packed_accum32(lo1, hi1, a_hi, b_hi);  // 高 nibble（元素 1,3,5...）
    }
    // 标量尾部（剩余 <64 元素）：读打包字节逐元素解包点积（精确，无越界读）
    int64_t tail = 0;
    for (; i < K; ++i) {
        const size_t j = i >> 1;
        const int sh = 4 * (static_cast<int>(i) & 1);
        const int a_i = static_cast<int>((a_packed[j] >> sh) & 0x0F);
        const int b_i = static_cast<int>(static_cast<int8_t>(
            static_cast<uint8_t>((b_packed[j] >> sh) & 0x0F) << 4) >> 4);
        tail += static_cast<int64_t>(a_i) * static_cast<int64_t>(b_i);
    }
    __m256i lo = _mm256_add_epi32(lo0, lo1);
    __m256i hi = _mm256_add_epi32(hi0, hi1);
    int32_t vl[8], vh[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(vl), lo);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(vh), hi);
    int64_t sum = tail;
    for (int t = 0; t < 8; ++t) sum += static_cast<int64_t>(vl[t]) + static_cast<int64_t>(vh[t]);
    return sum;
}

} // namespace sgn::simd
#endif