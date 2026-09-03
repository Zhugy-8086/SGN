// gemm_avx2vnni.cpp - mkern/gemm gemm_i8 AVX2-VNNI 内核（4 行 × 16 列 tile）
//
// 背景：fixes_相关修复/mkern微内核层实施计划_2026_09_03.md §3.2（R3）。
// 结构：消费 pack_b_i8 的 VNNI 面板布局（见 gemm_api.h），每个 vpdpbusd 的
// 16 个 int32 lane 对应 16 个输出列（32 字节半面板 = 8 列 × 4 k），累加器即
// C 列向量（免水平归约）；A 侧 4 字节 k 组 set1 广播（每组被 2 个半面板复用），
// B 面板组被 4 行复用。每 dpbusd 32 MAC，瓶颈为 dpbusd 端口。
//
// 编译：CMake 对本文件加 -mavx2 -mavxvnni（per-file，同 simd 层纪律）。
// 整数运算：多累加器/分块不改变整数结果 → 与标量锚点 bit-exact。
// 溢出安全域：K ≤ 65536（每 lane 累计全部 K 个乘积，见 gemm_api.h 契约）。

#include "mkern/gemm/gemm_api.h"

#include <cstring>

#if defined(__AVX2__) && defined(__AVXVNNI__)
#include <immintrin.h>

namespace sgn::mkern::gemm {
namespace {

// 一个满 4 k 字节组的 A 广播向量（定长 4 字节读 → 内联单 mov，热循环零开销）。
inline __m256i broadcast_group4(const uint8_t* a) {
    uint32_t d;
    std::memcpy(&d, a, 4);
    return _mm256_set1_epi32(static_cast<int32_t>(d));
}

// K 尾组（K%4 ≠ 0 的最后 1 组，≤3 有效字节）零填充装配——每行仅调用一次（冷路径）。
inline __m256i broadcast_group_tail(const uint8_t* a, int64_t rem) {
    uint32_t d = 0;
    std::memcpy(&d, a, static_cast<size_t>(rem));
    return _mm256_set1_epi32(static_cast<int32_t>(d));
}

// 4 行 × 16 列 tile 的 K 全扫（含尾组），acc[8] = 4 行 × 2 半面板（各 8 列）。
inline void tile_4x16_kloop(__m256i acc[8],
                            const uint8_t* const ar[4], const int8_t* Bp,
                            int64_t K, int64_t n0) {
    const int64_t ng = (K + 3) / 4;
    const int64_t gfull = K / 4;   // 满组数（K%4 尾组单独处理，见函数尾）
    const int64_t panel = n0 / 16;
    for (int64_t g = 0; g < gfull; ++g) {
        // 2 个 8 列半面板 = 1 个 16 列面板；vb 面板组被 4 行复用
        const int8_t* vb_lo = Bp + (panel * ng + g) * 64;       // 列 0..7
        const int8_t* vb_hi = vb_lo + 32;                        // 列 8..15
        __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vb_lo));
        __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vb_hi));
        for (int r = 0; r < 4; ++r) {
            __m256i va = broadcast_group4(ar[r] + g * 4);
            acc[2 * r]     = _mm256_dpbusd_epi32(acc[2 * r],     va, vb0);
            acc[2 * r + 1] = _mm256_dpbusd_epi32(acc[2 * r + 1], va, vb1);
        }
    }
    if (gfull < ng) {  // K%4 尾组（pack 已零填充 vb 侧；va 侧此处零填充）
        const int64_t rem = K - gfull * 4;
        const int8_t* vb_lo = Bp + (panel * ng + gfull) * 64;
        const int8_t* vb_hi = vb_lo + 32;
        __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vb_lo));
        __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vb_hi));
        for (int r = 0; r < 4; ++r) {
            __m256i va = broadcast_group_tail(ar[r] + gfull * 4, rem);
            acc[2 * r]     = _mm256_dpbusd_epi32(acc[2 * r],     va, vb0);
            acc[2 * r + 1] = _mm256_dpbusd_epi32(acc[2 * r + 1], va, vb1);
        }
    }
}

// tile 存回 C（末面板列 n0+j ≥ N 时跳过——pack 零填充列照算不存）
inline void tile_4x16_store(int32_t* C, int64_t N, int64_t n0,
                            const __m256i acc[8], bool accum) {
    for (int r = 0; r < 4; ++r) {
        alignas(32) int32_t v[16];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(v),     acc[2 * r]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(v + 8), acc[2 * r + 1]);
        int32_t* crow = C + r * N + n0;
        const int64_t cols = (n0 + 16 <= N) ? 16 : static_cast<int64_t>(N - n0);
        if (accum) {
            for (int64_t j = 0; j < cols; ++j) crow[j] += v[j];
        } else {
            for (int64_t j = 0; j < cols; ++j) crow[j] = v[j];
        }
    }
}

} // namespace

void gemm_i8_avx2vnni(int32_t* C, const uint8_t* A, const int8_t* Bp,
                      int64_t M, int64_t N, int64_t K, bool accum) {
    if (M == 0 || N == 0) return;
    if (K == 0) {
        if (!accum) std::memset(C, 0, sizeof(int32_t) * static_cast<size_t>(M) * static_cast<size_t>(N));
        return;
    }
    const int64_t nt = ((N + 15) / 16) * 16;   // 面板对齐列数（含末面板补零列）
    const int64_t m4 = M - (M % 4);

    for (int64_t m0 = 0; m0 < m4; m0 += 4) {
        const uint8_t* ar[4] = {A + (m0 + 0) * K, A + (m0 + 1) * K,
                                A + (m0 + 2) * K, A + (m0 + 3) * K};
        for (int64_t n0 = 0; n0 < nt; n0 += 16) {
            __m256i acc[8];
            // 累加器恒零起算：Σ 在 k 扫描中累积，store 时按 accum 语义
            // 覆盖写 / += 写回（勿在此 load C 老值——store 再 += 会双重累加）。
            for (int i = 0; i < 8; ++i) acc[i] = _mm256_setzero_si256();
            tile_4x16_kloop(acc, ar, Bp, K, n0);
            tile_4x16_store(C + m0 * N, N, n0, acc, accum);
        }
    }

    // M 尾行（<4）：与主 tile 同构的向量路径（行数 mt = M-m4，1..3），vb 组仍被
    // 尾行复用——勿回退标量收集（实测标量尾行仅 ~2 GMAC/s，与 M=4 的 135 GMAC/s
    // 差 67×，2026-09-03 提交后复查抓出并修复）。整数累加顺序无关 → bit-exact。
    {
        const int64_t ng = (K + 3) / 4;
        const int64_t mt = M - m4;
        const uint8_t* art[3] = {A + m4 * K, A + (m4 + 1) * K, A + (m4 + 2) * K};
        const int64_t gfull = K / 4;
        for (int64_t n0 = 0; n0 < nt; n0 += 16) {
            const int64_t panel = n0 / 16;
            __m256i acct[3][2];
            for (int r = 0; r < mt; ++r) {
                acct[r][0] = _mm256_setzero_si256();
                acct[r][1] = _mm256_setzero_si256();
            }
            for (int64_t g = 0; g < gfull; ++g) {
                const int8_t* vb_lo = Bp + (panel * ng + g) * 64;
                const int8_t* vb_hi = vb_lo + 32;
                __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vb_lo));
                __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vb_hi));
                for (int64_t r = 0; r < mt; ++r) {
                    __m256i va = broadcast_group4(art[r] + g * 4);
                    acct[r][0] = _mm256_dpbusd_epi32(acct[r][0], va, vb0);
                    acct[r][1] = _mm256_dpbusd_epi32(acct[r][1], va, vb1);
                }
            }
            if (gfull < ng) {  // K%4 尾组
                const int64_t rem = K - gfull * 4;
                const int8_t* vb_lo = Bp + (panel * ng + gfull) * 64;
                const int8_t* vb_hi = vb_lo + 32;
                __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vb_lo));
                __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vb_hi));
                for (int64_t r = 0; r < mt; ++r) {
                    __m256i va = broadcast_group_tail(art[r] + gfull * 4, rem);
                    acct[r][0] = _mm256_dpbusd_epi32(acct[r][0], va, vb0);
                    acct[r][1] = _mm256_dpbusd_epi32(acct[r][1], va, vb1);
                }
            }
            // 存回（末面板补零列照算不存）
            for (int64_t r = 0; r < mt; ++r) {
                alignas(32) int32_t v[16];
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(v),     acct[r][0]);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(v + 8), acct[r][1]);
                int32_t* crow = C + (m4 + r) * N + n0;
                const int64_t cols = (n0 + 16 <= N) ? 16 : static_cast<int64_t>(N - n0);
                if (accum) {
                    for (int64_t j = 0; j < cols; ++j) crow[j] += v[j];
                } else {
                    for (int64_t j = 0; j < cols; ++j) crow[j] = v[j];
                }
            }
        }
    }
}

} // namespace sgn::mkern::gemm
#endif
