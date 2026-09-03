// gemm_avx512vnni.cpp - mkern/gemm gemm_i8 AVX512-VNNI 内核（4 行 × 32 列 tile）
//
// 背景：fixes_相关修复/mkern微内核层实施计划_2026_09_03.md §3.2（R3，服务器加速）。
// 结构与 256 位版（gemm_avx2vnni.cpp）同构：消费 pack_b_i8 的 VNNI 面板布局，
// 每个 _mm512_dpbusd_epi32 的 16 个 int32 lane 对应一个 16 列面板（64 字节块 =
// 16 列 × 4 k），累加器即 C 列向量；A 侧 4 字节 k 组 set1 广播。tile 4 行 × 2 面板
// = 32 列，8 个 zmm 累加器。每次 dpbusd 64 MAC，dpbusd 端口为唯一瓶颈。
//
// 编译：CMake 对本文件加 -mavx512f -mavx512bw -mavx512vnni（符号常驻编译，
// 运行时由 gemm_dispatch 的 CPUID 检测选入；本机无 AVX-512 仅编译验证，
// 与 simd 层 avx512vnni 同一「待远程 EPYC 验证」状态——见实施计划 §五.3 欠账）。
// 整数运算 bit-exact；溢出安全域 K ≤ 65536（同 gemm_api.h 契约）。

#include "mkern/gemm/gemm_api.h"

#include <cstring>

#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
#include <immintrin.h>

namespace sgn::mkern::gemm {
namespace {

// 满 4 k 字节组广播（定长 4 字节读 → 内联单 mov，热循环零开销；变长尾组见下）
inline __m512i broadcast_group512(const uint8_t* a) {
    uint32_t d;
    std::memcpy(&d, a, 4);
    return _mm512_set1_epi32(static_cast<int32_t>(d));
}

// K 尾组（K%4 ≠ 0，≤3 有效字节）零填充装配——每行仅一次（冷路径）
inline __m512i broadcast_group512_tail(const uint8_t* a, int64_t rem) {
    uint32_t d = 0;
    std::memcpy(&d, a, static_cast<size_t>(rem));
    return _mm512_set1_epi32(static_cast<int32_t>(d));
}

} // namespace

void gemm_i8_avx512vnni(int32_t* C, const uint8_t* A, const int8_t* Bp,
                        int64_t M, int64_t N, int64_t K, bool accum) {
    if (M == 0 || N == 0) return;
    if (K == 0) {
        if (!accum) std::memset(C, 0, sizeof(int32_t) * static_cast<size_t>(M) * static_cast<size_t>(N));
        return;
    }
    const int64_t ng = (K + 3) / 4;
    const int64_t gfull = K / 4;
    const int64_t np = (N + 15) / 16;      // 16 列面板数（含末面板补零）
    const int64_t m4 = M - (M % 4);

    for (int64_t m0 = 0; m0 < m4; m0 += 4) {
        for (int64_t p0 = 0; p0 < np; p0 += 2) {   // 每次 2 面板 = 32 列
            const int64_t panels = (p0 + 2 <= np) ? 2 : 1;
            __m512i acc[4][2];
            for (int r = 0; r < 4; ++r) {
                for (int q = 0; q < panels; ++q) {
                    // 恒零起算（勿 load C 老值——store 再 += 会双重累加，同 256 位版）
                    acc[r][q] = _mm512_setzero_si512();
                }
            }
            for (int64_t g = 0; g < gfull; ++g) {
                __m512i vb[2];
                for (int q = 0; q < panels; ++q) {
                    vb[q] = _mm512_loadu_si512(
                        reinterpret_cast<const void*>(Bp + ((p0 + q) * ng + g) * 64));
                }
                for (int r = 0; r < 4; ++r) {
                    const __m512i va = broadcast_group512(A + (m0 + r) * K + g * 4);
                    for (int q = 0; q < panels; ++q) {
                        acc[r][q] = _mm512_dpbusd_epi32(acc[r][q], va, vb[q]);
                    }
                }
            }
            if (gfull < ng) {  // K%4 尾组
                const int64_t rem = K - gfull * 4;
                __m512i vb[2];
                for (int q = 0; q < panels; ++q) {
                    vb[q] = _mm512_loadu_si512(
                        reinterpret_cast<const void*>(Bp + ((p0 + q) * ng + gfull) * 64));
                }
                for (int r = 0; r < 4; ++r) {
                    const __m512i va = broadcast_group512_tail(A + (m0 + r) * K + gfull * 4, rem);
                    for (int q = 0; q < panels; ++q) {
                        acc[r][q] = _mm512_dpbusd_epi32(acc[r][q], va, vb[q]);
                    }
                }
            }
            // 存回（末面板列 ≥ N 的补零列跳过）
            for (int r = 0; r < 4; ++r) {
                for (int q = 0; q < panels; ++q) {
                    const int64_t n0 = (p0 + q) * 16;
                    const int64_t cols = (n0 + 16 <= N) ? 16 : static_cast<int64_t>(N - n0);
                    int32_t v[16];
                    _mm512_storeu_si512(reinterpret_cast<void*>(v), acc[r][q]);
                    int32_t* crow = C + (m0 + r) * N + n0;
                    if (accum) {
                        for (int64_t j = 0; j < cols; ++j) crow[j] += v[j];
                    } else {
                        for (int64_t j = 0; j < cols; ++j) crow[j] = v[j];
                    }
                }
            }
        }
    }

    // M 尾行（<4）：与主 tile 同构的向量路径（行数 mt = M-m4，1..3），vb 组仍被
    // 尾行复用（同 256 位版的 67× 悬崖教训）。
    {
        const int64_t mt = M - m4;
        const uint8_t* art[3] = {A + m4 * K, A + (m4 + 1) * K, A + (m4 + 2) * K};
        const int64_t gfull = K / 4;
        for (int64_t p0 = 0; p0 < np; p0 += 2) {
            const int64_t panels = (p0 + 2 <= np) ? 2 : 1;
            __m512i acct[3][2];
            for (int64_t r = 0; r < mt; ++r) {
                for (int q = 0; q < panels; ++q) {
                    acct[r][q] = _mm512_setzero_si512();
                }
            }
            for (int64_t g = 0; g < gfull; ++g) {
                __m512i vb[2];
                for (int q = 0; q < panels; ++q) {
                    vb[q] = _mm512_loadu_si512(
                        reinterpret_cast<const void*>(Bp + ((p0 + q) * ng + g) * 64));
                }
                for (int64_t r = 0; r < mt; ++r) {
                    const __m512i va = broadcast_group512(art[r] + g * 4);
                    for (int q = 0; q < panels; ++q) {
                        acct[r][q] = _mm512_dpbusd_epi32(acct[r][q], va, vb[q]);
                    }
                }
            }
            if (gfull < ng) {  // K%4 尾组
                const int64_t rem = K - gfull * 4;
                __m512i vb[2];
                for (int q = 0; q < panels; ++q) {
                    vb[q] = _mm512_loadu_si512(
                        reinterpret_cast<const void*>(Bp + ((p0 + q) * ng + gfull) * 64));
                }
                for (int64_t r = 0; r < mt; ++r) {
                    const __m512i va = broadcast_group512_tail(art[r] + gfull * 4, rem);
                    for (int q = 0; q < panels; ++q) {
                        acct[r][q] = _mm512_dpbusd_epi32(acct[r][q], va, vb[q]);
                    }
                }
            }
            for (int64_t r = 0; r < mt; ++r) {
                for (int q = 0; q < panels; ++q) {
                    const int64_t n0 = (p0 + q) * 16;
                    const int64_t cols = (n0 + 16 <= N) ? 16 : static_cast<int64_t>(N - n0);
                    int32_t v[16];
                    _mm512_storeu_si512(reinterpret_cast<void*>(v), acct[r][q]);
                    int32_t* crow = C + (m4 + r) * N + n0;
                    if (accum) {
                        for (int64_t j = 0; j < cols; ++j) crow[j] += v[j];
                    } else {
                        for (int64_t j = 0; j < cols; ++j) crow[j] = v[j];
                    }
                }
            }
        }
    }
}

} // namespace sgn::mkern::gemm
#endif
