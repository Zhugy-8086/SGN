// gemm_avx2.cpp - mkern/gemm gemm_i16 AVX2 内核（k-pair madd + 列 quad 交错）
//
// 背景：fixes_相关修复/mkern微内核层实施计划_2026_09_03.md §3.2/§五.4（R3）。
// 结构（2026-09-03 madd-quad 重设计，取代首版 mullo N 轴方案——首版 k 标量循环
// 每 MAC 摊 vpmulld 2 uop，实测仅 naive 0.42×）：
//   - k 按【对】处理：va = A 行内 4 字节（a[k0],a[k1]）广播成 8 int16 lane
//     [a_k0,a_k1]×4；vb = B 相邻两 k 行的 4 列切片 unpacklo 交错成
//     [c0k0,c0k1,c1k0,c1k1,...]；_mm_madd_epi16 一条完成 4 列 × 2 k = 8 MAC
//     （lane=列！），cvtepi32_epi64 扩 int64 累加——B 行片 1 次加载被 4 行复用。
//   - 溢出分析（bit-exact 守卫的依据）：pair 和 = a[k0]c[k0]+a[k1]c[k1]，每积
//     ≤ 2^30（仅 (-32768)² 达到），负向积 ≥ -(2^30-2^15) → 负溢出不可能；
//     正溢出需两积同时 = +2^30，即 a[k0]=a[k1]=-32768 且 c 侧同位匹配。
//     故守卫仅需整字比较 dword==0x80008000（a 侧双 -32768，极罕见）→ 该
//     (行,k对) 回退标量 int64（无需向量扫 c 侧）。比 dot16 的"批内任一 -32768
//     即回退"粗守卫便宜（其 16 元素批对应 8 个独立 pair，无法收窄到对）。
//   - M%4 尾行走同构向量路径（mt=1..3，B 行片仍复用）；N%8 尾列标量（无越界）。
//   - 修复首版潜在越界读：N%8≠0 时部分列向量全宽 load 读穿行尾（末行越出
//     缓冲——输出正确因 store 有界，但属 UB）；重写后热循环只做完整 8 列 vec，
//     尾列全走标量。
//
// 编译：CMake 对本文件加 -mavx2（per-file，同 simd 层纪律）。
// 整数运算：逐乘积/madd pair 精确 + int64 累加（顺序无关）→ 与标量锚点
// bit-exact，无 K 上界（同 dot16 语义）。

#include "mkern/gemm/gemm_api.h"

#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>

namespace sgn::mkern::gemm {
namespace {

constexpr int16_t kI16Min = -32768;

// A 是否含 (偶对齐 i, a[i]==a[i+1]==-32768) 模式（madd pair 溢出的必要条件）。
// 一次性预扫描（O(M·K)，远小于 MAC 量级）：干净输入走无守卫热路径（模板特化
// 去掉逐 k 对分支），脏输入走守卫版——两条路径都被 boundary 测试覆盖。
inline bool a_has_danger_pairs(const int16_t* A, int64_t M, int64_t K) {
    const int64_t npair = K / 2;
    for (int64_t m = 0; m < M; ++m) {
        const int16_t* a = A + m * K;
        for (int64_t p = 0; p < npair; ++p) {
            if (a[2 * p] == kI16Min && a[2 * p + 1] == kI16Min) return true;
        }
    }
    return false;
}

// 一个完整 8 列 n-vec 的 K 全扫（k 对主循环 + K 奇数尾），行数 mt ∈ [1,4]。
// acc[r][q]：行 r、quad q（列 n0+4q..+3）的 4×int64 累加器；ex[r][j]：回退/尾 k
// 的标量余量（整数加法可交换，store 时并入不改变结果）。
// kGuard=true 时逐 (行,k对) 检查 a 侧双 -32768 并回退标量（调用方须已确认 A
// 含危险对）；false 时热路径零守卫分支（预扫描保证不触发溢出）。
template <bool kGuard>
inline void tile_kp_madd(__m256i acc[][2], int64_t ex[][8],
                         const int16_t* const* ar, int mt,
                         const int16_t* B, int64_t N, int64_t K, int64_t n0) {
    const int64_t npair = K / 2;
    for (int64_t pr = 0; pr < npair; ++pr) {
        const int64_t k0 = 2 * pr;
        const int64_t k1 = k0 + 1;
        const int16_t* b0 = B + k0 * N + n0;        // 行 k0 的 8 列
        const int16_t* b1 = b0 + N;                 // 行 k1 = k0+1 的 8 列
        const __m128i v0q0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b0));
        const __m128i v1q0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b1));
        const __m128i v0q1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b0 + 4));
        const __m128i v1q1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b1 + 4));
        const __m128i inter0 = _mm_unpacklo_epi16(v0q0, v1q0);  // [c0k0,c0k1,c1k0,c1k1,...]
        const __m128i inter1 = _mm_unpacklo_epi16(v0q1, v1q1);
        for (int r = 0; r < mt; ++r) {
            uint32_t d;
            std::memcpy(&d, ar[r] + k0, 4);         // (a[k0],a[k1]) 相邻 4 字节
            if (kGuard && d == 0x80008000u) {
                // a 侧双 -32768：madd pair 和可能溢出（冷路径，保守整对回退）
                const int16_t* a = ar[r];
                for (int j = 0; j < 8; ++j) {
                    ex[r][j] += static_cast<int64_t>(a[k0]) * static_cast<int64_t>(b0[j])
                              + static_cast<int64_t>(a[k1]) * static_cast<int64_t>(b1[j]);
                }
                continue;
            }
            const __m128i va = _mm_set1_epi32(static_cast<int32_t>(d));
            acc[r][0] = _mm256_add_epi64(acc[r][0],
                _mm256_cvtepi32_epi64(_mm_madd_epi16(va, inter0)));
            acc[r][1] = _mm256_add_epi64(acc[r][1],
                _mm256_cvtepi32_epi64(_mm_madd_epi16(va, inter1)));
        }
    }
    if (K & 1) {  // K 奇数：最后一个 k 标量并入余量
        const int64_t k = K - 1;
        const int16_t* b = B + k * N + n0;
        for (int r = 0; r < mt; ++r) {
            for (int j = 0; j < 8; ++j) {
                ex[r][j] += static_cast<int64_t>(ar[r][k]) * static_cast<int64_t>(b[j]);
            }
        }
    }
}

// tile 存回（n0 起的完整 8 列；ex 余量按列并入）
inline void tile_kp_store(int64_t* C, int64_t N, int64_t n0, int mt,
                          const __m256i acc[][2], const int64_t ex[][8], bool accum) {
    for (int r = 0; r < mt; ++r) {
        alignas(32) int64_t v[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(v),     acc[r][0]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(v + 4), acc[r][1]);
        int64_t* crow = C + r * N + n0;
        if (accum) {
            for (int j = 0; j < 8; ++j) crow[j] += v[j] + ex[r][j];
        } else {
            for (int j = 0; j < 8; ++j) crow[j] = v[j] + ex[r][j];
        }
    }
}

// N%8 尾列（部分列向量）：标量逐 (行, 列) 全 k（无越界读；N%8≠0 才有成本）
inline void tail_cols_scalar(int64_t* C, const int16_t* const* ar, int mt,
                             const int16_t* B, int64_t N, int64_t K,
                             int64_t n8full, bool accum) {
    for (int64_t c = n8full; c < N; ++c) {
        for (int r = 0; r < mt; ++r) {
            int64_t s = 0;
            for (int64_t k = 0; k < K; ++k) {
                s += static_cast<int64_t>(ar[r][k]) * static_cast<int64_t>(B[k * N + c]);
            }
            int64_t* p = C + r * N + c;
            if (accum) *p += s;
            else       *p  = s;
        }
    }
}

} // namespace

void gemm_i16_avx2(int64_t* C, const int16_t* A, const int16_t* B,
                   int64_t M, int64_t N, int64_t K, bool accum) {
    if (M == 0 || N == 0) return;
    if (K == 0) {
        if (!accum) std::memset(C, 0, sizeof(int64_t) * static_cast<size_t>(M) * static_cast<size_t>(N));
        return;
    }
    const int64_t n8full = N - (N % 8);   // 完整 8 列 vec 数（尾列标量路径）
    const int64_t m4 = M - (M % 4);
    const bool dirty = a_has_danger_pairs(A, M, K);   // 一次性预扫描（见上方说明）

    for (int64_t m0 = 0; m0 < m4; m0 += 4) {
        const int16_t* ar[4] = {A + (m0 + 0) * K, A + (m0 + 1) * K,
                                A + (m0 + 2) * K, A + (m0 + 3) * K};
        for (int64_t n0 = 0; n0 < n8full; n0 += 8) {
            __m256i acc[4][2];
            int64_t ex[4][8];
            for (int r = 0; r < 4; ++r) {
                acc[r][0] = _mm256_setzero_si256();
                acc[r][1] = _mm256_setzero_si256();
                for (int j = 0; j < 8; ++j) ex[r][j] = 0;
            }
            if (dirty) tile_kp_madd<true>(acc, ex, ar, 4, B, N, K, n0);
            else       tile_kp_madd<false>(acc, ex, ar, 4, B, N, K, n0);
            tile_kp_store(C + m0 * N, N, n0, 4, acc, ex, accum);
        }
        if (n8full < N) {
            tail_cols_scalar(C + m0 * N, ar, 4, B, N, K, n8full, accum);
        }
    }

    // M 尾行（<4）：同构向量路径（mt 行，B 行片仍复用；勿回退标量——67× 悬崖教训）
    const int mt = static_cast<int>(M - m4);
    if (mt > 0) {
        const int16_t* art[3] = {A + m4 * K, A + (m4 + 1) * K, A + (m4 + 2) * K};
        for (int64_t n0 = 0; n0 < n8full; n0 += 8) {
            __m256i acc[3][2];
            int64_t ex[3][8];
            for (int r = 0; r < mt; ++r) {
                acc[r][0] = _mm256_setzero_si256();
                acc[r][1] = _mm256_setzero_si256();
                for (int j = 0; j < 8; ++j) ex[r][j] = 0;
            }
            if (dirty) tile_kp_madd<true>(acc, ex, art, mt, B, N, K, n0);
            else       tile_kp_madd<false>(acc, ex, art, mt, B, N, K, n0);
            tile_kp_store(C + m4 * N, N, n0, mt, acc, ex, accum);
        }
        if (n8full < N) {
            tail_cols_scalar(C + m4 * N, art, mt, B, N, K, n8full, accum);
        }
    }
}

} // namespace sgn::mkern::gemm
#endif
