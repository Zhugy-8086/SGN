/**
 * @file hc4_pshufb.c
 * @brief HC4 PSHUFB LUT - int4×int4→int8 查表乘法 C 实现
 * @version 1.6.0
 *
 * 实现思路：
 *   1. 预计算 16×16 乘法 LUT（256 字节）
 *   2. PSHUFB 批量乘法：对每个 b 值迭代，用 _mm256_shuffle_epi8 查表
 *   3. matmul：沿 k 维度累加 PSHUFB 乘积
 *
 * PSHUFB 关键技巧：
 *   _mm256_shuffle_epi8 是 per-lane 操作（128-bit lane 独立）
 *   需要用 _mm256_broadcastsi128_si256 把 16 字节 LUT 广播到两个 lane
 *
 * 参考：
 *   - 数学验证：architecture/msint_math_validation_results_2026_07_30.md
 */

#include "hc4_pshufb.h"

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <malloc.h>
#include <intrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* ============================================================================
 * AVX2 运行时检测
 * ============================================================================ */

static int g_avx2_cache = -1;

int hc4_pshufb_detect_avx2(void) {
    if (g_avx2_cache >= 0) return g_avx2_cache;
    int cpuinfo[4] = {0, 0, 0, 0};
    __cpuidex(cpuinfo, 7, 0);
    /* AVX2: CPUID.07H.00H:EBX[5] */
    if (!(cpuinfo[1] & (1 << 5))) {
        g_avx2_cache = 0;
        return 0;
    }
    g_avx2_cache = 1;
    return 1;
}

/* ============================================================================
 * 16×16 乘法 LUT（256 字节，64 字节对齐）
 * ============================================================================

 * LUT[b*16 + a] = (uint8_t)(a * b)
 * b=0:  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
 * b=1:  0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
 * b=2:  0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30
 * ...
 * b=15: 0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225
 */

/* 静态初始化 LUT（编译期计算）
 * MSVC 用 __declspec(align(64))，GCC/Clang 用 __attribute__((aligned(64))) */
#ifdef _MSC_VER
__declspec(align(64)) static uint8_t g_mul_lut[256];
#else
static uint8_t g_mul_lut[256] __attribute__((aligned(64)));
#endif

static void hc4_init_lut(void) {
    static volatile long initialized = 0;
    if (_InterlockedCompareExchange(&initialized, 1, 0) == 1) return;
    for (int b = 0; b < 16; ++b) {
        for (int a = 0; a < 16; ++a) {
            g_mul_lut[b * 16 + a] = (uint8_t)(a * b);
        }
    }
}

/* ============================================================================
 * 对齐分配
 * ============================================================================ */

#define HC4_PSHUFB_ALIGNED_ALLOC(n, type) \
    ((type*)_aligned_malloc((size_t)(n) * sizeof(type), 64))

#define HC4_PSHUFB_ALIGNED_FREE(ptr) \
    do { if ((ptr) != NULL) { _aligned_free(ptr); (ptr) = NULL; } } while (0)

/* ============================================================================
 * PSHUFB 单批次乘法：32 个 uint4 × uint4 → 32 个 uint8
 * ============================================================================ */

void hc4_pshufb_mul_32(const uint8_t* a, const uint8_t* b, uint8_t* out) {
    hc4_init_lut();

#ifdef __AVX2__
    if (hc4_pshufb_detect_avx2()) {
        __m256i a_vec = _mm256_loadu_si256((const __m256i*)a);
        __m256i b_vec = _mm256_loadu_si256((const __m256i*)b);
        __m256i result = _mm256_setzero_si256();

        /* 对每个 b 值 v（0-15），用 PSHUFB 查 LUT_v */
        for (int v = 0; v < 16; ++v) {
            /* 加载 16 字节 LUT_v，广播到两个 128-bit lane */
            __m128i lut_128 = _mm_load_si128((const __m128i*)&g_mul_lut[v * 16]);
            __m256i lut_256 = _mm256_broadcastsi128_si256(lut_128);

            /* 创建掩码：b == v 的位置为 0xFF，否则 0x00 */
            __m256i cmp_val = _mm256_set1_epi8((char)v);
            __m256i mask = _mm256_cmpeq_epi8(b_vec, cmp_val);

            /* PSHUFB 查表：prod[i] = LUT_v[a[i]]（a[i] < 16 时） */
            __m256i prod = _mm256_shuffle_epi8(lut_256, a_vec);

            /* 用掩码过滤：只保留 b == v 的位置 */
            prod = _mm256_and_si256(prod, mask);

            /* 累加：每位置只命中一次 b 值，用 OR 即可 */
            result = _mm256_or_si256(result, prod);
        }

        _mm256_storeu_si256((__m256i*)out, result);
        return;
    }
#endif

    /* 标量回退路径 */
    for (int i = 0; i < 32; ++i) {
        out[i] = g_mul_lut[b[i] * 16 + a[i]];
    }
}

/* ============================================================================
 * HC4 PSHUFB matmul
 * ============================================================================ */

void hc4_pshufb_matmul(const uint8_t* a, const uint8_t* b,
                        uint32_t m, uint32_t k, uint32_t n,
                        int32_t* out) {
    if (m == 0 || k == 0 || n == 0 || a == NULL || b == NULL || out == NULL) return;

    hc4_init_lut();

    /* 初始化输出为 0 */
    memset(out, 0, sizeof(int32_t) * m * n);

#ifdef __AVX2__
    if (hc4_pshufb_detect_avx2()) {
        /* 转置 B (k×n → n×k)，使 b_t[j][l] 沿 l 连续 */
        uint8_t* b_t = HC4_PSHUFB_ALIGNED_ALLOC((size_t)n * k, uint8_t);
        if (b_t == NULL) {
            /* 回退到标量 */
            for (uint32_t i = 0; i < m; ++i) {
                for (uint32_t j = 0; j < n; ++j) {
                    int32_t acc = 0;
                    for (uint32_t l = 0; l < k; ++l) {
                        acc += (int32_t)a[i * k + l] * (int32_t)b[l * n + j];
                    }
                    out[i * n + j] = acc;
                }
            }
            return;
        }

        for (uint32_t j = 0; j < n; ++j) {
            for (uint32_t l = 0; l < k; ++l) {
                b_t[(size_t)j * k + l] = b[(size_t)l * n + j];
            }
        }

        /* 预加载 16 个 LUT 到寄存器（循环外加载，减少内存访问） */
        __m256i lut_256[16];
        for (int v = 0; v < 16; ++v) {
            __m128i lut_128 = _mm_load_si128((const __m128i*)&g_mul_lut[v * 16]);
            lut_256[v] = _mm256_broadcastsi128_si256(lut_128);
        }

        /* 主循环：沿 k 维度步进 32，一次处理 32 个乘积 */
        int ii;
        #pragma omp parallel for
        for (ii = 0; ii < (int)m; ++ii) {
            uint32_t i = (uint32_t)ii;
            const uint8_t* a_row = a + (size_t)i * k;

            for (uint32_t j = 0; j < n; ++j) {
                const uint8_t* b_row = b_t + (size_t)j * k;
                int32_t acc = 0;
                uint32_t l = 0;

                /* AVX2 主循环：步进 32 */
                for (; l + 32 <= k; l += 32) {
                    __m256i a_vec = _mm256_loadu_si256((const __m256i*)(a_row + l));
                    __m256i b_vec = _mm256_loadu_si256((const __m256i*)(b_row + l));
                    __m256i prod = _mm256_setzero_si256();

                    /* 对每个 b 值 v，用 PSHUFB 查表 */
                    for (int v = 0; v < 16; ++v) {
                        __m256i cmp_val = _mm256_set1_epi8((char)v);
                        __m256i mask = _mm256_cmpeq_epi8(b_vec, cmp_val);
                        __m256i p = _mm256_shuffle_epi8(lut_256[v], a_vec);
                        p = _mm256_and_si256(p, mask);
                        prod = _mm256_or_si256(prod, p);
                    }

                    /* 累加 32 个 uint8 乘积到 int32
                     * SSSE3 优化: _mm256_sad_epu8 替代 unpack+madd
                     * sad 将 256-bit 分成 2 个 128-bit lane，每 lane 内 2 组 8 字节 SAD，
                     * 4 个 16-bit sum 分别位于 word 0 / 4 / 8 / 12（256-bit 全局
                     * word 索引）。原代码取 word 0/2/4/6——word2/6 恒为 0、
                     * word4 被重复计入 sum[8..15]，等效丢失每组后 16 字节乘积，
                     * 结果静默错误（存量失败 test_4/5/6 根因） */
                    __m256i zero = _mm256_setzero_si256();
                    __m256i sad = _mm256_sad_epu8(prod, zero);
                    acc += (int32_t)(uint16_t)_mm256_extract_epi16(sad, 0);   /* sum[0..7]   */
                    acc += (int32_t)(uint16_t)_mm256_extract_epi16(sad, 4);   /* sum[8..15]  */
                    acc += (int32_t)(uint16_t)_mm256_extract_epi16(sad, 8);   /* sum[16..23] */
                    acc += (int32_t)(uint16_t)_mm256_extract_epi16(sad, 12);  /* sum[24..31] */
                }

                /* 标量尾部 */
                for (; l < k; ++l) {
                    acc += (int32_t)a_row[l] * (int32_t)b_row[l];
                }

                out[(size_t)i * n + j] = acc;
            }
        }

        HC4_PSHUFB_ALIGNED_FREE(b_t);
        return;
    }
#endif

    /* 标量路径 */
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            int32_t acc = 0;
            for (uint32_t l = 0; l < k; ++l) {
                acc += (int32_t)a[i * k + l] * (int32_t)b[l * n + j];
            }
            out[i * n + j] = acc;
        }
    }
}

/* ============================================================================
 * 便捷封装：float → float 的 HC4 PSHUFB 量化 matmul
 * ============================================================================ */

void hc4_pshufb_quantized_matmul(const float* x, const float* w,
                                  uint32_t m, uint32_t k, uint32_t n,
                                  float* out) {
    if (m == 0 || k == 0 || n == 0 || x == NULL || w == NULL || out == NULL) return;

    /* 推导量化 scale：uint4 中心化，范围 [0, 15]，中心 8 对应 0 */
    /* q = round(w / scale) + 8, clamp to [0, 15] */
    /* w_approx = (q - 8) * scale */
    /* scale = max(|w|) / 7（uint4 有符号范围 [-7, 7]） */

    float x_max = 0.0f, w_max = 0.0f;
    size_t x_total = (size_t)m * k;
    size_t w_total = (size_t)k * n;
    for (size_t i = 0; i < x_total; ++i) {
        float a = fabsf(x[i]);
        if (a > x_max) x_max = a;
    }
    for (size_t i = 0; i < w_total; ++i) {
        float a = fabsf(w[i]);
        if (a > w_max) w_max = a;
    }

    float x_scale = (x_max > 0.0f) ? (x_max / 7.0f) : 1.0f;
    float w_scale = (w_max > 0.0f) ? (w_max / 7.0f) : 1.0f;
    float out_scale = x_scale * w_scale;

    /* 量化到 uint4（中心化） */
    uint8_t* x_q = HC4_PSHUFB_ALIGNED_ALLOC((size_t)m * k, uint8_t);
    uint8_t* w_q = HC4_PSHUFB_ALIGNED_ALLOC((size_t)k * n, uint8_t);
    if (x_q == NULL || w_q == NULL) {
        HC4_PSHUFB_ALIGNED_FREE(x_q);
        HC4_PSHUFB_ALIGNED_FREE(w_q);
        return;
    }

    float x_inv = 1.0f / x_scale;
    float w_inv = 1.0f / w_scale;

    for (size_t i = 0; i < x_total; ++i) {
        int32_t q = (int32_t)lroundf(x[i] * x_inv) + 8;
        if (q < 0) q = 0;
        else if (q > 15) q = 15;
        x_q[i] = (uint8_t)q;
    }

    for (size_t i = 0; i < w_total; ++i) {
        int32_t q = (int32_t)lroundf(w[i] * w_inv) + 8;
        if (q < 0) q = 0;
        else if (q > 15) q = 15;
        w_q[i] = (uint8_t)q;
    }

    /* PSHUFB matmul */
    int32_t* out_int = HC4_PSHUFB_ALIGNED_ALLOC((size_t)m * n, int32_t);
    if (out_int == NULL) {
        HC4_PSHUFB_ALIGNED_FREE(x_q);
        HC4_PSHUFB_ALIGNED_FREE(w_q);
        return;
    }

    hc4_pshufb_matmul(x_q, w_q, m, k, n, out_int);

    /* 反量化：减去偏移项
     * C[i][j] = sum_l (x_q[i][l] - 8) * (w_q[l][j] - 8) * out_scale
     *         = out_scale * (sum_l x_q*w_q - 8*sum_l x_q - 8*sum_l w_q + 64*k)
     * 这里直接用 out_int（已是 x_q*w_q 的和），需要减去偏移项
     */
    /* 预计算 sum_l x_q[i][l] 和 sum_l w_q[l][j] */
    int32_t* x_sum = (int32_t*)calloc(m, sizeof(int32_t));
    int32_t* w_sum = (int32_t*)calloc(n, sizeof(int32_t));
    if (x_sum == NULL || w_sum == NULL) {
        free(x_sum);
        free(w_sum);
        free(x_q);
        free(w_q);
        free(out_int);
        return;
    }

    for (uint32_t i = 0; i < m; ++i) {
        int32_t s = 0;
        for (uint32_t l = 0; l < k; ++l) {
            s += x_q[(size_t)i * k + l];
        }
        x_sum[i] = s;
    }

    for (uint32_t j = 0; j < n; ++j) {
        int32_t s = 0;
        for (uint32_t l = 0; l < k; ++l) {
            s += w_q[(size_t)l * n + j];
        }
        w_sum[j] = s;
    }

    /* C = out_scale * (out_int - 8*x_sum[i] - 8*w_sum[j] + 64*k) */
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            int32_t adjusted = out_int[(size_t)i * n + j]
                             - 8 * x_sum[i]
                             - 8 * w_sum[j]
                             + 64 * (int32_t)k;
            out[(size_t)i * n + j] = (float)adjusted * out_scale;
        }
    }

    free(x_sum);
    free(w_sum);
    HC4_PSHUFB_ALIGNED_FREE(x_q);
    HC4_PSHUFB_ALIGNED_FREE(w_q);
    HC4_PSHUFB_ALIGNED_FREE(out_int);
}
