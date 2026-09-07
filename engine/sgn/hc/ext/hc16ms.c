/**
 * @file hc16ms.c
 * @brief HC16MS - MSInt int16 容器多视角存储 C 实现（方案 B）
 * @version 1.5.0
 *
 * 实现思路：
 *   - HC16 视角: 直接读 int16，int16×int16→int64 累加（复用 hc16_net.c 逻辑）
 *   - HC8 视角: 2× int8 展开，int8×int8→int32 累加
 *   - 零拷贝切换: 同一块内存，按视角不同解释
 *
 * 矩阵组织（关键设计）：
 *   hc16ms_t 数组 a[m×k] 在不同视角下：
 *   - HC16: int16 矩阵 (m×k)，1 元素/hc16ms_t
 *   - HC8:  int8 矩阵 (m×2k)，2 元素/hc16ms_t（低字节 + 高字节）
 *
 * HC8 视角矩阵展开（小端序）：
 *   a[i][j] (hc16ms_t) → a_8[i][2j] = low_byte, a_8[i][2j+1] = high_byte
 *   即 HC8 视角下 A (m×k hc16ms_t) 等效于 A_8 (m×2k int8)
 *
 * 参考：
 *   - HC16 C 扩展: hc16_net.c（AVX2 madd 路径参考）
 *   - 存储优化研究: architecture/msint_storage_optimization_research_2026_07_30.md
 */

#include "hc16ms.h"

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
 * AVX2 向量化字节序/nibble 序交换（读取顺序切换）
 * ============================================================================
 * _mm256_shuffle_epi8 是 per-lane 操作，两个 128-bit lane 各自交换。
 * MSVC 不支持 __attribute__((aligned))；此处 mask 由 _mm256_setr_epi8 构造，
 * 不需要静态对齐数组（参考 hc4_pshufb.c 的对齐处理）。
 * ============================================================================ */

#ifdef __AVX2__
/* 32 个 int16 批量字节交换（16 个 hc16ms_t） */
static __m256i hc16ms_bswap16_avx2(__m256i v) {
    __m256i mask = _mm256_setr_epi8(
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14
    );
    return _mm256_shuffle_epi8(v, mask);
}

/* 32 个 int8 批量 nibble 交换 */
static __m256i hc16ms_nswap8_avx2(__m256i v) {
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), _mm256_set1_epi8(0x0F));
    __m256i lo = _mm256_and_si256(_mm256_slli_epi16(v, 4), _mm256_set1_epi8(0xF0));
    return _mm256_or_si256(hi, lo);
}
#endif

/* ============================================================================
 * AVX2 运行时检测（复用 hc16_net.c 的检测逻辑）
 * ============================================================================ */

static int g_avx2_cache = -1;

int hc16ms_detect_avx2(void) {
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
 * 对齐分配
 * ============================================================================ */

#define HC16MS_ALIGNED_ALLOC(n, type) \
    ((type*)_aligned_malloc((size_t)(n) * sizeof(type), 64))

/* NULL 检查与 hc8_net.c/hc4_pshufb.c 统一（安全审计 2026-08-16 H16M-1） */
#define HC16MS_ALIGNED_FREE(ptr) \
    do { if ((ptr) != NULL) { _aligned_free(ptr); (ptr) = NULL; } } while (0)

static void* hc16ms_aligned_calloc(size_t count, size_t size) {
    void* p = _aligned_malloc(count * size, 64);
    if (p != NULL) memset(p, 0, count * size);
    return p;
}
#define HC16MS_ALIGNED_CALLOC(n, type) ((type*)hc16ms_aligned_calloc((size_t)(n), sizeof(type)))

/* ============================================================================
 * 量化 scale 推导
 * ============================================================================ */

float hc16ms_quant_compute_scale_hc16(const float* w, uint32_t n) {
    if (n == 0 || w == NULL) return 1.0f;
    float max_abs = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        float a = fabsf(w[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0f) return 1.0f;
    return max_abs / 32767.0f;
}

float hc16ms_quant_compute_scale_hc8(const float* w, uint32_t n) {
    if (n == 0 || w == NULL) return 1.0f;
    float max_abs = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        float a = fabsf(w[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0f) return 1.0f;
    return max_abs / 127.0f;
}

/* ============================================================================
 * 切换视角读取（读取顺序切换）：按相反字节序/nibble 序读取
 * ============================================================================ */

int16_t hc16ms_read_hc16_swapped(const hc16ms_t* h) {
    return hc16ms_bswap16(h->raw);
}

void hc16ms_read_hc8_swapped(const hc16ms_t* h, int8_t* high, int8_t* low) {
    /* 与正向 read_hc8 相比 high/low 对调：
     *   正向: *high = p[1] (高字节), *low = p[0] (低字节)
     *   切换: *high = p[0] (低字节), *low = p[1] (高字节) */
    const int8_t* p = (const int8_t*)&h->raw;
    *high = p[0];  /* 低字节 */
    *low  = p[1];  /* 高字节 */
}

void hc16ms_read_hc4_swapped(const hc16ms_t* h,
                              uint8_t* h0, uint8_t* h1,
                              uint8_t* h2, uint8_t* h3) {
    /* nibble 顺序反转：正向 [h0,h1,h2,h3] = [高字节高nibble, 高字节低nibble,
     * 低字节高nibble, 低字节低nibble]，切换返回反转 [h3,h2,h1,h0]
     * 即 *h0=低字节低nibble, *h1=低字节高nibble, *h2=高字节低nibble, *h3=高字节高nibble */
    const uint8_t* p = (const uint8_t*)&h->raw;
    *h0 = p[0] & 0xF;          /* 低字节低 nibble (原 h3) */
    *h1 = (p[0] >> 4) & 0xF;   /* 低字节高 nibble (原 h2) */
    *h2 = p[1] & 0xF;          /* 高字节低 nibble (原 h1) */
    *h3 = (p[1] >> 4) & 0xF;   /* 高字节高 nibble (原 h0) */
}

/* ============================================================================
 * HC16 视角量化/反量化
 * ============================================================================ */

void hc16ms_quantize_hc16(const float* w, uint32_t n,
                           float scale,
                           hc16ms_t* out) {
    if (n == 0 || w == NULL || out == NULL) return;
    if (scale == 0.0f) scale = 1.0f;
    float inv_scale = 1.0f / scale;

    for (uint32_t i = 0; i < n; ++i) {
        float q_f = w[i] * inv_scale;
        int32_t q = (int32_t)lroundf(q_f);
        /* qmin=-32767 防 _mm256_madd_epi16 int32 溢出 */
        if (q < -32767) q = -32767;
        else if (q > 32767) q = 32767;
        hc16ms_write_hc16(&out[i], (int16_t)q);
    }
}

void hc16ms_dequantize_hc16(const hc16ms_t* h, uint32_t n,
                             float scale,
                             float* out) {
    if (n == 0 || h == NULL || out == NULL) return;
    for (uint32_t i = 0; i < n; ++i) {
        out[i] = (float)hc16ms_read_hc16(&h[i]) * scale;
    }
}

/* ============================================================================
 * HC8 视角量化/反量化（每个 hc16ms_t 存 2 个 int8）
 * ============================================================================ */

void hc16ms_quantize_hc8(const float* w, uint32_t n,
                          float scale,
                          hc16ms_t* out) {
    /* w 长度 = 2n（n 个 hc16ms_t，每个存 2 个 int8） */
    if (n == 0 || w == NULL || out == NULL) return;
    if (scale == 0.0f) scale = 1.0f;
    float inv_scale = 1.0f / scale;

    for (uint32_t i = 0; i < n; ++i) {
        /* w[2i] → 低字节, w[2i+1] → 高字节 */
        float q_f_low  = w[2*i]     * inv_scale;
        float q_f_high = w[2*i + 1] * inv_scale;
        int32_t q_low  = (int32_t)lroundf(q_f_low);
        int32_t q_high = (int32_t)lroundf(q_f_high);
        if (q_low < -127) q_low = -127;
        else if (q_low > 127) q_low = 127;
        if (q_high < -127) q_high = -127;
        else if (q_high > 127) q_high = 127;

        hc16ms_write_hc8(&out[i], (int8_t)q_high, (int8_t)q_low);
    }
}

void hc16ms_dequantize_hc8(const hc16ms_t* h, uint32_t n,
                            float scale,
                            float* out) {
    /* out 长度 = 2n */
    if (n == 0 || h == NULL || out == NULL) return;
    for (uint32_t i = 0; i < n; ++i) {
        int8_t high, low;
        hc16ms_read_hc8(&h[i], &high, &low);
        out[2*i]     = (float)low  * scale;  /* 低字节 */
        out[2*i + 1] = (float)high * scale;  /* 高字节 */
    }
}

/* ============================================================================
 * HC16 视角 matmul（int16×int16→int64 累加，AVX2 _mm256_madd_epi16）
 * ============================================================================ */

void hc16ms_matmul_hc16(const hc16ms_t* a, const hc16ms_t* b,
                         uint32_t m, uint32_t k, uint32_t n,
                         float a_scale, float b_scale,
                         float* out) {
    if (m == 0 || k == 0 || n == 0 || a == NULL || b == NULL || out == NULL) return;

    float out_combined_scale = a_scale * b_scale;

#ifdef __AVX2__
    if (hc16ms_detect_avx2()) {
        /* 转置 B (k×n → n×k)，使 b_t[j][l] 沿 l 连续 */
        int16_t* b_t = HC16MS_ALIGNED_CALLOC((size_t)n * k, int16_t);
        if (b_t == NULL) {
            /* 回退到标量 */
            for (uint32_t i = 0; i < m; ++i) {
                for (uint32_t j = 0; j < n; ++j) {
                    int64_t acc = 0;
                    for (uint32_t l = 0; l < k; ++l) {
                        acc += (int64_t)hc16ms_read_hc16(&a[i*k+l]) *
                               (int64_t)hc16ms_read_hc16(&b[l*n+j]);
                    }
                    out[i*n+j] = (float)acc * out_combined_scale;
                }
            }
            return;
        }

        for (uint32_t j = 0; j < n; ++j) {
            for (uint32_t l = 0; l < k; ++l) {
                b_t[(size_t)j * k + l] = hc16ms_read_hc16(&b[(size_t)l * n + j]);
            }
        }

        /* AVX2 主循环：沿 l 步进 16，_mm256_madd_epi16 一次处理 16 个 int16 */
        int ii;
        #pragma omp parallel for
        for (ii = 0; ii < (int)m; ++ii) {
            uint32_t i = (uint32_t)ii;
            const int16_t* a_row = (const int16_t*)(a + (size_t)i * k);
            float* out_row = out + (size_t)i * n;

            for (uint32_t j = 0; j < n; ++j) {
                const int16_t* b_row = b_t + (size_t)j * k;
                int64_t acc = 0;
                uint32_t l = 0;

                for (; l + 16 <= k; l += 16) {
                    __m256i a_vec = _mm256_loadu_si256((const __m256i*)(a_row + l));
                    __m256i b_vec = _mm256_loadu_si256((const __m256i*)(b_row + l));
                    __m256i prod = _mm256_madd_epi16(a_vec, b_vec);

                    int32_t tmp[8];
                    _mm256_storeu_si256((__m256i*)tmp, prod);
                    acc += (int64_t)tmp[0] + (int64_t)tmp[1] +
                           (int64_t)tmp[2] + (int64_t)tmp[3] +
                           (int64_t)tmp[4] + (int64_t)tmp[5] +
                           (int64_t)tmp[6] + (int64_t)tmp[7];
                }

                for (; l < k; ++l) {
                    acc += (int64_t)a_row[l] * (int64_t)b_row[l];
                }

                out_row[j] = (float)acc * out_combined_scale;
            }
        }

        HC16MS_ALIGNED_FREE(b_t);
        return;
    }
#endif

    /* 标量路径 */
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            int64_t acc = 0;
            for (uint32_t l = 0; l < k; ++l) {
                acc += (int64_t)hc16ms_read_hc16(&a[i*k+l]) *
                       (int64_t)hc16ms_read_hc16(&b[l*n+j]);
            }
            out[i*n+j] = (float)acc * out_combined_scale;
        }
    }
}

/* ============================================================================
 * HC16 切换视角 matmul（先对 A、B 做 bswap16，再调用正向 hc16ms_matmul_hc16）
 * ============================================================================ */

void hc16ms_matmul_hc16_swapped(const hc16ms_t* a, const hc16ms_t* b,
                                 uint32_t m, uint32_t k, uint32_t n,
                                 float a_scale, float b_scale, float* out) {
    if (m == 0 || k == 0 || n == 0 || a == NULL || b == NULL || out == NULL) return;

    /* 分配临时缓冲区，存放 bswap16 后的 A、B */
    hc16ms_t* a_sw = HC16MS_ALIGNED_CALLOC((size_t)m * k, hc16ms_t);
    hc16ms_t* b_sw = HC16MS_ALIGNED_CALLOC((size_t)k * n, hc16ms_t);
    if (a_sw == NULL || b_sw == NULL) {
        HC16MS_ALIGNED_FREE(a_sw);
        HC16MS_ALIGNED_FREE(b_sw);
        return;
    }

    /* 对 A、B 每个 hc16ms_t 的 raw 做 bswap16 */
    size_t a_n = (size_t)m * k;
    for (size_t i = 0; i < a_n; ++i) {
        a_sw[i].raw = hc16ms_bswap16(a[i].raw);
    }
    size_t b_n = (size_t)k * n;
    for (size_t i = 0; i < b_n; ++i) {
        b_sw[i].raw = hc16ms_bswap16(b[i].raw);
    }

    /* 调用现有正向 hc16ms_matmul_hc16 */
    hc16ms_matmul_hc16(a_sw, b_sw, m, k, n, a_scale, b_scale, out);

    HC16MS_ALIGNED_FREE(a_sw);
    HC16MS_ALIGNED_FREE(b_sw);
}

/* ============================================================================
 * HC8 视角 matmul（int8×int8→int32 累加）
 *
 * 矩阵组织：A (m×k hc16ms_t) → A_8 (m×2k int8), B (k×n hc16ms_t) → B_8 (2k×n int8)
 * 运算: C[i][j] = sum_{l=0..2k-1} A_8[i][l] * B_8[l][j]
 * ============================================================================ */

void hc16ms_matmul_hc8(const hc16ms_t* a, const hc16ms_t* b,
                        uint32_t m, uint32_t k, uint32_t n,
                        float a_scale, float b_scale,
                        float* out) {
    if (m == 0 || k == 0 || n == 0 || a == NULL || b == NULL || out == NULL) return;

    float out_combined_scale = a_scale * b_scale;
    uint32_t k2 = 2 * k;  /* HC8 视角下 k 维翻倍 */

    /* 展开为 int8 矩阵：A (m×k hc16ms_t) → A_8 (m×2k int8) */
    int8_t* a_8 = HC16MS_ALIGNED_CALLOC((size_t)m * k2, int8_t);
    int8_t* b_8 = HC16MS_ALIGNED_CALLOC((size_t)k2 * n, int8_t);
    if (a_8 == NULL || b_8 == NULL) {
        HC16MS_ALIGNED_FREE(a_8);
        HC16MS_ALIGNED_FREE(b_8);
        return;
    }

    /* 展开 A: a[i][j] (hc16ms_t) → a_8[i][2j]=low, a_8[i][2j+1]=high */
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < k; ++j) {
            int8_t high, low;
            hc16ms_read_hc8(&a[(size_t)i * k + j], &high, &low);
            a_8[(size_t)i * k2 + 2*j]     = low;
            a_8[(size_t)i * k2 + 2*j + 1] = high;
        }
    }

    /* 展开 B: b[j][l] (hc16ms_t) → b_8[2j][l]=low, b_8[2j+1][l]=high
     * 注意 B 是 (k×n)，b[j][l] 对应行优先 b[j*n + l]
     * 展开后 B_8 (2k×n): b_8[2j][l] = low, b_8[2j+1][l] = high */
    for (uint32_t j = 0; j < k; ++j) {
        for (uint32_t l = 0; l < n; ++l) {
            int8_t high, low;
            hc16ms_read_hc8(&b[(size_t)j * n + l], &high, &low);
            b_8[(size_t)(2*j)     * n + l] = low;
            b_8[(size_t)(2*j + 1) * n + l] = high;
        }
    }

    /* int8×int8→int32 累加（标量路径，后续可优化 AVX2 maddubs） */
    int ii;
    #pragma omp parallel for
    for (ii = 0; ii < (int)m; ++ii) {
        uint32_t i = (uint32_t)ii;
        for (uint32_t j = 0; j < n; ++j) {
            int32_t acc = 0;
            for (uint32_t l = 0; l < k2; ++l) {
                acc += (int32_t)a_8[(size_t)i * k2 + l] *
                       (int32_t)b_8[(size_t)l * n + j];
            }
            out[(size_t)i * n + j] = (float)acc * out_combined_scale;
        }
    }

    HC16MS_ALIGNED_FREE(a_8);
    HC16MS_ALIGNED_FREE(b_8);
}

/* ============================================================================
 * 便捷封装：float → float 的多视角量化 matmul
 * ============================================================================ */

void hc16ms_quantized_matmul_hc16(const float* x, const float* w,
                                   uint32_t m, uint32_t k, uint32_t n,
                                   float* out) {
    if (m == 0 || k == 0 || n == 0 || x == NULL || w == NULL || out == NULL) return;

    float x_scale = hc16ms_quant_compute_scale_hc16(x, m * k);
    float w_scale = hc16ms_quant_compute_scale_hc16(w, k * n);

    hc16ms_t* x_q = HC16MS_ALIGNED_CALLOC((size_t)m * k, hc16ms_t);
    hc16ms_t* w_q = HC16MS_ALIGNED_CALLOC((size_t)k * n, hc16ms_t);
    if (x_q == NULL || w_q == NULL) {
        HC16MS_ALIGNED_FREE(x_q);
        HC16MS_ALIGNED_FREE(w_q);
        return;
    }

    hc16ms_quantize_hc16(x, m * k, x_scale, x_q);
    hc16ms_quantize_hc16(w, k * n, w_scale, w_q);

    hc16ms_matmul_hc16(x_q, w_q, m, k, n, x_scale, w_scale, out);

    HC16MS_ALIGNED_FREE(x_q);
    HC16MS_ALIGNED_FREE(w_q);
}

void hc16ms_quantized_matmul_hc8(const float* x, const float* w,
                                  uint32_t m, uint32_t k, uint32_t n,
                                  float* out) {
    if (m == 0 || k == 0 || n == 0 || x == NULL || w == NULL || out == NULL) return;

    /* HC8 视角: x 长度 2*m*k, w 长度 2*k*n */
    float x_scale = hc16ms_quant_compute_scale_hc8(x, 2 * m * k);
    float w_scale = hc16ms_quant_compute_scale_hc8(w, 2 * k * n);

    hc16ms_t* x_q = HC16MS_ALIGNED_CALLOC((size_t)m * k, hc16ms_t);
    hc16ms_t* w_q = HC16MS_ALIGNED_CALLOC((size_t)k * n, hc16ms_t);
    if (x_q == NULL || w_q == NULL) {
        HC16MS_ALIGNED_FREE(x_q);
        HC16MS_ALIGNED_FREE(w_q);
        return;
    }

    hc16ms_quantize_hc8(x, m * k, x_scale, x_q);
    hc16ms_quantize_hc8(w, k * n, w_scale, w_q);

    hc16ms_matmul_hc8(x_q, w_q, m, k, n, x_scale, w_scale, out);

    HC16MS_ALIGNED_FREE(x_q);
    HC16MS_ALIGNED_FREE(w_q);
}
