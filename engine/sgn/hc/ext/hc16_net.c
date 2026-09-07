/**
 * @file hc16_net.c
 * @brief HC16 神经网络运算扩展（MSInt 位宽链中间层）C 实现
 * @version 1.5.0
 *
 * 实现思路（与 Python 数学验证对齐）：
 *   - 对称量化：scale = max(|w|) / 32767, q = round(w / scale) ∈ [-32767, 32767]
 *     （qmin=-32767 而非 -32768，避免 _mm256_madd_epi16 输出 int32 溢出）
 *   - 直接存 int16_t（有符号，无需 offset 偏移）
 *   - 矩阵乘标量路径：int16×int16→int64 累加（防溢出）
 *   - 矩阵乘 AVX2 路径：_mm256_madd_epi16 + store→int64 累加
 *
 * 溢出分析（关键）：
 *   - _mm256_madd_epi16：result[i] = a[2i]*b[2i] + a[2i+1]*b[2i+1]
 *   - 若用 -32768：(-32768)^2 * 2 = 2,147,483,648 > INT32_MAX (2,147,483,647) → 溢出 1！
 *   - 限制 qmin=-32767：32767^2 * 2 = 2,147,450,722 < INT32_MAX → 安全
 *   - 跨多次 madd 累加必须用 int64（K=1024 时 64 次 madd，int32 必溢出）
 *
 * AVX2 矩阵乘策略：
 *   1. 转置 B (k×n → n×k)，使 b_t[j][l] 沿 l 连续（匹配 a[i][l] 的访问模式）
 *   2. 对每个 (i, j)，沿 l 步进 16，用 _mm256_madd_epi16 一次处理 16 个 int16
 *   3. 每次 madd 输出 8 个 int32，store 到临时数组后逐个累加到 int64
 *   4. 尾部 l（k % 16）用标量 int64 累加
 *
 * 参考：
 *   - 数学验证：内部档案
 *   - HC8 实现：hc8_net.c（AVX-VNNI SEH 探针模式参考）
 */

#include "hc16_net.h"

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <malloc.h>  /* _aligned_malloc / _aligned_free (MSVC) */
#include <intrin.h> /* __cpuidex / __cpuid (MSVC CPUID intrinsics) */

#ifdef _OPENMP
#include <omp.h>
#endif

/* A 阶段：AVX2 intrinsics（编译时需要 /arch:AVX2）
 * MSVC /arch:AVX2 会定义 __AVX2__ 宏 */
#ifdef __AVX2__
#include <immintrin.h>
#endif

/* ============================================================================
 * AVX2 运行时检测
 * ============================================================================ */

static int g_avx2_cache = -1;  /* -1 = 未检测, 0 = 不支持, 1 = 支持 */

int hc16_detect_avx2(void) {
    if (g_avx2_cache >= 0) return g_avx2_cache;

    /* CPUID 检查：CPUID.07H.00H:EBX[5] = AVX2 支持 */
    int cpuinfo[4] = {0, 0, 0, 0};
    __cpuidex(cpuinfo, 7, 0);
    if (!(cpuinfo[1] & (1 << 5))) {
        g_avx2_cache = 0;
        return 0;
    }

    /* AVX2 是稳定指令集（Haswell 2013+ 所有 CPU 都支持），无需 SEH 探针 */
    g_avx2_cache = 1;
    return 1;
}

/* ============================================================================
 * 64 字节对齐分配（参考 hc8_net.c）
 * ============================================================================ */

#define HC16_ALIGNED_ALLOC(n, type) \
    ((type*)_aligned_malloc((size_t)(n) * sizeof(type), 64))

/* NULL 检查与 hc8_net.c/hc4_pshufb.c 统一（_aligned_free(NULL) 跨平台 UB
 * 防御——安全审计 2026-08-16 H16-1） */
#define HC16_ALIGNED_FREE(ptr) \
    do { if ((ptr) != NULL) { _aligned_free(ptr); (ptr) = NULL; } } while (0)

static void* hc16_aligned_calloc(size_t count, size_t size) {
    void* p = _aligned_malloc(count * size, 64);
    if (p != NULL) {
        memset(p, 0, count * size);
    }
    return p;
}
#define HC16_ALIGNED_CALLOC(n, type) ((type*)hc16_aligned_calloc((size_t)(n), sizeof(type)))

/* ============================================================================
 * 默认量化方案
 * ============================================================================ */

hc16_quant_schema_t hc16_quant_default_schema(void) {
    hc16_quant_schema_t s;
    s.scale = 1.0f;
    /* qmin=-32767 而非 -32768，避免 _mm256_madd_epi16 输出 int32 溢出
     * 证明：(-32768)^2 * 2 = 2,147,483,648 > INT32_MAX (2,147,483,647)
     *       (-32767)^2 * 2 = 2,147,450,722 < INT32_MAX → 安全 */
    s.qmin  = -32767;
    s.qmax  = 32767;
    return s;
}

/* ============================================================================
 * 量化 scale 推导
 * ============================================================================ */

float hc16_quant_compute_scale(const float* w, uint32_t n) {
    if (n == 0 || w == NULL) return 1.0f;

    float max_abs = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        float a = fabsf(w[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0f) return 1.0f;
    return max_abs / 32767.0f;
}

/* ============================================================================
 * 量化 / 反量化
 * ============================================================================ */

void hc16_quantize(const float* w, uint32_t n,
                   float scale,
                   const hc16_quant_schema_t* schema,
                   int16_t* out) {
    if (n == 0 || w == NULL || out == NULL || schema == NULL) return;

    /* 防止 scale=0 导致除零 */
    if (scale == 0.0f) scale = 1.0f;
    float inv_scale = 1.0f / scale;

    for (uint32_t i = 0; i < n; ++i) {
        /* q = round(w / scale) */
        float q_f = w[i] * inv_scale;
        int32_t q = (int32_t)lroundf(q_f);

        /* clamp 到 [qmin, qmax] */
        if (q < schema->qmin) q = schema->qmin;
        else if (q > schema->qmax) q = schema->qmax;

        /* 直接存 int16_t（有符号，无需偏移） */
        out[i] = (int16_t)q;
    }
}

void hc16_dequantize(const int16_t* h, uint32_t n,
                     float scale,
                     float* out) {
    if (n == 0 || h == NULL || out == NULL) return;

    for (uint32_t i = 0; i < n; ++i) {
        /* w = q * scale */
        out[i] = (float)h[i] * scale;
    }
}

/* ============================================================================
 * HC16 矩阵乘 — 标量路径（int64 累加器）
 * ============================================================================ */

void hc16_matmul_scalar(const int16_t* a, const int16_t* b,
                        uint32_t m, uint32_t k, uint32_t n,
                        float a_scale, float b_scale,
                        float* out) {
    if (m == 0 || k == 0 || n == 0 ||
        a == NULL || b == NULL || out == NULL) {
        return;
    }

    float out_combined_scale = a_scale * b_scale;

    /* 累加 + 反量化
     * int64 累加器：最大累加值 32767^2 * 4096 ≈ 4.4e12 << INT64_MAX (9.2e18) */
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            int64_t acc = 0;
            for (uint32_t l = 0; l < k; ++l) {
                int64_t a_q = (int64_t)a[i * k + l];
                int64_t b_q = (int64_t)b[l * n + j];
                acc += a_q * b_q;
            }
            out[i * n + j] = (float)acc * out_combined_scale;
        }
    }
}

/* ============================================================================
 * HC16 矩阵乘 — AVX2 优化路径（_mm256_madd_epi16 + int64 累加）
 * ============================================================================ */

#ifdef __AVX2__

void hc16_matmul_avx2(const int16_t* a, const int16_t* b,
                      uint32_t m, uint32_t k, uint32_t n,
                      float a_scale, float b_scale,
                      float* out) {
    if (m == 0 || k == 0 || n == 0 ||
        a == NULL || b == NULL || out == NULL) {
        return;
    }

    float out_combined_scale = a_scale * b_scale;

    /* 转置 B (k×n → n×k)，使 b_t[j][l] 沿 l 连续
     * 这样 a[i][l] 和 b_t[j][l] 都沿 l 连续，可用 _mm256_loadu_si256 加载 */
    int16_t* b_t = HC16_ALIGNED_CALLOC((size_t)n * k, int16_t);
    if (b_t == NULL) {
        /* 内存分配失败，回退到标量路径 */
        hc16_matmul_scalar(a, b, m, k, n, a_scale, b_scale, out);
        return;
    }

    for (uint32_t j = 0; j < n; ++j) {
        for (uint32_t l = 0; l < k; ++l) {
            b_t[(size_t)j * k + l] = b[(size_t)l * n + j];
        }
    }

    /* AVX2 矩阵乘主循环
     * 对每个 (i, j)，沿 l 步进 16，用 _mm256_madd_epi16 一次处理 16 个 int16
     * _mm256_madd_epi16：result[t] = a[2t]*b[2t] + a[2t+1]*b[2t+1] (t=0..7)
     * 输出 8 个 int32，store 后逐个累加到 int64 */
    /* OpenMP 2.0（MSVC /openmp）要求循环变量在 for 语句外声明 */
    int ii;
    #pragma omp parallel for
    for (ii = 0; ii < (int)m; ++ii) {
        uint32_t i = (uint32_t)ii;
        const int16_t* a_row = a + (size_t)i * k;
        float* out_row = out + (size_t)i * n;

        for (uint32_t j = 0; j < n; ++j) {
            const int16_t* b_row = b_t + (size_t)j * k;
            int64_t acc = 0;
            uint32_t l = 0;

            /* AVX2 主循环：每次处理 16 个 int16 元素 */
            for (; l + 16 <= k; l += 16) {
                __m256i a_vec = _mm256_loadu_si256((const __m256i*)(a_row + l));
                __m256i b_vec = _mm256_loadu_si256((const __m256i*)(b_row + l));
                __m256i prod = _mm256_madd_epi16(a_vec, b_vec);

                /* store 8 个 int32 并累加到 int64
                 * 每次 madd 输出 int32 ≤ 2*32767^2 = 2.15e9（接近 int32 上限）
                 * 跨多次 madd 必须用 int64 累加
                 * 用 storeu（不要求对齐）避免 MSVC __attribute__ 兼容问题 */
                int32_t tmp[8];
                _mm256_storeu_si256((__m256i*)tmp, prod);
                acc += (int64_t)tmp[0] + (int64_t)tmp[1] +
                       (int64_t)tmp[2] + (int64_t)tmp[3] +
                       (int64_t)tmp[4] + (int64_t)tmp[5] +
                       (int64_t)tmp[6] + (int64_t)tmp[7];
            }

            /* 尾部处理：剩余 0~15 个元素用标量 int64 累加 */
            for (; l < k; ++l) {
                acc += (int64_t)a_row[l] * (int64_t)b_row[l];
            }

            out_row[j] = (float)acc * out_combined_scale;
        }
    }

    HC16_ALIGNED_FREE(b_t);
}

#else /* !__AVX2__ */

void hc16_matmul_avx2(const int16_t* a, const int16_t* b,
                      uint32_t m, uint32_t k, uint32_t n,
                      float a_scale, float b_scale,
                      float* out) {
    /* AVX2 未编译启用，回退到标量路径 */
    hc16_matmul_scalar(a, b, m, k, n, a_scale, b_scale, out);
}

#endif /* __AVX2__ */

/* ============================================================================
 * HC16 矩阵乘 — 自动选择最优路径
 * ============================================================================ */

void hc16_matmul(const int16_t* a, const int16_t* b,
                 uint32_t m, uint32_t k, uint32_t n,
                 float a_scale, float b_scale,
                 float* out) {
#ifdef __AVX2__
    if (hc16_detect_avx2()) {
        hc16_matmul_avx2(a, b, m, k, n, a_scale, b_scale, out);
    } else {
        hc16_matmul_scalar(a, b, m, k, n, a_scale, b_scale, out);
    }
#else
    hc16_matmul_scalar(a, b, m, k, n, a_scale, b_scale, out);
#endif
}

/* ============================================================================
 * 便捷封装：float → float 的 HC16 量化 matmul
 * ============================================================================ */

void hc16_quantized_matmul(const float* x, const float* w,
                           uint32_t m, uint32_t k, uint32_t n,
                           const hc16_quant_schema_t* schema,
                           float* out) {
    if (m == 0 || k == 0 || n == 0 ||
        x == NULL || w == NULL || out == NULL || schema == NULL) {
        return;
    }

    /* 1. 推导 scale */
    float x_scale = hc16_quant_compute_scale(x, m * k);
    float w_scale = hc16_quant_compute_scale(w, k * n);

    /* 2. 量化到 int16 */
    int16_t* x_q = HC16_ALIGNED_CALLOC((size_t)m * k, int16_t);
    int16_t* w_q = HC16_ALIGNED_CALLOC((size_t)k * n, int16_t);
    if (x_q == NULL || w_q == NULL) {
        HC16_ALIGNED_FREE(x_q);
        HC16_ALIGNED_FREE(w_q);
        return;
    }

    hc16_quantize(x, m * k, x_scale, schema, x_q);
    hc16_quantize(w, k * n, w_scale, schema, w_q);

    /* 3. int16 matmul */
    hc16_matmul(x_q, w_q, m, k, n, x_scale, w_scale, out);

    HC16_ALIGNED_FREE(x_q);
    HC16_ALIGNED_FREE(w_q);
}

/* ============================================================================
 * Per-channel scale 扩展（v1.5.0 新增）
 * ============================================================================
 *
 * 设计原则：
 *   - 整数累加路径与 per-tensor 完全一致（AVX2 madd + int64 累加器）
 *   - 仅在量化/反量化/matmul 输出阶段按行/列组合 scale
 *   - 不破坏 per-tensor 路径（独立函数）
 *
 * 维度约定（与 hc16_net.h 一致）：
 *   - 量化/反量化：w 为 rows × per_row_size 行优先，scales 长度 rows
 *   - matmul: A(m×k) 行优先，a_scales 长度 m（每行一个 scale）
 *             B(k×n) 行优先，b_scales 长度 n（每列一个 scale）
 *             C(i,j) = acc * a_scales[i] * b_scales[j]
 */

void hc16_quantize_per_channel(const float* w, uint32_t rows, uint32_t per_row_size,
                               const float* scales,
                               const hc16_quant_schema_t* schema,
                               int16_t* out) {
    if (rows == 0 || per_row_size == 0 || w == NULL || out == NULL ||
        scales == NULL || schema == NULL) {
        return;
    }

    for (uint32_t i = 0; i < rows; ++i) {
        float scale = scales[i];
        /* 防止 scale=0 导致除零 */
        if (scale == 0.0f) scale = 1.0f;
        float inv_scale = 1.0f / scale;

        const float* w_row = w + (size_t)i * per_row_size;
        int16_t* out_row = out + (size_t)i * per_row_size;

        for (uint32_t j = 0; j < per_row_size; ++j) {
            float q_f = w_row[j] * inv_scale;
            int32_t q = (int32_t)lroundf(q_f);

            if (q < schema->qmin) q = schema->qmin;
            else if (q > schema->qmax) q = schema->qmax;

            out_row[j] = (int16_t)q;
        }
    }
}

void hc16_dequantize_per_channel(const int16_t* h, uint32_t rows, uint32_t per_row_size,
                                 const float* scales,
                                 float* out) {
    if (rows == 0 || per_row_size == 0 || h == NULL || out == NULL || scales == NULL) {
        return;
    }

    for (uint32_t i = 0; i < rows; ++i) {
        float scale = scales[i];
        const int16_t* h_row = h + (size_t)i * per_row_size;
        float* out_row = out + (size_t)i * per_row_size;

        for (uint32_t j = 0; j < per_row_size; ++j) {
            out_row[j] = (float)h_row[j] * scale;
        }
    }
}

/* ============================================================================
 * Per-channel matmul — 标量路径（int64 累加器）
 *
 * 与 per-tensor 标量路径结构一致，仅输出阶段按 (i, j) 组合 scale
 * ============================================================================ */

void hc16_matmul_per_channel_scalar(const int16_t* a, const int16_t* b,
                                    uint32_t m, uint32_t k, uint32_t n,
                                    const float* a_scales, const float* b_scales,
                                    float* out) {
    if (m == 0 || k == 0 || n == 0 ||
        a == NULL || b == NULL || out == NULL ||
        a_scales == NULL || b_scales == NULL) {
        return;
    }

    for (uint32_t i = 0; i < m; ++i) {
        float a_s = a_scales[i];
        for (uint32_t j = 0; j < n; ++j) {
            int64_t acc = 0;
            for (uint32_t l = 0; l < k; ++l) {
                int64_t a_q = (int64_t)a[i * k + l];
                int64_t b_q = (int64_t)b[l * n + j];
                acc += a_q * b_q;
            }
            out[i * n + j] = (float)acc * a_s * b_scales[j];
        }
    }
}

/* ============================================================================
 * Per-channel matmul — AVX2 优化路径
 *
 * 策略：与 per-tensor AVX2 路径结构一致
 *   1. 转置 B (k×n → n×k)，使 b_t[j][l] 沿 l 连续
 *   2. OpenMP 并行外层 i 行
 *   3. 内层 (i, j) 用 _mm256_madd_epi16 累加，输出阶段组合 a_scales[i] * b_scales[j]
 *
 * 性能预期：与 per-tensor AVX2 路径相当（scale 组合在输出阶段，开销可忽略）
 * ============================================================================ */

#ifdef __AVX2__

void hc16_matmul_per_channel_avx2(const int16_t* a, const int16_t* b,
                                  uint32_t m, uint32_t k, uint32_t n,
                                  const float* a_scales, const float* b_scales,
                                  float* out) {
    if (m == 0 || k == 0 || n == 0 ||
        a == NULL || b == NULL || out == NULL ||
        a_scales == NULL || b_scales == NULL) {
        return;
    }

    /* 转置 B (k×n → n×k)，使 b_t[j][l] 沿 l 连续 */
    int16_t* b_t = HC16_ALIGNED_CALLOC((size_t)n * k, int16_t);
    if (b_t == NULL) {
        hc16_matmul_per_channel_scalar(a, b, m, k, n, a_scales, b_scales, out);
        return;
    }

    for (uint32_t j = 0; j < n; ++j) {
        for (uint32_t l = 0; l < k; ++l) {
            b_t[(size_t)j * k + l] = b[(size_t)l * n + j];
        }
    }

    int ii;
    #pragma omp parallel for
    for (ii = 0; ii < (int)m; ++ii) {
        uint32_t i = (uint32_t)ii;
        const int16_t* a_row = a + (size_t)i * k;
        float* out_row = out + (size_t)i * n;
        float a_s = a_scales[i];

        for (uint32_t j = 0; j < n; ++j) {
            const int16_t* b_row = b_t + (size_t)j * k;
            int64_t acc = 0;
            uint32_t l = 0;

            /* AVX2 主循环：每次处理 16 个 int16 元素 */
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

            /* 尾部处理：剩余 0~15 个元素用标量 int64 累加 */
            for (; l < k; ++l) {
                acc += (int64_t)a_row[l] * (int64_t)b_row[l];
            }

            out_row[j] = (float)acc * a_s * b_scales[j];
        }
    }

    HC16_ALIGNED_FREE(b_t);
}

#else /* !__AVX2__ */

void hc16_matmul_per_channel_avx2(const int16_t* a, const int16_t* b,
                                  uint32_t m, uint32_t k, uint32_t n,
                                  const float* a_scales, const float* b_scales,
                                  float* out) {
    /* AVX2 未编译启用，回退到标量路径 */
    hc16_matmul_per_channel_scalar(a, b, m, k, n, a_scales, b_scales, out);
}

#endif /* __AVX2__ */

/* ============================================================================
 * Per-channel matmul — 自动选择最优路径
 * ============================================================================ */

void hc16_matmul_per_channel(const int16_t* a, const int16_t* b,
                             uint32_t m, uint32_t k, uint32_t n,
                             const float* a_scales, const float* b_scales,
                             float* out) {
#ifdef __AVX2__
    if (hc16_detect_avx2()) {
        hc16_matmul_per_channel_avx2(a, b, m, k, n, a_scales, b_scales, out);
    } else {
        hc16_matmul_per_channel_scalar(a, b, m, k, n, a_scales, b_scales, out);
    }
#else
    hc16_matmul_per_channel_scalar(a, b, m, k, n, a_scales, b_scales, out);
#endif
}
