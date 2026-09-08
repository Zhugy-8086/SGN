/**
 * @file hc8_net.c
 * @brief HC8 神经网络运算扩展（阶段 1.3 全整数路径）C 实现
 * @version 1.3.0
 *
 * 实现思路（与纯 Python 版本对齐）：
 *   - 对称量化：scale = max(|w|) / 127, q = round(w / scale) ∈ [-127, 127]
 *   - 偏移到无符号：q_u = q + 128 ∈ [1, 255]
 *   - HC8 只用 v[0] 存 q_u，v[1..5] = 0
 *   - 矩阵乘：int8 × int8 → int32 累加 → 反量化 → 重新量化
 *   - ReLU：q_u < 128 → 负值 → q_u = 128
 *
 * 性能：
 *   - 矩阵乘的累加循环用 int32，不会溢出（k 最大约 4096，每个元素 |q|≤127，
 *     最大累加值约 4096 * 127 * 127 ≈ 6.6e7，远小于 int32 上限 2.1e9）
 *   - 后续优化可用 SIMD（SSE2/AVX2）并行累加，约 4-8 倍加速
 *
 * 参考：
 *   - 纯 Python 实现：legacy/traditional/stage_1_3_int_path/hc_matmul.py
 *   - 静态预测试报告：legacy/traditional/stage_1_3_int_path/stage_1_3_preflight_static_review.md
 *     （已修复 BUG-1/2/3，本 C 实现继承修复后的语义）
 */

/* hc8_net.h 在本扩展目录（不在 engine/hc/sgn/include/hc/），
 * setup_net.py 会把本目录加入 include_dirs */
#include "hc8_net.h"

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <malloc.h>  /* _aligned_malloc / _aligned_free (MSVC) */
#include <intrin.h> /* __cpuidex / __cpuid (MSVC CPUID intrinsics) */
#include <excpt.h>  /* __try / __except (SEH, 结构化异常处理) */

#ifdef _OPENMP
#include <omp.h>
#endif

/* A 阶段：AVX2 intrinsics（编译时需要 /arch:AVX2）
 * MSVC /arch:AVX2 会定义 __AVX2__ 宏 */
#ifdef __AVX2__
#include <immintrin.h>
#endif

/* ============================================================================
 * AVX-VNNI 运行时检测（v1.5.1，2026-07-22）
 *
 * 问题：setup_net.py 用 define_macros=[("__AVXVNNI__","1")] 让编译器生成
 *   _mm256_dpbusd_epi32 指令，但 CPU 不一定真正支持 AVX-VNNI。
 *
 *   已知问题：部分 Zhaoxin/Hygon CPU 在 CPUID 中报告 AVX-VNNI 支持
 *   （CPUID.07H.01H:EAX[4]=1），但实际执行 VPDPBUSD 指令会触发
 *   STATUS_ILLEGAL_INSTRUCTION (0xC000001D)。
 *
 * 方案：两阶段检测
 *   1. CPUID 检查：CPUID.07H.01H:EAX[4] = 1？
 *   2. SEH 探针：实际执行一次 _mm256_dpbusd_epi32，用 __try/__except
 *      捕获 ILLEGAL_INSTRUCTION 异常
 *
 *   只有两阶段都通过才返回 1（支持 VNNI），否则回退标量路径。
 * ============================================================================ */

/* STATUS_ILLEGAL_INSTRUCTION = 0xC000001D (不依赖 windows.h) */
#define HC_STATUS_ILLEGAL_INSTRUCTION 0xC000001DL

static int g_avx_vnni_cache = -1;  /* -1 = 未检测, 0 = 不支持, 1 = 支持 */
static volatile int g_vnni_probe_sink = 0;  /* 防止编译器优化掉探针指令 */

int hc_detect_avx_vnni(void) {
    if (g_avx_vnni_cache >= 0) return g_avx_vnni_cache;

    /* 阶段 1: CPUID 检查 */
    int cpuinfo[4] = {0, 0, 0, 0};
    __cpuidex(cpuinfo, 7, 1);
    /* AVX-VNNI: CPUID.07H.01H:EAX[4] */
    if (!(cpuinfo[0] & (1 << 4))) {
        g_avx_vnni_cache = 0;
        return 0;
    }

    /* 阶段 2: SEH 探针 — 实际执行 VNNI 指令验证
     * 部分 CPU（如 Zhaoxin）CPUID 报告支持但实际不支持 */
#ifdef __AVXVNNI__
    __try {
        /* 用非零输入确保指令真正执行（全零可能被优化掉） */
        __m256i a = _mm256_set1_epi8(1);   /* 32 个 uint8, 值 1 */
        __m256i b = _mm256_set1_epi8(1);   /* 32 个 int8, 值 1 */
        __m256i c = _mm256_dpbusd_epi32(_mm256_setzero_si256(), a, b);
        /* 写入 volatile 变量防止优化 */
        g_vnni_probe_sink = _mm256_extract_epi32(c, 0);
        g_avx_vnni_cache = 1;
    } __except (GetExceptionCode() == HC_STATUS_ILLEGAL_INSTRUCTION ?
               EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        /* VNNI 指令触发非法指令异常 — CPU 不支持 */
        g_avx_vnni_cache = 0;
    }
#else
    g_avx_vnni_cache = 0;
#endif
    return g_avx_vnni_cache;
}

/* 调试用：返回 CPUID 7.1 的原始 EAX/EBX/ECX/EDX 值 */
void hc_cpuid_7_1_raw(int* eax, int* ebx, int* ecx, int* edx) {
    int cpuinfo[4] = {0, 0, 0, 0};
    __cpuidex(cpuinfo, 7, 1);
    if (eax) *eax = cpuinfo[0];
    if (ebx) *ebx = cpuinfo[1];
    if (ecx) *ecx = cpuinfo[2];
    if (edx) *edx = cpuinfo[3];
}

/* ============================================================================
 * A-0 阶段：64 字节对齐分配
 *
 * 目的：为 AVX-256 / AVX-512 intrinsics 提供 cache-line 对齐的内存
 *   - AVX-256 _mm256_load_si256 要求 32 字节对齐
 *   - AVX-512 _mm512_load_si512 要求 64 字节对齐（未来兼容）
 *   - 64 字节 = 1 个 cache line，对齐后 load 永不跨 line
 *
 * MSVC 实现：_aligned_malloc(size, alignment) / _aligned_free(ptr)
 *   注意：_aligned_malloc 返回的指针必须用 _aligned_free 释放（不能用 free）
 *
 * 数值不变：对齐分配只改分配方式，不改数据内容，max_diff=0（已通过对照测试验证）
 * ============================================================================ */

/* 对齐分配宏：分配 n 个 type 元素，64 字节对齐 */
#define HC_ALIGNED_ALLOC(n, type) \
    ((type*)_aligned_malloc((size_t)(n) * sizeof(type), 64))

/* 对齐释放宏：释放 _aligned_malloc 返回的指针 */
#define HC_ALIGNED_FREE(ptr) \
    do { if ((ptr) != NULL) { _aligned_free(ptr); (ptr) = NULL; } } while (0)

/* 对齐 calloc 宏：分配 n 个 type 元素并清零，64 字节对齐 */
static void* hc_aligned_calloc(size_t count, size_t size) {
    void* p = _aligned_malloc(count * size, 64);
    if (p != NULL) {
        memset(p, 0, count * size);
    }
    return p;
}
#define HC_ALIGNED_CALLOC(n, type) ((type*)hc_aligned_calloc((size_t)(n), sizeof(type)))



/* ============================================================================
 * 默认量化方案
 * ============================================================================ */

hc8_quant_schema_t hc8_quant_default_schema(void) {
    hc8_quant_schema_t s;
    s.scale  = 1.0f;
    s.qmin   = -127;
    s.qmax   = 127;
    s.offset = 128;
    return s;
}

/* ============================================================================
 * 量化 scale 推导
 * ============================================================================ */

float hc8_quant_compute_scale(const float* w, uint32_t n) {
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
 * 量化 / 反量化
 * ============================================================================ */

void hc8_quantize(const float* w, uint32_t n,
                  float scale,
                  const hc8_quant_schema_t* schema,
                  hc8_t* out) {
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

        /* 偏移到无符号：q_u = q + offset ∈ [1, 255] */
        int32_t q_u = q + schema->offset;
        if (q_u < 0) q_u = 0;
        else if (q_u > 255) q_u = 255;

        /* 只填 v[0]，v[1..5] = 0 */
        out[i].v[0] = (uint8_t)q_u;
        out[i].v[1] = 0;
        out[i].v[2] = 0;
        out[i].v[3] = 0;
        out[i].v[4] = 0;
        out[i].v[5] = 0;
    }
}

void hc8_dequantize(const hc8_t* h, uint32_t n,
                    float scale,
                    const hc8_quant_schema_t* schema,
                    float* out) {
    if (n == 0 || h == NULL || out == NULL || schema == NULL) return;

    for (uint32_t i = 0; i < n; ++i) {
        /* q_u = hc8.v[0], q = q_u - offset, w = q * scale */
        int32_t q_u = h[i].v[0];
        int32_t q = q_u - schema->offset;
        out[i] = (float)q * scale;
    }
}

/* ============================================================================
 * HC8 矩阵乘
 * ============================================================================ */

void hc8_matmul(const hc8_t* a, const hc8_t* b,
                uint32_t m, uint32_t k, uint32_t n,
                float a_scale, float b_scale,
                const hc8_quant_schema_t* schema,
                hc8_t* out, float* out_scale) {
    if (m == 0 || k == 0 || n == 0 ||
        a == NULL || b == NULL || out == NULL || schema == NULL || out_scale == NULL) {
        if (out_scale) *out_scale = 1.0f;
        return;
    }

    /* 步骤 1+2: 整数累加（纯整数路径）
     *   a_q[i][l] = a[i*k + l].v[0] - offset
     *   b_q[l][j] = b[l*n + j].v[0] - offset
     *   c_acc[i][j] = sum_l(a_q[i][l] * b_q[l][j])
     *
     * 用 int32 累加，不会溢出：
     *   k 最大约 4096，|q| ≤ 127
     *   最大累加值 ≈ 4096 * 127 * 127 ≈ 6.6e7 << INT32_MAX (2.1e9)
     */
    int32_t offset = schema->offset;
    float out_combined_scale = a_scale * b_scale;

    /* 临时 float 缓冲区存 c_float（用于重新量化） */
    uint32_t out_size = m * n;
    float* c_float = (float*)malloc(sizeof(float) * out_size);
    if (c_float == NULL) {
        *out_scale = 1.0f;
        return;
    }

    /* 累加 + 反量化 */
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            int32_t acc = 0;
            for (uint32_t l = 0; l < k; ++l) {
                int32_t a_q = (int32_t)a[i * k + l].v[0] - offset;
                int32_t b_q = (int32_t)b[l * n + j].v[0] - offset;
                acc += a_q * b_q;
            }
            c_float[i * n + j] = (float)acc * out_combined_scale;
        }
    }

    /* 步骤 3+4: 重新量化到 HC8
     *   基于 c_float 的 max(|w|) 推导新 scale
     *   量化到 HC8
     */
    float new_scale = hc8_quant_compute_scale(c_float, out_size);
    hc8_quantize(c_float, out_size, new_scale, schema, out);

    *out_scale = new_scale;

    free(c_float);
}

/* ============================================================================
 * HC8 整数 ReLU
 * ============================================================================ */

void hc8_relu(const hc8_t* x, uint32_t m, uint32_t n,
              const hc8_quant_schema_t* schema,
              hc8_t* out) {
    if (m == 0 || n == 0 || x == NULL || out == NULL || schema == NULL) return;

    uint32_t total = m * n;
    uint8_t offset = (uint8_t)schema->offset;

    for (uint32_t i = 0; i < total; ++i) {
        uint8_t q_u = x[i].v[0];
        /* q_u < offset 表示负值，ReLU 后应为 0（即 q=0, q_u=offset） */
        uint8_t new_q_u = (q_u < offset) ? offset : q_u;

        out[i].v[0] = new_q_u;
        out[i].v[1] = 0;
        out[i].v[2] = 0;
        out[i].v[3] = 0;
        out[i].v[4] = 0;
        out[i].v[5] = 0;
    }
}

/* ============================================================================
 * HC8 数组 ↔ bytes 互转
 * ============================================================================ */

void hc8_array_to_bytes(const hc8_t* h, uint32_t n, uint8_t* out) {
    if (n == 0 || h == NULL || out == NULL) return;
    /* hc8_t 是 6 字节紧缩结构（#pragma pack(1)），可以直接 memcpy */
    memcpy(out, h, (size_t)n * sizeof(hc8_t));
}

void hc8_bytes_to_array(const uint8_t* bytes, uint32_t n, hc8_t* out) {
    if (n == 0 || bytes == NULL || out == NULL) return;
    memcpy(out, bytes, (size_t)n * sizeof(hc8_t));
}

/* ============================================================================
 * UFP-1 残差量化：v[0..depth] 存量化残差
 * ============================================================================
 *
 * 设计见 hc_ufp_1_residual_design.md。
 *
 * 编码（递归残差）：
 *   remaining = w
 *   for layer in 0..depth:
 *       scale_layer = max(|remaining_array|) / 127   # 组级共享
 *       q = round(remaining / scale_layer), clamp(-127, 127)
 *       v[layer] = q + 128
 *       remaining = remaining - q * scale_layer
 *
 * 解码（累加所有层）：
 *   w_approx = sum_{layer=0..depth} (v[layer] - 128) * scales[layer]
 *
 * 矩阵乘用方案 C（存储高精度，运算 v[0]）：
 *   反量化到 float → 重新量化到 v[0] → 做 hc8_matmul
 *
 * 性能：与阶段 1.3 相同（矩阵乘仍是 v[0]），只是多了一次反量化-重量化的开销。
 */

void hc8_quantize_residual(const float* w, uint32_t n,
                           int depth,
                           const hc8_quant_schema_t* schema,
                           hc8_t* out,
                           hc8_residual_scales_t* out_scales) {
    if (n == 0 || w == NULL || out == NULL || schema == NULL || out_scales == NULL) return;

    /* depth 范围检查 */
    if (depth < 0) depth = 0;
    if (depth > 5) depth = 5;

    /* 初始化 out_scales 全 0 */
    for (int i = 0; i < 6; ++i) out_scales->scales[i] = 0.0f;

    /* 临时 buffer 存当前层的 remaining（float）
     * 第 0 层 remaining = w，之后每层 remaining = 上一层残差
     */
    float* remaining = (float*)malloc(sizeof(float) * n);
    if (remaining == NULL) {
        memset(out, 0, sizeof(hc8_t) * n);
        return;
    }
    memcpy(remaining, w, sizeof(float) * n);

    /* 逐层量化 */
    for (int layer = 0; layer <= depth; ++layer) {
        /* 求本层 max(|remaining|) */
        float max_abs = 0.0f;
        for (uint32_t i = 0; i < n; ++i) {
            float a = fabsf(remaining[i]);
            if (a > max_abs) max_abs = a;
        }

        /* max_abs = 0 表示残差已全部为 0，后续层 scale=0, v[layer]=128（q=0） */
        float scale_layer;
        if (max_abs == 0.0f) {
            scale_layer = 0.0f;
        } else {
            scale_layer = max_abs / 127.0f;
        }
        out_scales->scales[layer] = scale_layer;

        /* 量化本层 */
        if (scale_layer > 0.0f) {
            float inv_scale = 1.0f / scale_layer;
            for (uint32_t i = 0; i < n; ++i) {
                float q_f = remaining[i] * inv_scale;
                int32_t q = (int32_t)lroundf(q_f);
                if (q < schema->qmin) q = schema->qmin;
                else if (q > schema->qmax) q = schema->qmax;

                int32_t q_u = q + schema->offset;
                if (q_u < 0) q_u = 0;
                else if (q_u > 255) q_u = 255;

                out[i].v[layer] = (uint8_t)q_u;

                /* 更新 remaining 为本层残差 */
                remaining[i] = remaining[i] - (float)q * scale_layer;
            }
        } else {
            /* scale=0，所有元素 q=0, q_u=128，remaining 保持 0 */
            for (uint32_t i = 0; i < n; ++i) {
                out[i].v[layer] = (uint8_t)schema->offset;
                /* remaining[i] 已经是 0，无需更新 */
            }
        }
    }

    /* 未使用的层 v[layer+1..5] = 0 */
    for (uint32_t i = 0; i < n; ++i) {
        for (int layer = depth + 1; layer < 6; ++layer) {
            out[i].v[layer] = 0;
        }
    }

    free(remaining);
}

void hc8_dequantize_residual(const hc8_t* h, uint32_t n,
                             int depth,
                             const hc8_residual_scales_t* scales,
                             const hc8_quant_schema_t* schema,
                             float* out) {
    if (n == 0 || h == NULL || scales == NULL || schema == NULL || out == NULL) return;

    /* depth 范围检查 */
    if (depth < 0) depth = 0;
    if (depth > 5) depth = 5;

    /* 累加所有层：w_approx = sum_{layer} (v[layer] - 128) * scales[layer] */
    for (uint32_t i = 0; i < n; ++i) {
        float w = 0.0f;
        for (int layer = 0; layer <= depth; ++layer) {
            int32_t q_u = h[i].v[layer];
            int32_t q = q_u - schema->offset;
            w += (float)q * scales->scales[layer];
        }
        out[i] = w;
    }
}

void hc8_residual_matmul(const hc8_t* a, const hc8_t* b,
                         uint32_t m, uint32_t k, uint32_t n,
                         int a_depth, int b_depth,
                         const hc8_residual_scales_t* a_scales,
                         const hc8_residual_scales_t* b_scales,
                         const hc8_quant_schema_t* schema,
                         hc8_t* out, float* out_scale) {
    if (m == 0 || k == 0 || n == 0 ||
        a == NULL || b == NULL || out == NULL || schema == NULL || out_scale == NULL ||
        a_scales == NULL || b_scales == NULL) {
        if (out_scale) *out_scale = 1.0f;
        return;
    }

    /* 方案 C：反量化到 float → 重新量化到 v[0] → 做 hc8_matmul */

    /* 步骤 1: A、B 反量化到 float */
    uint32_t a_size = m * k;
    uint32_t b_size = k * n;
    float* a_float = (float*)malloc(sizeof(float) * a_size);
    float* b_float = (float*)malloc(sizeof(float) * b_size);
    if (a_float == NULL || b_float == NULL) {
        free(a_float);
        free(b_float);
        *out_scale = 1.0f;
        return;
    }

    hc8_dequantize_residual(a, a_size, a_depth, a_scales, schema, a_float);
    hc8_dequantize_residual(b, b_size, b_depth, b_scales, schema, b_float);

    /* 步骤 2: 重新量化到 v[0]（用各自的 scale_0）
     * 注意：这里用 a_float/b_float 的 max(|w|) 重新算 scale，不是用 a_scales->scales[0]
     * 因为我们要让 v[0] 能完整表示 a_float 的动态范围（避免再次量化损失）
     */
    float a_scale0 = hc8_quant_compute_scale(a_float, a_size);
    float b_scale0 = hc8_quant_compute_scale(b_float, b_size);

    hc8_t* a_v0 = (hc8_t*)malloc(sizeof(hc8_t) * a_size);
    hc8_t* b_v0 = (hc8_t*)malloc(sizeof(hc8_t) * b_size);
    if (a_v0 == NULL || b_v0 == NULL) {
        free(a_float); free(b_float);
        free(a_v0); free(b_v0);
        *out_scale = 1.0f;
        return;
    }

    hc8_quantize(a_float, a_size, a_scale0, schema, a_v0);
    hc8_quantize(b_float, b_size, b_scale0, schema, b_v0);

    /* 步骤 3: 做 v[0] 矩阵乘 */
    hc8_matmul(a_v0, b_v0, m, k, n, a_scale0, b_scale0, schema, out, out_scale);

    free(a_float); free(b_float);
    free(a_v0); free(b_v0);
}

void hc8_residual_relu(const hc8_t* x, uint32_t m, uint32_t n,
                       const hc8_quant_schema_t* schema,
                       hc8_t* out) {
    if (m == 0 || n == 0 || x == NULL || out == NULL || schema == NULL) return;

    /* ReLU 是非线性运算，跨层残差不再有效，输出总是 depth=0
     * 只对 v[0] 做 ReLU，v[1..5] 清零
     */
    uint32_t total = m * n;
    uint8_t offset = (uint8_t)schema->offset;

    for (uint32_t i = 0; i < total; ++i) {
        uint8_t q_u = x[i].v[0];
        uint8_t new_q_u = (q_u < offset) ? offset : q_u;

        out[i].v[0] = new_q_u;
        out[i].v[1] = 0;
        out[i].v[2] = 0;
        out[i].v[3] = 0;
        out[i].v[4] = 0;
        out[i].v[5] = 0;
    }
}

/* ============================================================================
 * UFP-2 方案 B：整数域累加矩阵乘
 * ============================================================================
 *
 * 设计见 hc_ufp_2_scheme_b_design.md。
 *
 * 与方案 C 的核心区别：
 *   - 方案 C：反量化 A/B 到 float32 → 重新量化到 v[0] → int8 矩阵乘
 *             精度损失：float32 只有 24 位尾数，48-bit 存储会截断
 *   - 方案 B：直接用 v[0..depth] 的 int8 值做矩阵乘
 *             整数累加精确，用 double 合并（52 位尾数，够 48-bit）
 *
 * 数学：
 *   C[i][j] = Σ_l Σ_m [a_scales[l] * b_scales[m]] * [Σ_k (a.v[l]-off) * (b.v[m]-off)]
 *                          ↑ double combined_scale          ↑ int32 精确累加
 *
 * 计算量：(a_depth+1) * (b_depth+1) 次 int8 矩阵乘
 *   depth=0: 1 次（与方案 C 等价）
 *   depth=5: 36 次
 */

void hc8_residual_matmul_b(const hc8_t* a, const hc8_t* b,
                           uint32_t m, uint32_t k, uint32_t n,
                           int a_depth, int b_depth,
                           const hc8_residual_scales_t* a_scales,
                           const hc8_residual_scales_t* b_scales,
                           const hc8_quant_schema_t* schema,
                           hc8_t* out, float* out_scale) {
    if (m == 0 || k == 0 || n == 0 ||
        a == NULL || b == NULL || out == NULL || schema == NULL || out_scale == NULL ||
        a_scales == NULL || b_scales == NULL) {
        if (out_scale) *out_scale = 1.0f;
        return;
    }

    /* depth 范围检查 */
    if (a_depth < 0) a_depth = 0;
    if (a_depth > 5) a_depth = 5;
    if (b_depth < 0) b_depth = 0;
    if (b_depth > 5) b_depth = 5;

    int32_t offset = schema->offset;
    uint32_t out_size = m * n;

    /* 用 double 累加，保留精度（52 位尾数，够 48-bit） */
    double* c_double = (double*)calloc(out_size, sizeof(double));
    if (c_double == NULL) {
        *out_scale = 1.0f;
        return;
    }

    /* 对每对 (l, m_idx) 做 int8 × int8 → int32 累加
     * 注意：m_idx 是 B 的层索引，不是矩阵维度 m
     */
    for (int l = 0; l <= a_depth; ++l) {
        if (a_scales->scales[l] == 0.0f) continue;
        for (int m_idx = 0; m_idx <= b_depth; ++m_idx) {
            if (b_scales->scales[m_idx] == 0.0f) continue;

            double combined_scale = (double)a_scales->scales[l] * (double)b_scales->scales[m_idx];

            for (uint32_t i = 0; i < m; ++i) {
                for (uint32_t j = 0; j < n; ++j) {
                    int32_t acc = 0;
                    for (uint32_t k_idx = 0; k_idx < k; ++k_idx) {
                        int32_t a_q = (int32_t)a[i * k + k_idx].v[l] - offset;
                        int32_t b_q = (int32_t)b[k_idx * n + j].v[m_idx] - offset;
                        acc += a_q * b_q;
                    }
                    c_double[i * n + j] += (double)acc * combined_scale;
                }
            }
        }
    }

    /* 重新量化 c_double 到 v[0]（输出 depth=0） */
    double max_abs = 0.0;
    for (uint32_t i = 0; i < out_size; ++i) {
        double v = fabs(c_double[i]);
        if (v > max_abs) max_abs = v;
    }

    float new_scale = (max_abs == 0.0) ? 1.0f : (float)(max_abs / 127.0);

    if (new_scale > 0.0f) {
        float inv_scale = 1.0f / new_scale;
        for (uint32_t i = 0; i < out_size; ++i) {
            float q_f = (float)c_double[i] * inv_scale;
            int32_t q = (int32_t)lroundf(q_f);
            if (q < schema->qmin) q = schema->qmin;
            else if (q > schema->qmax) q = schema->qmax;

            int32_t q_u = q + schema->offset;
            if (q_u < 0) q_u = 0;
            else if (q_u > 255) q_u = 255;

            out[i].v[0] = (uint8_t)q_u;
        }
    } else {
        for (uint32_t i = 0; i < out_size; ++i) {
            out[i].v[0] = (uint8_t)schema->offset;
        }
    }

    /* v[1..5] = 0 */
    for (uint32_t i = 0; i < out_size; ++i) {
        out[i].v[1] = 0;
        out[i].v[2] = 0;
        out[i].v[3] = 0;
        out[i].v[4] = 0;
        out[i].v[5] = 0;
    }

    *out_scale = new_scale;
    free(c_double);
}

/* ============================================================================
 * HC4 非对称拆分：int8 → int4+int4 运算路径
 * ============================================================================
 *
 * 设计见 hc_v1.4_asymmetric_split_design.md。
 *
 * 数学核心：
 *   a_q = a_u - 128 = (a_h*16 + a_l) - 128
 *   b_q = b_u - 128 = (b_h*16 + b_l) - 128
 *   Σ(a_q * b_q) = 256*Σ(a_h*b_h) + 16*Σ(a_h*b_l) + 16*Σ(a_l*b_h) + Σ(a_l*b_l)
 *                - 2048*Σ(a_h) - 128*Σ(a_l) - 2048*Σ(b_h) - 128*Σ(b_l) + 16384*k
 *
 * 关键发现：hc4_t.packed[l] 与 hc8_t.v[l] 二进制相同，拆分/合并零成本。
 */

/* HC8 → HC4 拆分（零成本，二进制相同） */
void hc8_split_to_hc4(const hc8_t* hc8, uint32_t n, hc4_t* out) {
    if (hc8 == NULL || out == NULL || n == 0) return;
    /* 二进制完全相同，直接拷贝 */
    memcpy(out, hc8, n * sizeof(hc4_t));
}

/* HC4 → HC8 合并（零成本，二进制相同，无损双射） */
void hc4_merge_to_hc8(const hc4_t* hc4, uint32_t n, hc8_t* out) {
    if (hc4 == NULL || out == NULL || n == 0) return;
    /* 二进制完全相同，直接拷贝 */
    memcpy(out, hc4, n * sizeof(hc8_t));
}

/**
 * int4×int4 矩阵乘 kernel（内部函数）
 *
 * 计算 S[i][j] = Σ_k a_nibble[i][k] * b_nibble[k][j]
 * 其中 a_nibble/b_nibble 由 ab_pair 决定 high/low
 *
 * @param a            HC4 矩阵 A（m×k）
 * @param b            HC4 矩阵 B（k×n）
 * @param m, k, n      维度
 * @param layer_l      A 的层索引（0-5）
 * @param layer_m      B 的层索引（0-5）
 * @param ab_pair      0=hh, 1=hl, 2=lh, 3=ll
 * @param acc_out      int32 累加器（m×n，累加到现有值上）
 */
static void hc4_matmul_kernel_int4(
    const hc4_t* a, const hc4_t* b,
    uint32_t m, uint32_t k, uint32_t n,
    int layer_l, int layer_m, int ab_pair,
    int32_t* acc_out
) {
    /* 根据 ab_pair 选择 nibble：
     *   ab_pair=0 (hh): a.high, b.high
     *   ab_pair=1 (hl): a.high, b.low
     *   ab_pair=2 (lh): a.low,  b.high
     *   ab_pair=3 (ll): a.low,  b.low
     */
    const int a_use_high = (ab_pair < 2);      /* 0,1 → high; 2,3 → low */
    const int b_use_high = (ab_pair % 2 == 0); /* 0,2 → high; 1,3 → low */

    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            int32_t acc = 0;
            for (uint32_t k_idx = 0; k_idx < k; ++k_idx) {
                uint8_t a_byte = a[i * k + k_idx].packed[layer_l];
                uint8_t b_byte = b[k_idx * n + j].packed[layer_m];
                uint8_t a_nibble = a_use_high ? ((a_byte >> 4) & 0x0F) : (a_byte & 0x0F);
                uint8_t b_nibble = b_use_high ? ((b_byte >> 4) & 0x0F) : (b_byte & 0x0F);
                acc += (int32_t)a_nibble * (int32_t)b_nibble;
            }
            acc_out[i * n + j] += acc;
        }
    }
}

/* HC4 残差矩阵乘（int4×int4→int32 累加 + double 合并） */
void hc4_residual_matmul_b(const hc4_t* a, const hc4_t* b,
                           uint32_t m, uint32_t k, uint32_t n,
                           int a_depth, int b_depth,
                           const hc8_residual_scales_t* a_scales,
                           const hc8_residual_scales_t* b_scales,
                           const hc8_quant_schema_t* schema,
                           hc8_t* out, float* out_scale) {
    if (m == 0 || k == 0 || n == 0 ||
        a == NULL || b == NULL || out == NULL || schema == NULL || out_scale == NULL ||
        a_scales == NULL || b_scales == NULL) {
        if (out_scale) *out_scale = 1.0f;
        return;
    }

    /* depth 范围检查 */
    if (a_depth < 0) a_depth = 0;
    if (a_depth > 5) a_depth = 5;
    if (b_depth < 0) b_depth = 0;
    if (b_depth > 5) b_depth = 5;

    uint32_t out_size = m * n;

    /* 用 double 累加，保留精度（52 位尾数，够 48-bit） */
    double* c_double = (double*)calloc(out_size, sizeof(double));
    if (c_double == NULL) {
        *out_scale = 1.0f;
        return;
    }

    /* 4 次 int4×int4 矩阵乘的累加器 */
    int32_t* S_hh = (int32_t*)calloc(out_size, sizeof(int32_t));
    int32_t* S_hl = (int32_t*)calloc(out_size, sizeof(int32_t));
    int32_t* S_lh = (int32_t*)calloc(out_size, sizeof(int32_t));
    int32_t* S_ll = (int32_t*)calloc(out_size, sizeof(int32_t));

    /* 偏移修正项预计算累加器（A1-4 修复，2026-09-08）：
     * sum_ah[l][i] = Σ_k a_h[i][k] 只依赖 (l,i)；sum_bh[m2][j] = Σ_k b_h[k][j]
     * 只依赖 (m2,j)。原标量版按 (i,j) 全矩阵累加，O(m·k·n + k·n·m)；
     * 现对齐 SIMD 版（hc4_residual_matmul_b_simd）预计算，O(depth·(m·k + k·n))，
     * 合并循环查表。数值与原实现逐位一致（每项求和顺序不变）。 */
    int32_t* sum_ah[6];
    int32_t* sum_al[6];
    int32_t* sum_bh[6];
    int32_t* sum_bl[6];
    int sums_ok = 1;
    for (int p = 0; p < 6; ++p) {
        sum_ah[p] = (int32_t*)calloc(m, sizeof(int32_t));
        sum_al[p] = (int32_t*)calloc(m, sizeof(int32_t));
        sum_bh[p] = (int32_t*)calloc(n, sizeof(int32_t));
        sum_bl[p] = (int32_t*)calloc(n, sizeof(int32_t));
        if (sum_ah[p] == NULL || sum_al[p] == NULL ||
            sum_bh[p] == NULL || sum_bl[p] == NULL) sums_ok = 0;
    }

    if (S_hh == NULL || S_hl == NULL || S_lh == NULL || S_ll == NULL || !sums_ok) {
        free(c_double); free(S_hh); free(S_hl); free(S_lh); free(S_ll);
        for (int p = 0; p < 6; ++p) {
            free(sum_ah[p]); free(sum_al[p]); free(sum_bh[p]); free(sum_bl[p]);
        }
        *out_scale = 1.0f;
        return;
    }

    /* 预计算偏移修正项（只对 depth 内的肢；求和顺序与原 (i,j) 版相同 → bit-exact） */
    for (int l = 0; l <= a_depth; ++l) {
        for (uint32_t i = 0; i < m; ++i) {
            int32_t sh = 0, sl = 0;
            for (uint32_t k_idx = 0; k_idx < k; ++k_idx) {
                uint8_t a_byte = a[i * k + k_idx].packed[l];
                sh += (int32_t)((a_byte >> 4) & 0x0F);
                sl += (int32_t)(a_byte & 0x0F);
            }
            sum_ah[l][i] = sh;
            sum_al[l][i] = sl;
        }
    }
    for (int m2 = 0; m2 <= b_depth; ++m2) {
        for (uint32_t j = 0; j < n; ++j) {
            int32_t sh = 0, sl = 0;
            for (uint32_t k_idx = 0; k_idx < k; ++k_idx) {
                uint8_t b_byte = b[k_idx * n + j].packed[m2];
                sh += (int32_t)((b_byte >> 4) & 0x0F);
                sl += (int32_t)(b_byte & 0x0F);
            }
            sum_bh[m2][j] = sh;
            sum_bl[m2][j] = sl;
        }
    }

    /* 对每对 (l, m_idx) 做 4 次 int4×int4 矩阵乘 + 偏移修正 */
    for (int l = 0; l <= a_depth; ++l) {
        if (a_scales->scales[l] == 0.0f) continue;
        for (int m_idx = 0; m_idx <= b_depth; ++m_idx) {
            if (b_scales->scales[m_idx] == 0.0f) continue;

            /* 清零累加器（每个 (l, m_idx) 对独立累加；sum_* 已预计算无需清零） */
            memset(S_hh, 0, out_size * sizeof(int32_t));
            memset(S_hl, 0, out_size * sizeof(int32_t));
            memset(S_lh, 0, out_size * sizeof(int32_t));
            memset(S_ll, 0, out_size * sizeof(int32_t));

            /* 4 次 int4×int4 矩阵乘 */
            hc4_matmul_kernel_int4(a, b, m, k, n, l, m_idx, 0, S_hh);
            hc4_matmul_kernel_int4(a, b, m, k, n, l, m_idx, 1, S_hl);
            hc4_matmul_kernel_int4(a, b, m, k, n, l, m_idx, 2, S_lh);
            hc4_matmul_kernel_int4(a, b, m, k, n, l, m_idx, 3, S_ll);

            /* 合并：acc = 256*S_hh + 16*S_hl + 16*S_lh + S_ll - 偏移修正 + 16384*k */
            double combined_scale = (double)a_scales->scales[l] * (double)b_scales->scales[m_idx];
            int32_t k_const = (int32_t)k;

            for (uint32_t i = 0; i < m; ++i) {
                for (uint32_t j = 0; j < n; ++j) {
                    uint32_t idx = i * n + j;
                    int32_t total = 256 * S_hh[idx] + 16 * S_hl[idx] + 16 * S_lh[idx] + S_ll[idx]
                                  - 2048 * sum_ah[l][i] - 128 * sum_al[l][i]
                                  - 2048 * sum_bh[m_idx][j] - 128 * sum_bl[m_idx][j]
                                  + 16384 * k_const;
                    c_double[idx] += (double)total * combined_scale;
                }
            }
        }
    }

    /* 重新量化 c_double 到 v[0]（与 hc8_residual_matmul_b 一致） */
    double max_abs = 0.0;
    for (uint32_t i = 0; i < out_size; ++i) {
        double v = fabs(c_double[i]);
        if (v > max_abs) max_abs = v;
    }

    float new_scale = (max_abs == 0.0) ? 1.0f : (float)(max_abs / 127.0);

    if (new_scale > 0.0f) {
        float inv_scale = 1.0f / new_scale;
        for (uint32_t i = 0; i < out_size; ++i) {
            float q_f = (float)c_double[i] * inv_scale;
            int32_t q = (int32_t)lroundf(q_f);
            if (q < schema->qmin) q = schema->qmin;
            else if (q > schema->qmax) q = schema->qmax;

            int32_t q_u = q + schema->offset;
            if (q_u < 0) q_u = 0;
            else if (q_u > 255) q_u = 255;

            out[i].v[0] = (uint8_t)q_u;
        }
    } else {
        for (uint32_t i = 0; i < out_size; ++i) {
            out[i].v[0] = (uint8_t)schema->offset;
        }
    }

    /* v[1..5] = 0 */
    for (uint32_t i = 0; i < out_size; ++i) {
        out[i].v[1] = 0;
        out[i].v[2] = 0;
        out[i].v[3] = 0;
        out[i].v[4] = 0;
        out[i].v[5] = 0;
    }

    *out_scale = new_scale;
    free(c_double);
    free(S_hh); free(S_hl); free(S_lh); free(S_ll);
    for (int p = 0; p < 6; ++p) {
        free(sum_ah[p]); free(sum_al[p]); free(sum_bh[p]); free(sum_bl[p]);
    }
}

/* ============================================================================
 * SIMD 加速：SoA 数据布局 + AVX-VNNI/AVX2 kernel
 * ============================================================================
 *
 * 设计见 hc_simd_avx_vnni_design.md v1.2。
 *
 * 本文件先实现 E 阶段（数据布局重排）+ 标量版 _simd 函数（验证布局正确性）。
 * AVX-VNNI/AVX2 intrinsics 实现放在 hc8_net_simd.c（A 阶段）。
 *
 * 编译开关：
 *   - 有 AVX2 时：__AVX2__ 定义，intrinsics 可用
 *   - 有 AVX-VNNI 时：__AVXVNNI__ 定义，dpbusd 可用
 *   - 都没有时：_simd 函数回退到标量版
 */

/* ----------------------------------------------------------------------------
 * E 阶段：数据布局重排
 * -------------------------------------------------------------------------- */

/* HC8 AoS → SoA：把 hc8_t[N].v[l] 重排为 layer[l][N]（每层连续） */
void hc8_aos_to_soa(const hc8_t* aos, uint32_t n, hc8_soa_t* out) {
    if (aos == NULL || out == NULL || n == 0) return;

    out->n = n;
    for (int l = 0; l < 6; ++l) {
        out->layer[l] = HC_ALIGNED_ALLOC(n, uint8_t);  /* A-0: 64 字节对齐 */
        if (out->layer[l] == NULL) {
            /* 分配失败，释放已分配的 */
            for (int j = 0; j < l; ++j) { HC_ALIGNED_FREE(out->layer[j]); }
            out->n = 0;
            return;
        }
        for (uint32_t i = 0; i < n; ++i) {
            out->layer[l][i] = aos[i].v[l];
        }
    }
}

/* HC8 SoA 释放 */
void hc8_soa_free(hc8_soa_t* soa) {
    if (soa == NULL) return;
    for (int l = 0; l < 6; ++l) {
        HC_ALIGNED_FREE(soa->layer[l]);  /* A-0: 配套 _aligned_free */
    }
    soa->n = 0;
}

/* HC4 AoS → SoA 预解包：packed[l] → high[l] + low[l] */
void hc4_unpack_to_soa(const hc4_t* aos, uint32_t n, int depth, hc4_soa_t* out) {
    if (aos == NULL || out == NULL || n == 0) return;

    if (depth < 0) depth = 0;
    if (depth > 5) depth = 5;

    out->n = n;
    for (int l = 0; l < 6; ++l) {
        if (l > depth) {
            out->high[l] = NULL;
            out->low[l] = NULL;
            continue;
        }
        out->high[l] = HC_ALIGNED_ALLOC(n, uint8_t);  /* A-0: 64 字节对齐 */
        out->low[l]  = HC_ALIGNED_ALLOC(n, uint8_t);  /* A-0: 64 字节对齐 */
        if (out->high[l] == NULL || out->low[l] == NULL) {
            /* 分配失败，释放已分配的 */
            for (int j = 0; j <= l; ++j) {
                HC_ALIGNED_FREE(out->high[j]);
                HC_ALIGNED_FREE(out->low[j]);
            }
            out->n = 0;
            return;
        }
        for (uint32_t i = 0; i < n; ++i) {
            uint8_t byte = aos[i].packed[l];
            out->high[l][i] = (byte >> 4) & 0x0F;  /* 高 nibble，0-15 */
            out->low[l][i]  = byte & 0x0F;          /* 低 nibble，0-15 */
        }
    }
}

/* HC4 SoA 释放 */
void hc4_soa_free(hc4_soa_t* soa) {
    if (soa == NULL) return;
    for (int l = 0; l < 6; ++l) {
        HC_ALIGNED_FREE(soa->high[l]);  /* A-0: 配套 _aligned_free */
        HC_ALIGNED_FREE(soa->low[l]);
    }
    soa->n = 0;
}

/* 通用 uint8 矩阵转置：k×n（行优先）→ n×k（行优先） */
uint8_t* hc8_transpose_u8(const uint8_t* src, uint32_t k, uint32_t n) {
    if (src == NULL || k == 0 || n == 0) return NULL;
    uint8_t* dst = HC_ALIGNED_ALLOC((size_t)k * n, uint8_t);  /* A-0: 64 字节对齐 */
    if (dst == NULL) return NULL;
    for (uint32_t i = 0; i < k; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            dst[j * k + i] = src[i * n + j];
        }
    }
    return dst;
}

/* ----------------------------------------------------------------------------
 * HC8 残差矩阵乘 SIMD 版（标量实现，验证 SoA 布局正确性）
 *
 * 本函数先用标量实现验证 SoA 布局的正确性（与 hc8_residual_matmul_b 结果一致）。
 * AVX-VNNI intrinsics 实现在 hc8_net_simd.c（A 阶段）。
 *
 * 数学（与 hc8_residual_matmul_b 相同）：
 *   C[i][j] = Σ_l Σ_m [a_scales[l] * b_scales[m]] * [Σ_k (a.v[l]-off) * (b.v[m]-off)]
 *
 * SoA 优化点：
 *   - a_soa.layer[l][i*k+k_idx] 连续访问（跨步 1 字节，vs AoS 跨步 6 字节）
 *   - B 转置后 b_T[m][j*k+k_idx] 连续访问
 *   - 缓存友好
 * -------------------------------------------------------------------------- */

/* HC8 SoA 标量 kernel：计算 S[i][j] = Σ_k (a_u-128) * (b_u-128)
 *
 * @param a_u        A 的第 l 层，m×k 行优先，uint8（q_u = q+128）
 * @param b_T_u      B 的第 m 层转置后，n×k 行优先，uint8
 * @param m, k, n    维度
 * @param offset     量化偏移（128）
 * @param sum_a      预计算的 Σ_k (a_u-128)，长度 m（只依赖 i）
 * @param sum_b      预计算的 Σ_k (b_u-128)，长度 n（只依赖 j）
 * @param acc_out    double 累加器（m×n，累加 * combined_scale）
 * @param combined_scale  a_scales[l] * b_scales[m]
 */
static void hc8_matmul_kernel_soa_scalar(
    const uint8_t* a_u, const uint8_t* b_T_u,
    uint32_t m, uint32_t k, uint32_t n,
    int32_t offset,
    const int32_t* sum_a,  /* 长度 m */
    const int32_t* sum_b,  /* 长度 n */
    double* acc_out,
    double combined_scale
) {
    for (uint32_t i = 0; i < m; ++i) {
        const uint8_t* a_row = a_u + i * k;
        for (uint32_t j = 0; j < n; ++j) {
            const uint8_t* b_row = b_T_u + j * k;
            int32_t acc = 0;
            for (uint32_t k_idx = 0; k_idx < k; ++k_idx) {
                int32_t a_q = (int32_t)a_row[k_idx] - offset;
                int32_t b_q = (int32_t)b_row[k_idx] - offset;
                acc += a_q * b_q;
            }
            acc_out[i * n + j] += (double)acc * combined_scale;
        }
    }
}

void hc8_residual_matmul_b_simd(const hc8_t* a, const hc8_t* b,
                                uint32_t m, uint32_t k, uint32_t n,
                                int a_depth, int b_depth,
                                const hc8_residual_scales_t* a_scales,
                                const hc8_residual_scales_t* b_scales,
                                const hc8_quant_schema_t* schema,
                                hc8_t* out, float* out_scale) {
    if (m == 0 || k == 0 || n == 0 ||
        a == NULL || b == NULL || out == NULL || schema == NULL || out_scale == NULL ||
        a_scales == NULL || b_scales == NULL) {
        if (out_scale) *out_scale = 1.0f;
        return;
    }

    if (a_depth < 0) a_depth = 0;
    if (a_depth > 5) a_depth = 5;
    if (b_depth < 0) b_depth = 0;
    if (b_depth > 5) b_depth = 5;

    /* 1. AoS → SoA */
    hc8_soa_t a_soa, b_soa;
    hc8_aos_to_soa(a, m * k, &a_soa);
    hc8_aos_to_soa(b, k * n, &b_soa);

    /* 2. B 每层转置（k×n → n×k） */
    uint8_t* b_T[6] = {NULL};
    for (int l = 0; l <= b_depth; ++l) {
        b_T[l] = hc8_transpose_u8(b_soa.layer[l], k, n);
    }

    /* 3. 预计算偏移修正项
     *   sum_a[l][i] = Σ_k (a_soa.layer[l][i*k+k_idx] - 128)
     *   sum_b[m][j] = Σ_k (b_T[m][j*k+k_idx] - 128)
     *   只依赖 i 或 j，O(m*k) / O(k*n)
     */
    int32_t offset = schema->offset;  /* 128 */
    int32_t* sum_a[6] = {NULL};
    int32_t* sum_b[6] = {NULL};
    for (int l = 0; l <= a_depth; ++l) {
        if (a_scales->scales[l] == 0.0f) continue;
        sum_a[l] = HC_ALIGNED_CALLOC(m, int32_t);  /* A-0: 64 字节对齐 */
        for (uint32_t i = 0; i < m; ++i) {
            int32_t s = 0;
            for (uint32_t k_idx = 0; k_idx < k; ++k_idx) {
                s += (int32_t)a_soa.layer[l][i * k + k_idx] - offset;
            }
            sum_a[l][i] = s;
        }
    }
    for (int m_idx = 0; m_idx <= b_depth; ++m_idx) {
        if (b_scales->scales[m_idx] == 0.0f) continue;
        sum_b[m_idx] = HC_ALIGNED_CALLOC(n, int32_t);  /* A-0: 64 字节对齐 */
        for (uint32_t j = 0; j < n; ++j) {
            int32_t s = 0;
            for (uint32_t k_idx = 0; k_idx < k; ++k_idx) {
                s += (int32_t)b_T[m_idx][j * k + k_idx] - offset;
            }
            sum_b[m_idx][j] = s;
        }
    }

    /* 4. 对每对 (l, m_idx) 做 SoA 标量矩阵乘 + double 合并 */
    uint32_t out_size = m * n;
    double* c_double = HC_ALIGNED_CALLOC(out_size, double);  /* A-0: 64 字节对齐 */
    if (c_double == NULL) {
        *out_scale = 1.0f;
        goto cleanup;
    }

    for (int l = 0; l <= a_depth; ++l) {
        if (a_scales->scales[l] == 0.0f) continue;
        for (int m_idx = 0; m_idx <= b_depth; ++m_idx) {
            if (b_scales->scales[m_idx] == 0.0f) continue;

            double combined_scale = (double)a_scales->scales[l]
                                  * (double)b_scales->scales[m_idx];

            /* SoA 标量 kernel（这里暂不做偏移修正优化，直接在 kernel 内算 (a-128)*(b-128)） */
            hc8_matmul_kernel_soa_scalar(
                a_soa.layer[l], b_T[m_idx],
                m, k, n, offset,
                sum_a[l], sum_b[m_idx],
                c_double, combined_scale
            );
        }
    }

    /* 5. 重新量化 c_double 到 v[0]（与标量版相同） */
    {
        double max_abs = 0.0;
        for (uint32_t i = 0; i < out_size; ++i) {
            double v = fabs(c_double[i]);
            if (v > max_abs) max_abs = v;
        }
        float new_scale = (max_abs == 0.0) ? 1.0f : (float)(max_abs / 127.0);
        if (new_scale > 0.0f) {
            float inv_scale = 1.0f / new_scale;
            for (uint32_t i = 0; i < out_size; ++i) {
                float q_f = (float)c_double[i] * inv_scale;
                int32_t q = (int32_t)lroundf(q_f);
                if (q < schema->qmin) q = schema->qmin;
                else if (q > schema->qmax) q = schema->qmax;
                int32_t q_u = q + schema->offset;
                if (q_u < 0) q_u = 0;
                else if (q_u > 255) q_u = 255;
                out[i].v[0] = (uint8_t)q_u;
            }
        } else {
            for (uint32_t i = 0; i < out_size; ++i) {
                out[i].v[0] = (uint8_t)schema->offset;
            }
        }
        for (uint32_t i = 0; i < out_size; ++i) {
            out[i].v[1] = 0; out[i].v[2] = 0; out[i].v[3] = 0;
            out[i].v[4] = 0; out[i].v[5] = 0;
        }
        *out_scale = new_scale;
    }

    HC_ALIGNED_FREE(c_double);  /* A-0: 配套 _aligned_free */

cleanup:
    for (int l = 0; l < 6; ++l) HC_ALIGNED_FREE(sum_a[l]);   /* A-0 */
    for (int m_idx = 0; m_idx < 6; ++m_idx) HC_ALIGNED_FREE(sum_b[m_idx]);  /* A-0 */
    for (int l = 0; l < 6; ++l) HC_ALIGNED_FREE(b_T[l]);     /* A-0: transpose_u8 用对齐分配 */
    hc8_soa_free(&a_soa);
    hc8_soa_free(&b_soa);
}

/* ----------------------------------------------------------------------------
 * HC4 残差矩阵乘 SIMD 版（标量实现，验证 SoA per-nibble 布局正确性）
 *
 * 本函数先用标量实现验证 SoA per-nibble 布局的正确性。
 * AVX2 maddubs intrinsics 实现在 hc8_net_simd.c（A 阶段）。
 *
 * 数学（与 hc4_residual_matmul_b 相同）：
 *   对每对 (l, m_idx)：
 *     4 次 int4×int4 矩阵乘：S_hh, S_hl, S_lh, S_ll
 *     偏移修正：acc = 256*S_hh + 16*S_hl + 16*S_lh + S_ll
 *                     - 2048*sum_ah - 128*sum_al - 2048*sum_bh - 128*sum_bl + 16384*k
 *
 * SoA per-nibble 优化点：
 *   - high[l]/low[l] 连续 uint8 数组，缓存友好
 *   - 偏移修正项预计算优化：O(m*k*n) → O(m*k + k*n)
 * -------------------------------------------------------------------------- */

/* HC4 SoA 标量 kernel：计算 S[i][j] = Σ_k a_nibble[i][k] * b_nibble[k][j]
 *
 * @param a_u8       A 的某层某 nibble（high 或 low），m×k 行优先，uint8 值 0-15
 * @param b_T_u8     B 的某层某 nibble 转置后，n×k 行优先
 * @param m, k, n    维度
 * @param acc_out    int32 累加器（m×n，累加到现有值上）
 */
static void hc4_matmul_kernel_soa_scalar(
    const uint8_t* a_u8, const uint8_t* b_T_u8,
    uint32_t m, uint32_t k, uint32_t n,
    int32_t* acc_out
) {
    for (uint32_t i = 0; i < m; ++i) {
        const uint8_t* a_row = a_u8 + i * k;
        for (uint32_t j = 0; j < n; ++j) {
            const uint8_t* b_row = b_T_u8 + j * k;
            int32_t acc = 0;
            for (uint32_t k_idx = 0; k_idx < k; ++k_idx) {
                acc += (int32_t)a_row[k_idx] * (int32_t)b_row[k_idx];
            }
            acc_out[i * n + j] += acc;
        }
    }
}

/* ============================================================================
 * A-HC4-d0 阶段：AVX2 intrinsics kernel（int4×int4 → int32 累加）
 *
 * 数学：S[i][j] = Σ_k a_nibble[i][k] * b_nibble[k][j]，nibble 值 [0,15]
 *
 * intrinsics 链：
 *   _mm256_maddubs_epi16(a, b): 32 个 uint8×int8 → 16 个 int16
 *     注：nibble [0,15] < 128，作为 int8 读取仍为非负，结果与 uint8×uint8 一致
 *   _mm256_madd_epi16(prod, ones): 16 个 int16 × 1 → 8 个 int32 累加
 *
 * 寄存器使用：1 个累加器 + 2 个加载临时 + 2 个中间 + 1 个常量 = 6 个 YMM（<< 16，无溢出）
 *
 * 尾部处理：剩余元素用标量（MNIST k=784，主循环处理 768，尾部 16 个标量）
 * ============================================================================ */
#ifdef __AVX2__
static void hc4_matmul_kernel_soa_avx2(
    const uint8_t* a_u8, const uint8_t* b_T_u8,
    uint32_t m, uint32_t k, uint32_t n,
    int32_t* acc_out
) {
    const __m256i ones = _mm256_set1_epi16(1);

    for (uint32_t i = 0; i < m; ++i) {
        const uint8_t* a_row = a_u8 + i * k;
        for (uint32_t j = 0; j < n; ++j) {
            const uint8_t* b_row = b_T_u8 + j * k;

            __m256i acc = _mm256_setzero_si256();
            uint32_t k_idx = 0;

            /* 主循环：每次 32 个元素
             * maddubs: 32 个 uint8×int8 → 16 个 int16（每对相邻相加）
             * madd: 16 个 int16 × ones → 8 个 int32（每对相邻相加）
             * 一轮把 32 个 nibble×nibble 累加到 8 个 int32 lane */
            for (; k_idx + 32 <= k; k_idx += 32) {
                __m256i a_vec = _mm256_loadu_si256((const __m256i*)(a_row + k_idx));
                __m256i b_vec = _mm256_loadu_si256((const __m256i*)(b_row + k_idx));
                __m256i prod16 = _mm256_maddubs_epi16(a_vec, b_vec);
                __m256i sum32  = _mm256_madd_epi16(prod16, ones);
                acc = _mm256_add_epi32(acc, sum32);
            }

            /* 尾部标量（k 不是 32 的倍数时） */
            int32_t tail = 0;
            for (; k_idx < k; ++k_idx) {
                tail += (int32_t)a_row[k_idx] * (int32_t)b_row[k_idx];
            }

            /* 水平合并 8 个 int32 lane + tail */
            int32_t partial[8];
            _mm256_storeu_si256((__m256i*)partial, acc);
            int32_t total = partial[0] + partial[1] + partial[2] + partial[3]
                          + partial[4] + partial[5] + partial[6] + partial[7]
                          + tail;

            acc_out[i * n + j] += total;
        }
    }
}
#endif /* __AVX2__ */

/* 编译时选择 kernel：AVX2 可用时用 intrinsics，否则回退标量 */
#ifdef __AVX2__
#define hc4_matmul_kernel_soa hc4_matmul_kernel_soa_avx2
#else
#define hc4_matmul_kernel_soa hc4_matmul_kernel_soa_scalar
#endif

void hc4_residual_matmul_b_simd(const hc4_t* a, const hc4_t* b,
                                uint32_t m, uint32_t k, uint32_t n,
                                int a_depth, int b_depth,
                                const hc8_residual_scales_t* a_scales,
                                const hc8_residual_scales_t* b_scales,
                                const hc8_quant_schema_t* schema,
                                hc8_t* out, float* out_scale) {
    if (m == 0 || k == 0 || n == 0 ||
        a == NULL || b == NULL || out == NULL || schema == NULL || out_scale == NULL ||
        a_scales == NULL || b_scales == NULL) {
        if (out_scale) *out_scale = 1.0f;
        return;
    }

    if (a_depth < 0) a_depth = 0;
    if (a_depth > 5) a_depth = 5;
    if (b_depth < 0) b_depth = 0;
    if (b_depth > 5) b_depth = 5;

    /* 1. AoS → SoA per-nibble 预解包 */
    hc4_soa_t a_soa, b_soa;
    hc4_unpack_to_soa(a, m * k, a_depth, &a_soa);
    hc4_unpack_to_soa(b, k * n, b_depth, &b_soa);

    /* 2. B 每层每 nibble 转置（k×n → n×k） */
    uint8_t* b_T_high[6] = {NULL};
    uint8_t* b_T_low[6] = {NULL};
    for (int l = 0; l <= b_depth; ++l) {
        b_T_high[l] = hc8_transpose_u8(b_soa.high[l], k, n);
        b_T_low[l]  = hc8_transpose_u8(b_soa.low[l],  k, n);
    }

    /* 3. 预计算偏移修正项（O(m*k + k*n) 优化）
     *   sum_ah[l][i] = Σ_k a_soa.high[l][i*k+k_idx]（只依赖 i）
     *   sum_al[l][i] = Σ_k a_soa.low[l][i*k+k_idx]
     *   sum_bh[m][j] = Σ_k b_soa.high[m][k*n + j*n + j]... 用转置后：b_T_high[m][j*k+k_idx]
     *   sum_bl[m][j] = Σ_k b_T_low[m][j*k+k_idx]
     */
    int32_t* sum_ah[6] = {NULL};
    int32_t* sum_al[6] = {NULL};
    int32_t* sum_bh[6] = {NULL};
    int32_t* sum_bl[6] = {NULL};

    for (int l = 0; l <= a_depth; ++l) {
        if (a_scales->scales[l] == 0.0f) continue;
        sum_ah[l] = HC_ALIGNED_CALLOC(m, int32_t);  /* A-0: 64 字节对齐 */
        sum_al[l] = HC_ALIGNED_CALLOC(m, int32_t);  /* A-0: 64 字节对齐 */
        for (uint32_t i = 0; i < m; ++i) {
            int32_t sh = 0, sl = 0;
            for (uint32_t k_idx = 0; k_idx < k; ++k_idx) {
                sh += (int32_t)a_soa.high[l][i * k + k_idx];
                sl += (int32_t)a_soa.low[l][i * k + k_idx];
            }
            sum_ah[l][i] = sh;
            sum_al[l][i] = sl;
        }
    }
    for (int m_idx = 0; m_idx <= b_depth; ++m_idx) {
        if (b_scales->scales[m_idx] == 0.0f) continue;
        sum_bh[m_idx] = HC_ALIGNED_CALLOC(n, int32_t);  /* A-0: 64 字节对齐 */
        sum_bl[m_idx] = HC_ALIGNED_CALLOC(n, int32_t);  /* A-0: 64 字节对齐 */
        for (uint32_t j = 0; j < n; ++j) {
            int32_t sh = 0, sl = 0;
            for (uint32_t k_idx = 0; k_idx < k; ++k_idx) {
                sh += (int32_t)b_T_high[m_idx][j * k + k_idx];
                sl += (int32_t)b_T_low[m_idx][j * k + k_idx];
            }
            sum_bh[m_idx][j] = sh;
            sum_bl[m_idx][j] = sl;
        }
    }

    /* 4. 对每对 (l, m_idx) 做 4 次 SoA 标量 int4×int4 矩阵乘 + 偏移修正 + double 合并 */
    uint32_t out_size = m * n;
    double* c_double = HC_ALIGNED_CALLOC(out_size, double);    /* A-0: 64 字节对齐 */
    int32_t* S_hh = HC_ALIGNED_CALLOC(out_size, int32_t);      /* A-0: 64 字节对齐 */
    int32_t* S_hl = HC_ALIGNED_CALLOC(out_size, int32_t);      /* A-0: 64 字节对齐 */
    int32_t* S_lh = HC_ALIGNED_CALLOC(out_size, int32_t);      /* A-0: 64 字节对齐 */
    int32_t* S_ll = HC_ALIGNED_CALLOC(out_size, int32_t);      /* A-0: 64 字节对齐 */

    if (c_double == NULL || S_hh == NULL || S_hl == NULL || S_lh == NULL || S_ll == NULL) {
        *out_scale = 1.0f;
        HC_ALIGNED_FREE(c_double); HC_ALIGNED_FREE(S_hh); HC_ALIGNED_FREE(S_hl);
        HC_ALIGNED_FREE(S_lh); HC_ALIGNED_FREE(S_ll);
        goto cleanup;
    }

    int32_t k_const = (int32_t)k;

    for (int l = 0; l <= a_depth; ++l) {
        if (a_scales->scales[l] == 0.0f) continue;
        for (int m_idx = 0; m_idx <= b_depth; ++m_idx) {
            if (b_scales->scales[m_idx] == 0.0f) continue;

            /* 清零 4 个累加器 */
            memset(S_hh, 0, out_size * sizeof(int32_t));
            memset(S_hl, 0, out_size * sizeof(int32_t));
            memset(S_lh, 0, out_size * sizeof(int32_t));
            memset(S_ll, 0, out_size * sizeof(int32_t));

            /* 4 次 SoA 矩阵乘（AVX2 可用时用 intrinsics，否则标量） */
            hc4_matmul_kernel_soa(a_soa.high[l], b_T_high[m_idx], m, k, n, S_hh);
            hc4_matmul_kernel_soa(a_soa.high[l], b_T_low[m_idx],  m, k, n, S_hl);
            hc4_matmul_kernel_soa(a_soa.low[l],  b_T_high[m_idx], m, k, n, S_lh);
            hc4_matmul_kernel_soa(a_soa.low[l],  b_T_low[m_idx],  m, k, n, S_ll);

            /* 偏移修正 + double 合并（用预计算的 sum_i / sum_j） */
            double combined_scale = (double)a_scales->scales[l]
                                  * (double)b_scales->scales[m_idx];

            for (uint32_t i = 0; i < m; ++i) {
                int32_t sah = sum_ah[l][i];
                int32_t sal = sum_al[l][i];
                for (uint32_t j = 0; j < n; ++j) {
                    uint32_t idx = i * n + j;
                    int32_t total = 256 * S_hh[idx] + 16 * S_hl[idx]
                                  + 16 * S_lh[idx] + S_ll[idx]
                                  - 2048 * sah - 128 * sal
                                  - 2048 * sum_bh[m_idx][j] - 128 * sum_bl[m_idx][j]
                                  + 16384 * k_const;
                    c_double[idx] += (double)total * combined_scale;
                }
            }
        }
    }

    /* 5. 重新量化 c_double 到 v[0]（与标量版相同） */
    {
        double max_abs = 0.0;
        for (uint32_t i = 0; i < out_size; ++i) {
            double v = fabs(c_double[i]);
            if (v > max_abs) max_abs = v;
        }
        float new_scale = (max_abs == 0.0) ? 1.0f : (float)(max_abs / 127.0);
        if (new_scale > 0.0f) {
            float inv_scale = 1.0f / new_scale;
            for (uint32_t i = 0; i < out_size; ++i) {
                float q_f = (float)c_double[i] * inv_scale;
                int32_t q = (int32_t)lroundf(q_f);
                if (q < schema->qmin) q = schema->qmin;
                else if (q > schema->qmax) q = schema->qmax;
                int32_t q_u = q + schema->offset;
                if (q_u < 0) q_u = 0;
                else if (q_u > 255) q_u = 255;
                out[i].v[0] = (uint8_t)q_u;
            }
        } else {
            for (uint32_t i = 0; i < out_size; ++i) {
                out[i].v[0] = (uint8_t)schema->offset;
            }
        }
        for (uint32_t i = 0; i < out_size; ++i) {
            out[i].v[1] = 0; out[i].v[2] = 0; out[i].v[3] = 0;
            out[i].v[4] = 0; out[i].v[5] = 0;
        }
        *out_scale = new_scale;
    }

    HC_ALIGNED_FREE(c_double); HC_ALIGNED_FREE(S_hh); HC_ALIGNED_FREE(S_hl);
    HC_ALIGNED_FREE(S_lh); HC_ALIGNED_FREE(S_ll);

cleanup:
    for (int l = 0; l < 6; ++l) {
        HC_ALIGNED_FREE(sum_ah[l]); HC_ALIGNED_FREE(sum_al[l]);
        HC_ALIGNED_FREE(sum_bh[l]); HC_ALIGNED_FREE(sum_bl[l]);
        HC_ALIGNED_FREE(b_T_high[l]); HC_ALIGNED_FREE(b_T_low[l]);
    }
    hc4_soa_free(&a_soa);
    hc4_soa_free(&b_soa);
}

/* ============================================================================
 * A-HC8 阶段：AVX-VNNI kernel + SBE C 化（2026-07-22）
 *
 * 目标：
 *   1. hc8_matmul_kernel_vnni: VNNI 加速 int8×int8→int32 累加
 *   2. sbe_quantize_weight_blocks_c: 权重 per-block 量化 + 预处理（转置+有符号转换+预计算sum）
 *   3. sbe_matmul_c: SBE 分块 matmul（C 循环 + VNNI kernel + float 累加）
 *
 * VNNI 数学（方案 2 from design doc）：
 *   想要: result = Σ_k (a_u - 128) * (b_u - 128)
 *   VNNI 计算: vnni = Σ_k a_u * b_signed, 其中 b_signed = (int8)(b_u - 128)
 *   修正: result = vnni - 128 * sum_b_signed
 *   其中 sum_b_signed[j] = Σ_k b_signed[j][k] 预计算一次
 *
 * 编译条件：__AVXVNNI__（setup_net.py define_macros 启用）
 * 回退：无 VNNI 时用标量（功能等价，性能低）
 * ============================================================================ */

/* ============================================================================
 * P1 通用 AVX2 优化辅助函数（v1.7.1，2026-07-23）
 *
 * 对 sbe_matmul_c 和 sbe_matmul_smoothed_c 都有效：
 *   - sbe_accumulate_float_avx2: int32→float 累加 y += c_int32 * scale
 *   - sbe_quantize_block_avx2: max(|x|) + round + clip → uint8
 *
 * 编译条件：__AVX2__（setup_net.py /arch:AVX2 启用）
 * 回退：无 AVX2 时用标量（功能等价，性能低）
 * ============================================================================ */

/* P1-a: float 累加 AVX2+FMA
 *
 * y[i*n + j] += (float)c_int32[i*n + j] * combined_scale  for j in [0, n)
 *
 * AVX2 一次处理 8 个 float：
 *   - _mm256_cvtepi32_ps: int32×8 → float×8
 *   - _mm256_fmadd_ps:    float×8 FMA（y = a*b + y）
 *
 * tail 处理：n % 8 个剩余元素用标量
 */
static void sbe_accumulate_float_avx2(
    const int32_t* c_int32,
    float combined_scale,
    uint32_t m, uint32_t n,
    float* y  /* m×n, row-major, 累加语义 */
) {
#ifdef __AVX2__
    if (n >= 8) {
        __m256 v_scale = _mm256_set1_ps(combined_scale);
        uint32_t n_main = n & ~7u;  /* 8 的倍数部分 */

        for (uint32_t i = 0; i < m; ++i) {
            const int32_t* c_row = c_int32 + (size_t)i * n;
            float* y_row = y + (size_t)i * n;

            /* AVX2 主循环：一次 8 个元素 */
            uint32_t j = 0;
            for (; j < n_main; j += 8) {
                __m256i c_vec = _mm256_loadu_si256((const __m256i*)(c_row + j));
                __m256 c_f = _mm256_cvtepi32_ps(c_vec);
                __m256 y_vec = _mm256_loadu_ps(y_row + j);
                y_vec = _mm256_fmadd_ps(c_f, v_scale, y_vec);
                _mm256_storeu_ps(y_row + j, y_vec);
            }

            /* tail：剩余 1-7 个元素用标量 */
            for (; j < n; ++j) {
                y_row[j] += (float)c_row[j] * combined_scale;
            }
        }
        return;
    }
#endif
    /* 标量回退（无 AVX2 或 n < 8） */
    for (uint32_t i = 0; i < m; ++i) {
        const int32_t* c_row = c_int32 + (size_t)i * n;
        float* y_row = y + (size_t)i * n;
        for (uint32_t j = 0; j < n; ++j) {
            y_row[j] += (float)c_row[j] * combined_scale;
        }
    }
}

/* P1-b: 量化 AVX2（max + round + clip）
 *
 * 步骤：
 *   1. x_scale = max(|x_block|) / 127  （全零返回 1.0）
 *   2. x_u[i*stride + ki] = clamp(round(x[i*stride + ki] * inv_scale), -127, 127) + 128
 *
 * AVX2 加速：
 *   - abs:  _mm256_andnot_ps(sign_mask, v)
 *   - max:  _mm256_max_ps
 *   - round: _mm256_cvtps_epi32（默认 round-to-nearest-even）
 *   - clip:  _mm256_max_epi32 + _mm256_min_epi32
 *
 * 返回 x_scale（调用者用做 combined_scale 计算）
 * 注意：与标量版数学等价（lroundf 和 _mm256_cvtps_epi32 都是 banker's rounding）
 */
static float sbe_quantize_block_avx2(
    const float* x,        /* 输入，可能是 stride > k_block 的行优先矩阵 */
    uint32_t stride,       /* 行步长（x 的列数，用于跨行寻址） */
    uint32_t m, uint32_t k_block,
    uint8_t* x_u           /* 输出: m×k_block, contiguous */
) {
    /* 1. 计算 max(|x|) */
    float x_max_abs = 0.0f;
#ifdef __AVX2__
    if (k_block >= 8) {
        const __m256 sign_mask = _mm256_set1_ps(-0.0f);
        __m256 v_max = _mm256_setzero_ps();
        uint32_t k_main = k_block & ~7u;

        for (uint32_t i = 0; i < m; ++i) {
            const float* x_row = x + (size_t)i * stride;
            uint32_t ki = 0;
            for (; ki < k_main; ki += 8) {
                __m256 v = _mm256_loadu_ps(x_row + ki);
                v = _mm256_andnot_ps(sign_mask, v);  /* abs */
                v_max = _mm256_max_ps(v_max, v);
            }
            /* tail 标量 */
            for (; ki < k_block; ++ki) {
                float v = fabsf(x_row[ki]);
                if (v > x_max_abs) x_max_abs = v;
            }
        }

        /* horizontal max v_max → x_max_abs */
        if (k_main > 0) {
            /* _mm256_max_ps 是 per-lane 的，需要跨 lane reduce */
            __m128 hi = _mm256_extractf128_ps(v_max, 1);
            __m128 lo = _mm256_castps256_ps128(v_max);
            __m128 m128 = _mm_max_ps(hi, lo);
            __m128 shuf = _mm_shuffle_ps(m128, m128, _MM_SHUFFLE(2, 3, 0, 1));
            m128 = _mm_max_ps(m128, shuf);
            shuf = _mm_shuffle_ps(m128, m128, _MM_SHUFFLE(1, 0, 3, 2));
            m128 = _mm_max_ps(m128, shuf);
            float lane_max = _mm_cvtss_f32(m128);
            if (lane_max > x_max_abs) x_max_abs = lane_max;
        }
    } else
#endif
    {
        /* 标量回退 */
        for (uint32_t i = 0; i < m; ++i) {
            const float* x_row = x + (size_t)i * stride;
            for (uint32_t ki = 0; ki < k_block; ++ki) {
                float v = fabsf(x_row[ki]);
                if (v > x_max_abs) x_max_abs = v;
            }
        }
    }

    float x_scale = (x_max_abs == 0.0f) ? 1.0f : x_max_abs / 127.0f;
    float inv_x_scale = 1.0f / x_scale;

    /* 2. 量化 → x_u */
#ifdef __AVX2__
    if (k_block >= 8) {
        const __m256 v_inv = _mm256_set1_ps(inv_x_scale);
        const __m256i v_lo = _mm256_set1_epi32(-127);
        const __m256i v_hi = _mm256_set1_epi32(127);
        const __m256i v_offset = _mm256_set1_epi32(128);
        uint32_t k_main = k_block & ~7u;

        for (uint32_t i = 0; i < m; ++i) {
            const float* x_row = x + (size_t)i * stride;
            uint8_t* x_u_row = x_u + (size_t)i * k_block;

            uint32_t ki = 0;
            for (; ki < k_main; ki += 8) {
                __m256 v = _mm256_loadu_ps(x_row + ki);
                v = _mm256_mul_ps(v, v_inv);
                __m256i q = _mm256_cvtps_epi32(v);       /* round to int32×8 */
                q = _mm256_max_epi32(q, v_lo);            /* clip -127 */
                q = _mm256_min_epi32(q, v_hi);            /* clip 127 */
                q = _mm256_add_epi32(q, v_offset);        /* + 128 → [1, 255] */

                /* int32×8 → uint8×8 packed
                 * _mm256_packus_epi32 是 per-lane 的（输出 [lo0-3, 0-3, hi4-7, 0-7]）
                 * 需要 vpermq 重排为 [lo0-3, hi4-7, 0, 0] 后再 packus_epi16
                 *
                 * 简化方案：拆成两个 128-bit，分别 pack
                 *   __m128i lo = _mm256_castsi256_si128(q)       // [a0,a1,a2,a3]
                 *   __m128i hi = _mm256_extracti128_si256(q, 1)  // [a4,a5,a6,a7]
                 *   __m128i p16 = _mm_packus_epi32(lo, hi)       // uint16×8: [a0,a1,a2,a3,a4,a5,a6,a7]
                 *   __m128i p8 = _mm_packus_epi16(p16, _mm_setzero_si128())  // uint8×16: [a0..a7, 0..0]
                 *   _mm_storel_epi64(dst, p8)  // 存低 8 字节
                 */
                __m128i lo = _mm256_castsi256_si128(q);
                __m128i hi = _mm256_extracti128_si256(q, 1);
                __m128i p16 = _mm_packus_epi32(lo, hi);   /* int32×8 → uint16×8, 顺序正确 */
                __m128i p8 = _mm_packus_epi16(p16, _mm_setzero_si128());  /* uint16×8 → uint8×8 */
                _mm_storel_epi64((__m128i*)(x_u_row + ki), p8);  /* 存 8 字节 */
            }

            /* tail 标量 */
            for (; ki < k_block; ++ki) {
                int32_t q = (int32_t)lroundf(x_row[ki] * inv_x_scale);
                if (q < -127) q = -127;
                else if (q > 127) q = 127;
                x_u_row[ki] = (uint8_t)(q + 128);
            }
        }
        return x_scale;
    }
#endif

    /* 标量回退 */
    for (uint32_t i = 0; i < m; ++i) {
        const float* x_row = x + (size_t)i * stride;
        uint8_t* x_u_row = x_u + (size_t)i * k_block;
        for (uint32_t ki = 0; ki < k_block; ++ki) {
            int32_t q = (int32_t)lroundf(x_row[ki] * inv_x_scale);
            if (q < -127) q = -127;
            else if (q > 127) q = 127;
            x_u_row[ki] = (uint8_t)(q + 128);
        }
    }
    return x_scale;
}

/* P1-b-1: 计算 per-block 量化 scale = max(|x|) / 127（仅 max，不量化）
 *
 * 从 sbe_quantize_block_avx2 拆出，供 sbe_matmul_c 两阶段并行使用：
 *   阶段 1：串行调用本函数计算各 group 的 x_scale
 *   阶段 2：按 m 分块并行，用预设 x_scale 量化（sbe_quantize_block_with_scale_avx2）
 *
 * 返回值与 sbe_quantize_block_avx2 相同（max_abs==0 时返回 1.0）
 */
static float sbe_compute_block_scale_avx2(
    const float* x,        /* 输入，可能是 stride > k_block 的行优先矩阵 */
    uint32_t stride,       /* 行步长（x 的列数，用于跨行寻址） */
    uint32_t m, uint32_t k_block
) {
    float x_max_abs = 0.0f;
#ifdef __AVX2__
    if (k_block >= 8) {
        const __m256 sign_mask = _mm256_set1_ps(-0.0f);
        __m256 v_max = _mm256_setzero_ps();
        uint32_t k_main = k_block & ~7u;

        for (uint32_t i = 0; i < m; ++i) {
            const float* x_row = x + (size_t)i * stride;
            uint32_t ki = 0;
            for (; ki < k_main; ki += 8) {
                __m256 v = _mm256_loadu_ps(x_row + ki);
                v = _mm256_andnot_ps(sign_mask, v);  /* abs */
                v_max = _mm256_max_ps(v_max, v);
            }
            /* tail 标量 */
            for (; ki < k_block; ++ki) {
                float v = fabsf(x_row[ki]);
                if (v > x_max_abs) x_max_abs = v;
            }
        }

        /* horizontal max v_max → x_max_abs */
        if (k_main > 0) {
            __m128 hi = _mm256_extractf128_ps(v_max, 1);
            __m128 lo = _mm256_castps256_ps128(v_max);
            __m128 m128 = _mm_max_ps(hi, lo);
            __m128 shuf = _mm_shuffle_ps(m128, m128, _MM_SHUFFLE(2, 3, 0, 1));
            m128 = _mm_max_ps(m128, shuf);
            shuf = _mm_shuffle_ps(m128, m128, _MM_SHUFFLE(1, 0, 3, 2));
            m128 = _mm_max_ps(m128, shuf);
            float lane_max = _mm_cvtss_f32(m128);
            if (lane_max > x_max_abs) x_max_abs = lane_max;
        }
    } else
#endif
    {
        /* 标量回退 */
        for (uint32_t i = 0; i < m; ++i) {
            const float* x_row = x + (size_t)i * stride;
            for (uint32_t ki = 0; ki < k_block; ++ki) {
                float v = fabsf(x_row[ki]);
                if (v > x_max_abs) x_max_abs = v;
            }
        }
    }

    return (x_max_abs == 0.0f) ? 1.0f : x_max_abs / 127.0f;
}

/* P1-b-2: 用预设 scale 量化 x_block → x_u
 *
 * 与 sbe_compute_block_scale_avx2 配合使用，数学等价于 sbe_quantize_block_avx2
 * 的量化部分。供 sbe_matmul_c 阶段 2 并行调用（各线程用同一 group 的预设 scale
 * 量化各自的 m_priv 行，量化结果与一次性量化全部 m 行完全相同，因为量化是逐元素独立的）。
 */
static void sbe_quantize_block_with_scale_avx2(
    const float* x,        /* 输入，可能是 stride > k_block 的行优先矩阵 */
    uint32_t stride,       /* 行步长（x 的列数，用于跨行寻址） */
    uint32_t m, uint32_t k_block,
    float x_scale,
    uint8_t* x_u           /* 输出: m×k_block, contiguous */
) {
    float inv_x_scale = 1.0f / x_scale;

#ifdef __AVX2__
    if (k_block >= 8) {
        const __m256 v_inv = _mm256_set1_ps(inv_x_scale);
        const __m256i v_lo = _mm256_set1_epi32(-127);
        const __m256i v_hi = _mm256_set1_epi32(127);
        const __m256i v_offset = _mm256_set1_epi32(128);
        uint32_t k_main = k_block & ~7u;

        for (uint32_t i = 0; i < m; ++i) {
            const float* x_row = x + (size_t)i * stride;
            uint8_t* x_u_row = x_u + (size_t)i * k_block;

            uint32_t ki = 0;
            for (; ki < k_main; ki += 8) {
                __m256 v = _mm256_loadu_ps(x_row + ki);
                v = _mm256_mul_ps(v, v_inv);
                __m256i q = _mm256_cvtps_epi32(v);       /* round to int32×8 */
                q = _mm256_max_epi32(q, v_lo);            /* clip -127 */
                q = _mm256_min_epi32(q, v_hi);            /* clip 127 */
                q = _mm256_add_epi32(q, v_offset);        /* + 128 → [1, 255] */

                __m128i lo = _mm256_castsi256_si128(q);
                __m128i hi = _mm256_extracti128_si256(q, 1);
                __m128i p16 = _mm_packus_epi32(lo, hi);   /* int32×8 → uint16×8 */
                __m128i p8 = _mm_packus_epi16(p16, _mm_setzero_si128());  /* uint16×8 → uint8×8 */
                _mm_storel_epi64((__m128i*)(x_u_row + ki), p8);  /* 存 8 字节 */
            }

            /* tail 标量 */
            for (; ki < k_block; ++ki) {
                int32_t q = (int32_t)lroundf(x_row[ki] * inv_x_scale);
                if (q < -127) q = -127;
                else if (q > 127) q = 127;
                x_u_row[ki] = (uint8_t)(q + 128);
            }
        }
        return;
    }
#endif

    /* 标量回退 */
    for (uint32_t i = 0; i < m; ++i) {
        const float* x_row = x + (size_t)i * stride;
        uint8_t* x_u_row = x_u + (size_t)i * k_block;
        for (uint32_t ki = 0; ki < k_block; ++ki) {
            int32_t q = (int32_t)lroundf(x_row[ki] * inv_x_scale);
            if (q < -127) q = -127;
            else if (q > 127) q = 127;
            x_u_row[ki] = (uint8_t)(q + 128);
        }
    }
}

/* VNNI kernel：计算 c_int32[i][j] = Σ_k a_u[i][k] * b_signed_T[j][k] - 128 * sum_b[j]
 *
 * 输入：
 *   a_u:        m×k, row-major, uint8 (q + 128, 范围 [1, 255])
 *   b_signed_T: n×k, row-major, int8  (b_u - 128, 范围 [-127, 127])
 *   sum_b:      n, 预计算 Σ_k b_signed_T[j][k]
 *
 * 输出：
 *   c_int32:    m×n, row-major, int32 累加结果
 *              c_int32[i][j] = Σ_k (a_u[i][k] - 128) * (b_u[k][j] - 128)
 *              = VNNI(a_u, b_signed) - 128 * sum_b[j]
 */
static void hc8_matmul_kernel_vnni(
    const uint8_t* a_u,
    const int8_t* b_signed_T,
    const int32_t* sum_b,
    uint32_t m, uint32_t k, uint32_t n,
    int32_t* c_int32,  /* m×output_stride, 调用者分配；有效列数 = n */
    uint32_t output_stride
) {
    /* 三级路径选择：
     *   1. AVX-VNNI（_mm256_dpbusd_epi32）：32 元素/指令，需要 CPU 支持
     *   2. AVX2 nibble split（_mm256_maddubs_epi16 + _mm256_madd_epi16）：32 元素/~12 指令
     *   3. 标量回退：32 元素/32 指令
     *
     * nibble split 原理：
     *   a_u ∈ [0, 255], b_signed ∈ [-127, 127]
     *   直接 maddubs 会溢出（255*127*2 = 64770 > 32767）
     *   拆分 a_u = a_high*16 + a_low，其中 a_high, a_low ∈ [0, 15]
     *   maddubs(a_high, b) max = 15*127*2 = 3810 < 32767 ✓
     *   maddubs(a_low, b)  max = 15*127*2 = 3810 < 32767 ✓
     *   合并在 int32 域：result = sum_high*16 + sum_low（int32 无溢出）
     */
    int use_vnni = 0;
#ifdef __AVXVNNI__
    use_vnni = hc_detect_avx_vnni();
#endif

    if (use_vnni) {
        /* ===== 路径 1: AVX-VNNI ===== */
        for (uint32_t i = 0; i < m; ++i) {
            const uint8_t* a_row = a_u + i * k;
            for (uint32_t j = 0; j < n; ++j) {
                const int8_t* b_row = b_signed_T + j * k;

                __m256i acc = _mm256_setzero_si256();
                uint32_t k_idx = 0;

                for (; k_idx + 32 <= k; k_idx += 32) {
                    __m256i a_vec = _mm256_loadu_si256((const __m256i*)(a_row + k_idx));
                    __m256i b_vec = _mm256_loadu_si256((const __m256i*)(b_row + k_idx));
                    acc = _mm256_dpbusd_epi32(acc, a_vec, b_vec);
                }

                int32_t tail = 0;
                for (; k_idx < k; ++k_idx) {
                    tail += (int32_t)a_row[k_idx] * (int32_t)b_row[k_idx];
                }

                int32_t partial[8];
                _mm256_storeu_si256((__m256i*)partial, acc);
                int32_t vnni_result = partial[0] + partial[1] + partial[2] + partial[3]
                                    + partial[4] + partial[5] + partial[6] + partial[7]
                                    + tail;
                c_int32[i * output_stride + j] = vnni_result - 128 * sum_b[j];
            }
        }
    }
#ifdef __AVX2__
    else {
        /* ===== 路径 2: AVX2 nibble split =====
         * 当 CPU 不支持 VNNI（如 Zhaoxin/Hygon CPUID errata）时使用。
         * 用 _mm256_maddubs_epi16（uint8×int8→int16 pairwise）+ nibble 拆分
         * 避免 int16 溢出，性能约为标量的 2.7x。 */
        const __m256i mask_0F = _mm256_set1_epi8(0x0F);
        const __m256i ones_16 = _mm256_set1_epi16(1);

        for (uint32_t i = 0; i < m; ++i) {
            const uint8_t* a_row = a_u + (size_t)i * k;
            int32_t* c_row = c_int32 + (size_t)i * output_stride;

            for (uint32_t j = 0; j < n; ++j) {
                const int8_t* b_row = b_signed_T + (size_t)j * k;

                __m256i acc = _mm256_setzero_si256();
                uint32_t k_idx = 0;

                for (; k_idx + 32 <= k; k_idx += 32) {
                    __m256i a_vec = _mm256_loadu_si256((const __m256i*)(a_row + k_idx));
                    __m256i b_vec = _mm256_loadu_si256((const __m256i*)(b_row + k_idx));

                    /* 拆分 a_u 为高低 nibble（各 [0, 15]） */
                    __m256i a_low  = _mm256_and_si256(a_vec, mask_0F);
                    __m256i a_high = _mm256_and_si256(_mm256_srli_epi16(a_vec, 4), mask_0F);

                    /* maddubs: uint8×int8→int16 pairwise sum
                     * max pairwise = 15*127*2 = 3810 < 32767 ✓ */
                    __m256i prod_h16 = _mm256_maddubs_epi16(a_high, b_vec);
                    __m256i prod_l16 = _mm256_maddubs_epi16(a_low,  b_vec);

                    /* int16→int32 pairwise sum（madd with ones） */
                    __m256i sum_h32 = _mm256_madd_epi16(prod_h16, ones_16);
                    __m256i sum_l32 = _mm256_madd_epi16(prod_l16, ones_16);

                    /* 合并在 int32 域：a_u*b = a_high*16*b + a_low*b */
                    sum_h32 = _mm256_slli_epi32(sum_h32, 4);  /* * 16 */
                    acc = _mm256_add_epi32(acc, _mm256_add_epi32(sum_h32, sum_l32));
                }

                /* 尾部标量 */
                int32_t tail = 0;
                for (; k_idx < k; ++k_idx) {
                    tail += (int32_t)a_row[k_idx] * (int32_t)b_row[k_idx];
                }

                /* 水平合并 8 个 int32 lane + tail - 偏移修正 */
                int32_t partial[8];
                _mm256_storeu_si256((__m256i*)partial, acc);
                int32_t result = partial[0] + partial[1] + partial[2] + partial[3]
                               + partial[4] + partial[5] + partial[6] + partial[7]
                               + tail;
                c_row[j] = result - 128 * sum_b[j];
            }
        }
    }
#else
    else {
        /* ===== 路径 3: 标量回退（无 AVX2） ===== */
        for (uint32_t i = 0; i < m; ++i) {
            const uint8_t* a_row = a_u + i * k;
            for (uint32_t j = 0; j < n; ++j) {
                const int8_t* b_row = b_signed_T + j * k;
                int32_t acc = 0;
                for (uint32_t k_idx = 0; k_idx < k; ++k_idx) {
                    acc += (int32_t)a_row[k_idx] * (int32_t)b_row[k_idx];
                }
                c_int32[i * output_stride + j] = acc - 128 * sum_b[j];
            }
        }
    }
#endif
}

/* ============================================================================
 * SBE C 化：权重 per-block 量化 + 预处理
 *
 * 把 (k, n) 权重矩阵按 k 维度分 groups 块，每块独立量化。
 * 每块预处理为 VNNI 友好格式：
 *   - b_signed_T: n×k_block, int8, 转置 + 有符号转换
 *   - sum_b: n, int32, 预计算 Σ_k b_signed_T[j][k]
 *   - scale: float, 该块的量化 scale
 *
 * 输出布局（flat 数组，pybind11 友好）：
 *   w_signed_flat:  groups * n * k_block 个 int8（每块 n×k_block, row-major）
 *   w_sum_b_flat:   groups * n 个 int32
 *   w_scales:       groups 个 float
 * ============================================================================ */
void sbe_quantize_weight_blocks_c(
    const float* w,           /* k×n, row-major */
    uint32_t groups, uint32_t k_block, uint32_t k, uint32_t n,
    int8_t* w_signed_flat,    /* 输出: groups * n * k_block, 预分配 */
    int32_t* w_sum_b_flat,    /* 输出: groups * n, 预分配 */
    float* w_scales           /* 输出: groups, 预分配 */
) {
    if (w == NULL || w_signed_flat == NULL || w_sum_b_flat == NULL || w_scales == NULL) {
        return;
    }
    if (groups == 0 || k_block == 0 || k == 0 || n == 0) {
        return;
    }

    for (uint32_t g = 0; g < groups; ++g) {
        uint32_t start = g * k_block;
        /* 1. 计算 per-block scale = max(|w_block|) / 127 */
        float max_abs = 0.0f;
        for (uint32_t ki = 0; ki < k_block; ++ki) {
            const float* w_row = w + (start + ki) * n;
            for (uint32_t j = 0; j < n; ++j) {
                float v = fabsf(w_row[j]);
                if (v > max_abs) max_abs = v;
            }
        }
        float scale = (max_abs == 0.0f) ? 1.0f : max_abs / 127.0f;
        w_scales[g] = scale;
        float inv_scale = 1.0f / scale;

        /* 2. 量化 + 转置 + 有符号转换 */
        int8_t* b_signed_T = w_signed_flat + g * n * k_block;  /* n×k_block */
        int32_t* sum_b = w_sum_b_flat + g * n;                  /* n */

        for (uint32_t j = 0; j < n; ++j) {
            int32_t s = 0;
            for (uint32_t ki = 0; ki < k_block; ++ki) {
                /* w[start+ki, j] → 量化 → 转置到 [j, ki] */
                float w_val = w[(start + ki) * n + j];
                int32_t q = (int32_t)lroundf(w_val * inv_scale);
                if (q < -127) q = -127;
                else if (q > 127) q = 127;
                /* q_u = q + 128, b_signed = q_u - 128 = q */
                int8_t b_signed = (int8_t)q;
                b_signed_T[j * k_block + ki] = b_signed;
                s += (int32_t)b_signed;
            }
            sum_b[j] = s;
        }
    }
}

/* ============================================================================
 * SBE C 化：分块 matmul（C 循环 + VNNI kernel + float 累加）
 *
 * 对每个 group g：
 *   1. 提取 x_block: x[:, g*k_block : (g+1)*k_block]（strided, 需拷贝到连续缓冲）
 *   2. per-block 量化 x_block → x_u (uint8, m×k_block)
 *   3. VNNI matmul: c_int32 = Σ_k (x_u - 128) * (b_signed - 0) - 128 * sum_b
 *   4. float 累加: y += c_int32 * x_scale * w_scale
 * ============================================================================ */
void sbe_matmul_c(
    const float* x,            /* m×k, row-major */
    const int8_t* w_signed,    /* groups * n * k_block, row-major (n×k_block per block) */
    const int32_t* w_sum_b,    /* groups * n */
    const float* w_scales,     /* groups */
    uint32_t groups, uint32_t k_block, uint32_t m, uint32_t k, uint32_t n,
    float* y                   /* m×n, output, 调用者负责清零 */
) {
    if (x == NULL || w_signed == NULL || w_sum_b == NULL || w_scales == NULL || y == NULL) {
        return;
    }
    if (groups == 0 || k_block == 0 || m == 0 || k == 0 || n == 0) {
        return;
    }

#ifdef _OPENMP
    /* ============================================================
     * OpenMP 并行路径（v2，2026-07-24 优化）
     *
     * 问题（v1：按 groups 并行 + critical reduce）：
     *   - CNN SBE 的 groups = C_in（conv1 仅 3），10 核 CPU 只能利用 3 核
     *   - critical reduce 串行执行 O(m*n)，m*n 可达 921600，抵消并行收益
     *   - 实测 CPU 利用率仅 50-60%
     *
     * 方案（v2：按 m 维度并行，两阶段计算）：
     *   阶段 1（串行）：计算各 group 的 x_scale（O(m*k)，比 matmul 轻得多）
     *   阶段 2（并行）：按 m 分块，每块独立处理所有 groups
     *     - 各线程写 y 的不同行区间，无需 critical（消除 reduce 瓶颈）
     *     - 并行度 = m / M_BLOCK，远大于核心数（CNN: m=28800~1568）
     *     - 复用批量 AVX2 kernel（M_BLOCK 行一起量化+matmul，效率高）
     *
     * 数值等价性：
     *   - x_scale 仍按 per-block 全局 max 计算（与 v1 / 串行路径完全相同）
     *   - 量化是逐元素独立的，分块量化结果与整体量化完全相同
     *   - VNNI matmul、float 累加逻辑不变
     * ============================================================ */

    /* 阶段 1: 计算各 group 的 x_scale（串行，O(m*k_block*groups)）
     * 这步只做 max(|x|)，比阶段 2 的 matmul（O(m*k_block*n)）轻 n 倍 */
    float* x_scales = (float*)malloc((size_t)groups * sizeof(float));
    if (x_scales == NULL) return;

    for (uint32_t g = 0; g < groups; ++g) {
        x_scales[g] = sbe_compute_block_scale_avx2(
            x + (size_t)g * k_block, k, m, k_block
        );
    }

    /* 阶段 2: 按 m 分块并行
     * M_BLOCK 选择：足够大以复用 AVX2 批量 kernel + cache 局部性，
     *               足够小以平衡负载（m/M_BLOCK >> num_threads） */
    {
        const uint32_t M_BLOCK = 64;  /* 每块行数（2 的幂，cache 友好） */

        #pragma omp parallel
        {
            /* 线程私有缓冲区（按 M_BLOCK 分配，复用批量 AVX2 kernel） */
            uint8_t* x_u_priv = (uint8_t*)malloc((size_t)M_BLOCK * k_block);
            int32_t* c_int32_priv = (int32_t*)malloc((size_t)M_BLOCK * n * sizeof(int32_t));

            if (x_u_priv != NULL && c_int32_priv != NULL) {
                /* MSVC OpenMP 2.0 要求循环变量是 signed int */
                long i_start;
                #pragma omp for schedule(dynamic)
                for (i_start = 0; i_start < (long)m; i_start += (long)M_BLOCK) {
                    uint32_t m_priv = (uint32_t)((long)m - i_start);
                    if (m_priv > M_BLOCK) m_priv = M_BLOCK;

                    for (uint32_t g = 0; g < groups; ++g) {
                        /* 1. 量化 x[i_start:i_start+m_priv, g*kb:(g+1)*kb] → x_u_priv
                         *    用阶段 1 预算的 x_scales[g]（数学等价于 v1 的 per-block 量化） */
                        sbe_quantize_block_with_scale_avx2(
                            x + (size_t)i_start * k + (size_t)g * k_block,
                            k, m_priv, k_block, x_scales[g], x_u_priv
                        );

                        /* 2. VNNI matmul → c_int32_priv */
                        const int8_t* b_signed_T = w_signed + (size_t)g * n * k_block;
                        const int32_t* sum_b = w_sum_b + (size_t)g * n;
                        memset(c_int32_priv, 0, (size_t)m_priv * n * sizeof(int32_t));
                        hc8_matmul_kernel_vnni(x_u_priv, b_signed_T, sum_b, m_priv, k_block, n, c_int32_priv, n);

                        /* 3. float 累加 y[i_start:..., :] += c_int32 * scale（AVX2+FMA）
                         *    各线程写 y 的不同行区间，无需 critical */
                        float combined_scale = x_scales[g] * w_scales[g];
                        sbe_accumulate_float_avx2(c_int32_priv, combined_scale, m_priv, n, y + (size_t)i_start * n);
                    }
                }
            }

            free(x_u_priv);
            free(c_int32_priv);
        }
    }

    free(x_scales);
#else
    /* 串行路径（无 OpenMP 时的回退） */
    uint8_t* x_u = (uint8_t*)malloc((size_t)m * k_block);
    int32_t* c_int32 = (int32_t*)malloc((size_t)m * n * sizeof(int32_t));

    if (x_u == NULL || c_int32 == NULL) {
        free(x_u); free(c_int32);
        return;
    }

    for (uint32_t g = 0; g < groups; ++g) {
        uint32_t start = g * k_block;

        float x_scale = sbe_quantize_block_avx2(
            x + start, k, m, k_block, x_u
        );

        const int8_t* b_signed_T = w_signed + (size_t)g * n * k_block;
        const int32_t* sum_b = w_sum_b + (size_t)g * n;

        memset(c_int32, 0, (size_t)m * n * sizeof(int32_t));
        hc8_matmul_kernel_vnni(x_u, b_signed_T, sum_b, m, k_block, n, c_int32, n);

        float combined_scale = x_scale * w_scales[g];
        sbe_accumulate_float_avx2(c_int32, combined_scale, m, n, y);
    }

    free(x_u);
    free(c_int32);
#endif
}

/* ============================================================================
 * SBE Conv2d 前向融合（v2.1.0-conv-fusion-omp，2026-07-28）
 *
 * 在 C 层融合 im2col + SBE matmul，消除 Python 层 im2col + transpose + reshape
 * 开销（诊断显示 81% 时间花在 Python 层）。
 *
 * 数学等价于：
 *   1. im2col: x (B, C_in, H, W) → x_col_2d (B*L, K), K=C_in*kh*kw, L=H_out*W_out
 *   2. sbe_matmul: x_col_2d × w → y_col_2d (B*L, C_out)
 *   3. reshape: y_col_2d → y (B, C_out, H_out, W_out)
 *
 * v2.1 改动（P0 优化）：
 *   - OpenMP 并行：按 m 分块，参照 sbe_matmul_c 的并行框架
 *   - workspace buffer 复用：y_col_2d 静态池，避免每次 forward malloc/free
 *   - 对齐分配：所有缓冲区用 HC_ALIGNED_ALLOC（64 字节对齐），AVX2 kernel 友好
 *
 * v2.2 改动（P1.1 优化）：
 *   - 阶段 3 转置用 AVX2 gather (_mm256_i32gather_ps) 向量化
 *   - 一次读取 8 个 strided 元素（stride=n），向量化加 bias 后 storeu
 *   - 尾部标量回退，无 AVX2 时整体标量回退
 *
 * 数值等价性：
 *   - x_scale 保持 per-m_block 计算（与 v2.0 一致，测试已通过）
 *   - 各线程写 y_col_2d 的不同行区间，无数据竞争
 *   - 阶段 3 gather + add + store 与标量逐元素加法完全等价，max_diff = 0
 * ============================================================================ */

/* P1.2: y_col_2d workspace 池已删除 —— 阶段 2 直接写入 y 的 (B, n, L) 布局 */

void sbe_conv2d_forward_c(
    const float* x,            /* (B, C_in, H, W), row-major */
    const int8_t* w_signed,    /* groups * n * k_block, row-major (n×k_block per block) */
    const int32_t* w_sum_b,    /* groups * n */
    const float* w_scales,     /* groups */
    uint32_t B, uint32_t C_in, uint32_t H, uint32_t W,
    uint32_t C_out, uint32_t kh, uint32_t kw,
    uint32_t stride, uint32_t padding,
    uint32_t groups, uint32_t k_block,
    const float* bias,         /* (C_out,) 或 NULL */
    float* y                   /* (B, C_out, H_out, W_out), row-major, 调用者分配 */
) {
    if (x == NULL || w_signed == NULL || w_sum_b == NULL || w_scales == NULL || y == NULL) {
        return;
    }

    /* 计算输出尺寸 */
    uint32_t H_out = (H + 2 * padding - kh) / stride + 1;
    uint32_t W_out = (W + 2 * padding - kw) / stride + 1;
    uint32_t L = H_out * W_out;
    uint32_t K = C_in * kh * kw;
    uint32_t M = B * L;  /* 总 "行数" (B*L 个 patch) */
    uint32_t n = C_out;

    /* ---- P1.2: 直接写入 y 的 (B, C_out, L) 布局，消除阶段 3 转置 ----
     * 旧版：kernel 写 y_col_2d (M, n) row-major → 阶段 3 转置到 y (B, n, L) + bias
     * 新版：阶段 2 float 累加直写 y[(b*n + j)*L + l]，bias 在 g==0 时一并加入
     * 优化：g==0 用 store (=)，g>0 用 accumulate (+=)；无需 memset y
     * 线程安全：每个 (b, l) 仅由一个线程处理（global_row 唯一），g 按 0→groups-1 顺序执行
     */

    const uint32_t M_BLOCK = 64;  /* 每块行数（与 sbe_matmul_c 一致，cache 友好） */

#ifdef _OPENMP
    /* ============================================================
     * OpenMP 并行路径（P0.1）
     *
     * 参照 sbe_matmul_c 的并行框架，按 m 分块并行：
     *   - 每线程私有缓冲区（x_col_priv、x_u_priv、c_int32_priv）
     *   - 各线程写 y_col_2d 的不同行区间，无数据竞争
     *   - im2col 在分块内做（每线程独立）
     *   - x_scale 保持 per-m_block 计算（与 v2.0 数值一致）
     * ============================================================ */
    #pragma omp parallel
    {
        /* 线程私有缓冲区（P0.3: 对齐分配）*/
        float* x_col_priv = (float*)HC_ALIGNED_ALLOC((size_t)M_BLOCK * K, float);
        uint8_t* x_u_priv = (uint8_t*)HC_ALIGNED_ALLOC((size_t)M_BLOCK * k_block, uint8_t);
        int32_t* c_int32_priv = (int32_t*)HC_ALIGNED_ALLOC((size_t)M_BLOCK * (L > n ? L : n), int32_t);

        if (x_col_priv != NULL && x_u_priv != NULL && c_int32_priv != NULL) {
            /* MSVC OpenMP 2.0 要求循环变量是 signed int */
            long i_start;
            #pragma omp for schedule(dynamic)
            for (i_start = 0; i_start < (long)M; i_start += (long)M_BLOCK) {
                uint32_t m_start = (uint32_t)i_start;
                uint32_t m_priv = M - m_start;
                if (m_priv > M_BLOCK) m_priv = M_BLOCK;

                /* ---- 阶段 1: im2col（C 实现，避免 Python stride tricks 开销）----
                 * x_col_priv[m_priv × K]，row-major
                 * x_col_priv[row, c*kh*kw + i*kw + j] = x_padded[b, c, h_idx + i, w_idx + j]
                 * 其中 b = (m_start + row) / L, l = (m_start + row) % L
                 *      h_idx = (l / W_out) * stride - padding
                 *      w_idx = (l % W_out) * stride - padding
                 */
                for (uint32_t row = 0; row < m_priv; ++row) {
                    uint32_t global_row = m_start + row;
                    uint32_t b = global_row / L;
                    uint32_t l = global_row % L;
                    uint32_t h_out = l / W_out;
                    uint32_t w_out = l % W_out;

                    float* x_col_row = x_col_priv + (size_t)row * K;

                    for (uint32_t c = 0; c < C_in; ++c) {
                        for (uint32_t i = 0; i < kh; ++i) {
                            int h_idx = (int)(h_out * stride + i) - (int)padding;
                            for (uint32_t j = 0; j < kw; ++j) {
                                int w_idx = (int)(w_out * stride + j) - (int)padding;
                                float val = 0.0f;
                                if (h_idx >= 0 && h_idx < (int)H &&
                                    w_idx >= 0 && w_idx < (int)W) {
                                    val = x[((b * C_in + c) * H + h_idx) * W + w_idx];
                                }
                                x_col_row[(c * kh + i) * kw + j] = val;
                            }
                        }
                    }
                }

                /* ---- 阶段 2: sbe_matmul（复用 sbe_matmul_c 的核心逻辑）----
                 * 对每个 group g：
                 *   1. 计算 x_scale（per-block max，与 v2.0 一致）
                 *   2. 量化 x_col_priv[:, g*kb:(g+1)*kb] → x_u_priv
                 *   3. VNNI matmul → c_int32_priv
                 *   4. float 累加 y_col_2d[m_start:..., :] += c_int32 * x_scale * w_scale
                 *      各线程写不同行区间，thread-safe
                 */
                for (uint32_t g = 0; g < groups; ++g) {
                    uint32_t kb_start = g * k_block;

                    /* 1. 计算 x_scale */
                    float x_scale = sbe_compute_block_scale_avx2(
                        x_col_priv + kb_start, K, m_priv, k_block
                    );

                    /* 2. 量化 */
                    sbe_quantize_block_with_scale_avx2(
                        x_col_priv + kb_start, K, m_priv, k_block, x_scale, x_u_priv
                    );

                    /* 3. VNNI matmul → c_int32_priv (m_priv × L 步长, 前 n 列有效)
                     *    kernel 保证写入所有 (i, j∈[0,n)) 位置，无需 memset */
                    const int8_t* b_signed_T = w_signed + (size_t)g * n * k_block;
                    const int32_t* sum_b = w_sum_b + (size_t)g * n;
                    hc8_matmul_kernel_vnni(x_u_priv, b_signed_T, sum_b, m_priv, k_block, n, c_int32_priv, L);

                    /* 4. float 累加直写 y（(B, n, L) 布局），bias 下沉到 g==0
                     *    AVX2 向量化 c_int32→float 转换 + 乘法 + bias；
                     *    y strided store 用标量展开（AVX2 无 scatter）
                     *    优化：g==0 用 store (=)，g>0 用 accumulate (+=) */
                    float combined_scale = x_scale * w_scales[g];
                    int add_bias = (bias != NULL && g == 0);
                    int use_store = (g == 0);  /* g==0 首次写入，无需 read-modify-write */
#ifdef __AVX2__
                    __m256 v_scale = _mm256_set1_ps(combined_scale);
#endif
                    for (uint32_t i = 0; i < m_priv; ++i) {
                        uint32_t global_row = m_start + i;
                        uint32_t b = global_row / L;
                        uint32_t l = global_row % L;
                        const int32_t* c_row = c_int32_priv + (size_t)i * L;
                        float* y_base = y + (size_t)b * n * L + l;
                        uint32_t j = 0;
#ifdef __AVX2__
                        for (; j + 8 <= n; j += 8) {
                            __m256i c_vec = _mm256_loadu_si256((const __m256i*)(c_row + j));
                            __m256 c_f = _mm256_cvtepi32_ps(c_vec);
                            __m256 val = _mm256_mul_ps(c_f, v_scale);
                            if (add_bias) {
                                __m256 b_vec = _mm256_loadu_ps(bias + j);
                                val = _mm256_add_ps(val, b_vec);
                            }
                            float tmp[8];
                            _mm256_storeu_ps(tmp, val);
                            if (use_store) {
                                y_base[(size_t)(j + 0) * L] = tmp[0];
                                y_base[(size_t)(j + 1) * L] = tmp[1];
                                y_base[(size_t)(j + 2) * L] = tmp[2];
                                y_base[(size_t)(j + 3) * L] = tmp[3];
                                y_base[(size_t)(j + 4) * L] = tmp[4];
                                y_base[(size_t)(j + 5) * L] = tmp[5];
                                y_base[(size_t)(j + 6) * L] = tmp[6];
                                y_base[(size_t)(j + 7) * L] = tmp[7];
                            } else {
                                y_base[(size_t)(j + 0) * L] += tmp[0];
                                y_base[(size_t)(j + 1) * L] += tmp[1];
                                y_base[(size_t)(j + 2) * L] += tmp[2];
                                y_base[(size_t)(j + 3) * L] += tmp[3];
                                y_base[(size_t)(j + 4) * L] += tmp[4];
                                y_base[(size_t)(j + 5) * L] += tmp[5];
                                y_base[(size_t)(j + 6) * L] += tmp[6];
                                y_base[(size_t)(j + 7) * L] += tmp[7];
                            }
                        }
#endif
                        for (; j < n; ++j) {
                            float val = (float)c_row[j] * combined_scale;
                            if (add_bias) val += bias[j];
                            if (use_store) y_base[(size_t)j * L] = val;
                            else            y_base[(size_t)j * L] += val;
                        }
                    }
                }
            }
        }

        HC_ALIGNED_FREE(x_col_priv);
        HC_ALIGNED_FREE(x_u_priv);
        HC_ALIGNED_FREE(c_int32_priv);
    }
#else
    /* ---- 串行路径（无 OpenMP 时的回退，P0.3: 对齐分配）---- */
    float* x_col_block = (float*)HC_ALIGNED_ALLOC((size_t)M_BLOCK * K, float);
    uint8_t* x_u = (uint8_t*)HC_ALIGNED_ALLOC((size_t)M_BLOCK * k_block, uint8_t);
    int32_t* c_int32 = (int32_t*)HC_ALIGNED_ALLOC((size_t)M_BLOCK * (L > n ? L : n), int32_t);

    if (x_col_block == NULL || x_u == NULL || c_int32 == NULL) {
        HC_ALIGNED_FREE(x_col_block);
        HC_ALIGNED_FREE(x_u);
        HC_ALIGNED_FREE(c_int32);
        return;
    }

    for (uint32_t m_start = 0; m_start < M; m_start += M_BLOCK) {
        uint32_t m_priv = M - m_start;
        if (m_priv > M_BLOCK) m_priv = M_BLOCK;

        /* 阶段 1: im2col */
        for (uint32_t row = 0; row < m_priv; ++row) {
            uint32_t global_row = m_start + row;
            uint32_t b = global_row / L;
            uint32_t l = global_row % L;
            uint32_t h_out = l / W_out;
            uint32_t w_out = l % W_out;

            float* x_col_row = x_col_block + (size_t)row * K;

            for (uint32_t c = 0; c < C_in; ++c) {
                for (uint32_t i = 0; i < kh; ++i) {
                    int h_idx = (int)(h_out * stride + i) - (int)padding;
                    for (uint32_t j = 0; j < kw; ++j) {
                        int w_idx = (int)(w_out * stride + j) - (int)padding;
                        float val = 0.0f;
                        if (h_idx >= 0 && h_idx < (int)H &&
                            w_idx >= 0 && w_idx < (int)W) {
                            val = x[((b * C_in + c) * H + h_idx) * W + w_idx];
                        }
                        x_col_row[(c * kh + i) * kw + j] = val;
                    }
                }
            }
        }

        /* 阶段 2: sbe_matmul */
        for (uint32_t g = 0; g < groups; ++g) {
            uint32_t kb_start = g * k_block;

            float x_scale = sbe_compute_block_scale_avx2(
                x_col_block + kb_start, K, m_priv, k_block
            );

            sbe_quantize_block_with_scale_avx2(
                x_col_block + kb_start, K, m_priv, k_block, x_scale, x_u
            );

            const int8_t* b_signed_T = w_signed + (size_t)g * n * k_block;
            const int32_t* sum_b = w_sum_b + (size_t)g * n;
            hc8_matmul_kernel_vnni(x_u, b_signed_T, sum_b, m_priv, k_block, n, c_int32, L);

            /* float 累加直写 y（(B, n, L) 布局），bias 下沉到 g==0
             * g==0 用 store (=)，g>0 用 accumulate (+=) */
            float combined_scale = x_scale * w_scales[g];
            int use_store = (g == 0);
            for (uint32_t i = 0; i < m_priv; ++i) {
                uint32_t global_row = m_start + i;
                uint32_t b = global_row / L;
                uint32_t l = global_row % L;
                const int32_t* c_row = c_int32 + (size_t)i * L;
                for (uint32_t j = 0; j < n; ++j) {
                    float val = (float)c_row[j] * combined_scale;
                    if (bias != NULL && g == 0) {
                        val += bias[j];
                    }
                    if (use_store) y[(size_t)(b * n + j) * L + l] = val;
                    else            y[(size_t)(b * n + j) * L + l] += val;
                }
            }
        }
    }

    HC_ALIGNED_FREE(x_col_block);
    HC_ALIGNED_FREE(x_u);
    HC_ALIGNED_FREE(c_int32);
#endif

    /* P1.2: 阶段 3 转置循环已删除 —— 阶段 2 float 累加直接写入 y 的 (B, n, L) 布局 */
}

/* ============================================================================
 * Triple-int8 缩放 C 化（v1.9.0，2026-07-24）
 *
 * 数学等价于 sbe_conv2d.py 的 rescale_to_triple_int8_sbe：
 *   scale = max(|x|) / 127
 *   C_high = round(x / scale)          ∈ [-127, 127]
 *   ε1 = (x/scale - C_high) * 256
 *   C_mid = round(ε1)                  ∈ [-128, 128]
 *   ε2 = (ε1 - C_mid) * 256
 *   C_low = round(ε2)                  ∈ [-128, 128]
 *
 * 性能（vs Python numpy float64 版本）：
 *   - AVX2 一次 8 个 float（vs numpy 逐元素）
 *   - OpenMP 并行（vs numpy 单线程元素级操作）
 *   - 一次遍历完成（vs numpy 多次遍历 + float64 转换 + 数组分配）
 *   - 实测 conv2 (2M 元素): 60ms → < 2ms（30x+ 加速）
 *
 * 数值精度：全部 float32（残差 ∈ [-0.5, 0.5]，float32 精度 >> round 需求）
 * ============================================================================ */
void sbe_rescale_to_triple_c(
    const float* x, uint32_t n,
    float* C_high, float* C_mid, float* C_low, float* out_scale
) {
    if (x == NULL || n == 0 || C_high == NULL || C_mid == NULL || C_low == NULL || out_scale == NULL) {
        if (out_scale) *out_scale = 1.0f;
        return;
    }

    /* 阶段 1: 求 max(|x|)（AVX2 加速，串行 — max 计算比分解轻得多，不需并行） */
    float x_max_abs = 0.0f;
    long n_long = (long)n;
    long n_main = n_long & ~7L;  /* 8 的倍数部分 */
#ifdef __AVX2__
    if (n_main > 0) {
        const __m256 sign_mask = _mm256_set1_ps(-0.0f);
        __m256 v_max = _mm256_setzero_ps();
        for (long ii = 0; ii < n_main; ii += 8) {
            __m256 v = _mm256_loadu_ps(x + ii);
            v = _mm256_andnot_ps(sign_mask, v);  /* abs */
            v_max = _mm256_max_ps(v_max, v);
        }
        /* horizontal reduce v_max → x_max_abs */
        __m128 hi = _mm256_extractf128_ps(v_max, 1);
        __m128 lo = _mm256_castps256_ps128(v_max);
        __m128 m128 = _mm_max_ps(hi, lo);
        __m128 shuf = _mm_shuffle_ps(m128, m128, _MM_SHUFFLE(2, 3, 0, 1));
        m128 = _mm_max_ps(m128, shuf);
        shuf = _mm_shuffle_ps(m128, m128, _MM_SHUFFLE(1, 0, 3, 2));
        m128 = _mm_max_ps(m128, shuf);
        x_max_abs = _mm_cvtss_f32(m128);
    }
    for (long ii = n_main; ii < n_long; ++ii) {
        float a = fabsf(x[ii]);
        if (a > x_max_abs) x_max_abs = a;
    }
#else
    for (uint32_t i = 0; i < n; ++i) {
        float a = fabsf(x[i]);
        if (a > x_max_abs) x_max_abs = a;
    }
#endif

    float scale = (x_max_abs == 0.0f) ? 1.0f : x_max_abs / 127.0f;
    *out_scale = scale;
    float inv_scale = 1.0f / scale;

    /* 全零输入：三分量全 0 */
    if (x_max_abs == 0.0f) {
        memset(C_high, 0, n * sizeof(float));
        memset(C_mid, 0, n * sizeof(float));
        memset(C_low, 0, n * sizeof(float));
        return;
    }

    /* 阶段 2: 分解为 3 个 int8 分量（AVX2 + OpenMP 并行）
     * 循环步进固定 8，OpenMP 2.0 友好 */
#ifdef __AVX2__
    const __m256 v_inv = _mm256_set1_ps(inv_scale);
    const __m256 v_256 = _mm256_set1_ps(256.0f);
    const __m256 v_lo127 = _mm256_set1_ps(-127.0f);
    const __m256 v_hi127 = _mm256_set1_ps(127.0f);
    const __m256 v_lo128 = _mm256_set1_ps(-128.0f);
    const __m256 v_hi128 = _mm256_set1_ps(128.0f);
    /* MSVC C 模式下 const int 不是编译时常量，_mm256_round_ps 需要宏 */
#define SBE_ROUND_MODE (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)

    long ii;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (ii = 0; ii < n_main; ii += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + ii);
        __m256 x_norm = _mm256_mul_ps(x_vec, v_inv);

        /* C_high = round(x_norm), clip [-127, 127] */
        __m256 ch = _mm256_round_ps(x_norm, SBE_ROUND_MODE);
        ch = _mm256_max_ps(ch, v_lo127);
        ch = _mm256_min_ps(ch, v_hi127);
        _mm256_storeu_ps(C_high + ii, ch);

        /* ε1 = (x_norm - C_high) * 256 */
        __m256 r1 = _mm256_sub_ps(x_norm, ch);
        r1 = _mm256_mul_ps(r1, v_256);

        /* C_mid = round(ε1), clip [-128, 128] */
        __m256 cm = _mm256_round_ps(r1, SBE_ROUND_MODE);
        cm = _mm256_max_ps(cm, v_lo128);
        cm = _mm256_min_ps(cm, v_hi128);
        _mm256_storeu_ps(C_mid + ii, cm);

        /* ε2 = (ε1 - C_mid) * 256 */
        __m256 r2 = _mm256_sub_ps(r1, cm);
        r2 = _mm256_mul_ps(r2, v_256);

        /* C_low = round(ε2), clip [-128, 128] */
        __m256 cl = _mm256_round_ps(r2, SBE_ROUND_MODE);
        cl = _mm256_max_ps(cl, v_lo128);
        cl = _mm256_min_ps(cl, v_hi128);
        _mm256_storeu_ps(C_low + ii, cl);
    }

    /* tail 标量（0-7 个元素，不并行） */
    for (ii = n_main; ii < n_long; ++ii) {
        float x_norm = x[ii] * inv_scale;
        float ch = roundf(x_norm);
        if (ch < -127.0f) ch = -127.0f;
        else if (ch > 127.0f) ch = 127.0f;
        C_high[ii] = ch;
        float r1 = (x_norm - ch) * 256.0f;
        float cm = roundf(r1);
        if (cm < -128.0f) cm = -128.0f;
        else if (cm > 128.0f) cm = 128.0f;
        C_mid[ii] = cm;
        float r2 = (r1 - cm) * 256.0f;
        float cl = roundf(r2);
        if (cl < -128.0f) cl = -128.0f;
        else if (cl > 128.0f) cl = 128.0f;
        C_low[ii] = cl;
    }
#else
    /* 标量路径（无 AVX2） */
    long ii;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (ii = 0; ii < n_long; ++ii) {
        float x_norm = x[ii] * inv_scale;
        float ch = roundf(x_norm);
        if (ch < -127.0f) ch = -127.0f;
        else if (ch > 127.0f) ch = 127.0f;
        C_high[ii] = ch;
        float r1 = (x_norm - ch) * 256.0f;
        float cm = roundf(r1);
        if (cm < -128.0f) cm = -128.0f;
        else if (cm > 128.0f) cm = 128.0f;
        C_mid[ii] = cm;
        float r2 = (r1 - cm) * 256.0f;
        float cl = roundf(r2);
        if (cl < -128.0f) cl = -128.0f;
        else if (cl > 128.0f) cl = 128.0f;
        C_low[ii] = cl;
    }
#endif
}

/* ============================================================================
 * SBE + Smoothing C 分块 matmul（v1.7.0，2026-07-23）
 *
 * 设计见 smoothing_c_external_optimization.md
 *
 * 与 sbe_matmul_c 数学等价，但更精确：
 *   对每个 group 的 x_block 做 per-row mean shift 后再量化，
 *   主项 INT8 matmul + 修正项 float 累加。
 *
 * 数学：
 *   y = Σ_g x_block_g @ w_block_g
 *     = Σ_g [(x_block_g - c_mean_g) @ w_block_g + c_mean_g @ w_block_g]
 *
 *   修正项优化: c_mean @ w_block = c_mean * w_sum
 *     其中 w_sum[j] = w_scale * Σ_k w_int8[j][k] = w_scale * w_sum_b[j]
 *
 * 关键：与 sbe_matmul_c 共享 VNNI kernel（hc8_matmul_kernel_vnni）。
 *       Smoothing 只影响预处理（mean+shift+quantize）和后处理（修正项累加）。
 * ============================================================================ */
void sbe_matmul_smoothed_c(
    const float* x,            /* m×k, row-major */
    const int8_t* w_signed,    /* groups * n * k_block, row-major (n×k_block per block) */
    const int32_t* w_sum_b,    /* groups * n */
    const float* w_scales,     /* groups */
    uint32_t groups, uint32_t k_block, uint32_t m, uint32_t k, uint32_t n,
    float* y                   /* m×n, output, 调用者负责清零 */
) {
    if (x == NULL || w_signed == NULL || w_sum_b == NULL || w_scales == NULL || y == NULL) {
        return;
    }
    if (groups == 0 || k_block == 0 || m == 0 || k == 0 || n == 0) {
        return;
    }

    /* 临时缓冲区 */
    uint8_t* x_u = (uint8_t*)malloc((size_t)m * k_block);        /* 连续量化 x_shifted block */
    int32_t* c_int32 = (int32_t*)malloc((size_t)m * n * sizeof(int32_t));  /* VNNI 输出 */
    float* c_mean = (float*)malloc((size_t)m * sizeof(float));   /* per-row mean of current block */
    float* x_shifted = (float*)malloc((size_t)m * k_block * sizeof(float));  /* shifted block (连续) */

    if (x_u == NULL || c_int32 == NULL || c_mean == NULL || x_shifted == NULL) {
        free(x_u); free(c_int32); free(c_mean); free(x_shifted);
        return;
    }

    for (uint32_t g = 0; g < groups; ++g) {
        uint32_t start = g * k_block;

        /* 1. per-row mean: c_mean[i] = mean(x_block[i, :]) */
        for (uint32_t i = 0; i < m; ++i) {
            const float* x_row = x + (size_t)i * k + start;
            float sum = 0.0f;
            for (uint32_t ki = 0; ki < k_block; ++ki) {
                sum += x_row[ki];
            }
            c_mean[i] = sum / (float)k_block;
        }

        /* 2. shift: x_shifted[i, ki] = x[i, start+ki] - c_mean[i]
         *    写入连续缓冲区（stride = k_block），供 P1-b 量化使用 */
        for (uint32_t i = 0; i < m; ++i) {
            const float* x_row = x + (size_t)i * k + start;
            float* xs_row = x_shifted + (size_t)i * k_block;
            float mean_i = c_mean[i];
            for (uint32_t ki = 0; ki < k_block; ++ki) {
                xs_row[ki] = x_row[ki] - mean_i;
            }
        }

        /* 3. P1-b: max(|x_shifted|) + 量化 → x_u（AVX2 加速）
         *    stride=k_block（连续），输入是 x_shifted */
        float x_scale = sbe_quantize_block_avx2(
            x_shifted, k_block, m, k_block, x_u
        );

        /* 4. VNNI matmul（与 sbe_matmul_c 共享 kernel） */
        const int8_t* b_signed_T = w_signed + (size_t)g * n * k_block;
        const int32_t* sum_b = w_sum_b + (size_t)g * n;

        memset(c_int32, 0, (size_t)m * n * sizeof(int32_t));
        hc8_matmul_kernel_vnni(x_u, b_signed_T, sum_b, m, k_block, n, c_int32, n);

        /* 5. P1-a: float 累加主项 y += c_int32 * x_scale * w_scale（AVX2+FMA） */
        float combined_scale = x_scale * w_scales[g];
        sbe_accumulate_float_avx2(c_int32, combined_scale, m, n, y);

        /* 6. 修正项累加: y[i, j] += c_mean[i] * (w_sum_b[j] * w_scale)
         *    这是 outer product (m,1)×(1,n)→(m,n)
         *    P1-a 的 AVX2 加速：对每行 i，用 _mm256_fmadd_ps 累加 8 个 j 元素 */
        float w_scale_g = w_scales[g];
        const int32_t* wsb_row = w_sum_b + (size_t)g * n;
#ifdef __AVX2__
        if (n >= 8) {
            uint32_t n_main = n & ~7u;
            for (uint32_t i = 0; i < m; ++i) {
                float mean_i = c_mean[i] * w_scale_g;
                __m256 v_mean = _mm256_set1_ps(mean_i);
                float* y_row = y + (size_t)i * n;

                uint32_t j = 0;
                for (; j < n_main; j += 8) {
                    /* w_sum_b 是 int32，转 float 后乘 mean */
                    __m256i wsb_vec = _mm256_loadu_si256((const __m256i*)(wsb_row + j));
                    __m256 wsb_f = _mm256_cvtepi32_ps(wsb_vec);
                    __m256 y_vec = _mm256_loadu_ps(y_row + j);
                    y_vec = _mm256_fmadd_ps(v_mean, wsb_f, y_vec);
                    _mm256_storeu_ps(y_row + j, y_vec);
                }
                /* tail 标量 */
                for (; j < n; ++j) {
                    y_row[j] += mean_i * (float)wsb_row[j];
                }
            }
        } else
#endif
        {
            /* 标量回退 */
            for (uint32_t i = 0; i < m; ++i) {
                float mean_i = c_mean[i] * w_scale_g;
                float* y_row = y + (size_t)i * n;
                for (uint32_t j = 0; j < n; ++j) {
                    y_row[j] += mean_i * (float)wsb_row[j];
                }
            }
        }
    }

    free(x_u);
    free(c_int32);
    free(c_mean);
    free(x_shifted);
}

/* ============================================================================
 * SBE per-channel 量化 + matmul（2026-07-23）
 *
 * 与 per-block SBE 数学等价，但 w_scales 从 (groups,) 改为 (groups, n)，
 * 每个 block 内每个输出通道独立 scale，量化精度更高。
 *
 * 复用 per-block 版本的 VNNI kernel（hc8_matmul_kernel_vnni）和
 * 输入量化（sbe_quantize_block_avx2），仅新增：
 *   - sbe_accumulate_float_perchannel_avx2: per-channel scale 向量版 float 累加
 *   - sbe_quantize_weight_blocks_perchannel_c: per-channel 量化
 *   - sbe_matmul_perchannel_c: per-channel matmul
 * ============================================================================ */

/* per-channel float 累加 AVX2+FMA
 *
 * y[i*n + j] += (float)c_int32[i*n + j] * combined_scale[j]  for j in [0, n)
 *
 * 与 sbe_accumulate_float_avx2 的区别：
 *   - combined_scale 是 (n,) 向量而非标量
 *   - AVX2 用 _mm256_loadu_ps 加载 8 个 scale，而非 _mm256_set1_ps 广播
 */
static void sbe_accumulate_float_perchannel_avx2(
    const int32_t* c_int32,
    const float* combined_scale,  /* n, per-channel */
    uint32_t m, uint32_t n,
    float* y  /* m×n, row-major, 累加语义 */
) {
#ifdef __AVX2__
    if (n >= 8) {
        uint32_t n_main = n & ~7u;  /* 8 的倍数部分 */

        for (uint32_t i = 0; i < m; ++i) {
            const int32_t* c_row = c_int32 + (size_t)i * n;
            float* y_row = y + (size_t)i * n;

            uint32_t j = 0;
            for (; j < n_main; j += 8) {
                __m256i c_vec = _mm256_loadu_si256((const __m256i*)(c_row + j));
                __m256 c_f = _mm256_cvtepi32_ps(c_vec);
                __m256 s_vec = _mm256_loadu_ps(combined_scale + j);  /* per-channel scale */
                __m256 y_vec = _mm256_loadu_ps(y_row + j);
                y_vec = _mm256_fmadd_ps(c_f, s_vec, y_vec);
                _mm256_storeu_ps(y_row + j, y_vec);
            }

            /* tail：剩余 1-7 个元素用标量 */
            for (; j < n; ++j) {
                y_row[j] += (float)c_row[j] * combined_scale[j];
            }
        }
        return;
    }
#endif
    /* 标量回退（无 AVX2 或 n < 8） */
    for (uint32_t i = 0; i < m; ++i) {
        const int32_t* c_row = c_int32 + (size_t)i * n;
        float* y_row = y + (size_t)i * n;
        for (uint32_t j = 0; j < n; ++j) {
            y_row[j] += (float)c_row[j] * combined_scale[j];
        }
    }
}

void sbe_quantize_weight_blocks_perchannel_c(
    const float* w,           /* k×n, row-major */
    uint32_t groups, uint32_t k_block, uint32_t k, uint32_t n,
    int8_t* w_signed_flat,    /* 输出: groups * n * k_block */
    int32_t* w_sum_b_flat,    /* 输出: groups * n */
    float* w_scales           /* 输出: groups * n, 布局 (groups, n) */
) {
    if (w == NULL || w_signed_flat == NULL || w_sum_b_flat == NULL || w_scales == NULL) {
        return;
    }
    if (groups == 0 || k_block == 0 || k == 0 || n == 0) {
        return;
    }

    for (uint32_t g = 0; g < groups; ++g) {
        uint32_t start = g * k_block;
        int8_t* b_signed_T = w_signed_flat + (size_t)g * n * k_block;  /* n×k_block */
        int32_t* sum_b = w_sum_b_flat + (size_t)g * n;                  /* n */
        float* scales_g = w_scales + (size_t)g * n;                     /* n, per-channel */

        /* 1. 计算 per-channel scale: scale[j] = max(|w_block[:, j]|) / 127 */
        for (uint32_t j = 0; j < n; ++j) {
            float max_abs = 0.0f;
            for (uint32_t ki = 0; ki < k_block; ++ki) {
                float v = fabsf(w[(size_t)(start + ki) * n + j]);
                if (v > max_abs) max_abs = v;
            }
            float scale = (max_abs == 0.0f) ? 1.0f : max_abs / 127.0f;
            scales_g[j] = scale;
        }

        /* 2. 量化 + 转置 + 有符号转换（每列用独立 scale） */
        for (uint32_t j = 0; j < n; ++j) {
            float inv_scale = 1.0f / scales_g[j];
            int32_t s = 0;
            for (uint32_t ki = 0; ki < k_block; ++ki) {
                float w_val = w[(size_t)(start + ki) * n + j];
                int32_t q = (int32_t)lroundf(w_val * inv_scale);
                if (q < -127) q = -127;
                else if (q > 127) q = 127;
                int8_t b_signed = (int8_t)q;
                b_signed_T[(size_t)j * k_block + ki] = b_signed;
                s += (int32_t)b_signed;
            }
            sum_b[j] = s;
        }
    }
}

void sbe_matmul_perchannel_c(
    const float* x,            /* m×k, row-major */
    const int8_t* w_signed,    /* groups * n * k_block, row-major (n×k_block per block) */
    const int32_t* w_sum_b,    /* groups * n */
    const float* w_scales,     /* groups * n, 布局 (groups, n) */
    uint32_t groups, uint32_t k_block, uint32_t m, uint32_t k, uint32_t n,
    float* y                   /* m×n, output, 调用者负责清零 */
) {
    if (x == NULL || w_signed == NULL || w_sum_b == NULL || w_scales == NULL || y == NULL) {
        return;
    }
    if (groups == 0 || k_block == 0 || m == 0 || k == 0 || n == 0) {
        return;
    }

    /* 临时缓冲区 */
    uint8_t* x_u = (uint8_t*)malloc((size_t)m * k_block);
    int32_t* c_int32 = (int32_t*)malloc((size_t)m * n * sizeof(int32_t));
    float* combined_scale = (float*)malloc((size_t)n * sizeof(float));  /* per-channel */

    if (x_u == NULL || c_int32 == NULL || combined_scale == NULL) {
        free(x_u); free(c_int32); free(combined_scale);
        return;
    }

    for (uint32_t g = 0; g < groups; ++g) {
        uint32_t start = g * k_block;

        /* 1+2. per-block 量化 x_block（x 仍用 per-block scale，与 per-block 版本一致） */
        float x_scale = sbe_quantize_block_avx2(
            x + start, k, m, k_block, x_u
        );

        /* 3. VNNI matmul（int8×int8→int32，与 scale 无关） */
        const int8_t* b_signed_T = w_signed + (size_t)g * n * k_block;
        const int32_t* sum_b = w_sum_b + (size_t)g * n;
        const float* scales_g = w_scales + (size_t)g * n;  /* per-channel */

        memset(c_int32, 0, (size_t)m * n * sizeof(int32_t));
        hc8_matmul_kernel_vnni(x_u, b_signed_T, sum_b, m, k_block, n, c_int32, n);

        /* 4. per-channel float 累加: combined_scale[j] = x_scale * w_scales[g, j] */
        for (uint32_t j = 0; j < n; ++j) {
            combined_scale[j] = x_scale * scales_g[j];
        }
        sbe_accumulate_float_perchannel_avx2(c_int32, combined_scale, m, n, y);
    }

    free(x_u);
    free(c_int32);
    free(combined_scale);
}

/* ============================================================================
 * 正交多视角 matmul（HC 树并行解读，v1.6.0，2026-07-23）
 *
 * 设计见 hc_tree_multiview_unified.md（方案 C 核心交付物）。
 * Python 参考实现：legacy/traditional/stage_2_3_int_path/hc_tree_unified.py
 *
 * 核心思想：分解累加值 C_acc (int64) 为 n_views 个 uint8 位段视角，
 * 权重保持 int8，做 n_views 次独立的 uint8×int8 matmul 后加权累加。
 *
 * 数学：
 *   C_next[i][j] = Σ_v [ Σ_l c_v[i][l] * b[l][j] ] * 2^(8v)
 *   其中 c_v[i][l] = (C_acc[i][l] >> 8v) & 0xFF（uint8, 0-255）
 *
 * 与 Python hc_tree_unified.matmul_multiview 数学等价（max_diff=0）。
 *
 * 约束：4 视角要求 C_acc >= 0（ReLU 后），8 视角才能处理有符号 int64。
 * ============================================================================ */

void hc8_multiview_matmul(
    const int64_t* c_acc,
    const int8_t* b_int8,
    uint32_t m, uint32_t k, uint32_t n,
    int n_views,
    int64_t* out
) {
    /* 清零输出 */
    memset(out, 0, (size_t)m * n * sizeof(int64_t));

    /* 按视角外循环（与 Python 参考实现结构一致） */
    for (int v = 0; v < n_views; v++) {
        int shift = 8 * v;
        int64_t weight = (int64_t)1 << shift;  /* 2^(8v)，用乘法避免负数左移 UB */

        for (uint32_t i = 0; i < m; i++) {
            for (uint32_t j = 0; j < n; j++) {
                int64_t view_acc = 0;
                for (uint32_t l = 0; l < k; l++) {
                    /* 提取 C_acc[i][l] 的第 v 个字节（uint8, 0-255） */
                    int64_t c_val = c_acc[(size_t)i * k + l];
                    uint8_t c_byte = (uint8_t)((c_val >> shift) & 0xFF);
                    /* 权重保持 int8（有符号） */
                    int8_t b_val = b_int8[(size_t)l * n + j];
                    /* uint8 × int8 → int64（有符号累加） */
                    view_acc += (int64_t)c_byte * (int64_t)b_val;
                }
                /* 加权累加：view_acc * 2^(8v) */
                out[(size_t)i * n + j] += view_acc * weight;
            }
        }
    }
}

/* ============================================================================
 * SSSE3 优化: PABSB 绝对值 L1 距离
 * ============================================================================
 * 用 _mm256_sub_epi8 + _mm256_abs_epi8 + _mm256_sad_epu8 替代手动 abs 序列
 * 饱和处理 (-128→127) 天然实现 Level clamp
 */
int32_t hc8_l1_distance_avx2(const uint8_t* a, const uint8_t* b, uint32_t n) {
    if (n == 0 || a == NULL || b == NULL) return 0;
    
    int32_t total = 0;
    uint32_t i = 0;
    
#ifdef __AVX2__
    // AVX2: 每次处理 32 个 int8
    for (; i + 32 <= n; i += 32) {
        __m256i a_vec = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i b_vec = _mm256_loadu_si256((const __m256i*)(b + i));
        // PSUBB + PABSB = |a-b|, 2 条指令替代 4-5 条
        __m256i diff = _mm256_sub_epi8(a_vec, b_vec);
        __m256i abs_diff = _mm256_abs_epi8(diff);
        // _mm256_sad_epu8: 8 个 uint8 abs diff → 累加到 16-bit，4 组
        // 结果: 低 64 位 = sum[0..7], 高 64 位 = sum[8..15]
        // 需要 2 次 sad 覆盖 32 个字节
        __m256i zero = _mm256_setzero_si256();
        __m256i sad = _mm256_sad_epu8(abs_diff, zero);
        // 提取 4 个 16-bit 累加值（sad_epu8 返回 128-bit：低 64-bit=2 个 16-bit，高 64-bit=2 个 16-bit）
        // 索引 0/2：低 64-bit 的两块，索引 1/3：高 64-bit 的两块
        total += (int32_t)_mm256_extract_epi16(sad, 0);
        total += (int32_t)_mm256_extract_epi16(sad, 2);
        total += (int32_t)_mm256_extract_epi16(sad, 4);
        total += (int32_t)_mm256_extract_epi16(sad, 6);
    }
#endif
    
    // 标量尾部
    for (; i < n; ++i) {
        int32_t d = (int32_t)a[i] - (int32_t)b[i];
        total += (d < 0) ? -d : d;
    }
    return total;
}

/* ============================================================================
 * SSSE3 优化: PSIGNB 符号门控更新
 * ============================================================================
 * delta[i] = PSIGNB(delta[i], mask[i])
 *   mask[i] = 0  → delta[i] = 0
 *   mask[i] > 0  → delta[i] 不变
 *   mask[i] < 0  → delta[i] = -delta[i]
 * 1 条指令替代 PCMPEQB + PXOR + PADDB 3+ 条
 */
void hc8_sign_gated_update(uint8_t* delta, const int8_t* mask, uint32_t n) {
    if (n == 0 || delta == NULL || mask == NULL) return;
    
    uint32_t i = 0;
    
#ifdef __AVX2__
    for (; i + 32 <= n; i += 32) {
        __m256i d_vec = _mm256_loadu_si256((const __m256i*)(delta + i));
        __m256i m_vec = _mm256_loadu_si256((const __m256i*)(mask + i));
        __m256i result = _mm256_sign_epi8(d_vec, m_vec);
        _mm256_storeu_si256((__m256i*)(delta + i), result);
    }
#endif
    
    // 标量尾部
    for (; i < n; ++i) {
        int8_t m = mask[i];
        int8_t d = (int8_t)delta[i];
        if (m == 0) {
            delta[i] = 0;
        } else if (m < 0) {
            delta[i] = (uint8_t)(-d);
        }
    }
}
