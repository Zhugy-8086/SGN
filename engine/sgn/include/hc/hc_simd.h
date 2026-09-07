/**
 * @file hc_simd.h
 * @brief SGN SIMD 批量运算接口
 * @version 2.0.0
 *
 * 提供 HC16/HC64 的 SIMD 加速批量操作。
 * 编译时需定义 SGN_USE_SIMD 并链接 hc_simd.c。
 * 未定义 SGN_USE_SIMD 时，所有函数自动退化为标量循环。
 *
 * 支持平台：
 *   - x86/x64: SSE2 (默认), AVX2
 *   - ARM: NEON (待实现)
 *
 * 依赖：hc16.h, hc64.h
 */

#ifndef SGN_HC_SIMD_H
#define SGN_HC_SIMD_H

#include "hc/hc8.h"
#include "hc/hc16.h"
#include "hc/hc32.h"
#include "hc/hc64.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CPU ISA 运行时检测与回退链（P0/P1）
 *
 * P0：运行时 CPUID 检测（SSE4.1/4.2/AVX2），补齐仅有编译期宏、无运行时检测的缺口。
 * P1：回退链 AVX2 → SSE4.1(128-bit) → 标量。dispatch 用 sgn_cpu_effective_isa(desired)
 *     同时受「编译期支持的上限」（SGN_COMPILED_ISA_MAX）与「运行时检测」双重约束，
 *     从最高档向下回退到两者都支持的一档。
 * 非 x86 平台（ARM/GPU）无 CPUID，检测函数恒返回标量档，天然走标量（项目硬约束）。
 * ============================================================================ */

typedef enum {
    SGN_CPU_ISA_UNKNOWN  = 0,   /* 未检测 / 未知 */
    SGN_CPU_ISA_SCALAR   = 1,   /* 无 SIMD 可用，退化为标量循环 */
    SGN_CPU_ISA_SSE2     = 2,   /* x86-64 基线，所有 batch 函数均可用 */
    SGN_CPU_ISA_SSE4_1   = 3,
    SGN_CPU_ISA_SSE4_2   = 4,
    SGN_CPU_ISA_AVX2     = 5,
    SGN_CPU_ISA_AVX512   = 6
} sgn_cpu_isa_t;

/* 编译期支持的最高 ISA 等级（由编译宏推导，见 hc_simd.c 实现） */
extern const sgn_cpu_isa_t SGN_COMPILED_ISA_MAX;

/* P0：运行时 CPUID 检测（结果缓存，返回 1=支持 / 0=不支持） */
int  sgn_cpu_supports_sse4_1(void);
int  sgn_cpu_supports_sse4_2(void);
int  sgn_cpu_supports_avx2(void);
/* 返回当前 CPU 运行时最高可用 ISA 等级（受编译期上限约束） */
sgn_cpu_isa_t sgn_cpu_detect_isa(void);

/* P1：回退链查询。
 * 给定 desired 目标等级，返回「编译期支持」且「运行时支持」的最高可用等级（<= desired）。
 * 例如 sgn_cpu_effective_isa(SGN_CPU_ISA_AVX2) 在无 AVX2 平台回退到 SSE4/SSE2/标量。 */
sgn_cpu_isa_t sgn_cpu_effective_isa(sgn_cpu_isa_t desired);

/* 返回 ISA 等级的可读名字（用于日志/测试）。 */
const char* sgn_cpu_isa_name(sgn_cpu_isa_t level);

/* ============================================================================
 * HC8 SIMD 批量操作
 * ============================================================================ */

/**
 * 批量 HC8 饱和加法：out[i] = saturate(a[i] + b[i])
 */
void hc8_add_sat_batch(const hc8_t* a, const hc8_t* b,
                                  hc8_t* out, uint32_t n);

/**
 * 批量 HC8 比较：out[i] = (a[i] < b[i]) ? 1 : 0
 */
void hc8_less_batch(const hc8_t* a, const hc8_t* b,
                         uint8_t* out, uint32_t n);

/**
 * 批量 HC8 软阈值：out[i] = max(a[i] - Lambda, 0)
 */
void hc8_soft_threshold_batch(const hc8_t* a, const hc8_t* Lambda,
                                     hc8_t* out, uint32_t n);

/* ============================================================================
 * HC16 SIMD 批量操作
 * ============================================================================ */

/**
 * 批量 HC16 饱和加法：out[i] = saturate(a[i] + b[i])
 * 利用 SIMD 并行检测进位掩码，逐层传播。
 */
void hc16_add_sat_batch(const hc16_t* a, const hc16_t* b,
                                   hc16_t* out, uint32_t n);

/**
 * 批量 HC16 比较：out[i] = (a[i] < b[i]) ? 1 : 0
 * 字典序比较，SIMD 可高效处理 16 位无符号比较。
 */
void hc16_less_batch(const hc16_t* a, const hc16_t* b,
                          uint8_t* out, uint32_t n);

/**
 * 批量 HC16 标量乘法（定点缩放）：out[i] = a[i] * factor_q16 >> 16
 * factor_q16 = factor * 65536，避免浮点。
 */
void hc16_scale_batch(const hc16_t* a, uint32_t factor_q16,
                           hc16_t* out, uint32_t n);

/**
 * 批量 HC16 软阈值：out[i] = max(a[i] - Lambda, 0)
 */
void hc16_soft_threshold_batch(const hc16_t* a, const hc16_t* Lambda,
                                    hc16_t* out, uint32_t n);

/* ============================================================================
 * HC64 SIMD 批量操作
 * ============================================================================ */

/**
 * 批量 HC64 饱和加法：out[i] = saturate(a[i] + b[i])
 * HC64 每个元素 16 字节，恰好占满一个 SSE 寄存器。
 */
void hc64_add_sat_batch(const hc64_t* a, const hc64_t* b,
                                   hc64_t* out, uint32_t n);

/**
 * 批量 HC64 比较：out[i] = (a[i] < b[i]) ? 1 : 0
 */
void hc64_less_batch(const hc64_t* a, const hc64_t* b,
                         uint8_t* out, uint32_t n);

/* ============================================================================
 * HC32 SIMD 批量操作
 * ============================================================================ */

/**
 * 批量 HC32 饱和加法：out[i] = saturate(a[i] + b[i])
 */
void hc32_add_sat_batch(const hc32_t* a, const hc32_t* b,
                                   hc32_t* out, uint32_t n);

/**
 * 批量 HC32 比较：out[i] = (a[i] < b[i]) ? 1 : 0
 */
void hc32_less_batch(const hc32_t* a, const hc32_t* b,
                         uint8_t* out, uint32_t n);

/**
 * 批量 HC32 标量乘法（定点缩放）：out[i] = a[i] * factor_q32 >> 32
 * factor_q32 = factor * 2^32，避免浮点（Q32 定点）。
 */
void hc32_scale_batch(const hc32_t* a, uint32_t factor_q32,
                           hc32_t* out, uint32_t n);

/**
 * 批量 HC32 软阈值：out[i] = max(a[i] - Lambda, 0)
 */
void hc32_soft_threshold_batch(const hc32_t* a, const hc32_t* Lambda,
                                    hc32_t* out, uint32_t n);

/* ============================================================================
 * MSint 精度升级（S6/S7）— 仅符号展宽（补码视图）
 *
 * MSint 中同一块内存按不同位宽读取；"精度升级"是从窄位宽展宽到宽位宽，
 * 供 MSint 升级路径使用。逐元素独立（无跨层依赖）。
 * 语义：out[i] = (wider)a[i]，a[i] 视为有符号补码，符号扩展。
 * ============================================================================ */

/**
 * S6：int8 数组 → int16 数组（符号展宽）
 * out[i] = (int16_t)a[i]；SIMD 用 PMOVSXBW（SSE4.1）/ SSE2 unpack 符号扩展。
 */
void msint_widen_i8_i16_batch(const int8_t* a, int16_t* out, uint32_t n);

/**
 * S7：int16 数组 → int32 数组（符号展宽）
 * out[i] = (int32_t)a[i]；SIMD 用 PMOVSXWD（SSE4.1）/ SSE2 unpack 符号扩展。
 */
void msint_widen_i16_i32_batch(const int16_t* a, int32_t* out, uint32_t n);

#ifdef __cplusplus
}
#endif

#endif /* SGN_HC_SIMD_H */
