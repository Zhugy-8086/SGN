/**
 * @file hpdc_sandbox.h
 * @brief HPDC 投影沙盒 ABI - 浮点编程接口
 * @version 1.0.0
 *
 * 投影沙盒：HPDC 的规范级组件，提供 HC 和 float64 的双向转换，
 * 让用户可以写正常的浮点代码（除法、梯度、缩放等），
 * 而库负责处理 HC 的内部表示。
 *
 * 核心思想：存储用 HC，计算用 float。
 *
 * 本模块依赖于 hpdc_core.h，但不依赖 Trie、存储、网络等其他模块。
 */

#ifndef HPDC_SANDBOX_H
#define HPDC_SANDBOX_H

#ifdef __cplusplus
extern "C" {
#endif

#include "hc/hpdc_core.h"

/* ============================================================================
 * 沙盒句柄（不透明类型）
 * ============================================================================ */

/**
 * 投影沙盒句柄。
 * 内部包含精度等级、整数部分位宽、缩放因子 R = 2^int_bits 等状态。
 */
typedef struct sandbox sandbox_t;

/* ============================================================================
 * 创建与销毁
 * ============================================================================ */

/**
 * 创建投影沙盒。
 *
 * @param prec        精度等级（决定 HC 的小数层数）
 * @param int_bits    整数部分位宽，决定 R = 2^int_bits。
 *                    推荐值：HC8 为 8，HC16 为 16。
 * @return 沙盒句柄，失败返回 NULL
 */
sandbox_t* sandbox_create(precision_t prec, uint32_t int_bits);

/** 创建沙盒（自动 int_bits：HC8→8, HC16→16, HC32→32, HC64→64） */
sandbox_t* sandbox_create_auto(hc_kind_t kind);

/**
 * 销毁沙盒，释放资源。
 * @param sb 沙盒句柄
 */
void sandbox_destroy(sandbox_t* sb);

/* ============================================================================
 * 正向投影：HC 到 float64
 * ============================================================================ */

/**
 * 将 HC8 投影到 float64，值域 [0, 1)。
 * 公式：phi = physical_value(H) / R
 *
 * @param sb 沙盒句柄
 * @param h  HC8 值
 * @return phi 在 [0, 1) 范围内的浮点数
 */
double sandbox_project_hc8(sandbox_t* sb, hc8_t h);

/**
 * 将 HC16 投影到 float64。
 */
double sandbox_project_hc16(sandbox_t* sb, hc16_t h);

#if SGN_PC_EXTENSION
/**
 * 将 HC32 投影到 float64（PC 端）。
 */
double sandbox_project_hc32(sandbox_t* sb, hc32_t h);
#endif

/* ============================================================================
 * 反向投影：float64 到 HC
 * ============================================================================ */

/**
 * 将 [0,1) 范围内的 float64 逆投影为 HC8。
 * 公式：H = inverse(phi) = saturate(phi * R)
 *
 * @param sb  沙盒句柄
 * @param phi 浮点数，应在 [0, 1) 范围内
 * @return HC8 值（饱和在 [0, R)）
 */
hc8_t sandbox_inverse_hc8(sandbox_t* sb, double phi);

/**
 * 逆投影为 HC16。
 */
hc16_t sandbox_inverse_hc16(sandbox_t* sb, double phi);

#if SGN_PC_EXTENSION
hc32_t sandbox_inverse_hc32(sandbox_t* sb, double phi);
#endif

/* ============================================================================
 * 批量操作（PC 端，SIMD 友好）
 * ============================================================================ */

#if SGN_PC_EXTENSION

/**
 * 批量投影 HC8 数组到 float64 数组。
 * @param sb   沙盒句柄
 * @param in   输入 HC8 数组
 * @param out  输出 float64 数组（长度 n）
 * @param n    元素个数
 */
void sandbox_project_hc8_batch(sandbox_t* sb,
                                   const hc8_t* in,
                                   double* out,
                                   uint32_t n);

void sandbox_project_hc16_batch(sandbox_t* sb,
                                    const hc16_t* in,
                                    double* out,
                                    uint32_t n);

void sandbox_project_hc32_batch(sandbox_t* sb,
                                    const hc32_t* in,
                                    double* out,
                                    uint32_t n);

/**
 * 批量逆投影 float64 数组到 HC8。
 */
void sandbox_inverse_hc8_batch(sandbox_t* sb,
                                   const double* in,
                                   hc8_t* out,
                                   uint32_t n);

void sandbox_inverse_hc16_batch(sandbox_t* sb,
                                    const double* in,
                                    hc16_t* out,
                                    uint32_t n);

void sandbox_inverse_hc32_batch(sandbox_t* sb,
                                    const double* in,
                                    hc32_t* out,
                                    uint32_t n);

#endif /* SGN_PC_EXTENSION */

/* ============================================================================
 * 算术辅助函数（魔法：在浮点空间完成运算，然后转回 HC）
 * ============================================================================ */

/**
 * 除法：a / b（支持任意除数，不限于奇数）。
 * 实现：pa = project(a), pb = project(b)，若 pb==0 返回 HC_MAX，否则 inverse(pa/pb)。
 *
 * @param sb 沙盒句柄
 * @param a  被除数
 * @param b  除数
 * @return 商（HC8），除零时返回 HC_MAX
 */
hc8_t sandbox_divide_hc8(sandbox_t* sb, hc8_t a, hc8_t b);

hc16_t sandbox_divide_hc16(sandbox_t* sb, hc16_t a, hc16_t b);

#if SGN_PC_EXTENSION
hc32_t sandbox_divide_hc32(sandbox_t* sb, hc32_t a, hc32_t b);
#endif

hc64_t sandbox_divide_hc64(sandbox_t* sb, hc64_t a, hc64_t b);

/**
 * 梯度更新：H_new = H - eta * grad
 * 在浮点空间完成减法，再转回 HC（饱和）。
 *
 * @param sb    沙盒句柄
 * @param h     当前 HC 值
 * @param grad  梯度（浮点）
 * @param eta   学习率（浮点）
 * @return 更新后的 HC 值
 */
hc8_t sandbox_gradient_hc8(sandbox_t* sb,
                                   hc8_t h,
                                   double grad,
                                   double eta);

hc16_t sandbox_gradient_hc16(sandbox_t* sb,
                                     hc16_t h,
                                     double grad,
                                     double eta);

#if SGN_PC_EXTENSION
hc32_t sandbox_gradient_hc32(sandbox_t* sb,
                                     hc32_t h,
                                     double grad,
                                     double eta);
#endif

hc64_t sandbox_gradient_hc64(sandbox_t* sb,
                                     hc64_t h,
                                     double grad,
                                     double eta);

/**
 * 缩放：result = H * factor
 *
 * @param sb     沙盒句柄
 * @param h      当前 HC 值
 * @param factor 缩放因子（浮点）
 * @return 缩放后的 HC 值
 */
hc8_t sandbox_scale_hc8(sandbox_t* sb, hc8_t h, double factor);

hc16_t sandbox_scale_hc16(sandbox_t* sb, hc16_t h, double factor);

#if SGN_PC_EXTENSION
hc32_t sandbox_scale_hc32(sandbox_t* sb, hc32_t h, double factor);
#endif

hc64_t sandbox_scale_hc64(sandbox_t* sb, hc64_t h, double factor);

/**
 * 有符号 HC 加法（通过浮点中间值）。
 */
shc8_t sandbox_signed_add(sandbox_t* sb, shc8_t a, shc8_t b);

/**
 * 有符号 HC 减法。
 */
shc8_t sandbox_signed_sub(sandbox_t* sb, shc8_t a, shc8_t b);

/* ============================================================================
 * 可扩展性：投影方案切换（为未来扩展预留）
 * ============================================================================ */

/**
 * 设置当前沙盒使用的投影方案名称。
 * 默认方案为 "default"（物理值/R）。
 * 官方插件可通过注册表增加新方案，如 "log_domain"、"float128"。
 *
 * @param sb          沙盒句柄
 * @param scheme_name 方案名称（不能为 NULL）
 * @return 0 成功，非 0 失败（方案不存在）
 */
int sandbox_set_scheme(sandbox_t* sb, const char* scheme_name);

/**
 * 获取当前投影方案名称。
 * @param sb 沙盒句柄
 * @return 方案名称字符串，默认 "default"
 */
const char* sandbox_get_scheme(const sandbox_t* sb);

/* ============================================================================
 * HC 种类辅助（从 hc.h 移入，避免基础层依赖上层）
 * ============================================================================ */

uint32_t sandbox_default_int_bits(hc_kind_t kind);
bool     sandbox_int_bits_valid(hc_kind_t kind, uint32_t int_bits);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HPDC_SANDBOX_H */