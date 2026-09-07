/**
 * @file sgn_macros.h
 * @brief SGN 便利宏 - 简化常见操作
 * @version 1.0.0
 *
 * 可选组件，不包含也不影响功能。
 * 嵌入式项目如需最小体积，可不包含此文件。
 *
 * 用法：
 *   #include "hc/hc8.h"
 *   #include "hc/sgn_macros.h"
 *
 *   hc8_t a = HC8_FROM_DOUBLE(3.14);
 *   hc8_t b = HC8_FROM_DOUBLE(2.0);
 *   hc8_t c = HC8_ADD(a, b);
 *   double v = HC8_TO_DOUBLE(c);
 */

#ifndef SGN_MACROS_H
#define SGN_MACROS_H

#include "hc/hc8.h"
#include "hc/hc16.h"
#include "hc/hc32.h"
#include "hc/hc64.h"

/* ============================================================================
 * 类型转换宏
 * ============================================================================ */

/** 创建 HC8 并从 double 转换（饱和） */
#define HC8_FROM_DOUBLE(v) hc8_from_double((v), SGN_OVERFLOW_SATURATE)

/** 创建 HC16 并从 double 转换（饱和） */
#define HC16_FROM_DOUBLE(v) hc16_from_double((v), SGN_OVERFLOW_SATURATE)

/** 创建 HC32 并从 double 转换（饱和） */
#define HC32_FROM_DOUBLE(v) hc32_from_double((v), SGN_OVERFLOW_SATURATE)

/** 创建 HC64 并从 double 转换（饱和） */
#define HC64_FROM_DOUBLE(v) hc64_from_double((v), SGN_OVERFLOW_SATURATE)

/* ============================================================================
 * 输出转换宏
 * ============================================================================ */

/** HC8 转 double */
#define HC8_TO_DOUBLE(h) hc8_to_double(h)

/** HC16 转 double */
#define HC16_TO_DOUBLE(h) hc16_to_double(h)

/** HC32 转 double */
#define HC32_TO_DOUBLE(h) hc32_to_double(h)

/** HC64 转 double */
#define HC64_TO_DOUBLE(h) hc64_to_double(h)

/* ============================================================================
 * 算术宏
 * ============================================================================ */

/** HC8 饱和加法 */
#define HC8_ADD(a, b) hc8_add_sat(&(a), &(b))

/** HC16 饱和加法 */
#define HC16_ADD(a, b) hc16_add_sat(&(a), &(b))

/** HC32 饱和加法 */
#define HC32_ADD(a, b) hc32_add_sat(&(a), &(b))

/** HC64 饱和加法 */
#define HC64_ADD(a, b) hc64_add_sat(&(a), &(b))

/** HC8 减法 */
#define HC8_SUB(a, b) hc8_sub(&(a), &(b))

/** HC16 减法 */
#define HC16_SUB(a, b) hc16_sub(&(a), &(b))

/** HC32 减法 */
#define HC32_SUB(a, b) hc32_sub(&(a), &(b))

/** HC64 减法 */
#define HC64_SUB(a, b) hc64_sub(&(a), &(b))

/* ============================================================================
 * 比较宏
 * ============================================================================ */

/** HC8 比较：a < b */
#define HC8_LESS(a, b) hc8_less(&(a), &(b))

/** HC16 比较：a < b */
#define HC16_LESS(a, b) hc16_less(&(a), &(b))

/** HC32 比较：a < b */
#define HC32_LESS(a, b) hc32_less(&(a), &(b))

/** HC64 比较：a < b */
#define HC64_LESS(a, b) hc64_less(&(a), &(b))

/** HC8 相等 */
#define HC8_EQUAL(a, b) hc8_equal(&(a), &(b))

/** HC16 相等 */
#define HC16_EQUAL(a, b) hc16_equal(&(a), &(b))

/** HC32 相等 */
#define HC32_EQUAL(a, b) hc32_equal(&(a), &(b))

/** HC64 相等 */
#define HC64_EQUAL(a, b) hc64_equal(&(a), &(b))

/* ============================================================================
 * 形态学宏
 * ============================================================================ */

/** HC8 软阈值：max(X - Lambda, 0) */
#define HC8_SOFT_THRESH(X, L) hc8_soft_threshold(&(X), &(L))

/** HC16 软阈值 */
#define HC16_SOFT_THRESH(X, L) hc16_soft_threshold(&(X), &(L))

/** HC32 软阈值 */
#define HC32_SOFT_THRESH(X, L) hc32_soft_threshold(&(X), &(L))

/** HC64 软阈值 */
#define HC64_SOFT_THRESH(X, L) hc64_soft_threshold(&(X), &(L))

/* ============================================================================
 * 有符号 HC8 宏
 * ============================================================================ */

/** 有符号 HC8 加法 */
#define SHC8_ADD(a, b) shc8_add(&(a), &(b))

/** 有符号 HC8 减法 */
#define SHC8_SUB(a, b) shc8_sub(&(a), &(b))

/** 有符号 HC8 比较：a < b */
#define SHC8_LESS(a, b) shc8_less(&(a), &(b))

/** 有符号 HC8 软阈值 */
#define SHC8_SOFT_THRESH(X, L) shc8_soft_threshold(&(X), &(L))

/** 无符号 HC8 转有符号 HC8 */
#define HC8_TO_SHC8(x) hc8_to_shc8(x)

/** 有符号 HC8 转无符号 HC8（失败返回零） */
#define SHC8_TO_HC8(x, ok) shc8_to_hc8(x, ok)

#endif /* SGN_MACROS_H */
