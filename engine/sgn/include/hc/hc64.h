/**
 * @file hc64.h
 * @brief SGN HC64 类型声明与运算接口
 * @version 2.0.0
 *
 * HC64: 2层 × 64位 = 128位小数精度，范围 [0, 2^64)
 * 每层基数 2^64，超度量树深度 3（含 int_part）
 *
 * 注意：double 尾数仅 53 位，无法精确表示 2^64 以下所有整数。
 * from_double 采用拆分为两个 32 位部分的策略以避免精度丢失。
 *
 * 依赖：hc.h
 */

#ifndef SGN_HC64_H
#define SGN_HC64_H

#include "hc/hc.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(1)

/** HC64: 2层 × 64位小数 */
typedef struct {
    uint64_t v[2];
} hc64_t;

#pragma pack()

/* 常量 */
extern hc64_t SGN_HC64_ZERO;
extern hc64_t SGN_HC64_MAX;

/* 比较 */
bool hc64_less(const hc64_t* SGN_RESTRICT a, const hc64_t* SGN_RESTRICT b);
bool hc64_equal(const hc64_t* SGN_RESTRICT a, const hc64_t* SGN_RESTRICT b);

/* 加法 */
hc64_t hc64_add_sat(const hc64_t* SGN_RESTRICT a, const hc64_t* SGN_RESTRICT b);
hc64_t hc64_add_wrap(const hc64_t* SGN_RESTRICT a, const hc64_t* SGN_RESTRICT b);

/* 减法 */
hc64_t hc64_sub(const hc64_t* SGN_RESTRICT a, const hc64_t* SGN_RESTRICT b);

/* 软阈值 */
hc64_t hc64_soft_threshold(const hc64_t* SGN_RESTRICT X, const hc64_t* SGN_RESTRICT Lambda);

/* 移位 */
hc64_t hc64_shift_right(const hc64_t* SGN_RESTRICT a, uint8_t shift);

/* 物理值转换 */
double      hc64_to_double(hc64_t h);
hc64_t  hc64_from_double(double v, overflow_t policy);

/* 校验和 */
uint8_t hc64_checksum(const hc64_t* hc);

#ifdef __cplusplus
}
#endif

#endif /* SGN_HC64_H */
