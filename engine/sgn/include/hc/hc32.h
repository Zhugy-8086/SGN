/**
 * @file hc32.h
 * @brief SGN HC32 类型声明与运算接口
 * @version 2.0.0
 *
 * HC32: 3层 × 32位 = 96位小数精度，范围 [0, 2^32)
 * 每层基数 2^32，超度量树深度 4（含 int_part）
 *
 * 依赖：hc.h
 */

#ifndef SGN_HC32_H
#define SGN_HC32_H

#include "hc/hc.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(1)

/** HC32: 3层 × 32位小数 */
typedef struct {
    uint32_t v[3];
} hc32_t;

#pragma pack()

/* 常量 */
extern hc32_t SGN_HC32_ZERO;
extern hc32_t SGN_HC32_MAX;

/* 比较 */
bool hc32_less(const hc32_t* SGN_RESTRICT a, const hc32_t* SGN_RESTRICT b);
bool hc32_equal(const hc32_t* SGN_RESTRICT a, const hc32_t* SGN_RESTRICT b);

/* 加法 */
hc32_t hc32_add_sat(const hc32_t* SGN_RESTRICT a, const hc32_t* SGN_RESTRICT b);
hc32_t hc32_add_wrap(const hc32_t* SGN_RESTRICT a, const hc32_t* SGN_RESTRICT b);

/* 减法 */
hc32_t hc32_sub(const hc32_t* SGN_RESTRICT a, const hc32_t* SGN_RESTRICT b);

/* 软阈值 */
hc32_t hc32_soft_threshold(const hc32_t* SGN_RESTRICT X, const hc32_t* SGN_RESTRICT Lambda);

/* 移位 */
hc32_t hc32_shift_right(const hc32_t* SGN_RESTRICT a, uint8_t shift);

/* 物理值转换 */
double      hc32_to_double(hc32_t h);
hc32_t  hc32_from_double(double v, overflow_t policy);

/* 校验和 */
uint8_t hc32_checksum(const hc32_t* hc);

#ifdef __cplusplus
}
#endif

#endif /* SGN_HC32_H */
