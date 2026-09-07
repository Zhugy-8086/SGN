/**
 * @file hc16.h
 * @brief SGN HC16 类型声明与运算接口
 * @version 2.0.0
 *
 * HC16: 4层 × 16位 = 64位小数精度，范围 [0, 65536)
 * 每层基数 65536，超度量树深度 5（含 int_part）
 *
 * 依赖：hc.h
 */

#ifndef SGN_HC16_H
#define SGN_HC16_H

#include "hc/hc.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(1)

/** HC16: 4层 × 16位小数 */
typedef struct {
    uint16_t v[4];
} hc16_t;

/** HC16 Lamport 逻辑时钟 */
typedef struct {
    uint32_t level;
    uint16_t sec;
    uint16_t ss[4];
} hc16_lamport_t;

/** HC16 时间类型 */
typedef struct {
    uint16_t sec;
    uint16_t ss[4];
} hc16_time_t;

#pragma pack()

/* 常量 */
extern hc16_t SGN_HC16_ZERO;
extern hc16_t SGN_HC16_MAX;

/* 比较 */
bool hc16_less(const hc16_t* SGN_RESTRICT a, const hc16_t* SGN_RESTRICT b);
bool hc16_equal(const hc16_t* SGN_RESTRICT a, const hc16_t* SGN_RESTRICT b);

/* 加法 */
hc16_t hc16_add_sat(const hc16_t* SGN_RESTRICT a, const hc16_t* SGN_RESTRICT b);
hc16_t hc16_add_wrap(const hc16_t* SGN_RESTRICT a, const hc16_t* SGN_RESTRICT b);

/* 减法 */
hc16_t hc16_sub(const hc16_t* SGN_RESTRICT a, const hc16_t* SGN_RESTRICT b);

/* 软阈值 */
hc16_t hc16_soft_threshold(const hc16_t* SGN_RESTRICT X, const hc16_t* SGN_RESTRICT Lambda);

/* 移位 */
hc16_t hc16_shift_right(const hc16_t* SGN_RESTRICT a, uint8_t shift);

/* 物理值转换 */
double      hc16_to_double(hc16_t h);
hc16_t  hc16_from_double(double v, overflow_t policy);

/* 校验和 */
uint8_t hc16_checksum(const hc16_t* hc);

#ifdef __cplusplus
}
#endif

#endif /* SGN_HC16_H */
