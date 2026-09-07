/**
 * @file hc8.h
 * @brief SGN HC8 类型声明与运算接口
 * @version 2.0.0
 *
 * HC8: 6层 × 8位 = 48位小数精度，范围 [0, 256)
 * 每层基数 256，超度量树深度 7（含 int_part）
 *
 * 依赖：hc.h
 */

#ifndef SGN_HC8_H
#define SGN_HC8_H

#include "hc/hc.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(1)
/** HC8: 6层 × 8位小数（紧缩存储，嵌入式友好） */
typedef struct {
    uint8_t v[6];
} hc8_t;
#pragma pack()

/** 有符号 HC8（自然对齐） */
typedef struct {
    uint8_t   sign;      /**< 0=正/零，1=负 */
    uint8_t   int_part;
    hc8_t     frac;
} shc8_t;

/** 合体模式：level + HC8（自然对齐） */
typedef struct {
    uint32_t  level;
    uint8_t   int_part;
    hc8_t     frac;
} leveled_hpdc8_t;

/* 常量 */
extern hc8_t SGN_HC8_ZERO;
extern hc8_t SGN_HC8_MAX;

/* 比较 */
bool hc8_less(const hc8_t* SGN_RESTRICT a, const hc8_t* SGN_RESTRICT b);
bool hc8_equal(const hc8_t* SGN_RESTRICT a, const hc8_t* SGN_RESTRICT b);
bool shc8_less(const shc8_t* SGN_RESTRICT a, const shc8_t* SGN_RESTRICT b);

/* 加法 */
hc8_t hc8_add_sat(const hc8_t* SGN_RESTRICT a, const hc8_t* SGN_RESTRICT b);
hc8_t hc8_add_wrap(const hc8_t* SGN_RESTRICT a, const hc8_t* SGN_RESTRICT b);
leveled_hpdc8_t leveled_add(const leveled_hpdc8_t* SGN_RESTRICT a,
                                     const leveled_hpdc8_t* SGN_RESTRICT b);
shc8_t shc8_add(const shc8_t* SGN_RESTRICT a, const shc8_t* SGN_RESTRICT b);

/* 减法 */
hc8_t hc8_sub(const hc8_t* SGN_RESTRICT a, const hc8_t* SGN_RESTRICT b);
shc8_t shc8_sub(const shc8_t* SGN_RESTRICT a, const shc8_t* SGN_RESTRICT b);

/* 软阈值 */
hc8_t hc8_soft_threshold(const hc8_t* SGN_RESTRICT X, const hc8_t* SGN_RESTRICT Lambda);
shc8_t shc8_soft_threshold(const shc8_t* SGN_RESTRICT X, const shc8_t* SGN_RESTRICT Lambda);

/* 移位与掩码 */
hc8_t hc8_shift_right(const hc8_t* SGN_RESTRICT a, uint8_t shift);
hc8_t hc8_mask_threshold(const hc8_t* SGN_RESTRICT X, uint8_t mask);

/* 有符号/无符号互转 */
shc8_t hc8_to_shc8(hc8_t x);
hc8_t  shc8_to_hc8(shc8_t x, bool* ok);

/* 物理值转换 */
double     hc8_to_double(hc8_t h);
hc8_t  hc8_from_double(double v, overflow_t policy);

/* 校验和 */
uint8_t hc8_checksum(const hc8_t* hc);

#ifdef __cplusplus
}
#endif

#endif /* SGN_HC8_H */
