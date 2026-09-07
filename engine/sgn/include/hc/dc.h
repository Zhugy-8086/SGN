/**
 * @file dc.h
 * @brief SGN DC（十进制定点数）类型与运算
 * @version 2.0.0
 *
 * DC: Discrete/Decimal Coordinate，十进制定点数。
 * 物理值 = index / 10^level
 * 用于空间变量（扫描、坐标、计数），与 HC 体系对偶。
 *
 * 依赖：hc.h (for precision_t, hc8_t)
 */

#ifndef SGN_DC_H
#define SGN_DC_H

#include "hc/hc.h"
#include "hc/hc8.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(1)

/** DC 十进制定点数 */
typedef struct {
    int64_t  index;   /**< 缩放后的整数（有符号） */
    uint32_t level;   /**< 十进制指数（小数位数） */
} dc_t;

#pragma pack()

/* 比较 */
bool dc_less(const dc_t* a, const dc_t* b);
bool dc_equal(const dc_t* a, const dc_t* b);

/* 跨精度转换 */
dc_t dc_to_level(const dc_t* a, uint32_t target_level);

/* 算术 */
dc_t dc_add(const dc_t* a, const dc_t* b);
dc_t dc_sub(const dc_t* a, const dc_t* b);
dc_t dc_mul(const dc_t* a, const dc_t* b);

/* 与 double 互转 */
double   dc_to_double(const dc_t* a);
dc_t dc_from_double(double v, uint32_t level);

/* 与 HC8 互转 */
hc8_t dc_to_hc8(const dc_t* a, precision_t prec);
dc_t  hc8_to_dc(const uint8_t* hc8_v, uint32_t level);

/* JSON 序列化 */
uint32_t dc_serialize(const dc_t* a, char* out_buf, uint32_t buf_size);
int      dc_deserialize(const char* json, dc_t* out);

#ifdef __cplusplus
}
#endif

#endif /* SGN_DC_H */
