/**
 * @file hc.h
 * @brief SGN 通用 HC 基础设施 - 元数据、错误码、类型枚举、宏
 * @version 2.0.0
 *
 * 所有 HC 变体共享的通用定义。
 * 编译时链接 hc.c 即可获得元数据查表、CRC、GF256 等通用功能。
 *
 * 依赖：无（最底层）
 */

#ifndef SGN_HC_H
#define SGN_HC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* ============================================================================
 * 版本信息
 * ============================================================================ */

#define SGN_ABI_MAJOR 2
#define SGN_ABI_MINOR 0
#define SGN_ABI_PATCH 0

/* ============================================================================
 * HC 变体种类
 * ============================================================================ */

typedef enum {
    SGN_HC_KIND_8  = 0,
    SGN_HC_KIND_16 = 1,
    SGN_HC_KIND_32 = 2,
    SGN_HC_KIND_64 = 3,
    SGN_HC_KIND_COUNT
} hc_kind_t;

/**
 * HC 类型元数据（编译期常量，表驱动）。
 * 消除所有硬编码的层位数、基数、上限。
 */
typedef struct {
    uint32_t    layers;       /**< 分数层数 */
    uint32_t    bits;         /**< 每层位数 */
    uint32_t    elem_bytes;   /**< 每层字节数 (bits/8) */
    double      base;         /**< 基数 (256, 65536, 2^32, 2^64) */
    double      max_phys;     /**< 物理值上限（饱和用） */
    size_t      total_bytes;  /**< 结构体总字节数 */
    const char* name;         /**< 调试名称 */
} hc_meta_t;

/** 获取 HC 变体元数据（只读，编译期初始化） */
const hc_meta_t* hc_meta_get(hc_kind_t kind);

/* ============================================================================
 * 精度等级
 * ============================================================================ */

typedef enum {
    SGN_PREC_ARCHIVE = 0,
    SGN_PREC_FAST    = 2,
    SGN_PREC_STD     = 4,
    SGN_PREC_PREC    = 6,
    SGN_PREC_FUTURE  = 8
} precision_t;

typedef enum {
    SGN_MODE_HC_ONLY    = 0,
    SGN_MODE_LEVEL_ONLY = 1,
    SGN_MODE_COMBINED   = 2
} sgn_mode_t;
/* 注：原 mode_t 与 POSIX/MinGW 系统类型冲突，重命名为 sgn_mode_t */

typedef enum {
    SGN_OVERFLOW_SATURATE = 0,
    SGN_OVERFLOW_WRAP     = 1,
    SGN_OVERFLOW_CARRY    = 2
} overflow_t;

/** 从精度等级推导默认 HC 种类 */
hc_kind_t precision_to_hc_kind(precision_t prec);
uint32_t      precision_layers(precision_t prec);

/* ============================================================================
 * 错误码
 * ============================================================================ */

typedef enum {
    SGN_OK                   = 0,
    SGN_ERR_NOMEM            = 1,
    SGN_ERR_OUT_OF_RANGE     = 2,
    SGN_ERR_INVALID_ARG      = 3,
    SGN_ERR_TMR_FAULT        = 4,
    SGN_ERR_TMR_MULTI_FAULT  = 5,
    SGN_ERR_RS_UNCORRECTABLE = 6,
    SGN_ERR_FILE_CORRUPT     = 7,
    SGN_ERR_VERSION_MISMATCH = 8,
} error_t;

const char* error_string(error_t err);

/* ============================================================================
 * 通用物理值 API（表驱动，修复 HC16 scale bug）
 * ============================================================================ */

/** 通用物理值计算：通过 kind 查表 */
double hc_physical_value(const void* hc_raw, hc_kind_t kind);

/** 通用 double→HC 转换 */
void hc_from_double(double v, overflow_t policy,
                      void* out_hc, hc_kind_t kind);

/* ============================================================================
 * 沙盒联动辅助（声明在各自模块头文件中）
 * ============================================================================ */

/* 已移至 hpdc_sandbox.h 和 hpdc_trie.h */

/* ============================================================================
 * ABI 版本查询
 * ============================================================================ */

uint32_t abi_version(void);

/* ============================================================================
 * 配置宏
 * ============================================================================ */

#ifndef SGN_MAX_TEMPLATES
#   define SGN_MAX_TEMPLATES 256
#endif

#ifndef SGN_MAX_CANDIDATES
#   define SGN_MAX_CANDIDATES 16
#endif

/* ============================================================================
 * restrict 可移植宏（提升编译器优化能力）
 * C99 使用 restrict，C++/MSVC 使用 __restrict，否则退化为空。
 * ============================================================================ */

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#   define SGN_RESTRICT restrict
#elif defined(__cplusplus) && (defined(_MSC_VER) || defined(__GNUC__) || defined(__clang__))
#   define SGN_RESTRICT __restrict
#else
#   define SGN_RESTRICT
#endif

#ifdef __cplusplus
}
#endif

#endif /* SGN_HC_H */
