/**
 * @file sgn_abi_dynamic.h
 * @brief 动态化改进核心 - 替换硬编码为表驱动
 * @version 1.1.0-draft
 *
 * 改进点：
 *   1. 引入 hc_meta_t 元数据表，消除 HC8/16/32 的层数、基数硬编码
 *   2. 修复 HC16 物理值 scale bug（起始 scale 应为 1.0）
 *   3. 精度等级 enum 和实际层数的映射函数
 *   4. HC32 最小实现占位（可编译链接）
 *   5. Trie 操作接受动态深度参数
 *   6. 沙盒 int_bits 和 HC 类型自动联动
 */

#ifndef SGN_ABI_DYNAMIC_H
#define SGN_ABI_DYNAMIC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "hc/hc.h"

/* ============================================================================
 * DYNAMIC HC METADATA SYSTEM（新增）
 * ============================================================================ */

/* 注意：hc_kind_t 和 hc_meta_t 已在 hc.h 中定义 */

/** 获取 HC 变体元数据（只读，编译期初始化） */
const hc_meta_t* hc_meta_get(hc_kind_t kind);

/** 从精度等级推导默认 HC 种类和层数 */
hc_kind_t precision_to_hc_kind(precision_t prec);
uint32_t      precision_layers(precision_t prec);

/* ============================================================================
 * GENERIC PHYSICAL VALUE API（新增）
 * ============================================================================ */

/** 通用物理值计算：通过 kind 查表，自动处理 8/16/32/64 */
double hc_physical_value(const void* hc_raw, hc_kind_t kind);

/** 通用 double 到 HC 转换：自动根据 kind 决定层数、基数、饱和上限 */
void hc_from_double(double v, overflow_t policy,
                      void* out_hc, hc_kind_t kind);

/* ============================================================================
 * HC32 MINIMAL IMPLEMENTATION（新增声明）
 * ============================================================================ */

double hc32_to_double(hc32_t h);
hc32_t hc32_from_double(double v, overflow_t policy);
bool hc32_less(const hc32_t* a, const hc32_t* b);
bool hc32_equal(const hc32_t* a, const hc32_t* b);

/* ============================================================================
 * SANDBOX INT_BITS AUTO-COUPLING（新增）
 * ============================================================================ */

/** 根据 HC kind 推荐 int_bits，避免 HC8 用 int_bits=16 的浪费 */
uint32_t sandbox_default_int_bits(hc_kind_t kind);

/** 沙盒创建时自动校验 int_bits 是否适配 HC kind */
bool sandbox_int_bits_valid(hc_kind_t kind, uint32_t int_bits);

/* ============================================================================
 * TRIE DYNAMIC DEPTH（新增）
 * ============================================================================ */

/** 根据 HC kind 返回 Trie 遍历深度（HC8=6, HC16=4, HC32=3, HC64=2） */
uint32_t trie_depth_for_hc(hc_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif /* SGN_ABI_DYNAMIC_H */
