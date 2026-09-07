/**
 * @file hpdc_core.h
 * @brief 向后兼容头文件 - 转发到新的模块化头文件
 * @version 2.0.0
 *
 * 新代码应直接包含 sgn/hc8.h、sgn/hc16.h 等。
 * 本文件仅为已有代码提供向后兼容。
 */

#ifndef HPDC_CORE_H
#define HPDC_CORE_H

#include "hc/hc.h"
#include "hc/hc8.h"
#include "hc/hc16.h"
#include "hc/hc32.h"
#include "hc/hc64.h"
#include "hc/dc.h"

/* 向后兼容宏 */
#define HPDC_CORE_ABI_MAJOR SGN_ABI_MAJOR
#define HPDC_CORE_ABI_MINOR SGN_ABI_MINOR
#define HPDC_CORE_ABI_PATCH SGN_ABI_PATCH

#ifndef SGN_HPDC_FRAC_LAYERS
#define SGN_HPDC_FRAC_LAYERS 6
#endif

#endif /* HPDC_CORE_H */
