/**
 * @file sgn.h
 * @brief SGN 技术栈统一入口（PC 开发/测试用）
 * @version 2.0.0
 *
 * 包含此文件即获得全部 HC 类型、DC、沙盒、Trie、引擎、存储、网络功能。
 * 嵌入式项目应按需包含单个头文件（如 hc8.h、hc16.h）以最小化链接体积。
 */

#ifndef SGN_H
#define SGN_H

#include "hc/hc.h"
#include "hc/hc8.h"
#include "hc/hc16.h"
#include "hc/hc32.h"
#include "hc/hc64.h"
#include "hc/dc.h"

#ifdef SGN_USE_SIMD
#include "hc/hc_simd.h"
#endif

#ifdef SGN_PC_EXTENSION
#include "hc/hpdc_sandbox.h"
#include "hc/hpdc_trie.h"
#include "hc/hpdc_engine.h"
#include "hc/hpdc_storage.h"
#include "hc/hpdc_network.h"
#include "hc/hpdc_plugin.h"
#endif

#ifdef __cplusplus
#include "hc/hpdc_cpp.hpp"
#endif

#endif /* SGN_H */
