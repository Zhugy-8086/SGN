/**
 * @file hpdc_normative.h
 * @brief HPDC 官方规范注册�?ABI - 内核级扩展接�? * @version 1.0.0
 *
 * 本文件定义官方插件（NORMATIVE/HYBRID）修改核心行为的接口�? * 第三方插件禁止包含此头文件，也不得调用其中的注册函数�? *
 * 包含以下能力�? *   - 注册新的沙盒投影方案（如对数域、float128�? *   - 注册自定�?HC 运算符（覆盖或新增）
 *   - 激活预留精度等级（�?SGN_PREC_FUTURE�? *   - 替换引擎策略（WTA 竞争算法、LRU 淘汰策略�? *
 * 依赖：hpdc_core.h, hpdc_sandbox.h, hpdc_trie.h, hpdc_engine.h
 *
 * 注意：此文件仅在需要扩展内核语义时使用，不属于基础 ABI�? */

#ifndef HPDC_NORMATIVE_H
#define HPDC_NORMATIVE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "hc/hpdc_core.h"
#include "hc/hpdc_sandbox.h"
#include "hc/hpdc_trie.h"
#include "hc/hpdc_engine.h"

/* ============================================================================
 * 规范注册表结构体（由 ABI 核心填充，插件在 register_normative 中调用）
 * ============================================================================ */

/**
 * 规范注册表�? * 包含一系列函数指针，插件通过调用这些函数向核心注入新行为�? * 注册表本身由核心管理，插件只能调用，不能修改结构体内容�? */
typedef struct normative_registry {
    /* ----- 沙盒投影方案注册 ----- */
    
    /**
     * 注册新的投影方案（如 "log_domain", "float128"）�?     * @param scheme_name       方案唯一标识�?     * @param project           正向投影函数：HC8 �?double（phi in [0,1)�?     * @param inverse           反向投影函数：double �?HC8（饱和）
     * @param supported_prec    支持的精度掩码（bit0=HC8, bit1=HC16, bit2=HC32�?     * @return 0 成功，非 0 失败
     */
    int (*register_sandbox_converter)(
        const char* scheme_name,
        double (*project)(const hc8_t* h, double R, void* user_data),
        hc8_t (*inverse)(double phi, double R, void* user_data),
        uint32_t supported_prec,
        void* user_data
    );
    
    /* ----- HC 运算符注册（覆盖或新增）----- */
    
    /**
     * 注册自定�?HC8 加法运算符�?     * @param op_name   操作名（�?"log_add"�?     * @param add_func  实现函数（饱和或自定义）
     * @param flags     保留�?�?     * @return 0 成功
     */
    int (*register_hc8_add_op)(
        const char* op_name,
        hc8_t (*add_func)(const hc8_t* a, const hc8_t* b),
        uint32_t flags
    );
    
    /**
     * 注册自定�?HC8 减法运算符�?     */
    int (*register_hc8_sub_op)(
        const char* op_name,
        hc8_t (*sub_func)(const hc8_t* a, const hc8_t* b),
        uint32_t flags
    );
    
    /**
     * 注册自定�?HC8 软阈值变体�?     */
    int (*register_hc8_soft_threshold_op)(
        const char* op_name,
        hc8_t (*threshold_func)(const hc8_t* X, const hc8_t* Lambda),
        uint32_t flags
    );
    
    /* ----- 精度等级激活（预留等级�?---- */
    
    /**
     * 激活预留精度等级（�?SGN_PREC_FUTURE）�?     * @param grade         精度枚举值（必须是预留的等级�?     * @param frac_layers   小数层数�? 或更多）
     * @param bits_per_layer 每层位数（固�?8�?     * @param description   描述字符�?     * @return 0 成功
     */
    int (*activate_precision_grade)(
        precision_t grade,
        uint32_t frac_layers,
        uint32_t bits_per_layer,
        const char* description
    );
    
    /* ----- 引擎策略替换 ----- */
    
    /**
     * 注册自定�?WTA 竞争策略�?     * @param name      策略名称
     * @param compete   竞争实现函数
     * @param flags     保留
     * @return 0 成功
     */
    int (*register_wta_strategy)(
        const char* name,
        void (*compete)(const trie_node_t* trie,
                        const hc8_t* query,
                        uint8_t K,
                        uint16_t* winner_ids,
                        uint8_t* winner_sims),
        uint32_t flags
    );
    
    /**
     * 注册自定�?LRU 淘汰策略�?     * @param name       策略名称
     * @param find_evict 淘汰查找函数
     * @param flags      保留
     * @return 0 成功
     */
    int (*register_lru_policy)(
        const char* name,
        uint16_t (*find_evict)(const hc8_t* counters, uint16_t N, const uint8_t* is_core),
        uint32_t flags
    );
    
    /**
     * 注册自定�?Trie 收集候选策略（如基于汉明距离的剪枝）�?     * @param name            策略名称
     * @param collect         收集函数
     * @param flags           保留
     * @return 0 成功
     */
    int (*register_candidate_collector)(
        const char* name,
        void (*collect)(const trie_node_t* root,
                        const hc8_t* query,
                        int k,
                        uint16_t* cand_ids,
                        uint16_t* cand_count),
        uint32_t flags
    );
    
    /* ----- 未来扩展预留字段 ----- */
    void* reserved[8];
} normative_registry_t;

/* ============================================================================
 * 辅助宏（用于插件实现�? * ============================================================================ */

/**
 * 导出插件描述符的简化宏（供插件实现使用）�? * 使用方式：在插件 .c 文件中：
 *   SGN_PLUGIN_EXPORT_DESC(my_plugin_desc)
 */
#define SGN_PLUGIN_EXPORT_DESC(desc_var) \
    extern "C" plugin_desc_t plugin_get_desc(void) { return desc_var; }

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HPDC_NORMATIVE_H */