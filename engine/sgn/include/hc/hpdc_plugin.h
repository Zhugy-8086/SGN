/**
 * @file hpdc_plugin.h
 * @brief HPDC 插件架构 ABI - 动态加载、能力掩码、生命周期管�? * @version 1.0.0
 *
 * 提供插件系统基础设施�? *   - 插件类型（官方内核级 / 第三方外围级 / 混合级）
 *   - 能力掩码�?4位，严格隔离�? *   - 动态加�?卸载�?so / .dll�? *   - 生命周期回调（init, shutdown, register�? *   - 规范晋升路径（候�?�?冻结�? *
 * 本模块仅依赖 hpdc_core.h，不依赖沙盒、Trie 等具体组件�? * 与内核扩展（如注册新投影方案）的接口�?hpdc_normative.h 中定义�? *
 * 设计原则�? *   - 第三方插件被强制限制�?EXTENSION 能力，不可修改核心语�? *   - 官方插件（NORMATIVE/HYBRID）可持有内核级能�? *   - 晋升路径：ACTIVE �?CANDIDATE �?FROZEN（冻结为核心 ABI 部分�? */

#ifndef HPDC_PLUGIN_H
#define HPDC_PLUGIN_H

#ifdef __cplusplus
extern "C" {
#endif

#include "hc/hpdc_core.h"
#include <stdint.h>
#include <stdbool.h>

/* ============================================================================
 * 插件类型
 * ============================================================================ */

/**
 * 插件分类：决定了可持有的能力范围�? */
typedef enum {
    SGN_PLUGIN_TYPE_NORMATIVE = 0,  /**< 内核级：可修改核心行为（沙盒转换器、HC运算符等�?*/
    SGN_PLUGIN_TYPE_EXTENSION = 1,  /**< 外围级：沙箱隔离，仅扩展存储/网络/监控/编解码器 */
    SGN_PLUGIN_TYPE_HYBRID    = 2,  /**< 混合级：部分内核 + 部分外围（官方全栈扩展） */
} plugin_type_t;

/* ============================================================================
 * 能力掩码�?4位）
 * ============================================================================ */

/* 内核级能力（仅官方插件可持有�?*/
#define SGN_PLUGIN_CAP_SANDBOX_CONVERTER  (1ULL << 0)  /**< 注册/替换投影方案 */
#define SGN_PLUGIN_CAP_HC_OPERATOR        (1ULL << 1)  /**< 新增或覆�?HC 算术运算�?*/
#define SGN_PLUGIN_CAP_PRECISION_GRADE    (1ULL << 2)  /**< 激活预留精度等级（�?FUTURE�?*/
#define SGN_PLUGIN_CAP_ENGINE_ALGORITHM   (1ULL << 3)  /**< 替换引擎策略（WTA/LRU 等） */

/* 外围能力（第三方和官方均可持有） */
#define SGN_PLUGIN_CAP_STORAGE_DRIVER     (1ULL << 4)  /**< 存储后端扩展（SQLite/Redis/S3�?*/
#define SGN_PLUGIN_CAP_NETWORK_DRIVER     (1ULL << 5)  /**< 网络协议扩展（MQTT/WebSocket�?*/
#define SGN_PLUGIN_CAP_MONITOR_HOOK       (1ULL << 6)  /**< 监控与遥测钩�?*/
#define SGN_PLUGIN_CAP_ENCODER            (1ULL << 7)  /**< 自定义编解码器（LZ4/自定义加密） */

/* 掩码定义 */
#define SGN_PLUGIN_CAP_NORMATIVE_MASK \
    (SGN_PLUGIN_CAP_SANDBOX_CONVERTER | SGN_PLUGIN_CAP_HC_OPERATOR | \
     SGN_PLUGIN_CAP_PRECISION_GRADE   | SGN_PLUGIN_CAP_ENGINE_ALGORITHM)

#define SGN_PLUGIN_CAP_EXTENSION_MASK \
    (SGN_PLUGIN_CAP_STORAGE_DRIVER | SGN_PLUGIN_CAP_NETWORK_DRIVER | \
     SGN_PLUGIN_CAP_MONITOR_HOOK   | SGN_PLUGIN_CAP_ENCODER)

/* ============================================================================
 * 插件事件类型（监控钩子）
 * ============================================================================ */

typedef enum {
    SGN_EVENT_HC_OP       = 0,  /**< HC 运算（加减比较软阈值等�?*/
    SGN_EVENT_WTA_COMPETE = 1,  /**< WTA 竞争触发 */
    SGN_EVENT_STORAGE_IO  = 2,  /**< 存储读写事件 */
    SGN_EVENT_PLUGIN_LOAD = 3,  /**< 插件加载/卸载生命周期事件 */
} monitor_event_t;

/* ============================================================================
 * 不透明类型前向声明
 * ============================================================================ */

typedef struct plugin_handle plugin_handle_t;   /**< 插件句柄 */
typedef struct plugin_ctx plugin_ctx_t;         /**< 插件上下文（sandbox, user_data, error�?*/
typedef struct normative_registry normative_registry_t; /**< 官方注册表（定义�?hpdc_normative.h�?*/
typedef struct extension_registry extension_registry_t; /**< 外围注册表（定义�?hpdc_plugin.h 内部或单独文件） */

/* ============================================================================
 * 插件描述符与生命周期回调
 * ============================================================================ */

/**
 * 插件描述符：每个插件必须提供此结构体（通过导出符号 plugin_get_desc）�? * 注意：register_normative �?register_extension 根据插件类型择一或同时实现�? */
typedef struct {
    uint32_t api_version;           /**< 插件 API 版本（当�?1�?*/
    plugin_type_t type;         /**< 插件类型 */
    const char* name;               /**< 唯一标识符（�?"sgn.log_domain"�?*/
    const char* version;            /**< 插件语义化版�?*/
    const char* author;             /**< 作�?组织 */
    const char* abi_requirement;    /**< ABI 版本范围（如 ">=1.0.0 <2.0.0"�?*/
    uint64_t capabilities;          /**< 能力掩码（按位或�?*/

    /** 初始化回调：插件加载后、注册前调用一�?*/
    int (*init)(plugin_ctx_t* ctx);

    /** 清理回调：卸载前调用 */
    int (*shutdown)(plugin_ctx_t* ctx);

    /** 官方插件专用：注册内核钩子（init 中调用） */
    int (*register_normative)(plugin_ctx_t* ctx, normative_registry_t* reg);

    /** 第三�?混合插件专用：注册外围扩展（init 中调用） */
    int (*register_extension)(plugin_ctx_t* ctx, extension_registry_t* reg);
} plugin_desc_t;

/* ============================================================================
 * 插件上下文结构体（不透明，内部实现可见）
 * ============================================================================ */

/**
 * 插件上下文（每个插件实例持有）。具体定义在实现文件中，这里仅声明�? * 用户无需直接访问，通过回调函数参数传入�? */
struct plugin_ctx {
    void*       sandbox;        /**< 当前活动沙盒（如需要） */
    void*       user_data;      /**< 插件私有数据 */
    error_t last_error;     /**< 最近错误码 */
    char        error_msg[256]; /**< 错误描述 */
};

/* ============================================================================
 * 插件管理�?API
 * ============================================================================ */

/**
 * 动态加载插件（.so / .dll）�? * @param path 动态库路径
 * @return 插件句柄，失败返�?NULL（可通过 error_string 获取详情�? */
plugin_handle_t* plugin_load(const char* path);

/**
 * 卸载插件�? * @param h 插件句柄
 * @return 错误码（若插件已冻结为规范，返回 SGN_ERR_INVALID_ARG�? */
error_t plugin_unload(plugin_handle_t* h);

/**
 * 静态注册内置插件（用于编译进核心的官方插件）�? * @param desc 插件描述符（必须保持有效，ABI 会复制内部内容）
 * @return 错误�? */
error_t plugin_register_static(const plugin_desc_t* desc);

/**
 * 查询插件是否已加载（ACTIVE/CANDIDATE/FROZEN 状态）�? * @param name 插件名称
 * @return true 已加�? */
bool plugin_is_loaded(const char* name);

/**
 * 获取当前已加载（ACTIVE/CANDIDATE/FROZEN）的插件数量�? * @return 数量
 */
uint32_t plugin_count(void);

/**
 * 按索引获取已加载插件的名称�? * @param idx 索引�? �?plugin_count()-1�? * @return 名称字符串，索引无效返回 NULL
 */
const char* plugin_get_name(uint32_t idx);

/**
 * 获取指定插件的能力掩码�? * @param name 插件名称
 * @return 能力掩码，若插件不存在返�?0
 */
uint64_t plugin_get_capabilities(const char* name);

/**
 * 晋升路径：将官方插件标记为“规范候选”�? * 只有 NORMATIVE �?HYBRID 类型且状态为 ACTIVE 的插件可晋升�? * @param name 插件名称
 * @return 错误码（成功 SGN_OK�? */
error_t plugin_propose_normative(const char* name);

/**
 * 查询插件是否为规范候选（CANDIDATE 状态）�? * @param name 插件名称
 * @return true 是候�? */
bool plugin_is_normative_candidate(const char* name);

/**
 * 将规范候选冻结为核心 ABI（内部调用，�?ABI 维护者在 Minor 版本发布时执行）�? * 冻结后插件不可卸载，成为核心永久组成部分�? * @param name 插件名称
 * @return 错误�? */
error_t plugin_freeze_normative(const char* name);

/* ============================================================================
 * 外围扩展注册表（Extension Registry�? 第三方插件使�? * ============================================================================ */

/**
 * 外围注册表结构体（由 ABI 核心填充，插件在 register_extension 中调用）�? * 注意：实际函数指针在实现中由核心提供，此仅为声明�? */
struct extension_registry {
    /** 注册存储后端驱动 */
    int (*register_storage_driver)(
        const char* name,
        int (*write)(const char* path, const void* hdr, const void* records, void* ud),
        int (*read)(const char* path, void* hdr, void* records, uint32_t max, void* ud),
        void* user_data
    );

    /** 注册网络协议驱动 */
    int (*register_network_driver)(
        const char* name,
        int (*pack)(uint8_t type, const uint8_t* payload, uint8_t len, uint8_t* frame, void* ud),
        int (*parse)(const uint8_t* frame, void* out, void* ud),
        void* user_data
    );

    /** 注册监控钩子 */
    int (*register_monitor_hook)(
        monitor_event_t event,
        void (*callback)(monitor_event_t event, const void* data, uint32_t len, void* ud),
        void* user_data
    );

    /** 注册自定义编解码�?*/
    int (*register_encoder)(
        const char* name,
        uint32_t (*encode)(const hc8_t* in, uint16_t N, void* out, void* ud),
        uint32_t (*decode)(const void* in, uint32_t len, hc8_t* out, void* ud),
        void* user_data
    );
};

/* ============================================================================
 * 错误码扩展（插件相关�? * ============================================================================ */

/* 插件特定的错误码（在 error_t 基础上扩展，实际可复�?SGN_ERR_INVALID_ARG 等） */
/* 无需新增枚举，使用通用错误码即可�?*/

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HPDC_PLUGIN_H */