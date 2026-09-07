/**
 * @file hpdc_trie.h
 * @brief HPDC Trie 索引 ABI - 超度量树索引
 * @version 1.0.0
 *
 * 基于 HC 字节层的 256 叉超度量树（Trie），用于模板索引和快速前缀匹配。
 * 每个节点对应 HC 的一个字节层，叶子节点存储 template_id。
 *
 * 特性：
 *   - 静态内存池，零 malloc（嵌入式友好）
 *   - 前缀剪枝：给定查询 HC，快速收集前 k 层匹配的候选模板
 *   - 插入、懒删除、遍历
 *
 * 依赖：hpdc_core.h
 */

#ifndef HPDC_TRIE_H
#define HPDC_TRIE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "hc/hpdc_core.h"
#include <stdint.h>
#include <stdbool.h>

/* ============================================================================
 * Trie 节点与内存池
 * ============================================================================ */

#pragma pack(push, 1)

/**
 * Trie 节点。
 * 每个节点对应 HC 的一个字节层（共 6 层）。
 */
typedef struct sgn_trie_node {
    uint8_t             byte;           /**< 分支键值（0-255） */
    uint16_t            template_id;    /**< 如果是叶子节点，存储模板 ID；否则为 0xFFFF */
    uint16_t            child_count;    /**< 子节点个数 */
    struct sgn_trie_node* children;     /**< 指向子节点数组的指针（内存池内） */
} trie_node_t;

/**
 * 静态内存池，零动态分配。
 * 最大支持 2048 个节点（约 1024 个模板 + 内部节点）。
 */
typedef struct {
    trie_node_t nodes[2048];    /**< 节点存储 */
    uint16_t        next_free;      /**< 下一个空闲节点索引 */
} trie_pool_t;

#pragma pack(pop)

/* 节点类型判断 */
#define SGN_TRIE_IS_LEAF(node)     ((node)->template_id != 0xFFFF)
#define SGN_TRIE_IS_INTERNAL(node) ((node)->template_id == 0xFFFF)

/* ============================================================================
 * 内存池操作
 * ============================================================================ */

/**
 * 初始化静态内存池。
 * @param pool 内存池指针
 */
void trie_pool_init(trie_pool_t* pool);

/**
 * 从内存池分配一个新节点。
 * @param pool 内存池
 * @return 节点指针，失败返回 NULL
 */
trie_node_t* trie_pool_alloc(trie_pool_t* pool);

/* ============================================================================
 * 查找操作
 * ============================================================================ */

/**
 * 在节点的子节点中查找指定字节的分支。
 * @param node   父节点
 * @param target 目标字节
 * @return 子节点指针，未找到返回 NULL
 */
trie_node_t* trie_find_child(const trie_node_t* node, uint8_t target);

/**
 * 沿 HC 路径查找节点（最多 6 层）。
 * @param root  根节点
 * @param hc    HC8 值
 * @param depth 要匹配的层数（1-6），0 返回根节点
 * @return 路径终点节点，若某层缺失则返回能匹配到的最深节点
 */
trie_node_t* trie_find_path(trie_node_t* root, const hc8_t* hc, int depth);

/* ============================================================================
 * 插入操作
 * ============================================================================ */

/**
 * 在父节点下插入一个新的子节点（字节 b）。
 * 若已存在则返回已有节点。
 * @param parent 父节点
 * @param b      分支字节
 * @param pool   内存池
 * @return 子节点指针，失败返回 NULL
 */
trie_node_t* trie_insert_child(trie_node_t* parent, uint8_t b, trie_pool_t* pool);

/**
 * 插入一个完整的 HC8 签名作为模板。
 * 沿 6 层路径插入节点（若缺失则创建），最后一层设为叶子节点，存储 template_id。
 * @param root   根节点
 * @param sig    HC8 签名
 * @param tid    模板 ID（0-65534，0xFFFF 为无效）
 * @param pool   内存池
 * @return true 成功，false 失败（内存不足或路径冲突）
 */
bool trie_insert_template(trie_node_t* root, const hc8_t* sig,
                              uint16_t tid, trie_pool_t* pool);

/* ============================================================================
 * 删除操作
 * ============================================================================ */

/**
 * 懒删除：将叶子节点的 template_id 设置为 0xFFFF。
 * 不会物理释放节点或剪枝。
 * @param leaf 叶子节点
 */
void trie_lazy_delete(trie_node_t* leaf);

/* ============================================================================
 * 收集与遍历
 * ============================================================================ */

/**
 * 收集节点下所有叶子的 template_id。
 * @param node      起始节点
 * @param out_ids   输出数组（调用方提供足够空间）
 * @param out_count 输出实际数量
 * @param max_count 输出数组最大容量
 */
void trie_collect_leaves(const trie_node_t* node,
                             uint16_t* out_ids,
                             uint16_t* out_count,
                             uint16_t max_count);

/**
 * 收集与查询 HC 前缀匹配的前 k 层候选模板。
 * 沿查询 HC 的前 k 层下降，收集该子树下所有叶子。
 * @param root      根节点
 * @param query     查询 HC8
 * @param k         前缀匹配层数（1-6）
 * @param cand_ids  输出候选模板 ID 数组
 * @param cand_count 输出候选数量
 */
void trie_collect_candidates(const trie_node_t* root,
                                 const hc8_t* query,
                                 int k,
                                 uint16_t* cand_ids,
                                 uint16_t* cand_count);

/* ============================================================================
 * HC 种类辅助（从 hc.h 移入，避免基础层依赖上层）
 * ============================================================================ */

uint32_t trie_depth_for_hc(hc_kind_t kind);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HPDC_TRIE_H */
