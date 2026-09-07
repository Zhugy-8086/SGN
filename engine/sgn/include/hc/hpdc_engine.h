/**
 * @file hpdc_engine.h
 * @brief HPDC 引擎算法 ABI - WTA、LRU、形态学
 * @version 1.0.0
 *
 * 提供基于 HC 和 Trie 的核心推理算法：
 *   - Winner-Take-All (WTA) 竞争：前缀剪枝 + 精确 Hamming 匹配
 *   - 软衰减（soft decay）
 *   - LRU 计数与淘汰
 *   - 模板合并（OR/AND）
 *   - 形态学算子（膨胀/腐蚀/开/闭）
 *   - 排序网络（bitonic sort）
 * 依赖：hpdc_core.h, hpdc_trie.h
 */

#ifndef HPDC_ENGINE_H
#define HPDC_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "hc/hpdc_core.h"
#include "hc/hpdc_trie.h"

/* ============================================================================
 * WTA 竞争
 * ============================================================================ */

/**
 * 计算两个 HC8 值在字节层上匹配的层数（Hamming 匹配）。
 * @param a 第一个 HC8
 * @param b 第二个 HC8
 * @return 相同字节的层数（0-6）。
 */
uint8_t match_bits_hamming(const hc8_t* a, const hc8_t* b);

/**
 * Winner-Take-All 竞争。
 * 算法：先对 Trie 前缀剪枝收集候选（第 2 层匹配），
 *       再对候选精确计算 Hamming 相似度，取 Top-K 数量。
 *
 * @param template_trie 模板库的 Trie 根节点。
 * @param template_sigs 模板库的 HC8 签名数组（按 template_id 索引）。
 * @param query_sig     查询 HC8 签名
 * @param K             要返回的 Top-K 数量
 * @param winner_ids    输出：获胜模板 ID 数组（调用方提供长度 >= K）。
 * @param winner_sims   输出：对应的匹配位数数组（0-6）。
 */
void wta_compete(const trie_node_t* template_trie,
                     const hc8_t* template_sigs,
                     const hc8_t* query_sig,
                     uint8_t K,
                     uint16_t* winner_ids,
                     uint8_t* winner_sims);

/* ============================================================================
 * 软衰减
 * ============================================================================ */

/**
 * 对 HC8 计数器数组进行软衰减：counter[i] = max(counter[i] - Lambda, 0)。
 * @param hit_counters 计数器数组
 * @param N            数组长度
 * @param Lambda       衰减阈值。
 */
void soft_decay_all(hc8_t* hit_counters, uint16_t N, const hc8_t* Lambda);

/* ============================================================================
 * LRU 计数与淘汰
 * ============================================================================ */

/**
 * LRU 命中：计数器增加 delta_frac0（仅最低字节增加，饱和）。
 * @param counter      指向 HC8 计数器的指针
 * @param delta_frac0  增加量（0-255）。
 */
void lru_hit(hc8_t* counter, uint8_t delta_frac0);

/**
 * LRU 全局衰减：所有计数器减去 lambda_frac0（仅最低字节）。
 * @param counters     计数器数组
 * @param N            数组长度
 * @param lambda_frac0 衰减量（0-255）。
 */
void lru_decay_all(hc8_t* counters, uint16_t N, uint8_t lambda_frac0);

/**
 * 查找最久未使用的候选（最小计数器值），跳过核心模板。
 * @param counters 计数器数组
 * @param N        数组长度
 * @param is_core  核心标记数组（1=核心，不可淘汰；可为 NULL）。
 * @return 被淘汰的模板索引（0~N-1），若全部为核心则返回 0xFFFF
 */
uint16_t lru_find_evict(const hc8_t* counters, uint16_t N, const uint8_t* is_core);

/**
 * 将计数器值大于阈值的模板提升为核心。
 * @param counters       计数器数组
 * @param is_core        核心标记数组（输出）
 * @param N              数组长度
 * @param threshold_int  整数部分阈值。
 */
void lru_promote_core(const hc8_t* counters, uint8_t* is_core,
                          uint16_t N, uint8_t threshold_int);

/* ============================================================================
 * 模板合并（OR/AND）
 * ============================================================================ */

/**
 * 逐字节 OR 合并：mask_a |= mask_b
 * @param mask_a   目标掩码
 * @param mask_b   源掩码
 * @param D_bytes  字节数。
 */
void merge_or(uint8_t* mask_a, const uint8_t* mask_b, uint8_t D_bytes);

/**
 * 逐字节 AND 合并：mask_a &= mask_b
 */
void merge_and(uint8_t* mask_a, const uint8_t* mask_b, uint8_t D_bytes);

/* ============================================================================
 * 形态学算子（基于 Trie 的邻域操作）
 * ============================================================================ */

/**
 * 膨胀：给定活跃模板集合，扩展 k 步（在 Trie 上找邻居）。
 * @param trie        Trie 根节点。
 * @param active_ids  活跃模板 ID 数组
 * @param N           活跃数量
 * @param k           膨胀步长（字节层数）
 * @param result_ids  输出结果数组
 * @param result_count 输出结果数量
 * @param max_results 输出缓冲区最大容量
 */
void dilate(const trie_node_t* trie, const uint16_t* active_ids,
                uint16_t N, int k, uint16_t* result_ids, uint16_t* result_count,
                uint16_t max_results);

/**
 * 腐蚀：保留那些在 k 步邻域内全部属于集合 S 的模板。
 * @param trie          Trie 根节点。
 * @param S_bitmap      集合位图（按 template_id 索引）。
 * @param k             腐蚀步长
 * @param result_ids    输出结果数组
 * @param result_count  输出结果数量
 * @param max_results   输出缓冲区最大容量
 */
void erode(const trie_node_t* trie, const uint8_t* S_bitmap,
               int k, uint16_t* result_ids, uint16_t* result_count,
               uint16_t max_results);

/**
 * 开运算：腐蚀后膨胀。
 */
void morph_open(const trie_node_t* trie, uint8_t* S_bitmap, int k,
              uint16_t* result_ids, uint16_t* result_count,
              uint16_t max_results);

/**
 * 闭运算：膨胀后腐蚀。
 */
void morph_close(const trie_node_t* trie, uint8_t* S_bitmap, int k,
               uint16_t* result_ids, uint16_t* result_count,
               uint16_t max_results);

/* ============================================================================
 * 排序网络
 * ============================================================================ */

/**
 * Bitonic 排序网络，对 HC8 数组进行排序（升序，按字典序）。
 * @param arr 数组
 * @param N   长度（必须为 2 的幂，否则退化为插入排序）。
 */
void sortnet_bitonic_hc8(hc8_t* arr, uint16_t N);

/* ============================================================================
 * 其他辅助
 * ============================================================================ */

/**
 * 逐层中位数（用于 Quorum 共识）。
 * @param values HC8 数组
 * @param N      数组长度
 * @return 层式中位数 HC8
 */
hc8_t median_layerwise(const hc8_t* values, uint16_t N);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HPDC_ENGINE_H */
