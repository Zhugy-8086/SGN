// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 zhugy-8086
//
// pool_allocator.h - SGN 通用内存池分配器（轻量、零第三方依赖）
//
// 设计目标（见 engine_infrastructure_plan_2026_08_14.md P1）：
//   1. size-class 分桶：请求字节数就近归入固定 size-class，减少碎片
//   2. thread_local 无锁：每线程独立池，无跨线程竞争
//   3. free-list 复用：释放的内存块进入对应桶的 free-list，下次分配直接复用
//   4. 全局可用分配器：可经 sgn::set_allocator() 接入 Storage
//   5. 外部后端兼容：分配/释放走 sgn::AllocFn/DeallocFn 签名（与 allocator.h 一致），
//      未来可替换底层（如接 oneDNN/ONNX Runtime 的池）
//
// 内存模型：
//   - 池不持有物理内存，只复用调用方 release 的块。
//   - 池为空时 allocate 走底层默认分配器（不预分配，避免驻留浪费）。
//   - 池化释放：块回收到 thread_local free-list，直到 clear_pool() 或析构。
//
// 用法：
//   sgn::set_allocator(&sgn::pool_alloc, &sgn::pool_dealloc);   // 启用池化
//   ... 训练 ...
//   sgn::set_allocator(nullptr, nullptr);                        // 恢复默认（可选）
//   sgn::clear_pool();                                           // 清空池（重置/测试）

#pragma once

#include "allocator.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace sgn {

// ============================================================================
// size-class 定义
// ============================================================================
// 桶按"块字节数"划分。请求向上取整到最近的 size-class。
// 覆盖 float 数组常见规模：16B(~4 float) ~ 16MB。
constexpr size_t kPoolMinBytes = 16;
constexpr size_t kPoolMaxBytes = 16 * 1024 * 1024;  // 16MB 上限，超过不走池

namespace detail {

// 分桶函数：把 nbytes 映射到 size-class（2 的幂间隔，向上取整到 2^k）。
// 返回该桶对应的"块大小"（即桶内所有块的实际字节数，>= nbytes）。
// 用 std::bit_ceil 单一实现（替代原循环 + 不可达 break 分支——审计 P-3），
// 并与 pool_index 共享同一换算逻辑，防止两份实现漂移（审计 P-4）。
// 前置条件：nbytes <= kPoolMaxBytes（调用方 pool_alloc/pool_dealloc 已保证）。
inline size_t pool_round_up(size_t nbytes) {
    if (nbytes <= kPoolMinBytes) return kPoolMinBytes;
    size_t block = std::bit_ceil(nbytes);  // nbytes > 16 时无溢出风险
    return block > kPoolMaxBytes ? kPoolMaxBytes : block;
}

// 每个 size-class 的 free-list（存放空闲块的裸指针）。
// thread_local：每线程独立，无锁。
// RAII 持有容器：析构（线程退出）时释放池内全部缓存块，避免"容器只释放指针数组
// 本身、指针指向的内存块永久泄露"的问题（安全审计 2026-08-15 问题 1）。
struct PoolFreeLists {
    std::vector<std::vector<void*>> lists;  // 下标 = size-class 编号
    ~PoolFreeLists() {
        for (auto& free_list : lists) {
            for (void* p : free_list) sgn_std_dealloc(p, 0);
        }
    }
};

inline PoolFreeLists& pool_free_lists() {
    static thread_local PoolFreeLists pool;
    return pool;
}

// 获取桶编号（按块大小）。前置条件：block_size 是 pool_round_up 的输出
// （2 的幂，>= kPoolMinBytes）。idx = log2(block / kPoolMinBytes)。
// 与 pool_round_up 构成互逆换算，二者必须保持一致（审计 P-4）。
inline size_t pool_index(size_t block_size) {
    return std::countr_zero(block_size) - std::countr_zero(kPoolMinBytes);
}

}  // namespace detail

// ============================================================================
// 池化分配/释放
// ============================================================================
// 分配：先查对应桶 free-list，有则复用；无则走默认分配器。
inline void* pool_alloc(size_t nbytes, size_t /*alignment*/) {
    if (nbytes == 0) return nullptr;
    if (nbytes > kPoolMaxBytes) {
        return sgn_std_alloc(nbytes, kDefaultAlignment);
    }
    size_t block = detail::pool_round_up(nbytes);
    if (block > kPoolMaxBytes) {
        return sgn_std_alloc(nbytes, kDefaultAlignment);  // 溢出保护
    }
    size_t idx = detail::pool_index(block);
    auto& lists = detail::pool_free_lists().lists;
    if (idx >= lists.size()) {
        lists.resize(idx + 1);
    }
    auto& free_list = lists[idx];
    if (!free_list.empty()) {
        void* p = free_list.back();
        free_list.pop_back();
        return p;
    }
    // 池空，走底层分配整块（传 kDefaultAlignment 保持分配契约——审计 A-1）
    return sgn_std_alloc(block, kDefaultAlignment);
}

// 释放：块回收到对应桶 free-list（若在池化范围内）。
inline void pool_dealloc(void* ptr, size_t nbytes) {
    if (ptr == nullptr) return;
    // 对称于 pool_alloc 的入口判断：超大块在分配时直接走底层（未入池），
    // 释放时也必须直接走底层，否则会被 pool_round_up 截断进 16MB 桶，
    // 污染桶内块大小一致性（安全审计 2026-08-15 问题 2）。
    if (nbytes > kPoolMaxBytes) {
        sgn_std_dealloc(ptr, nbytes);
        return;
    }
    size_t block = detail::pool_round_up(nbytes);
    size_t idx = detail::pool_index(block);
    auto& lists = detail::pool_free_lists().lists;
    if (idx >= lists.size()) {
        lists.resize(idx + 1);
    }
    lists[idx].push_back(ptr);
}

// 清空池（释放所有缓存块，用于测试/重置）。
// 注 1：仅清空【当前线程】的池（thread_local 语义）——多线程训练时若需彻底
//       释放，必须在每条 worker 线程内分别调用；主线程调用不会影响其他线程
//       的缓存块（安全审计 2026-08-16 P-1）。
// 注 2：池内块恒由底层默认分配器（sgn_std_alloc）分配，故用 sgn_std_dealloc
//       释放是配对的（安全审计 2026-08-15 问题 3）。
inline void clear_pool() {
    auto& lists = detail::pool_free_lists().lists;
    for (auto& free_list : lists) {
        for (void* p : free_list) {
            sgn_std_dealloc(p, 0);
        }
        free_list.clear();
    }
    lists.clear();
}

}  // namespace sgn
