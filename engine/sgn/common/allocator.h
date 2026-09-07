// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 zhugy-8086
//
// allocator.h - SGN 统一内存分配器抽象（轻量、零第三方依赖）
//
// 设计目标（见 engine_infrastructure_plan_2026_08_14.md P1）：
//   1. 定义可插拔分配器接口，让 Storage 从 `new float[]` 解耦
//   2. 默认 StdAllocator 保持现有行为（new/delete），切换不改变数值
//   3. 全局可设置（set_allocator），未来可接：
//      - PoolAllocator（本计划 P1 第 2 步）
//      - 外部后端的内存管理（如 oneDNN / ONNX Runtime / TensorRT 的 allocator）
//      - 内存统计/对齐/线程本地分配
//   4. header-only + 内联，函数指针形式（与 logger 的 sink 一致的插拔哲学）
//
// 接口说明：
//   using AllocFn = void*(*)(size_t nbytes, size_t alignment);
//   using DeallocFn = void(*)(void* ptr, size_t nbytes);
//   按字节数 + 对齐分配/释放。默认 64 字节对齐（AVX-512 对齐上限）。
//
// 用法（Storage 侧）：
//   auto deleter = [](float* p) { deallocate(p, nbytes); };
//   std::shared_ptr<float[]>(ptr, deleter)
//   其中 allocate/deallocate 经全局 allocator 分发。

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>

// 仅支持 64 位平台：n * sizeof(float) 的溢出保护依赖 size_t >= 8 字节
// （安全审计 2026-08-16 A-2）
static_assert(sizeof(size_t) >= 8, "SGN allocator assumes 64-bit size_t");

namespace sgn {

// ============================================================================
// 分配器函数签名
// ============================================================================
// 按字节数 + 对齐分配。约定：0 字节返回 nullptr；分配失败抛 std::bad_alloc
// （默认实现走 aligned new；池化实现空池时同样走 aligned new）。
// 释放时回传原始 nbytes（供池化释放/统计）。
using AllocFn = void* (*)(size_t nbytes, size_t alignment);
using DeallocFn = void (*)(void* ptr, size_t nbytes);

// 默认对齐：AVX-512 需要 64 字节，向上兼容未来 SIMD 宽度。
constexpr size_t kDefaultAlignment = 64;

// ============================================================================
// 默认分配器（C++17 aligned new/delete，统一按 kDefaultAlignment 对齐）
// ============================================================================
// 注意：allocate/deallocate 必须配对使用同一对齐。为简单与安全，默认实现
// 固定用 kDefaultAlignment（对齐参数保留仅供未来池化/统计使用，当前忽略）。
inline void* sgn_std_alloc(size_t nbytes, size_t /*alignment*/) {
    if (nbytes == 0) return nullptr;
    return ::operator new(nbytes, std::align_val_t{kDefaultAlignment});
}

inline void sgn_std_dealloc(void* ptr, size_t /*nbytes*/) {
    if (ptr == nullptr) return;
    // 必须与 sgn_std_alloc 的 aligned new 配对（同一对齐值）
    ::operator delete(ptr, std::align_val_t{kDefaultAlignment});
}

// ============================================================================
// 全局分配器（可插拔）
// ============================================================================
namespace detail {
// 全局分配器指针用 atomic 存储：set_allocator（写）与分配/释放（读）可能
// 在不同线程并发发生，普通函数指针的并发读写是数据竞争 UB
// （安全审计 2026-08-16 P-5；与 logger.h 的 atomic sink 保持一致）。
// relaxed 序足够：仅保证指针本身的原子可见性，无跨变量顺序依赖。
inline std::atomic<AllocFn>& alloc_ref() {
    static std::atomic<AllocFn> fn{&sgn_std_alloc};
    return fn;
}
inline std::atomic<DeallocFn>& dealloc_ref() {
    static std::atomic<DeallocFn> fn{&sgn_std_dealloc};
    return fn;
}
}  // namespace detail

// 设置全局分配器。任一传 nullptr 则恢复默认。
// 注意：应在训练开始前（单线程阶段）调用；运行期切换不保证已分配块的行为，
// 除非 Storage 采用"分配时捕获 deallocator"的配对策略（见 tensor.cpp）。
inline void set_allocator(AllocFn alloc, DeallocFn dealloc) {
    detail::alloc_ref().store(alloc ? alloc : &sgn_std_alloc,
                              std::memory_order_relaxed);
    detail::dealloc_ref().store(dealloc ? dealloc : &sgn_std_dealloc,
                                std::memory_order_relaxed);
}

// ============================================================================
// 便捷包装（供 Storage 等调用）
// ============================================================================
// 分配 n 个 float（按默认对齐），返回裸指针；失败抛 std::bad_alloc。
inline float* sgn_allocate_floats(size_t n) {
    // 溢出保护：n*sizeof(float) 溢出 size_t 会导致实际分配远小于 n，
    // 后续 fill 越界写（安全审计 2026-08-15 问题 4）。
    if (n > std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::bad_alloc();
    }
    return static_cast<float*>(
        detail::alloc_ref().load(std::memory_order_relaxed)(
            n * sizeof(float), kDefaultAlignment));
}

// 释放（需回传原始字节数）。
// 注意：此函数在释放时做全局查找——若分配与释放之间全局分配器被切换，
// 会导致"用错分配器释放"（审计 P-2/P-6）。需要严格配对的调用方
// （如 Storage）应在分配时捕获 deallocator（见 tensor.cpp）。
inline void sgn_deallocate_floats(float* ptr, size_t n) {
    detail::dealloc_ref().load(std::memory_order_relaxed)(ptr, n * sizeof(float));
}

}  // namespace sgn
