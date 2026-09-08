// packed_backend.h - MSInt PackedBackend C++ 实现
//
// 对标 Python engine.ms_int.backends.PackedBackend
//
// 设计：
//   - 所有槽位打包进一个 uint64_t（总 bits ≤ 64）
//   - 第一个槽位在高位（与 Python 实现一致）
//   - 支持有符号/无符号槽位
//   - get_all() 标量实现，get_all_simd() AVX2 优化（等宽 ≥8 bit 场景）
//
// 数学依据：
//   - 位操作打包/解包：位移 + 掩码
//   - 符号位处理：补码表示
//   - AVX2 向量化：等宽槽位批量提取 + 字节反转
#pragma once

#include <cstdint>
#include <vector>
#include <string>

namespace sgn {

// ============================================================
// SlotSpec - 槽位规格（预计算 offset）
// ============================================================

struct SlotSpec {
    int bits;       // 位宽（8/12/16/20/24 等）
    bool is_signed; // 是否有符号
    int offset;     // 预计算的位偏移（第一个槽位在高位）

    SlotSpec() : bits(0), is_signed(false), offset(0) {}
    SlotSpec(int b, bool s, int off) : bits(b), is_signed(s), offset(off) {}
};

// ============================================================
// PackedBackend - 打包后端 C++ 实现
// ============================================================

class PackedBackend {
public:
    // 从槽位规格列表构造（offset 已预计算）
    explicit PackedBackend(const std::vector<SlotSpec>& slots, uint64_t packed = 0);

    // 从 bits 列表构造（自动计算 offset，第一个槽位在高位）
    // signed_flags 为空时默认全部无符号
    static PackedBackend from_bits(const std::vector<int>& bits_list,
                                   const std::vector<bool>& signed_flags = {},
                                   uint64_t packed = 0);

    // ---- 基本接口（对标 Python PackedBackend）----

    // 读取第 index 个槽位（含符号位处理）
    int64_t get(int index) const;

    // 写入第 index 个槽位
    void set(int index, int64_t value);

    // 批量读取所有槽位（标量实现）
    std::vector<int64_t> get_all() const;

    // 批量读取所有槽位（AVX2 优化，等宽 ≥8 bit 场景）
    // 非等宽或 bits < 8 时自动回退到标量
    std::vector<int64_t> get_all_simd() const;

    // ---- 批量 API（一次调用处理整个数组，消除 pybind11 逐元素开销）----

    // 批量读取多个 packed 值的所有槽位
    // signed_flags：逐槽位符号标志，空 = 全无符号（与 from_bits 约定一致；
    // A1-1 修复 2026-09-08：backward_int16 等 signed 场景此前无法走批量 API）
    // 返回扁平化 2D 数组 [n_values * n_slots]，行优先
    static std::vector<int64_t> batch_get_all(
        const std::vector<int>& bits_list,
        const std::vector<uint64_t>& packed_values,
        const std::vector<bool>& signed_flags = {}
    );

    // ---- 属性 ----

    uint64_t packed_value() const { return packed_; }
    int slot_count() const { return static_cast<int>(slots_.size()); }
    int total_bits() const { return total_bits_; }
    const std::vector<SlotSpec>& slots() const { return slots_; }

    // 序列化（A1-3 修复：含 slots 数组，独立恢复完整状态）
    std::string serialize() const;  // 返回 JSON 字符串

    // 从 serialize() 输出重建（手写固定格式解析，字段顺序与 serialize 输出一致）
    static PackedBackend deserialize(const std::string& json);

private:
    uint64_t packed_;
    std::vector<SlotSpec> slots_;
    int total_bits_;

    // 检查是否可以用 AVX2 优化（所有槽位等宽且 bits ∈ {8, 16, 32}）
    bool can_use_simd_() const;

    // AVX2 批量读取 8-bit 等宽无符号槽位
    std::vector<int64_t> get_all_simd_8bit_unsigned_() const;

    // AVX2 批量读取 16-bit 等宽无符号槽位
    std::vector<int64_t> get_all_simd_16bit_unsigned_() const;
};

} // namespace sgn
