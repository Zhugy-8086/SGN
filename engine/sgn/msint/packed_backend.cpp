// packed_backend.cpp - PackedBackend C++ 实现
//
// 对标 Python engine.ms_int.backends.PackedBackend
//
// 核心算法：
//   - get(index): raw = (packed >> offset) & mask; 符号位扩展
//   - set(index): packed &= ~(mask << offset); packed |= (value & mask) << offset
//   - get_all_simd(): AVX2 字节提取 + shuffle 反转（等宽 8/16 bit 无符号）
#include "packed_backend.h"

#include "mkern/simd/simd_api.h"

#include <stdexcept>
#include <sstream>
#include <cstring>

namespace sgn {

// ============================================================
// 构造
// ============================================================

PackedBackend::PackedBackend(const std::vector<SlotSpec>& slots, uint64_t packed)
    : packed_(packed), slots_(slots), total_bits_(0) {
    for (const auto& s : slots_) {
        total_bits_ += s.bits;
    }
    if (total_bits_ <= 0) {
        throw std::invalid_argument("槽位总位数必须 > 0");
    }
    if (total_bits_ > 64) {
        throw std::invalid_argument("槽位总位数不能超过 64（当前 " +
                                   std::to_string(total_bits_) + "）");
    }
}

PackedBackend PackedBackend::from_bits(const std::vector<int>& bits_list,
                                       const std::vector<bool>& signed_flags,
                                       uint64_t packed) {
    if (bits_list.empty()) {
        throw std::invalid_argument("bits_list 不能为空");
    }
    // 第一个槽位在高位：offset 从 total_bits 递减
    int total = 0;
    for (int b : bits_list) {
        if (b <= 0) {
            throw std::invalid_argument("每个槽位 bits 必须 > 0");
        }
        total += b;
    }
    if (total > 64) {
        throw std::invalid_argument("槽位总位数不能超过 64（当前 " +
                                   std::to_string(total) + "）");
    }

    std::vector<SlotSpec> slots;
    slots.reserve(bits_list.size());
    int offset = total;
    for (size_t i = 0; i < bits_list.size(); ++i) {
        offset -= bits_list[i];
        bool is_signed = (!signed_flags.empty() && i < signed_flags.size())
                         ? signed_flags[i] : false;
        slots.emplace_back(bits_list[i], is_signed, offset);
    }
    return PackedBackend(slots, packed);
}

// ============================================================
// get / set（标量，对标 Python PackedBackend.get/set）
// ============================================================

int64_t PackedBackend::get(int index) const {
    if (index < 0 || index >= static_cast<int>(slots_.size())) {
        throw std::out_of_range("槽位索引 " + std::to_string(index) +
                                " 超出范围 [0, " +
                                std::to_string(slots_.size()) + ")");
    }
    const SlotSpec& slot = slots_[index];
#ifdef __BMI__
    // BEXTR: 单指令提取位字段 (start=slot.offset, len=slot.bits)
    uint64_t raw = _bextr_u64(packed_, slot.offset, slot.bits);
#else
    // 回退：手动 shift + mask
    uint64_t mask = (slot.bits >= 64) ? ~0ULL : ((1ULL << slot.bits) - 1);
    uint64_t raw = (packed_ >> slot.offset) & mask;
#endif

    // 符号位处理：补码表示（M1 修复 2026-09-07：bits=64 时 1ULL<<64 为 UB，
    // 此时 raw 本身即补码 int64 位型，直接转回）
    if (slot.is_signed && (raw & (1ULL << (slot.bits - 1)))) {
        return slot.bits >= 64
                   ? static_cast<int64_t>(raw)
                   : static_cast<int64_t>(raw)
                         - static_cast<int64_t>(1ULL << slot.bits);
    }
    return static_cast<int64_t>(raw);
}

void PackedBackend::set(int index, int64_t value) {
    if (index < 0 || index >= static_cast<int>(slots_.size())) {
        throw std::out_of_range("槽位索引 " + std::to_string(index) +
                                " 超出范围 [0, " +
                                std::to_string(slots_.size()) + ")");
    }
    const SlotSpec& slot = slots_[index];
#ifdef __BMI2__
    // BZHI: 单指令生成 mask（高位清零），正确处理 bits=64 边界
    uint64_t mask = _bzhi_u64(~0ULL, slot.bits);
#else
    uint64_t mask = (slot.bits >= 64) ? ~0ULL : ((1ULL << slot.bits) - 1);
#endif

    uint64_t shifted_mask = mask << slot.offset;
#ifdef __BMI__
    // ANDN: 单指令清除位字段 (~shifted_mask & packed_)
    packed_ = _andn_u64(shifted_mask, packed_);
#else
    // 回退：手动取反再与
    packed_ &= ~shifted_mask;
#endif
    // 写入新值（负数自动用补码，int64_t → uint64_t 隐式转换）
    packed_ |= (static_cast<uint64_t>(value) & mask) << slot.offset;
}

// ============================================================
// get_all（标量）
// ============================================================

std::vector<int64_t> PackedBackend::get_all() const {
    std::vector<int64_t> result;
    result.reserve(slots_.size());
    for (int i = 0; i < static_cast<int>(slots_.size()); ++i) {
        result.push_back(get(i));
    }
    return result;
}

// ============================================================
// get_all_simd（AVX2 优化）
// ============================================================

bool PackedBackend::can_use_simd_() const {
    if (slots_.empty()) return false;
    int first_bits = slots_[0].bits;
    if (first_bits < 8) return false;
    // 所有槽位等宽且无符号
    for (const auto& s : slots_) {
        if (s.bits != first_bits || s.is_signed) return false;
    }
    // 只支持 8/16 bit（32 bit 最多 2 个槽位，SIMD 无优势）
    return first_bits == 8 || first_bits == 16;
}

std::vector<int64_t> PackedBackend::get_all_simd() const {
    if (!can_use_simd_()) {
        return get_all();  // 回退到标量
    }
    if (slots_[0].bits == 8) {
        return get_all_simd_8bit_unsigned_();
    }
    return get_all_simd_16bit_unsigned_();
}

std::vector<int64_t> PackedBackend::get_all_simd_8bit_unsigned_() const {
    // 8-bit 等宽无符号，最多 8 个槽位
    // packed_ 的字节排列（little-endian）：
    //   byte[0] = bits 0-7 = slot[n-1]（最后一个槽位，offset=0）
    //   byte[7] = bits 56-63 = slot[0]（第一个槽位，offset=56）
    // 需反转字节顺序 → simd::reverse_bytes8（SSSE3 PSHUFB / 标量回退，见 simd 原语层）
    int n = static_cast<int>(slots_.size());
    std::vector<int64_t> result(n);
    simd::reverse_bytes8(packed_, n, result.data());
    return result;
}

std::vector<int64_t> PackedBackend::get_all_simd_16bit_unsigned_() const {
    // 16-bit 等宽无符号，最多 4 个槽位
    // packed_ 的 16-bit 单元排列（little-endian）：
    //   word[0] = bits 0-15 = slot[n-1]
    //   word[3] = bits 48-63 = slot[0]
    int n = static_cast<int>(slots_.size());
    std::vector<int64_t> result(n);

    // 直接用标量提取 16-bit 单元（4 个槽位时 SIMD 无明显优势）
    // 但保持接口一致，用位操作
    for (int i = 0; i < n; ++i) {
        result[i] = static_cast<int64_t>(
            (packed_ >> slots_[i].offset) & 0xFFFF
        );
    }
    return result;
}

// ============================================================
// 序列化（简单 JSON）
// ============================================================

std::string PackedBackend::serialize() const {
    std::ostringstream oss;
    oss << "{\"backend\":\"packed\",\"packed_value\":"
        << packed_ << ",\"total_bits\":" << total_bits_ << "}";
    return oss.str();
}

// ============================================================
// batch_get_all — 批量读取多个 packed 值
// ============================================================

std::vector<int64_t> PackedBackend::batch_get_all(
    const std::vector<int>& bits_list,
    const std::vector<uint64_t>& packed_values
) {
    if (bits_list.empty() || packed_values.empty()) {
        return {};
    }

    // 计算总位数和偏移（与 from_bits 相同的逻辑）
    int total = 0;
    for (int b : bits_list) {
        if (b <= 0) throw std::invalid_argument("每个 bits 必须 > 0");
        total += b;
    }
    if (total > 64) {
        throw std::invalid_argument("总位数超过 64");
    }

    int n_slots = static_cast<int>(bits_list.size());
    std::vector<int> offsets(n_slots);
    std::vector<uint64_t> masks(n_slots);
    int offset = total;
    for (int i = 0; i < n_slots; ++i) {
        offset -= bits_list[i];
        offsets[i] = offset;
        masks[i] = (bits_list[i] >= 64) ? ~0ULL : ((1ULL << bits_list[i]) - 1);
    }

    int n_values = static_cast<int>(packed_values.size());
    std::vector<int64_t> result(static_cast<size_t>(n_values) * n_slots);

    // 检查是否可用 8-bit 等宽快速路径
    bool use_8bit_fast = (n_slots >= 4 && n_slots <= 8);
    if (use_8bit_fast) {
        for (int b : bits_list) {
            if (b != 8) { use_8bit_fast = false; break; }
        }
    }

    // 8-bit 等宽快速路径 → simd::batch_reverse_u8（SSSE3/AVX2 PSHUFB 批量反转 +
    // 标量回退，见 simd 原语层；use_8bit_fast 保证 n_slots ∈ [4,8]，满足原语前提）
    if (use_8bit_fast) {
        simd::batch_reverse_u8(packed_values.data(), n_values, n_slots, result.data());
    } else {
        // 通用标量路径（回退，兼容 GPU/ARM 等非 x86 平台）
        for (int i = 0; i < n_values; ++i) {
            uint64_t pv = packed_values[i];
            for (int j = 0; j < n_slots; ++j) {
                result[static_cast<size_t>(i) * n_slots + j] =
                    static_cast<int64_t>((pv >> offsets[j]) & masks[j]);
            }
        }
    }

    return result;
}

} // namespace sgn
