// msint_view.cpp - MSInt 视角 C++ 实现（bitsplit / concat）
//
// 对标 Python engine.ms_int.core.ViewEngine._resolve_bitsplit / _resolve_concat
#include "msint_view.h"

#include <stdexcept>
#include <string>

namespace sgn {

// ============================================================
// bitsplit
// ============================================================

std::vector<int64_t> MSIntView::bitsplit(int64_t raw, int total_bits, int target_bits) {
    if (target_bits <= 0) {
        throw std::invalid_argument("target_bits 必须 > 0");
    }
    if (total_bits <= 0) {
        throw std::invalid_argument("total_bits 必须 > 0");
    }
    if (target_bits > total_bits) {
        throw std::invalid_argument("target_bits(" + std::to_string(target_bits) +
                                    ") 不能大于 total_bits(" +
                                    std::to_string(total_bits) + ")");
    }

    // 负数转补码（用 total_bits 位宽）
    uint64_t raw_unsigned = static_cast<uint64_t>(raw);
    if (raw < 0) {
        uint64_t mask = (total_bits >= 64) ? ~0ULL : ((1ULL << total_bits) - 1);
        raw_unsigned = raw_unsigned & mask;
    }

    uint64_t mask = (target_bits >= 64) ? ~0ULL : ((1ULL << target_bits) - 1);
    int n_parts = (total_bits + target_bits - 1) / target_bits;

    std::vector<int64_t> result;
    result.reserve(n_parts);
    uint64_t remaining = raw_unsigned;
    for (int i = 0; i < n_parts; ++i) {
        result.push_back(static_cast<int64_t>(remaining & mask));
        remaining >>= target_bits;
    }
    return result;
}

int64_t MSIntView::bitsplit_index(int64_t raw, int total_bits, int target_bits, int idx) {
    if (idx < 0) {
        throw std::out_of_range("bitsplit index " + std::to_string(idx) + " < 0");
    }
    auto parts = bitsplit(raw, total_bits, target_bits);
    if (idx >= static_cast<int>(parts.size())) {
        throw std::out_of_range("bitsplit index " + std::to_string(idx) +
                                " 超出范围 [0, " +
                                std::to_string(parts.size()) + ")");
    }
    return parts[idx];
}

// ============================================================
// concat
// ============================================================

uint64_t MSIntView::concat(const std::vector<int64_t>& values,
                           const std::vector<int>& bits_list) {
    if (values.size() != bits_list.size()) {
        throw std::invalid_argument("values 长度(" +
                                    std::to_string(values.size()) +
                                    ") != bits_list 长度(" +
                                    std::to_string(bits_list.size()) + ")");
    }
    if (values.empty()) {
        return 0;
    }

    // 检查总位数
    int total = 0;
    for (int b : bits_list) {
        if (b <= 0) {
            throw std::invalid_argument("每个 bits 必须 > 0");
        }
        total += b;
    }
    if (total > 64) {
        throw std::invalid_argument("总位数 " + std::to_string(total) + " > 64");
    }

    // 第一个值在高位，最后一个在低位
    // result = (result << bits_i) | (value_i & mask_i)
    uint64_t result = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        uint64_t mask = (bits_list[i] >= 64) ? ~0ULL : ((1ULL << bits_list[i]) - 1);
        uint64_t val_unsigned = static_cast<uint64_t>(values[i]) & mask;
        result = (result << bits_list[i]) | val_unsigned;
    }
    return result;
}

int64_t MSIntView::concat_signed(const std::vector<int64_t>& values,
                                 const std::vector<int>& bits_list) {
    return static_cast<int64_t>(concat(values, bits_list));
}

} // namespace sgn
