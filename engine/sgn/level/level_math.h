#pragma once
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <climits>

namespace sgn {

// bits → max_range: (1 << bits) - 1
// bits < 0 表示未设置，返回 255（旧默认值）
// bits == 0 返回 0
// bits >= 32 返回 0xFFFFFFFF（用 int64_t 避免溢出）
inline int64_t bits_to_max_range(int32_t bits) {
    if (bits < 0) return 255;
    if (bits == 0) return 0;
    if (bits >= 32) return 0xFFFFFFFFLL;
    return (1LL << bits) - 1;
}

// max_range → bits: ceil(log2(max_range + 1))
// 返回 -1 表示未设置（max_range <= 0）
inline int32_t max_range_to_bits(int64_t max_range) {
    if (max_range <= 0) return -1;
    double exact = std::log2(static_cast<double>(max_range) + 1.0);
    return static_cast<int32_t>(std::ceil(exact));
}

} // namespace sgn