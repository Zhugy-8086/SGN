// multiscale_view.cpp - MSInt 1:N 多精度解释实现
#include "multiscale_view.h"

#include <stdexcept>
#include <string>
#include <algorithm>

#include "split_dot.h"

namespace sgn {

// 默认精度层次：total_bits/2 递减到 4（须为 2 的幂且能整除 total_bits）
std::vector<int> MultiScaleView::default_levels(int total_bits) {
    if (total_bits <= 0) {
        throw std::invalid_argument("total_bits 必须 > 0");
    }
    std::vector<int> levels;
    int b = total_bits / 2;
    // 逐级减半，直到 4；要求 total_bits % b == 0（total_bits 为 2 的幂时天然满足）
    while (b >= 4 && total_bits % b == 0) {
        levels.push_back(b);
        if (b == 4) break;
        b /= 2;
    }
    if (levels.empty()) {
        levels.push_back(16);  // 回退
    }
    return levels;
}

std::map<int, std::vector<int64_t>> MultiScaleView::interpret_levels(
    int64_t value, int total_bits, const std::vector<int>& split_bits_list) {
    if (split_bits_list.empty()) {
        throw std::invalid_argument("split_bits_list 不能为空");
    }
    std::map<int, std::vector<int64_t>> result;
    for (int b : split_bits_list) {
        result[b] = SplitDot::split_parts(value, total_bits, b);
    }
    return result;
}

std::map<int, std::vector<int64_t>> MultiScaleView::interpret(
    int64_t value, int total_bits) {
    return interpret_levels(value, total_bits, default_levels(total_bits));
}

std::map<int, std::vector<std::vector<int64_t>>> MultiScaleView::interpret_batch(
    const std::vector<int64_t>& values, int total_bits,
    const std::vector<int>& split_bits_list) {
    std::vector<int> levels = split_bits_list.empty() ? default_levels(total_bits)
                                                       : split_bits_list;
    std::map<int, std::vector<std::vector<int64_t>>> result;
    for (int b : levels) {
        result[b].resize(values.size());
    }
    for (size_t i = 0; i < values.size(); ++i) {
        auto parts_by_level = interpret_levels(values[i], total_bits, levels);
        for (int b : levels) {
            result[b][i] = std::move(parts_by_level[b]);
        }
    }
    return result;
}

bool MultiScaleView::is_exact(int64_t value, int total_bits, int split_bits) {
    if (split_bits <= 0) {
        throw std::invalid_argument("split_bits 必须 > 0");
    }
    if (total_bits % split_bits != 0) {
        throw std::invalid_argument("total_bits 必须能被 split_bits 整除");
    }
    auto parts = SplitDot::split_parts(value, total_bits, split_bits);
    // 重建（任意精度；此处用 __int128 覆盖 total_bits<=64 的有符号范围）
#if defined(__SIZEOF_INT128__)
    __int128 acc = 0;
    for (size_t k = 0; k < parts.size(); ++k) {
        acc += static_cast<__int128>(parts[k]) << (static_cast<int>(k) * split_bits);
    }
    return static_cast<int64_t>(acc) == value;
#else
    // 回退：用 int64 重建（total_bits<=64 时对符合符号扩展的拆分成立）
    // 仅当总位数不超过 63 位且符号正确时才可靠；此处保守处理：
    // 直接按位比较拆分-重建位模式。
    uint64_t lo = 0;
    for (size_t k = 0; k < parts.size(); ++k) {
        uint64_t p = static_cast<uint64_t>(parts[k]);
        uint64_t mask = (split_bits >= 64) ? ~0ULL : ((1ULL << split_bits) - 1);
        lo |= (p & mask) << (static_cast<int>(k) * split_bits);
    }
    return static_cast<int64_t>(lo) == value;
#endif
}

} // namespace sgn
