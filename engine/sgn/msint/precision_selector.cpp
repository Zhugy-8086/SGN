// precision_selector.cpp - MSInt Level 逐元素精度选择实现
#include "precision_selector.h"

#include <stdexcept>
#include <string>

#include "split_dot.h"

namespace sgn {

PrecisionSelector::PrecisionSelector(int total_bits,
                                     std::vector<int> options,
                                     std::vector<int64_t> thresholds) {
    if (total_bits <= 0) {
        throw std::invalid_argument("total_bits 必须 > 0");
    }
    if (options.empty()) {
        throw std::invalid_argument("options 不能为空");
    }
    // 校验从粗到细（split_bits 严格递减）且都能整除 total_bits
    for (size_t i = 0; i < options.size(); ++i) {
        if (options[i] <= 0) {
            throw std::invalid_argument("每个 split_bits 必须 > 0");
        }
        if (total_bits % options[i] != 0) {
            throw std::invalid_argument(
                "total_bits(" + std::to_string(total_bits) + ") 必须能被 split_bits(" +
                std::to_string(options[i]) + ") 整除");
        }
        if (i > 0 && options[i] >= options[i - 1]) {
            throw std::invalid_argument(
                "options 必须按从粗到细排列（split_bits 递减）");
        }
    }
    if (thresholds.size() != options.size() - 1) {
        throw std::invalid_argument(
            "thresholds 长度(" + std::to_string(thresholds.size()) +
            ") 必须等于 options 长度-1(" + std::to_string(options.size() - 1) + ")");
    }
    for (size_t i = 1; i < thresholds.size(); ++i) {
        if (thresholds[i] <= thresholds[i - 1]) {
            throw std::invalid_argument("thresholds 必须严格升序");
        }
    }
    total_bits_ = total_bits;
    options_ = std::move(options);
    thresholds_ = std::move(thresholds);
}

PrecisionSelector PrecisionSelector::default_selector() {
    return PrecisionSelector(32, {16, 8, 4}, {100, 1000});
}

int PrecisionSelector::select(int64_t importance) const {
    // 从最粗开始；importance 越过每个阈值就向更细移动一级
    size_t idx = 0;
    for (size_t k = 0; k < thresholds_.size(); ++k) {
        if (importance >= thresholds_[k]) {
            idx = k + 1;
        } else {
            break;
        }
    }
    return options_[idx];
}

std::vector<int64_t> PrecisionSelector::interpret(int64_t value, int64_t importance) const {
    int split_bits = select(importance);
    return SplitDot::split_parts(value, total_bits_, split_bits);
}

std::vector<std::vector<int64_t>> PrecisionSelector::interpret_batch(
    const std::vector<int64_t>& values,
    const std::vector<int64_t>& importances) const {
    if (values.size() != importances.size()) {
        throw std::invalid_argument(
            "values 长度(" + std::to_string(values.size()) +
            ") != importances 长度(" + std::to_string(importances.size()) + ")");
    }
    std::vector<std::vector<int64_t>> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = interpret(values[i], importances[i]);
    }
    return result;
}

} // namespace sgn
