#pragma once
#include <cstdint>
#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "level_constants.h"
#include "../common/precision_budget.h"

namespace sgn {

// BitsAllocator: bits 分配适配器
//
// 集成 PrecisionBudget::allocate() 进行贪心边际分配，
// 并实现 allocate_with_hysteresis() 带滞后机制。
//
// 数学依据：
//   - 贪心边际分配（exp34 M-凸性，O(n log n) 全局最优）
//   - 成本信号 c_i = grad_l2² × in_dim（exp26 确认）
//   - 滞后机制 ±2（exp23，防止 bits 频繁切换）
class BitsAllocator {
public:
    BitsAllocator(uint32_t total_bits = LevelConstants::DEFAULT_TOTAL_BITS,
                  uint8_t b_min = LevelConstants::DEFAULT_BITS_MIN,
                  uint8_t b_max = LevelConstants::DEFAULT_BITS_MAX,
                  uint8_t hysteresis_delta = LevelConstants::HYSTERESIS_DELTA)
        : total_bits_(total_bits), b_min_(b_min), b_max_(b_max),
          hysteresis_delta_(hysteresis_delta) {}

    // 从 (grad_l2, in_dim) 列表构建 PrecisionBudget
    // 成本信号 c_i = grad_l2² × in_dim（exp26 确认）
    PrecisionBudget build_budget(const std::vector<std::pair<double, int32_t>>& costs) const {
        std::vector<LayerCost> layers;
        layers.reserve(costs.size());
        for (const auto& [grad_l2, in_dim] : costs) {
            layers.emplace_back(grad_l2 * grad_l2 * static_cast<double>(in_dim),
                                b_min_, b_max_);
        }
        return PrecisionBudget(total_bits_, layers);
    }

    // 贪心边际分配（无滞后机制）
    std::vector<uint8_t> allocate(const std::vector<std::pair<double, int32_t>>& costs) const {
        auto pb = build_budget(costs);
        return pb.allocate();
    }

    // 带滞后机制的贪心分配（bits 变化限制 ±delta，exp23 设计）
    std::vector<uint8_t> allocate_with_hysteresis(
        const std::vector<std::pair<double, int32_t>>& costs,
        const std::vector<uint8_t>& prev_bits) const {

        auto raw = allocate(costs);

        // 无上一轮数据时返回原始分配
        if (prev_bits.empty() || prev_bits.size() != raw.size()) {
            return raw;
        }

        // 应用滞后约束：clamp 到 [prev - delta, prev + delta]
        std::vector<uint8_t> clamped(raw.size());
        int32_t total = 0;
        for (size_t i = 0; i < raw.size(); ++i) {
            int32_t lo = std::max(static_cast<int32_t>(b_min_),
                                  static_cast<int32_t>(prev_bits[i]) - static_cast<int32_t>(hysteresis_delta_));
            int32_t hi = std::min(static_cast<int32_t>(b_max_),
                                  static_cast<int32_t>(prev_bits[i]) + static_cast<int32_t>(hysteresis_delta_));
            clamped[i] = static_cast<uint8_t>(std::max(lo, std::min(hi, static_cast<int32_t>(raw[i]))));
            total += clamped[i];
        }

        // clamp 后总 bits 可能变化，需重新分配差额
        int32_t diff = static_cast<int32_t>(total_bits_) - total;
        if (diff > 0) {
            // 预算剩余：按边际收益从大到小分配给未达上限的层
            while (diff > 0) {
                double best_gain = -1.0;
                size_t best_idx = raw.size();
                for (size_t i = 0; i < raw.size(); ++i) {
                    int32_t hi = std::min(static_cast<int32_t>(b_max_),
                                          static_cast<int32_t>(prev_bits[i]) + static_cast<int32_t>(hysteresis_delta_));
                    if (static_cast<int32_t>(clamped[i]) >= hi) continue;
                    double c = costs[i].first * costs[i].first * static_cast<double>(costs[i].second);
                    double gain = PrecisionBudget::marginal_gain(c, clamped[i]);
                    if (gain > best_gain) {
                        best_gain = gain;
                        best_idx = i;
                    }
                }
                if (best_idx >= raw.size()) break;
                clamped[best_idx]++;
                diff--;
            }
        } else if (diff < 0) {
            // 预算超支：按边际收益从小到大回收
            diff = -diff;
            while (diff > 0) {
                double worst_gain = std::numeric_limits<double>::max();
                size_t worst_idx = raw.size();
                for (size_t i = 0; i < raw.size(); ++i) {
                    int32_t lo = std::max(static_cast<int32_t>(b_min_),
                                          static_cast<int32_t>(prev_bits[i]) - static_cast<int32_t>(hysteresis_delta_));
                    if (static_cast<int32_t>(clamped[i]) <= lo) continue;
                    double c = costs[i].first * costs[i].first * static_cast<double>(costs[i].second);
                    double gain = PrecisionBudget::marginal_gain(c, clamped[i] - 1);
                    if (gain < worst_gain) {
                        worst_gain = gain;
                        worst_idx = i;
                    }
                }
                if (worst_idx >= raw.size()) break;
                clamped[worst_idx]--;
                diff--;
            }
        }

        return clamped;
    }

    // Accessors
    uint32_t total_bits() const { return total_bits_; }
    uint8_t b_min() const { return b_min_; }
    uint8_t b_max() const { return b_max_; }
    uint8_t hysteresis_delta() const { return hysteresis_delta_; }

private:
    uint32_t total_bits_;
    uint8_t b_min_;
    uint8_t b_max_;
    uint8_t hysteresis_delta_;
};

} // namespace sgn