#pragma once
#include <cstdint>
#include <string>
#include <memory>
#include <optional>
#include <algorithm>

#include "level_context.h"
#include "neuron_level_stats.h"
#include "level_constants.h"

namespace sgn {

// LevelStrategy: level 策略接口
//
// 所有策略实现必须提供：
//   - 运算时的 level 决策
//   - 自适应建议（可选）
class LevelStrategy {
public:
    virtual ~LevelStrategy() = default;

    // 策略名称
    virtual std::string name() const = 0;

    // 默认 level
    virtual int32_t default_level() const = 0;

    // 获取指定运算的 level
    virtual int32_t get_level_for_operation(
        LevelOperation operation,
        int32_t neuron_id = -1,
        const NeuronLevelStats* stats = nullptr) const = 0;

    // 根据统计建议新的 level（可选实现）
    // 返回建议的 level，或 std::nullopt 表示不调整
    virtual std::optional<int32_t> suggest_adaptation(const NeuronLevelStats& stats) const {
        (void)stats;
        return std::nullopt;
    }

    // 验证率触发的主动降级（可选实现）
    // 返回建议的 level，或 std::nullopt 表示不降级
    virtual std::optional<int32_t> suggest_demotion(const NeuronLevelStats& stats) const {
        (void)stats;
        return std::nullopt;
    }
};

// StandardStrategy: 标准策略 - 固定 level，不自适应
class StandardStrategy : public LevelStrategy {
public:
    explicit StandardStrategy(int32_t level = 0)
        : level_(level) {}

    std::string name() const override {
        return "standard(L" + std::to_string(level_) + ")";
    }

    int32_t default_level() const override {
        return level_;
    }

    int32_t get_level_for_operation(
        LevelOperation operation,
        int32_t neuron_id = -1,
        const NeuronLevelStats* stats = nullptr) const override {
        (void)operation;
        (void)neuron_id;
        (void)stats;
        return level_;
    }

    int32_t level() const { return level_; }

private:
    int32_t level_;
};

// AdaptiveStrategy: 自适应策略
//
// 规则（v5.1.9-fix 三段式，与 Python 对齐）：
//   - suggest_demotion（最高优先级）：验证率 < 阈值 → 主动降级
//   - suggest_adaptation：
//     - 升 level：匹配值方差 < 阈值/4 → 细粒度（受 peak_level 硬上限约束）
//     - 降 level（variance 路径）：方差 > 阈值*4 → 粗粒度
//   - v1.4-rc20 方案 D（grad_variance_threshold > 0 时启用，HC8 STE 训练专用）：
//     - 梯度方差 > threshold → 降 level（粗粒度，减少 STE 噪声）
//     - 梯度方差 < threshold/4 → 升 level（细粒度）
//     - 与 match_variance 触发独立，任一满足即触发
//     - 优先级：grad 降级 > match 升/降级 > grad 升级
class AdaptiveStrategy : public LevelStrategy {
public:
    AdaptiveStrategy(
        int32_t base_level = 0,
        double variance_threshold = 100.0,
        int32_t history_window = 50,
        double demotion_verification_threshold = 0.5,
        int32_t demotion_min_samples = 30,
        double grad_variance_threshold = 0.0)
        : base_level_(base_level)
        , variance_threshold_(variance_threshold)
        , history_window_(history_window)
        , demotion_verification_threshold_(demotion_verification_threshold)
        , demotion_min_samples_(demotion_min_samples)
        , grad_variance_threshold_(grad_variance_threshold) {}

    std::string name() const override {
        return "adaptive(base=L" + std::to_string(base_level_) + ")";
    }

    int32_t default_level() const override {
        return base_level_;
    }

    int32_t get_level_for_operation(
        LevelOperation operation,
        int32_t neuron_id = -1,
        const NeuronLevelStats* stats = nullptr) const override {
        (void)operation;
        (void)neuron_id;
        if (stats) {
            return stats->current_level;
        }
        return base_level_;
    }

    // 方差触发的升 level 或粗粒度降 level
    // v1.4-rc20 方案 D: STE 梯度方差触发（grad_variance_threshold > 0 且
    // 有梯度方差历史时启用），优先级：
    //   grad 降级 > match 升/降级 > grad 升级
    std::optional<int32_t> suggest_adaptation(const NeuronLevelStats& stats) const override {
        // 样本不足时不调整（用 history_window_，不是 demotion_min_samples_）
        if (static_cast<int32_t>(stats.match_history.size()) < history_window_) {
            return std::nullopt;
        }

        // 获取有效方差阈值
        double effective_threshold = stats.get_effective_variance_threshold(variance_threshold_);
        int32_t mad = stats.match_variance();
        int32_t current = stats.current_level;

        // 方案 D 有效梯度方差阈值（per-neuron 覆盖优先，回退策略实例阈值）
        double effective_grad_threshold =
            stats.get_effective_grad_variance_threshold(grad_variance_threshold_);
        bool grad_d_enabled =
            effective_grad_threshold > 0.0 && stats.grad_variance_history.size() >= 2;

        // 方案 D 降级（优先级最高）：梯度方差大 → 粗粒度，减少 STE 噪声
        if (grad_d_enabled && stats.grad_variance() > effective_grad_threshold) {
            int32_t new_level = std::max(current - 1, LevelConstants::LEVEL_MIN);
            if (new_level != current) {
                return new_level;
            }
        }

        // 升 level：方差小 → 可更高精度
        // v5.1.9-fix P1-2: peak_level 作为升 level 的硬上限
        if (mad >= 0 && mad < effective_threshold / 4.0) {
            // current+1 用 int64 中间量防有符号溢出 UB（审计 L4-1，实践被
            // LEVEL_MAX=2 约束不可达）
            int32_t new_level = static_cast<int32_t>(
                std::min<int64_t>(std::min<int64_t>(static_cast<int64_t>(current) + 1,
                                                    LevelConstants::LEVEL_MAX),
                                  stats.peak_level));
            if (new_level != current) {
                return new_level;
            }
        }

        // 降 level（variance 路径）：方差大 → 降低精度提高稳定性
        if (mad > effective_threshold * 4.0) {
            int32_t new_level = std::max(current - 1, LevelConstants::LEVEL_MIN);
            if (new_level != current) {
                return new_level;
            }
        }

        // 方案 D 升级（优先级最低）：梯度方差小 → 细粒度
        if (grad_d_enabled && stats.grad_variance() < effective_grad_threshold / 4.0) {
            int32_t new_level = static_cast<int32_t>(
                std::min<int64_t>(std::min<int64_t>(static_cast<int64_t>(current) + 1,
                                                    LevelConstants::LEVEL_MAX),
                                  stats.peak_level));
            if (new_level != current) {
                return new_level;
            }
        }

        return std::nullopt;
    }

    // 验证率差时主动降级（v5.1.9-fix）
    std::optional<int32_t> suggest_demotion(const NeuronLevelStats& stats) const override {
        // 样本不足，不做降级判断（用 total_count，不是 match_history.size()）
        if (stats.total_count < demotion_min_samples_) {
            return std::nullopt;
        }

        // 验证率足够好，不降级
        if (stats.verification_rate() >= demotion_verification_threshold_) {
            return std::nullopt;
        }

        // 已是最粗粒度，不降级
        if (stats.current_level <= LevelConstants::LEVEL_MIN) {
            return std::nullopt;
        }

        // 主动降一级
        return std::max(stats.current_level - 1, LevelConstants::LEVEL_MIN);
    }

    // Accessors
    double variance_threshold() const { return variance_threshold_; }
    int32_t history_window() const { return history_window_; }
    double demotion_verification_threshold() const { return demotion_verification_threshold_; }
    int32_t demotion_min_samples() const { return demotion_min_samples_; }
    // v1.4-rc20 方案 D: STE 梯度方差阈值（0 = 禁用）
    double grad_variance_threshold() const { return grad_variance_threshold_; }

private:
    int32_t base_level_;
    double variance_threshold_;
    int32_t history_window_;
    double demotion_verification_threshold_;
    int32_t demotion_min_samples_;
    double grad_variance_threshold_;
};

} // namespace sgn