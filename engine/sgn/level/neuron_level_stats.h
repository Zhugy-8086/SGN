#pragma once
#include <cstdint>
#include <vector>
#include <deque>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>

#include "level_constants.h"

namespace sgn {

// NeuronLevelStats: 神经元 level 统计
//
// 用于自适应策略，基于 MAD（平均绝对偏差）动态调整 level。
//
// 数学依据：
//   - MAD 替代方差：纯整数计算，对极端值更鲁棒（v5.1.9-fix P2-3）
//   - 匹配值历史保留最近 100 个样本
//   - 验证率 = verified_count / total_count
struct NeuronLevelStats {
    int32_t neuron_id;
    int32_t current_level;
    int32_t peak_level;
    int32_t verified_count;
    int32_t total_count;
    int32_t level_change_count;
    int32_t last_match;
    double last_verification_rate;

    // 匹配值历史（用于计算 MAD）
    std::deque<int32_t> match_history;

    // 方差阈值覆盖（None=用默认）
    bool has_variance_override;
    double variance_override_value;

    // v1.4-rc20 方案 D: STE 梯度方差信号（HC8 STE 训练专用）
    // 与 match_variance（MAD，纯整数）不同，grad_variance 是 float，反映
    // 量化噪声经 STE 直通到反向传播的梯度分布（legacy level.py 方案 D 移植）
    std::deque<double> grad_variance_history;  // 窗口 100
    double last_grad_variance;
    bool has_grad_variance_threshold_override;
    double grad_variance_threshold_override;

    // v5.1.9-fix P2-3: 纯整数 MAD
    NeuronLevelStats()
        : neuron_id(0), current_level(LevelConstants::LEVEL_DEFAULT),
          peak_level(LevelConstants::LEVEL_DEFAULT),
          verified_count(0), total_count(0), level_change_count(0),
          last_match(0), last_verification_rate(1.0),
          has_variance_override(false), variance_override_value(0.0),
          last_grad_variance(0.0),
          has_grad_variance_threshold_override(false),
          grad_variance_threshold_override(0.0) {}

    NeuronLevelStats(int32_t nid, int32_t level = LevelConstants::LEVEL_DEFAULT)
        : neuron_id(nid), current_level(level),
          peak_level(level),
          verified_count(0), total_count(0), level_change_count(0),
          last_match(0), last_verification_rate(1.0),
          has_variance_override(false), variance_override_value(0.0),
          last_grad_variance(0.0),
          has_grad_variance_threshold_override(false),
          grad_variance_threshold_override(0.0) {}

    // 计算匹配值离散度（MAD：平均绝对偏差，纯整数）
    // n < 2 时返回 0
    int32_t match_variance() const {
        size_t n = match_history.size();
        if (n < 2) return 0;

        // 整数化均值（四舍五入）
        int64_t total = 0;
        for (auto m : match_history) {
            total += static_cast<int64_t>(m);
        }
        int32_t mean = static_cast<int32_t>((total + static_cast<int64_t>(n) / 2) / static_cast<int64_t>(n));

        // 平均绝对偏差（纯整数）。mad_sum/n 的商理论上可达 max-min ≈ 2^32-1
        // （int32 极端值交替），C++20 前越界 cast 为 UB——clamp 防御
        // （安全审计 2026-08-16 L3-1，实践不可达）
        int64_t mad_sum = 0;
        for (auto m : match_history) {
            mad_sum += static_cast<int64_t>(std::abs(static_cast<int64_t>(m) - static_cast<int64_t>(mean)));
        }
        int64_t mad = mad_sum / static_cast<int64_t>(n);
        if (mad > INT32_MAX) return INT32_MAX;
        if (mad < INT32_MIN) return INT32_MIN;
        return static_cast<int32_t>(mad);
    }

    // 验证通过率
    double verification_rate() const {
        if (total_count == 0) return 0.0;
        return static_cast<double>(verified_count) / static_cast<double>(total_count);
    }

    // v1.4-rc20 方案 D: STE 梯度方差（滑动平均）
    // 样本不足（< 2）时返回 0.0
    double grad_variance() const {
        size_t n = grad_variance_history.size();
        if (n < 2) return 0.0;
        double total = 0.0;
        for (double g : grad_variance_history) total += g;
        return total / static_cast<double>(n);
    }

    // 更新统计。grad_variance > 0 时累积到 grad_variance_history 驱动
    // 方案 D 触发条件；默认 0.0 = 不更新（v5.1.9 行为，向后兼容）
    void update(int32_t match, bool verified, double grad_variance_in = 0.0) {
        last_match = match;
        total_count++;
        if (verified) {
            verified_count++;
        }
        // 保留最近 100 个匹配值
        match_history.push_back(match);
        if (match_history.size() > 100) {
            match_history.pop_front();
        }
        // 同步更新当前验证率
        last_verification_rate = verification_rate();
        // 方案 D: 仅当传入有效值时累积（> 0）
        if (grad_variance_in > 0.0) {
            last_grad_variance = grad_variance_in;
            grad_variance_history.push_back(grad_variance_in);
            if (grad_variance_history.size() > 100) {
                grad_variance_history.pop_front();
            }
        }
    }

    // 方案 D: 梯度方差阈值覆盖（per-neuron 差异化）
    void set_grad_variance_threshold(double threshold) {
        has_grad_variance_threshold_override = true;
        grad_variance_threshold_override = threshold;
    }

    void clear_grad_variance_threshold() {
        has_grad_variance_threshold_override = false;
        grad_variance_threshold_override = 0.0;
    }

    double get_effective_grad_variance_threshold(double default_threshold) const {
        if (has_grad_variance_threshold_override) {
            return grad_variance_threshold_override;
        }
        return default_threshold;
    }

    // 设置方差阈值覆盖
    void set_variance_threshold(double threshold) {
        has_variance_override = true;
        variance_override_value = threshold;
    }

    // 清除方差阈值覆盖
    void clear_variance_threshold() {
        has_variance_override = false;
        variance_override_value = 0.0;
    }

    // 获取有效方差阈值
    double get_effective_variance_threshold(double default_threshold) const {
        if (has_variance_override) {
            return variance_override_value;
        }
        return default_threshold;
    }

    // 序列化
    std::unordered_map<std::string, std::string> to_dict() const {
        std::unordered_map<std::string, std::string> d;
        d["neuron_id"] = std::to_string(neuron_id);
        d["current_level"] = std::to_string(current_level);
        d["peak_level"] = std::to_string(peak_level);
        d["verified_count"] = std::to_string(verified_count);
        d["total_count"] = std::to_string(total_count);
        d["level_change_count"] = std::to_string(level_change_count);
        d["last_match"] = std::to_string(last_match);
        d["last_verification_rate"] = std::to_string(last_verification_rate);
        // 方案 D 字段
        d["last_grad_variance"] = std::to_string(last_grad_variance);
        std::string gv_hist;
        for (double g : grad_variance_history) {
            if (!gv_hist.empty()) gv_hist += ",";
            gv_hist += std::to_string(g);
        }
        d["grad_variance_history"] = gv_hist;
        return d;
    }

    // 反序列化
    static NeuronLevelStats from_dict(const std::unordered_map<std::string, std::string>& d) {
        NeuronLevelStats stats;
        auto it = d.find("neuron_id");
        if (it != d.end()) stats.neuron_id = std::stoi(it->second);
        it = d.find("current_level");
        if (it != d.end()) stats.current_level = std::stoi(it->second);
        it = d.find("peak_level");
        if (it != d.end()) stats.peak_level = std::stoi(it->second);
        it = d.find("verified_count");
        if (it != d.end()) stats.verified_count = std::stoi(it->second);
        it = d.find("total_count");
        if (it != d.end()) stats.total_count = std::stoi(it->second);
        it = d.find("level_change_count");
        if (it != d.end()) stats.level_change_count = std::stoi(it->second);
        it = d.find("last_match");
        if (it != d.end()) stats.last_match = std::stoi(it->second);
        it = d.find("last_verification_rate");
        if (it != d.end()) stats.last_verification_rate = std::stod(it->second);
        // 方案 D 字段（旧数据缺失时保持默认 0.0/空）
        it = d.find("last_grad_variance");
        if (it != d.end()) stats.last_grad_variance = std::stod(it->second);
        it = d.find("grad_variance_history");
        if (it != d.end() && !it->second.empty()) {
            // 格式: "v1,v2,..."（逆序解析无影响，deque 按序 push_back）
            std::string hist = it->second;
            size_t pos = 0;
            while (pos < hist.size()) {
                size_t comma = hist.find(',', pos);
                std::string val = hist.substr(pos, comma - pos);
                if (!val.empty()) {
                    try {
                        stats.grad_variance_history.push_back(std::stod(val));
                    } catch (...) {}
                }
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        }
        return stats;
    }
};

} // namespace sgn