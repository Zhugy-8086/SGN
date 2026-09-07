#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include "value_spec.h"

namespace sgn {

// LayerCost: 单层成本信号
//
// 数学依据（exp26 + exp34）：
//   - 成本信号 c_i = grad_l2² × in_dim（exp26 确认，含 in_dim 因子）
//   - 不可用 log₂(var_i) 近似（exp26 B 失败，差 29×，缺 in_dim 因子）
//   - M-凸性保证贪心边际分配给出全局最优（exp34，gap ~ 1e-14）
//
// 误差模型：f_i(b) = c_i / (2^b - 1)（STE 方差，bits 越大误差越小）
struct LayerCost {
    double c;        // 成本信号 = grad_l2² × in_dim
    uint8_t b_min;   // 该层最小 bits（通常 4 或 8）
    uint8_t b_max;   // 该层最大 bits（通常 20 或 24）

    constexpr LayerCost(double cost = 0.0, uint8_t lo = 4, uint8_t hi = 24)
        : c(cost), b_min(lo), b_max(hi) {}
};

// PrecisionBudget: 精度预算接口
//
// Stage 3.0 统一抽象层的核心组件，负责在给定总 bits 预算下
// 为各层分配 bits，使总 STE 方差最小化。
//
// 数学依据：
//   - 主算法：贪心边际分配（exp34 证明 M-凸性，O(n log n) 全局最优）
//   - 成本信号：c_i = grad_l2² × in_dim（exp26 确认）
//   - 误差模型：f_i(b) = c_i / (2^b - 1)
//   - 反馈信号：log(var)（exp10-recheck power law, R²=0.903）
//
// 预留字段：
//   - level_f: 前向 ValueSpec（Level_f 接口预留，Stage 3.0.4 实现）
//   - level_b: 反向 ValueSpec（Level_b 接口预留，Stage 3.0.4 实现）
struct PrecisionBudget {
    // 精度预算配置
    uint32_t total_bits;          // 总 bits 预算 B_total
    std::vector<LayerCost> layers; // 各层成本信号

    // Level_f / Level_b 双向控制预留（默认 None，Stage 3.0.4 填充）
    // 当 level_f/level_b 非 None 时，调度器优先使用它们
    ValueSpec level_f;            // 前向精度规格（默认 bits=0 表示 None）
    ValueSpec level_b;            // 反向精度规格（默认 bits=0 表示 None）
    bool has_level_f = false;     // level_f 是否设置
    bool has_level_b = false;     // level_b 是否设置

    // 设置 level_f（前向精度规格）
    void set_level_f(ValueSpec spec) {
        level_f = spec;
        has_level_f = true;
    }

    // 设置 level_b（反向精度规格）
    void set_level_b(ValueSpec spec) {
        level_b = spec;
        has_level_b = true;
    }

    // 清除 level_f/level_b（回退到单 level 模式）
    void clear_level_f() { has_level_f = false; }
    void clear_level_b() { has_level_b = false; }

    // 贪心边际分配算法（exp34 M-凸性，O(n log n) 全局最优）
    //
    // 算法：
    //   1. 初始化 b_i = b_min_i, R = B_total - Σ b_min_i
    //   2. while R > 0:
    //        i* = argmax [f_i(b_i) - f_i(b_i+1)]  // 最大边际收益
    //        if b_{i*} >= b_max_{i*}: 跳过该层
    //        b_{i*} += 1; R -= 1
    //
    // 误差模型：f_i(b) = c_i / (2^b - 1)
    // 边际收益：Δ_i(b) = f_i(b) - f_i(b+1) = c_i / (2^b - 1) - c_i / (2^(b+1) - 1)
    //                   = c_i * (2^(b+1) - 2^b) / ((2^b - 1)(2^(b+1) - 1))
    //                   = c_i * 2^b / ((2^b - 1)(2^(b+1) - 1))
    //
    // 返回：各层分配的 bits 数组
    std::vector<uint8_t> allocate() const {
        if (layers.empty()) return {};
        // 1. 初始化每层到 b_min
        std::vector<uint8_t> bits(layers.size());
        uint32_t used = 0;
        for (size_t i = 0; i < layers.size(); ++i) {
            bits[i] = layers[i].b_min;
            used += layers[i].b_min;
        }
        int32_t remaining = static_cast<int32_t>(total_bits) - static_cast<int32_t>(used);
        // 预算不足：每层保持 b_min（可能不满足，但避免负数）
        if (remaining <= 0) return bits;

        // 2. 贪心分配：每次把 1 bit 分给边际收益最大的层
        while (remaining > 0) {
            bool any_alloc = false;
            double best_gain = -1.0;
            size_t best_idx = 0;
            for (size_t i = 0; i < layers.size(); ++i) {
                if (bits[i] >= layers[i].b_max) continue;
                double gain = marginal_gain(layers[i].c, bits[i]);
                if (gain > best_gain) {
                    best_gain = gain;
                    best_idx = i;
                    any_alloc = true;
                }
            }
            if (!any_alloc) break;  // 所有层都到 b_max
            bits[best_idx]++;
            remaining--;
        }
        return bits;
    }

    // 计算总 STE 方差：Σ c_i / (2^b_i - 1)
    double total_error(const std::vector<uint8_t>& bits) const {
        double total = 0.0;
        for (size_t i = 0; i < layers.size() && i < bits.size(); ++i) {
            total += layer_error(layers[i].c, bits[i]);
        }
        return total;
    }

    // 单层误差：f_i(b) = c_i / (2^b - 1)
    static double layer_error(double c, uint8_t b) {
        if (b == 0) return c;  // 0 bit 时误差等于成本（无量化）
        double denom = std::ldexp(1.0, b) - 1.0;  // 2^b - 1
        return c / denom;
    }

    // 边际收益：Δ_i(b) = f_i(b) - f_i(b+1) = c_i * 2^b / ((2^b - 1)(2^(b+1) - 1))
    static double marginal_gain(double c, uint8_t b) {
        if (b == 0) return c;  // 0→1 边际收益等于成本
        double two_b = std::ldexp(1.0, b);        // 2^b
        double two_b1 = std::ldexp(1.0, b + 1);   // 2^(b+1)
        double denom = (two_b - 1.0) * (two_b1 - 1.0);
        return c * two_b / denom;
    }
};

} // namespace sgn
