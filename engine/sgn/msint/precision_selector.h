// precision_selector.h - MSInt Level 逐元素精度选择（PrecisionSelector）
//
// 对标范式文档 §3.2 "Level 调度器可以控制'展开'到哪个精度层次" 与待验证假设 H4
//   "Level 调度可以逐元素选择最优拆分粒度"。
//
// 语义：
//   - 重要性 importance 越高 → 拆分越细（split_bits 越小，保留更多精度层级）
//   - 重要性 importance 越低 → 只用粗粒度（split_bits 越大，少搬运）
//   例如 total_bits=32，可选粒度 options={16,8,4}：
//     importance 低 → 16（粗粒度，2 部分）
//     importance 中 → 8 （4 部分）
//     importance 高 → 4 （最精细，8 部分）
//
// 与 MultiScaleView 组合：按选定粒度对元素做 1:N 解释，
// 实现"重要元素展开到细粒度、次要元素只用粗粒度"的解释性计算。
#pragma once

#include <cstdint>
#include <vector>

namespace sgn {

class PrecisionSelector {
public:
    // options: 可选拆分粒度（split_bits），须按从粗到细排列（split_bits 递减）。
    //   e.g. {16, 8, 4}：16 最粗、4 最细。
    // thresholds: 相邻粒度间的切换阈值，升序，长度 == options.size()-1。
    //   select(importance)：importance < t0 → options[0]（最粗）
    //                        t0 <= importance < t1 → options[1]
    //                        ...
    //                        importance >= t_{k-1} → options[k]（最细）
    // 空 thresholds 时恒选 options[0]。
    PrecisionSelector(int total_bits,
                      std::vector<int> options,
                      std::vector<int64_t> thresholds);

    // 返回默认构造：total_bits=32, options={16,8,4}, thresholds={100,1000}。
    static PrecisionSelector default_selector();

    // 根据重要性选择该元素的拆分粒度 split_bits。
    int select(int64_t importance) const;

    // 便捷：按选定粒度对单个值做 1:N 解释（返回 parts，低位在前）。
    std::vector<int64_t> interpret(int64_t value, int64_t importance) const;

    // 批量逐元素选择 + 解释：
    //   values 与 importances 等长；返回第 i 个元素按 select(importances[i])
    //   粒度解释的 parts（不同元素 parts 数可不同，故为锯齿形二维数组）。
    std::vector<std::vector<int64_t>> interpret_batch(
        const std::vector<int64_t>& values,
        const std::vector<int64_t>& importances) const;

    // Accessors
    int total_bits() const { return total_bits_; }
    const std::vector<int>& options() const { return options_; }
    const std::vector<int64_t>& thresholds() const { return thresholds_; }

private:
    int total_bits_;
    std::vector<int> options_;        // 从粗到细（split_bits 递减）
    std::vector<int64_t> thresholds_; // 升序，len = options.size()-1
};

} // namespace sgn
