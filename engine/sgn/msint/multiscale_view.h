// multiscale_view.h - MSInt 1:N 多精度解释（MultiScaleView）
//
// 对标范式文档 docs/msint_multisplit_paradigm/MSint多精度拆分计算范式.md §1.2
//   "一次搬运，多精度解释"：同一个 int 值可同时解释为多个精度层次，
//   每个层次都能独立参与计算。
//
// 与 MSIntView::bitsplit 的区别：
//   - MSIntView::bitsplit 只按一个 target_bits 拆分（单一精度）
//   - MultiScaleView 一次性产出多个精度层次（如 int32 → {int16×2, int8×4, int4×8}），
//     体现 1:N 多精度解释：一次读取，多粒度表示。
//
// 数学依据（复用 SplitDot::split_parts，位拆分严格可逆）：
//   value = Σ_{k=0}^{n-1} part_k * 2^(k*split_bits)
//   低位 part_0..part_{n-2} 无符号，最高位 part_{n-1} 符号扩展。
//   对任何精度层次 split_bits，重建后都 bit-exact 等于原值（含负数）。
#pragma once

#include <cstdint>
#include <map>
#include <vector>

namespace sgn {

class MultiScaleView {
public:
    // 一次性把一个 int64 解释为多个精度层次（1:N）。
    // total_bits=32 时默认产出 split_bits ∈ {16, 8, 4}（由 total_bits/2 递减到 4）。
    // 返回：有序 map<split_bits, vector<int64_t> parts>，parts 低位在前。
    //   {16: [l, h], 8: [ll, lh, hl, hh], 4: [l0..l7]}
    // 每个层次的 parts 都能独立重建出原值。
    static std::map<int, std::vector<int64_t>> interpret(int64_t value, int total_bits = 32);

    // 指定精度层次列表解释（split_bits_list 需能整除 total_bits）。
    static std::map<int, std::vector<int64_t>> interpret_levels(
        int64_t value, int total_bits, const std::vector<int>& split_bits_list);

    // 默认精度层次：total_bits/2 递减到 4（须为 2 的幂且能整除 total_bits）。
    // 空/非法时回退到 {16}。
    static std::vector<int> default_levels(int total_bits);

    // 批量 1:N 解释：返回按 level 分组的二维矩阵。
    //   result[split_bits] = K x n 矩阵，result[split_bits][i] 是第 i 个元素的 n 个 parts。
    // split_bits_list 为空时用 default_levels。
    static std::map<int, std::vector<std::vector<int64_t>>> interpret_batch(
        const std::vector<int64_t>& values, int total_bits = 32,
        const std::vector<int>& split_bits_list = {});

    // 一致性检查：指定精度层次重建后等于原值（含负数）。
    static bool is_exact(int64_t value, int total_bits, int split_bits);
};

} // namespace sgn
