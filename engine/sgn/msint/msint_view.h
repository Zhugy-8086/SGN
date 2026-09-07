// msint_view.h - MSInt 视角 C++ 实现（bitsplit / concat）
//
// 对标 Python engine.ms_int.core.ViewEngine._resolve_bitsplit / _resolve_concat
//
// 设计：
//   - bitsplit: 把 raw 按 target_bits 拆分成多个分片，低位在前
//   - concat: 把多个值按各自 bits 拼接为一个 int，第一个在高位
//   - 全静态方法，无状态
//
// 数学依据：
//   - bitsplit: result[i] = (raw >> (i * target_bits)) & mask
//   - concat: result = (result << bits_i) | value_i
#pragma once

#include <cstdint>
#include <vector>

namespace sgn {

class MSIntView {
public:
    // ---- bitsplit ----

    // 位拆分：把 raw 按 target_bits 拆分，低位在前
    // total_bits 是 raw 的有效位数（决定分片数量）
    // 负数自动转补码（用 total_bits 位宽）
    static std::vector<int64_t> bitsplit(int64_t raw, int total_bits, int target_bits);

    // 位拆分取第 idx 个分片（0-based，低位在前）
    static int64_t bitsplit_index(int64_t raw, int total_bits, int target_bits, int idx);

    // ---- concat ----

    // 拼接：把多个值按各自 bits 拼接为一个 uint64_t
    // 第一个值在高位，最后一个在低位
    // values 和 bits_list 等长
    static uint64_t concat(const std::vector<int64_t>& values,
                           const std::vector<int>& bits_list);

    // 拼接取结果（返回 int64_t，高位截断）
    static int64_t concat_signed(const std::vector<int64_t>& values,
                                 const std::vector<int>& bits_list);
};

} // namespace sgn
