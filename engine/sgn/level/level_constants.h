#pragma once
#include <cstdint>

namespace sgn {

// LevelConstants: 默认精度参数
//
// 数学依据：
//   - bits 是主杠杆（exp19/25/34）
//   - max_range = 2^bits - 1（消除位空洞）
//   - 贪心边际分配（exp34 M-凸性，gap ~ 1e-14）
//   - 成本信号 c_i = grad_l2² × in_dim（exp26 确认）
//   - 滞后机制 ±2（exp23）
struct LevelConstants {
    // 默认 bits 范围
    static constexpr uint8_t DEFAULT_BITS_MIN = 4;
    static constexpr uint8_t DEFAULT_BITS_MAX = 20;
    static constexpr uint8_t DEFAULT_BITS = 8;

    // 默认 max_range（对应 8 bits）
    static constexpr int32_t DEFAULT_MAX_RANGE = 255;

    // 总 bits 预算默认值
    static constexpr uint32_t DEFAULT_TOTAL_BITS = 124;

    // 滞后机制默认 delta（exp23）
    static constexpr uint8_t HYSTERESIS_DELTA = 2;

    // Level 范围（与 Python config.py 对齐）
    // Python: LEVEL_MIN_LOWER=-4, LEVEL_MAX_UPPER=2, LEVEL_DEFAULT=0
    static constexpr int32_t LEVEL_MIN = -4;
    static constexpr int32_t LEVEL_MAX = 2;
    static constexpr int32_t LEVEL_DEFAULT = 0;

    // 未设置标记
    static constexpr int32_t BITS_UNSET = -1;
};

} // namespace sgn