#pragma once
#include <cstdint>
#include <stdexcept>
#include <cmath>

namespace sgn {

// Scale 函数类型（exp25 确认 MAX 为默认最优）
enum class ScaleFn : uint8_t {
    MAX = 0,   // scale = max(|x|)，默认，所有 bits 下最优
    RMS = 1,   // scale = sqrt(mean(x^2))，保留接口
    L2 = 2,    // scale = ||x||_2，保留接口（低 bits 截断）
    P95 = 3,   // scale = percentile(|x|, 95)，保留接口
};

// ValueSpec: 统一精度规格
// bits 是主杠杆，scale 默认 MAX
struct ValueSpec {
    uint8_t bits;        // 主杠杆：8/12/16/20/24
    ScaleFn scale;       // 次要杠杆：默认 MAX

    // 构造
    constexpr ValueSpec(uint8_t b = 16, ScaleFn s = ScaleFn::MAX)
        : bits(b), scale(s) {}

    // 从 max_range 构造（兼容旧 Level 系统）
    static ValueSpec from_max_range(uint32_t max_range) {
        // max_range = 2^bits - 1 → bits = log2(max_range + 1)
        if (max_range == 0) return ValueSpec(0);
        // 0xFFFFFFFF 时 max_range+1 会 uint32 溢出为 0（审计 V-1），早返回
        if (max_range == 0xFFFFFFFFu) return ValueSpec(32);
        uint32_t v = max_range + 1;
        uint8_t b = 0;
        while (v > 1) { v >>= 1; b++; }
        // 验证 max_range 是 2^bits - 1
        if ((1u << b) - 1 != max_range) {
            // 非 2^n-1，向上取整
            b++;
        }
        return ValueSpec(b);
    }

    // 转换为 max_range（双向兼容）
    constexpr uint32_t to_max_range() const {
        return bits >= 32 ? 0xFFFFFFFF : ((1u << bits) - 1);
    }

    // 比较
    bool operator==(const ValueSpec& o) const { return bits == o.bits && scale == o.scale; }
    bool operator!=(const ValueSpec& o) const { return !(*this == o); }
};

} // namespace sgn
