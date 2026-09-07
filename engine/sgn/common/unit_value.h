#pragma once
#include <cstdint>
#include <cmath>
#include <vector>
#include "value_spec.h"

namespace sgn {

// UnitValue: 统一单位值
//
// 覆盖 per-tensor / per-layer 场景，减少 HC/MSint/Level 三库的重复适配。
//
// 设计依据（基于 HC/MSint/Level 单位值场景调研）：
//   - HC8WeightSchema (Py): dequantize(q, scale) = q * scale，per-tensor 标量 scale
//   - HC16 per-channel (C): dequantize(q[i][j], scales[i]) = q[i][j] * scales[i]
//   - Level: map(raw, max_range)，max_range 通过 ValueSpec.to_max_range() 桥接
//   - MSint: 无 scale（纯位运算），UnitValue 仅描述 slot 精度
//
// raw 是有符号整数（HC8WeightSchema 的 q 值，不含 offset）。
// scale 是浮点 scale 值，由外部提供（如 HC8: max(|w|)/127）。
struct UnitValue {
    int64_t raw;        // 原始整数值（有符号，容纳 int8/int16/int32）
    ValueSpec spec;     // 精度规格（bits + ScaleFn）

    // 构造
    constexpr UnitValue(int64_t r = 0, ValueSpec s = ValueSpec())
        : raw(r), spec(s) {}

    // 反量化：raw → float
    // 公式：x = raw * scale （HC8WeightSchema.dequantize 模式）
    // scale: 浮点 scale 值（per-tensor 标量或 per-channel 数组元素）
    float to_float(float scale) const {
        return static_cast<float>(raw) * scale;
    }

    // 量化：float → UnitValue
    // 公式：raw = round(x / scale).clamp(-signed_max, signed_max)
    // signed_max = 2^(bits-1) - 1（对称有符号范围，HC8: 127, HC16: 32767）
    //
    // x:     输入浮点值
    // scale: 浮点 scale 值（scale=0 时返回 raw=0，避免除零）
    // spec:  精度规格（决定 clamp 范围）
    static UnitValue from_float(float x, float scale, ValueSpec spec) {
        if (scale == 0.0f) return UnitValue(0, spec);
        int64_t smax = signed_max_for_bits(spec.bits);
        // bits > 24 时 smax 超出 float 精确整数域（2^24），float clamp 会丢
        // 精度——改用 double 中间量（HC 实际上限 24 bits，此分支为防御；
        // bits <= 24 的路径保持原 float 计算不变以确保 bit-exact——审计 V-2）
        if (spec.bits > 24) {
            double qd = std::round(static_cast<double>(x) / scale);
            if (qd > static_cast<double>(smax)) qd = static_cast<double>(smax);
            if (qd < -static_cast<double>(smax)) qd = -static_cast<double>(smax);
            return UnitValue(static_cast<int64_t>(qd), spec);
        }
        float q = std::round(x / scale);
        if (q > static_cast<float>(smax)) q = static_cast<float>(smax);
        if (q < -static_cast<float>(smax)) q = -static_cast<float>(smax);
        return UnitValue(static_cast<int64_t>(q), spec);
    }

    // 有符号对称最大值：2^(bits-1) - 1
    // HC8: 127, HC12: 2047, HC16: 32767, HC20: 524287, HC24: 8388607
    static constexpr int64_t signed_max_for_bits(uint8_t bits) {
        if (bits == 0) return 0;
        if (bits >= 64) return INT64_MAX;
        return (1LL << (bits - 1)) - 1;
    }

    // 比较
    bool operator==(const UnitValue& o) const { return raw == o.raw && spec == o.spec; }
    bool operator!=(const UnitValue& o) const { return !(*this == o); }
};

// per-layer 数组（每层一个 UnitValue，支持 per-layer bits 分配）
// per-tensor 场景用单个 UnitValue；per-layer 场景用 UnitValueArray
using UnitValueArray = std::vector<UnitValue>;

} // namespace sgn
