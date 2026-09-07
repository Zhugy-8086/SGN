#pragma once
#include <cstdint>
#include <cmath>
#include <string>
#include <unordered_map>
#include <memory>
#include <optional>

#include "value_spec.h"
#include "level_math.h"

namespace sgn {

// Level 操作类型
enum class LevelOperation : uint8_t {
    ADD = 0,
    SUB = 1,
    MUL = 2,
    COMPARE = 3,
    ASSIGN = 4,
};

// LevelContext: 运算上下文
//
// 字段：
//   target_level: 目标 level（0-255）
//   max_range: 最大范围（2^bits - 1）
//   bits: 位宽，-1 表示未设置
//   operation: 操作类型
//   source: 来源字符串
struct LevelContext {
    int32_t target_level;
    int32_t max_range;
    int32_t bits;        // -1 表示未设置
    LevelOperation operation;
    std::string source;

    LevelContext()
        : target_level(0), max_range(255), bits(-1),
          operation(LevelOperation::ASSIGN), source("") {}

    LevelContext(int32_t target_level, int32_t max_range,
                 int32_t bits = -1,
                 LevelOperation op = LevelOperation::ASSIGN,
                 const std::string& source = "")
        : target_level(target_level), max_range(max_range),
          bits(bits), operation(op), source(source) {}

    // 从 dict 反序列化（兼容旧数据：仅含 max_range 时自动反推 bits）。
    // corrupt 输入（非数字）会抛未捕获异常——逐字段 try/catch 失败保持默认
    // （安全审计 2026-08-16 L2-1）
    static LevelContext from_dict(const std::unordered_map<std::string, std::string>& d) {
        LevelContext ctx;
        auto it = d.find("target_level");
        if (it != d.end()) {
            try { ctx.target_level = std::stoi(it->second); } catch (...) {}
        }
        it = d.find("max_range");
        if (it != d.end()) {
            try { ctx.max_range = std::stoi(it->second); } catch (...) {}
        }
        it = d.find("bits");
        if (it != d.end()) {
            try { ctx.bits = std::stoi(it->second); } catch (...) {}
        } else {
            // 旧数据：从 max_range 反推 bits
            ctx.bits = -1;  // 由 Python 端处理反推
        }
        it = d.find("operation");
        if (it != d.end()) {
            static const std::unordered_map<std::string, LevelOperation> op_map = {
                {"ADD", LevelOperation::ADD},
                {"SUB", LevelOperation::SUB},
                {"MUL", LevelOperation::MUL},
                {"COMPARE", LevelOperation::COMPARE},
                {"ASSIGN", LevelOperation::ASSIGN},
            };
            auto op_it = op_map.find(it->second);
            if (op_it != op_map.end()) ctx.operation = op_it->second;
        }
        it = d.find("source");
        if (it != d.end()) ctx.source = it->second;
        return ctx;
    }

    // 序列化为 dict
    std::unordered_map<std::string, std::string> to_dict() const {
        std::unordered_map<std::string, std::string> d;
        d["target_level"] = std::to_string(target_level);
        d["max_range"] = std::to_string(max_range);
        d["bits"] = std::to_string(bits);
        switch (operation) {
            case LevelOperation::ADD: d["operation"] = "ADD"; break;
            case LevelOperation::SUB: d["operation"] = "SUB"; break;
            case LevelOperation::MUL: d["operation"] = "MUL"; break;
            case LevelOperation::COMPARE: d["operation"] = "COMPARE"; break;
            case LevelOperation::ASSIGN: d["operation"] = "ASSIGN"; break;
        }
        d["source"] = source;
        return d;
    }
};

// ============================================================
// UpgradedLevelContext: LevelContext 的 bits 升级版（Stage 3.0.4 Task 4.1/4.2）
// ============================================================
// 在 LevelContext 基础上增加：
//   1. bits ↔ max_range 双向自动同步（bits 是主杠杆，exp19/25/34）
//   2. level_f / level_b 前反向差异化精度接口（optional ValueSpec，None=跟随 bits）
//   3. get_effective_bits(direction) 优先级：level_f/level_b > bits > max_range 反推
//
// 对应 v5.1.9 Python 版 UpgradedLevelContext（legacy/level_scheduler/），
// C++ 实现于 2026-08-16（安全审计修复批次 2.5，消除 test_level_upgrade 17 失败）。
class UpgradedLevelContext : public LevelContext {
public:
    UpgradedLevelContext() : LevelContext() {
        sync_from_max_range_impl(false);
    }

    // bits 入口：自动同步 max_range
    explicit UpgradedLevelContext(int32_t bits)
        : LevelContext(0, 255, bits) {
        max_range = static_cast<int32_t>(bits_to_max_range(bits));
    }

    // max_range 入口：自动反推 bits（旧数据兼容路径）
    explicit UpgradedLevelContext(int32_t max_range_in, bool /*disambiguate*/)
        : LevelContext(0, max_range_in, -1) {
        sync_from_max_range_impl(false);
    }

    // === level_f / level_b 接口（Task 4.2）===

    bool has_level_f() const { return level_f_.has_value(); }
    bool has_level_b() const { return level_b_.has_value(); }

    const ValueSpec* level_f() const { return level_f_ ? &level_f_.value() : nullptr; }
    const ValueSpec* level_b() const { return level_b_ ? &level_b_.value() : nullptr; }

    void set_level_f(const ValueSpec& spec) { level_f_ = spec; }
    void set_level_b(const ValueSpec& spec) { level_b_ = spec; }
    void clear_level_f() { level_f_.reset(); }
    void clear_level_b() { level_b_.reset(); }

    // === bits 同步接口（Task 4.1）===

    // 设置 bits 并同步 max_range
    void set_bits(int32_t new_bits) {
        bits = new_bits;
        max_range = static_cast<int32_t>(bits_to_max_range(new_bits));
    }

    // 从 max_range 反推 bits（非 2^n-1 时向上取整）。
    // 返回是否为精确匹配；warn=true 且非精确时经 Python 绑定层发出
    // DeprecationWarning（C++ 侧返回标志，由绑定层决定警告方式）。
    bool sync_from_max_range(bool warn) {
        (void)warn;  // 警告由绑定层处理（见 level_bindings.cpp）
        return sync_from_max_range_impl(true);
    }

    // === effective 优先级（Task 4.2）===
    // 方向 "forward" → level_f，"backward" → level_b；均未设置时用 bits。

    int32_t get_effective_bits(const std::string& direction) const {
        if (direction == "forward" && level_f_) return level_f_->bits;
        if (direction == "backward" && level_b_) return level_b_->bits;
        return bits;
    }

    int32_t get_effective_max_range(const std::string& direction) const {
        return static_cast<int32_t>(bits_to_max_range(get_effective_bits(direction)));
    }

    // === 序列化（Task 4.4，兼容旧数据）===

    std::unordered_map<std::string, std::string> to_dict() const {
        auto d = LevelContext::to_dict();
        // level_f/level_b 编码为 "bits,scale_int"（scale 枚举转 int）
        if (level_f_) {
            d["level_f"] = std::to_string(level_f_->bits) + "," +
                           std::to_string(static_cast<int>(level_f_->scale));
        }
        if (level_b_) {
            d["level_b"] = std::to_string(level_b_->bits) + "," +
                           std::to_string(static_cast<int>(level_b_->scale));
        }
        return d;
    }

    static UpgradedLevelContext from_dict(
        const std::unordered_map<std::string, std::string>& d) {
        UpgradedLevelContext ctx;  // 默认构造已从 max_range=255 反推
        // 复用基类解析（含 try/catch 保护）
        LevelContext base = LevelContext::from_dict(d);
        ctx.target_level = base.target_level;
        ctx.max_range = base.max_range;
        ctx.bits = base.bits;
        ctx.operation = base.operation;
        ctx.source = base.source;

        // bits 缺失时从 max_range 反推（旧数据兼容）
        if (ctx.bits < 0) {
            ctx.sync_from_max_range_impl(false);
        }

        // level_f / level_b
        auto parse_spec = [](const std::string& s) -> std::optional<ValueSpec> {
            size_t comma = s.find(',');
            if (comma == std::string::npos) return std::nullopt;
            try {
                uint8_t b = static_cast<uint8_t>(std::stoi(s.substr(0, comma)));
                int sc = std::stoi(s.substr(comma + 1));
                return ValueSpec(b, static_cast<ScaleFn>(sc));
            } catch (...) {
                return std::nullopt;
            }
        };
        auto it = d.find("level_f");
        if (it != d.end()) ctx.level_f_ = parse_spec(it->second);
        it = d.find("level_b");
        if (it != d.end()) ctx.level_b_ = parse_spec(it->second);
        return ctx;
    }

private:
    // max_range → bits 反推实现。返回是否精确匹配 2^n-1。
    bool sync_from_max_range_impl(bool set_bits_flag) {
        if (max_range <= 0) return true;
        int32_t b = max_range_to_bits(static_cast<int64_t>(max_range));
        bool exact = (bits_to_max_range(b) == max_range);
        if (set_bits_flag || bits < 0) {
            bits = b;
        }
        return exact;
    }

    std::optional<ValueSpec> level_f_;  // 前向差异化精度（None=跟随 bits）
    std::optional<ValueSpec> level_b_;  // 反向差异化精度（None=跟随 bits）
};

} // namespace sgn