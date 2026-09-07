#include "level_scheduler.h"
#include <stdexcept>
#include <algorithm>

namespace sgn {

LevelScheduler::LevelScheduler(
    int32_t cache_size,
    int32_t adapt_interval,
    double default_variance_threshold)
    : adapt_interval_(adapt_interval > 0 ? adapt_interval : 50)
    , step_counter_(0)
    , default_variance_threshold_(default_variance_threshold)
    , cache_size_(cache_size) {
    // adapt_interval <= 0 会使 check_adaptation 模零 UB（审计 L1-1）——
    // 非法值回退默认 50

    // 注册内置策略
    register_strategy("standard", std::make_shared<StandardStrategy>());
    register_strategy("adaptive", std::make_shared<AdaptiveStrategy>());
    set_default_strategy("standard");
}

// ============================================================
// 策略管理
// ============================================================

void LevelScheduler::register_strategy(
    const std::string& name,
    std::shared_ptr<LevelStrategy> strategy) {
    strategies_[name] = std::move(strategy);
}

void LevelScheduler::bind_neuron(int32_t neuron_id, const std::string& strategy_name) {
    if (strategies_.find(strategy_name) == strategies_.end()) {
        throw std::runtime_error("Unknown strategy: " + strategy_name);
    }
    neuron_strategy_[neuron_id] = strategy_name;

    // 确保神经元统计存在
    if (neuron_stats_.find(neuron_id) == neuron_stats_.end()) {
        neuron_stats_.emplace(neuron_id, NeuronLevelStats(neuron_id));
    }
}

void LevelScheduler::set_default_strategy(const std::string& strategy_name) {
    if (strategies_.find(strategy_name) == strategies_.end()) {
        throw std::runtime_error("Unknown strategy: " + strategy_name);
    }
    default_strategy_name_ = strategy_name;
}

std::shared_ptr<LevelStrategy> LevelScheduler::get_strategy(const std::string& name) const {
    auto it = strategies_.find(name);
    if (it != strategies_.end()) {
        return it->second;
    }
    return nullptr;
}

// ============================================================
// Level 查询
// ============================================================

int32_t LevelScheduler::get_level(int32_t neuron_id, LevelOperation operation) const {
    // 查找神经元绑定的策略
    auto strategy_it = neuron_strategy_.find(neuron_id);
    std::shared_ptr<LevelStrategy> strategy;

    if (strategy_it != neuron_strategy_.end()) {
        auto strat_it = strategies_.find(strategy_it->second);
        if (strat_it != strategies_.end()) {
            strategy = strat_it->second;
        }
    }

    // 回退到默认策略
    if (!strategy) {
        auto default_it = strategies_.find(default_strategy_name_);
        if (default_it != strategies_.end()) {
            strategy = default_it->second;
        }
    }

    if (!strategy) {
        return LevelConstants::LEVEL_DEFAULT;
    }

    // 查找神经元统计
    auto stats_it = neuron_stats_.find(neuron_id);
    const NeuronLevelStats* stats = (stats_it != neuron_stats_.end()) ? &stats_it->second : nullptr;

    return strategy->get_level_for_operation(operation, neuron_id, stats);
}

const NeuronLevelStats* LevelScheduler::get_stats(int32_t neuron_id) const {
    auto it = neuron_stats_.find(neuron_id);
    if (it != neuron_stats_.end()) {
        return &it->second;
    }
    return nullptr;
}

NeuronLevelStats& LevelScheduler::get_or_create_stats(int32_t neuron_id) {
    auto it = neuron_stats_.find(neuron_id);
    if (it != neuron_stats_.end()) {
        return it->second;
    }
    auto result = neuron_stats_.emplace(neuron_id, NeuronLevelStats(neuron_id));
    return result.first->second;
}

// ============================================================
// 自适应
// ============================================================

void LevelScheduler::update_stats(int32_t neuron_id, int32_t match, bool verified,
                                  double grad_variance) {
    auto& stats = get_or_create_stats(neuron_id);
    stats.update(match, verified, grad_variance);

    // 步进计数器
    step_counter_++;

    // 自适应检查
    check_adaptation(neuron_id, stats);
}

void LevelScheduler::step() {
    step_counter_++;
}

void LevelScheduler::check_adaptation(int32_t neuron_id, NeuronLevelStats& stats) {
    // 检查间隔（adapt_interval_ <= 0 时模零 UB——审计 L1-1，守卫防御
    // from_dict 等路径注入的非法值）
    if (adapt_interval_ <= 0) {
        return;
    }
    if (step_counter_ % adapt_interval_ != 0) {
        return;
    }

    // 查找神经元绑定的策略
    auto strategy_it = neuron_strategy_.find(neuron_id);
    std::shared_ptr<LevelStrategy> strategy;

    if (strategy_it != neuron_strategy_.end()) {
        auto strat_it = strategies_.find(strategy_it->second);
        if (strat_it != strategies_.end()) {
            strategy = strat_it->second;
        }
    }

    // 回退到默认策略
    if (!strategy) {
        auto default_it = strategies_.find(default_strategy_name_);
        if (default_it != strategies_.end()) {
            strategy = default_it->second;
        }
    }

    if (!strategy) return;

    // v5.1.9-fix 三段式优先级：
    // 1. demotion（验证率差主动降级）— 最高优先级
    // 2. adaptation（方差触发的升/降 level）— 次优先级
    auto suggested = strategy->suggest_demotion(stats);

    // 优先级 2 - 方差触发的自适应（升或降）
    if (!suggested.has_value()) {
        suggested = strategy->suggest_adaptation(stats);
    }

    if (suggested.has_value()) {
        int32_t new_level = suggested.value();
        if (new_level != stats.current_level) {
            stats.current_level = new_level;
            stats.level_change_count++;
            if (new_level > stats.peak_level) {
                stats.peak_level = new_level;
            }
        }
    }
}

// ============================================================
// 序列化
// ============================================================

std::unordered_map<std::string, std::string> LevelScheduler::to_dict() const {
    std::unordered_map<std::string, std::string> d;
    d["default_strategy"] = default_strategy_name_;
    d["adapt_interval"] = std::to_string(adapt_interval_);
    d["step_counter"] = std::to_string(step_counter_);
    d["default_variance_threshold"] = std::to_string(default_variance_threshold_);

    // 序列化神经元统计
    std::string stats_str;
    for (const auto& [nid, stats] : neuron_stats_) {
        if (!stats_str.empty()) stats_str += ";";
        auto sd = stats.to_dict();
        std::string entry;
        for (const auto& [k, v] : sd) {
            if (!entry.empty()) entry += ",";
            entry += k + "=" + v;
        }
        stats_str += entry;
    }
    d["neuron_stats"] = stats_str;

    // 序列化神经元策略绑定
    std::string binding_str;
    for (const auto& [nid, sname] : neuron_strategy_) {
        if (!binding_str.empty()) binding_str += ";";
        binding_str += std::to_string(nid) + "=" + sname;
    }
    d["neuron_strategy"] = binding_str;

    return d;
}

std::unique_ptr<LevelScheduler> LevelScheduler::from_dict(
    const std::unordered_map<std::string, std::string>& d) {

    auto scheduler = std::make_unique<LevelScheduler>();

    auto it = d.find("default_strategy");
    if (it != d.end()) {
        try {
            scheduler->set_default_strategy(it->second);
        } catch (...) {
            // 默认策略不可用时保持默认
        }
    }

    // 数值字段反序列化：corrupt 输入（非数字/超范围）会抛未捕获的
    // std::invalid_argument/out_of_range——逐字段 try/catch，失败保持默认
    // （审计 L1-2）
    it = d.find("adapt_interval");
    if (it != d.end()) {
        try {
            int32_t v = std::stoi(it->second);
            if (v > 0) scheduler->adapt_interval_ = v;  // <=0 回绝（模零 UB，L1-1）
        } catch (...) {}
    }

    it = d.find("step_counter");
    if (it != d.end()) {
        try {
            scheduler->step_counter_ = std::stoi(it->second);
        } catch (...) {}
    }

    it = d.find("default_variance_threshold");
    if (it != d.end()) {
        try {
            scheduler->default_variance_threshold_ = std::stod(it->second);
        } catch (...) {}
    }

    // 反序列化神经元统计
    it = d.find("neuron_stats");
    if (it != d.end() && !it->second.empty()) {
        // 格式: "neuron_id=0,current_level=2,...;neuron_id=1,current_level=3,..."
        std::string stats_str = it->second;
        size_t pos = 0;
        while (pos < stats_str.size()) {
            size_t semicolon = stats_str.find(';', pos);
            std::string entry = stats_str.substr(pos, semicolon - pos);
            if (!entry.empty()) {
                std::unordered_map<std::string, std::string> fields;
                size_t epos = 0;
                while (epos < entry.size()) {
                    size_t comma = entry.find(',', epos);
                    std::string kv = entry.substr(epos, comma - epos);
                    size_t eq = kv.find('=');
                    if (eq != std::string::npos) {
                        fields[kv.substr(0, eq)] = kv.substr(eq + 1);
                    }
                    if (comma == std::string::npos) break;
                    epos = comma + 1;
                }
                if (!fields.empty()) {
                    auto stats = NeuronLevelStats::from_dict(fields);
                    scheduler->neuron_stats_[stats.neuron_id] = stats;
                }
            }
            if (semicolon == std::string::npos) break;
            pos = semicolon + 1;
        }
    }

    // 反序列化神经元策略绑定
    it = d.find("neuron_strategy");
    if (it != d.end() && !it->second.empty()) {
        std::string binding_str = it->second;
        size_t pos = 0;
        while (pos < binding_str.size()) {
            size_t semicolon = binding_str.find(';', pos);
            std::string entry = binding_str.substr(pos, semicolon - pos);
            size_t eq = entry.find('=');
            if (eq != std::string::npos) {
                int32_t nid = std::stoi(entry.substr(0, eq));
                std::string sname = entry.substr(eq + 1);
                scheduler->neuron_strategy_[nid] = sname;
            }
            if (semicolon == std::string::npos) break;
            pos = semicolon + 1;
        }
    }

    return scheduler;
}

} // namespace sgn