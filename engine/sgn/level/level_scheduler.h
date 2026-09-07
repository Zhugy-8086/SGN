#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <memory>
#include <optional>
#include <vector>
#include <utility>

#include "level_context.h"
#include "level_strategy.h"
#include "neuron_level_stats.h"
#include "level_constants.h"

namespace sgn {

// LevelScheduler: level 调度器
//
// 核心职责：
//   1. 定策略 - 给定运算类型和神经元，决定用什么 level
//   2. 管自适应 - 根据神经元训练历史，动态调整 level
//   3. 控输出 - 运算结果的 level 由策略决定，不由输入决定
//
// 当前迁移版本聚焦核心数学验证路径：
//   - 策略注册与管理
//   - level 查询
//   - 自适应检查
//   - 序列化/反序列化
//
// 辅助功能（保留在 Python 端）：
//   - max_range 旋钮表
//   - 映射函数
//   - 读取策略
//   - 元学习器
//   - 回调机制
//
// ⚠️ 线程安全（安全审计 2026-08-16 L1-3）：非线程安全——update_stats/step/
// get_or_create_stats 等会修改共享容器（unordered_map 的 emplace/rehash）。
// 并行训练（如 OpenMP 多线程同时 update 同一 scheduler）构成数据竞争，
// 调用方必须串行化（Level 是决策层，标量逻辑，无热路径并行需求——
// 见 project_memory "Level 调度器不需要指令集优化"同理）。
class LevelScheduler {
public:
    // adapt_interval 必须 > 0（==0 会使 check_adaptation 模零 UB——审计 L1-1）
    LevelScheduler(
        int32_t cache_size = 1024,
        int32_t adapt_interval = 50,
        double default_variance_threshold = 100.0);

    ~LevelScheduler() = default;

    // ============================================================
    // 策略管理
    // ============================================================

    // 注册策略
    void register_strategy(const std::string& name, std::shared_ptr<LevelStrategy> strategy);

    // 绑定神经元到策略
    void bind_neuron(int32_t neuron_id, const std::string& strategy_name);

    // 设置默认策略
    void set_default_strategy(const std::string& strategy_name);

    // 获取策略
    std::shared_ptr<LevelStrategy> get_strategy(const std::string& name) const;

    // ============================================================
    // Level 查询
    // ============================================================

    // 获取神经元当前 level
    int32_t get_level(int32_t neuron_id, LevelOperation operation = LevelOperation::ASSIGN) const;

    // 获取神经元统计
    const NeuronLevelStats* get_stats(int32_t neuron_id) const;

    // 获取或创建神经元统计
    NeuronLevelStats& get_or_create_stats(int32_t neuron_id);

    // ============================================================
    // 自适应
    // ============================================================

    // 更新统计（触发自适应检查）
    // grad_variance > 0 时累积方案 D STE 梯度方差历史（默认 0=不更新）
    void update_stats(int32_t neuron_id, int32_t match, bool verified,
                      double grad_variance = 0.0);

    // 步进计数器（用于自适应检查间隔）
    void step();

    // ============================================================
    // 序列化
    // ============================================================

    // 序列化全部状态
    std::unordered_map<std::string, std::string> to_dict() const;

    // 反序列化
    static std::unique_ptr<LevelScheduler> from_dict(
        const std::unordered_map<std::string, std::string>& d);

    // ============================================================
    // Accessors
    // ============================================================

    int32_t cache_size() const { return cache_size_; }
    int32_t adapt_interval() const { return adapt_interval_; }
    double default_variance_threshold() const { return default_variance_threshold_; }
    int32_t step_counter() const { return step_counter_; }

private:
    // 执行自适应检查
    void check_adaptation(int32_t neuron_id, NeuronLevelStats& stats);

    // 策略注册表
    std::unordered_map<std::string, std::shared_ptr<LevelStrategy>> strategies_;

    // 神经元 → 策略名 映射
    std::unordered_map<int32_t, std::string> neuron_strategy_;

    // 神经元统计
    std::unordered_map<int32_t, NeuronLevelStats> neuron_stats_;

    // 默认策略名
    std::string default_strategy_name_;

    // 自适应检查间隔
    int32_t adapt_interval_;
    int32_t step_counter_;

    // 默认方差阈值
    double default_variance_threshold_;

    // 缓存大小
    int32_t cache_size_;
};

} // namespace sgn