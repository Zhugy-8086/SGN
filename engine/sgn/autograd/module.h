// module.h - Module 基类 + Parameter/Buffer 包装类
//
// 框架双模式系统（framework_dual_mode_design.md v2）Phase 1：
//   稳定模式的核心抽象：参数注册、状态管理、序列化、模式切换。
//
// 设计要点：
//   1. Module 不管理 tape 生命周期 —— tape 由 record_scope() 统一管理
//   2. Parameter/Buffer 是 Tensor 轻量包装，通过 operator Tensor&() 隐式转换
//   3. Module 内部存储 NamedParam（直接存 Tensor），非 Parameter 对象
//   4. parameters() 递归收集：本层 → 遍历 children_ → 递归
//   5. named_parameters() 路径名：prefix + "." + name

#pragma once

#include "tensor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sgn_autograd {

// ============================================================================
// Parameter: 可学习参数包装（构造时自动 requires_grad = true）
// ============================================================================
class Parameter {
public:
    explicit Parameter(const std::vector<int64_t>& shape);
    explicit Parameter(const Tensor& t);

    // 隐式转换为 Tensor&，可无缝传给现有算子
    operator Tensor&() { return tensor_; }
    operator const Tensor&() const { return tensor_; }

    Tensor& tensor() { return tensor_; }
    const Tensor& tensor() const { return tensor_; }

private:
    Tensor tensor_;
};

// ============================================================================
// Buffer: 非学习状态缓冲区（构造时自动 requires_grad = false）
// ============================================================================
class Buffer {
public:
    explicit Buffer(const std::vector<int64_t>& shape);
    explicit Buffer(const Tensor& t);

    operator Tensor&() { return tensor_; }
    operator const Tensor&() const { return tensor_; }

    Tensor& tensor() { return tensor_; }
    const Tensor& tensor() const { return tensor_; }

private:
    Tensor tensor_;
};

}  // namespace sgn_autograd


namespace sgn::nn {

// ============================================================================
// Module: 神经网络模块基类
// ============================================================================
class Module {
public:
    Module() = default;
    virtual ~Module() = default;

    // 禁止拷贝（避免子模块所有权混乱）
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;

    // === 参数管理 ===
    // 注册可学习参数（name 为 None 时取消注册）
    void register_parameter(const std::string& name,
                            std::optional<sgn_autograd::Parameter> param);

    // 递归收集所有可学习参数（本层 + 子模块）
    std::vector<sgn_autograd::Tensor> parameters() const;

    // 递归收集所有命名参数（点分隔路径名，如 "features.0.weight"）
    std::vector<std::pair<std::string, sgn_autograd::Tensor>> named_parameters(
        const std::string& prefix = "") const;

    // === 缓冲区管理 ===
    void register_buffer(const std::string& name,
                         std::optional<sgn_autograd::Buffer> buf);

    std::vector<sgn_autograd::Tensor> buffers() const;

    std::vector<std::pair<std::string, sgn_autograd::Tensor>> named_buffers(
        const std::string& prefix = "") const;

    // === 子模块管理 ===
    void register_module(const std::string& name,
                         std::shared_ptr<Module> child);

    std::vector<std::shared_ptr<Module>> children() const { return children_; }

    // === 模式切换 ===
    void train(bool mode = true);   // 递归设置所有子模块
    void eval();
    bool is_training() const { return training_; }

    // === 梯度 ===
    void zero_grad();  // 清零参数梯度，不清 tape

    // === 序列化 ===
    // state_dict: dict[str, numpy.ndarray]，用 named_parameters() 的路径名作 key
    // 注意：返回类型依赖 pybind11，在 module_bindings.cpp 中实现
    // 这里声明虚函数，由 pybind11 trampoline 调用 C++ 实现

    // === forward ===
    // 纯虚函数，子类必须实现
    virtual sgn_autograd::Tensor forward(
        const std::vector<sgn_autograd::Tensor>& inputs) = 0;

    // 单输入便捷接口
    sgn_autograd::Tensor forward(sgn_autograd::Tensor input);

    // operator() → forward()（不管理 tape）
    sgn_autograd::Tensor operator()(
        const std::vector<sgn_autograd::Tensor>& inputs);
    sgn_autograd::Tensor operator()(sgn_autograd::Tensor input);

private:
    struct NamedParam {
        std::string name;
        sgn_autograd::Tensor tensor;
        bool requires_grad;  // true = Parameter, false = Buffer
    };

    std::vector<NamedParam> params_;          // 有序（本层参数）
    std::vector<std::shared_ptr<Module>> children_;
    std::vector<std::string> children_names_;  // 子模块注册名（与 children_ 平行）
    bool training_ = true;
};

}  // namespace sgn::nn