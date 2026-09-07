// module.cpp - Module 基类 + Parameter/Buffer 实现
//
// 框架双模式系统（framework_dual_mode_design.md v2）Phase 1。
//
// 实现说明：
//   - Parameter/Buffer 构造时自动设置 requires_grad
//   - register_parameter/register_buffer 支持 nullopt 取消注册
//   - parameters()/named_parameters() 递归收集子模块参数
//   - zero_grad() 通过 Tape::grad() 查询梯度并清零
//   - train()/eval() 递归设置所有子模块

#include "module.h"
#include "autograd.h"  // Tape::grad()

#include <algorithm>
#include <cstring>
#include <stdexcept>

// ============================================================================
// sgn_autograd::Parameter 实现
// ============================================================================
namespace sgn_autograd {

Parameter::Parameter(const std::vector<int64_t>& shape)
    : tensor_(shape) {
    tensor_.set_requires_grad(true);
}

Parameter::Parameter(const Tensor& t)
    : tensor_(t) {
    // 参数张量必须连续：算子内核（matmul/conv 等）按连续布局寻址，
    // 非连续参数（如 transpose 视图）会静默读错数据
    // （project_memory 约束 + 安全审计 2026-08-16 M-1）
    if (!tensor_.is_contiguous()) {
        throw std::invalid_argument(
            "Parameter: tensor must be contiguous (call .contiguous() first)");
    }
    tensor_.set_requires_grad(true);
}

// ============================================================================
// sgn_autograd::Buffer 实现
// ============================================================================
Buffer::Buffer(const std::vector<int64_t>& shape)
    : tensor_(shape) {
    tensor_.set_requires_grad(false);
}

Buffer::Buffer(const Tensor& t)
    : tensor_(t) {
    // 同 Parameter：非连续张量拒绝注册（审计 M-1）
    if (!tensor_.is_contiguous()) {
        throw std::invalid_argument(
            "Buffer: tensor must be contiguous (call .contiguous() first)");
    }
    tensor_.set_requires_grad(false);
}

}  // namespace sgn_autograd


// ============================================================================
// sgn::nn::Module 实现
// ============================================================================
namespace sgn::nn {

// === 参数管理 ===

void Module::register_parameter(const std::string& name,
                                std::optional<sgn_autograd::Parameter> param) {
    // 先移除同名参数（如果已存在）
    auto it = std::find_if(params_.begin(), params_.end(),
        [&name](const NamedParam& np) { return np.name == name; });
    if (it != params_.end()) {
        params_.erase(it);
    }

    if (param.has_value()) {
        NamedParam np;
        np.name = name;
        np.tensor = param->tensor();
        np.requires_grad = true;
        // 防御：非连续张量拒绝注册（Parameter 构造已检查，此处兜底——审计 M-1）
        if (!np.tensor.is_contiguous()) {
            throw std::invalid_argument(
                "register_parameter: tensor must be contiguous");
        }
        params_.push_back(std::move(np));
    }
    // param == nullopt → 移除参数（已在上方 erase，不添加）
}

std::vector<sgn_autograd::Tensor> Module::parameters() const {
    std::vector<sgn_autograd::Tensor> result;

    // 本层参数
    for (const auto& np : params_) {
        if (np.requires_grad) {
            result.push_back(np.tensor);
        }
    }

    // 递归收集子模块参数
    for (const auto& child : children_) {
        auto child_params = child->parameters();
        result.insert(result.end(), child_params.begin(), child_params.end());
    }

    return result;
}

std::vector<std::pair<std::string, sgn_autograd::Tensor>>
Module::named_parameters(const std::string& prefix) const {
    std::vector<std::pair<std::string, sgn_autograd::Tensor>> result;

    // 本层参数：路径名 = prefix + name
    for (const auto& np : params_) {
        if (np.requires_grad) {
            std::string full_name = prefix.empty() ? np.name : prefix + "." + np.name;
            result.emplace_back(full_name, np.tensor);
        }
    }

    // 递归收集子模块参数
    for (size_t i = 0; i < children_.size(); ++i) {
        const std::string& child_name = children_names_[i];
        std::string child_prefix = prefix.empty()
            ? child_name
            : prefix + "." + child_name;
        auto child_named = children_[i]->named_parameters(child_prefix);
        result.insert(result.end(), child_named.begin(), child_named.end());
    }

    return result;
}

// === 缓冲区管理 ===

void Module::register_buffer(const std::string& name,
                             std::optional<sgn_autograd::Buffer> buf) {
    auto it = std::find_if(params_.begin(), params_.end(),
        [&name](const NamedParam& np) { return np.name == name; });
    if (it != params_.end()) {
        params_.erase(it);
    }

    if (buf.has_value()) {
        NamedParam np;
        np.name = name;
        np.tensor = buf->tensor();
        np.requires_grad = false;
        // 防御：非连续张量拒绝注册（审计 M-1）
        if (!np.tensor.is_contiguous()) {
            throw std::invalid_argument(
                "register_buffer: tensor must be contiguous");
        }
        params_.push_back(std::move(np));
    }
}

std::vector<sgn_autograd::Tensor> Module::buffers() const {
    std::vector<sgn_autograd::Tensor> result;

    for (const auto& np : params_) {
        if (!np.requires_grad) {
            result.push_back(np.tensor);
        }
    }

    for (const auto& child : children_) {
        auto child_bufs = child->buffers();
        result.insert(result.end(), child_bufs.begin(), child_bufs.end());
    }

    return result;
}

std::vector<std::pair<std::string, sgn_autograd::Tensor>>
Module::named_buffers(const std::string& prefix) const {
    std::vector<std::pair<std::string, sgn_autograd::Tensor>> result;

    for (const auto& np : params_) {
        if (!np.requires_grad) {
            std::string full_name = prefix.empty() ? np.name : prefix + "." + np.name;
            result.emplace_back(full_name, np.tensor);
        }
    }

    for (size_t i = 0; i < children_.size(); ++i) {
        const std::string& child_name = children_names_[i];
        std::string child_prefix = prefix.empty()
            ? child_name
            : prefix + "." + child_name;
        auto child_named = children_[i]->named_buffers(child_prefix);
        result.insert(result.end(), child_named.begin(), child_named.end());
    }

    return result;
}

// === 子模块管理 ===

void Module::register_module(const std::string& name,
                             std::shared_ptr<Module> child) {
    // 重名子模块会导致 named_parameters 产生重复路径、load_state_dict 键
    // 冲突，注册时即拒绝（安全审计 2026-08-16 M-5）
    if (std::find(children_names_.begin(), children_names_.end(), name) !=
        children_names_.end()) {
        throw std::invalid_argument(
            "register_module: duplicate child name '" + name + "'");
    }
    children_.push_back(std::move(child));
    children_names_.push_back(name);
}

// === 模式切换 ===

void Module::train(bool mode) {
    training_ = mode;
    for (auto& child : children_) {
        child->train(mode);
    }
}

void Module::eval() {
    train(false);
}

// === 梯度 ===

void Module::zero_grad() {
    auto& tape = sgn_autograd::Tape::current();
    for (auto& p : parameters()) {
        // 显式可变接口（grad_mutable），替代 const_cast 突破 const
        // （安全审计 2026-08-16 M-3）；numel()==0 的 entry 一律跳过（AG-1）
        auto* g = tape.grad_mutable(p.id());
        if (g != nullptr && g->numel() > 0 && g->data() != nullptr) {
            std::memset(g->data(), 0, g->numel() * sizeof(float));
        }
    }
}

// === forward ===

sgn_autograd::Tensor Module::forward(sgn_autograd::Tensor input) {
    return forward(std::vector<sgn_autograd::Tensor>{std::move(input)});
}

sgn_autograd::Tensor Module::operator()(
    const std::vector<sgn_autograd::Tensor>& inputs) {
    return forward(inputs);
}

sgn_autograd::Tensor Module::operator()(sgn_autograd::Tensor input) {
    return forward(std::move(input));
}

}  // namespace sgn::nn