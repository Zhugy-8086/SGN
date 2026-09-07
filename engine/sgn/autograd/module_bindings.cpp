// module_bindings.cpp - Module 基类 pybind11 绑定
//
// 框架双模式系统（framework_dual_mode_design.md v2）Phase 1。
//
// 绑定内容：
//   - sgn.nn.Module（含 PyModule trampoline，支持 Python 子类化）
//   - sgn.nn.Parameter / sgn.nn.Buffer（Tensor 轻量包装）
//   - 初始化辅助函数（kaiming_uniform_ / uniform_ / fill_）
//
// 设计要点：
//   - Module 通过 trampoline 支持 Python 子类重写 forward
//   - state_dict/load_state_dict 在 Python 端实现（通过 named_parameters 构建）
//   - Parameter/Buffer 提供隐式转换，可直接传给 autograd 算子

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "module.h"
#include "tensor.h"
#include "autograd.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

// ============================================================================
// 辅助：Tensor → numpy array（拷贝）
// ============================================================================
static py::array_t<float> tensor_to_numpy_copy(const sgn_autograd::Tensor& t) {
    std::vector<py::ssize_t> shape(t.ndim());
    for (size_t i = 0; i < t.ndim(); ++i) {
        shape[i] = static_cast<py::ssize_t>(t.shape()[i]);
    }
    py::array_t<float> arr(shape);
    auto buf = arr.request();
    if (!t.is_contiguous()) {
        sgn_autograd::Tensor c = t.contiguous();
        std::memcpy(buf.ptr, c.data(), t.numel() * sizeof(float));
    } else {
        std::memcpy(buf.ptr, t.data(), t.numel() * sizeof(float));
    }
    return arr;
}

// ============================================================================
// PyModule: pybind11 trampoline 类，支持 Python 子类化 Module
// ============================================================================
class PyModule : public sgn::nn::Module {
public:
    using sgn::nn::Module::Module;  // 继承构造函数

    sgn_autograd::Tensor forward(
        const std::vector<sgn_autograd::Tensor>& inputs) override {
        PYBIND11_OVERRIDE_PURE(
            sgn_autograd::Tensor,      // 返回类型
            sgn::nn::Module,           // 父类
            forward,                   // 函数名
            inputs                     // 参数
        );
    }
};

// ============================================================================
// 注册函数：由 placeholder.cpp 调用
// ============================================================================
void register_nn(py::module_& m) {
    // 创建子模块 sgn.nn
    py::module_ nn_m = m.def_submodule("nn",
        "神经网络模块（稳定模式）\n\n"
        "提供 Module 基类、Parameter/Buffer 包装、标准层（Conv2d/Linear/BN 等）。\n"
        "与 sgn.autograd 自由函数（调试模式）共享底层引擎，可混合使用。");

    // ========================================================================
    // Parameter 类
    // ========================================================================
    py::class_<sgn_autograd::Parameter>(nn_m, "Parameter",
        "可学习参数包装（构造时自动 requires_grad=True）\n\n"
        "内部是普通 Tensor，通过隐式转换可无缝传给 autograd 算子。")
        .def(py::init([](const std::vector<int64_t>& shape) {
            return sgn_autograd::Parameter(shape);
        }), py::arg("shape"), "按形状分配（元素初始化为 0）")
        .def(py::init([](const sgn_autograd::Tensor& t) {
            return sgn_autograd::Parameter(t);
        }), py::arg("tensor"), "从已有 Tensor 构造（自动设置 requires_grad=True）")
        // 属性访问
        .def_property_readonly("shape", [](const sgn_autograd::Parameter& p) {
            return p.tensor().shape();
        })
        .def_property_readonly("ndim", [](const sgn_autograd::Parameter& p) {
            return p.tensor().ndim();
        })
        .def_property_readonly("numel", [](const sgn_autograd::Parameter& p) {
            return p.tensor().numel();
        })
        .def_property_readonly("requires_grad", [](const sgn_autograd::Parameter& p) {
            return p.tensor().requires_grad();
        })
        // 梯度
        .def_property_readonly("grad", [](const sgn_autograd::Parameter& p) -> py::object {
            const sgn_autograd::Tensor* g =
                sgn_autograd::Tape::current().grad(p.tensor().id());
            if (g == nullptr) {
                return py::none();
            }
            return py::reinterpret_steal<py::object>(
                tensor_to_numpy_copy(*g).release().ptr());
        })
        // numpy 互转
        .def("to_numpy", [](const sgn_autograd::Parameter& p) {
            return tensor_to_numpy_copy(p.tensor());
        }, "转换为 numpy array（拷贝）")
        .def("__repr__", [](const sgn_autograd::Parameter& p) {
            std::string s = "Parameter(shape=[";
            for (size_t i = 0; i < p.tensor().ndim(); ++i) {
                if (i > 0) s += ",";
                s += std::to_string(p.tensor().shape()[i]);
            }
            s += "])";
            return s;
        })
        // 访问内部 Tensor
        // 安全审计 2026-08-16 B1：reference_internal 绑定生命周期到 Parameter
        // （keeper）——Python 侧持有返回的 Tensor 引用不会延长 Parameter，
        // Parameter 被 GC 后引用即失效，调用方须保持 Parameter 存活
        .def("tensor", [](sgn_autograd::Parameter& p) -> sgn_autograd::Tensor& {
            return p.tensor();
        }, py::return_value_policy::reference_internal,
            "返回内部 Tensor（可直接传给 autograd 算子）。\n"
            "注意：引用生命周期绑定到本 Parameter——必须保持 Parameter 存活，\n"
            "Parameter 释放后此 Tensor 引用失效。");

    // ========================================================================
    // Buffer 类
    // ========================================================================
    py::class_<sgn_autograd::Buffer>(nn_m, "Buffer",
        "非学习状态缓冲区（构造时自动 requires_grad=False）\n\n"
        "用于 BatchNorm 的 running_mean/var 等非学习状态。")
        .def(py::init([](const std::vector<int64_t>& shape) {
            return sgn_autograd::Buffer(shape);
        }), py::arg("shape"), "按形状分配（元素初始化为 0）")
        .def(py::init([](const sgn_autograd::Tensor& t) {
            return sgn_autograd::Buffer(t);
        }), py::arg("tensor"), "从已有 Tensor 构造")
        .def_property_readonly("shape", [](const sgn_autograd::Buffer& b) {
            return b.tensor().shape();
        })
        .def("to_numpy", [](const sgn_autograd::Buffer& b) {
            return tensor_to_numpy_copy(b.tensor());
        }, "转换为 numpy array（拷贝）")
        .def("__repr__", [](const sgn_autograd::Buffer& b) {
            std::string s = "Buffer(shape=[";
            for (size_t i = 0; i < b.tensor().ndim(); ++i) {
                if (i > 0) s += ",";
                s += std::to_string(b.tensor().shape()[i]);
            }
            s += "])";
            return s;
        })
        // 访问内部 Tensor
        .def("tensor", [](sgn_autograd::Buffer& b) -> sgn_autograd::Tensor& {
            return b.tensor();
        }, py::return_value_policy::reference_internal,
            "返回内部 Tensor");

    // ========================================================================
    // Module 基类（含 trampoline，支持 Python 子类化）
    // ========================================================================
    py::class_<sgn::nn::Module, PyModule, std::shared_ptr<sgn::nn::Module>>(
        nn_m, "Module",
        "神经网络模块基类\n\n"
        "稳定模式的核心抽象。提供参数注册、状态管理、序列化、模式切换。\n"
        "子类需重写 forward() 方法。\n\n"
        "注意：Module 不管理 tape 生命周期 —— tape 由 record_scope() 统一管理。")
        .def(py::init<>())

        // === 参数管理 ===
        .def("register_parameter",
            [](py::object self, const std::string& name,
               py::object param) {
                auto* mod = self.cast<sgn::nn::Module*>();
                if (param.is_none()) {
                    mod->register_parameter(name, std::nullopt);
                    // 删除 Python 属性
                    if (py::hasattr(self, name.c_str())) {
                        py::delattr(self, name.c_str());
                    }
                } else {
                    auto p = param.cast<sgn_autograd::Parameter>();
                    mod->register_parameter(name, p);
                    // 设置 Python 属性，方便 self.weight 访问
                    py::setattr(self, name.c_str(), param);
                }
            },
            py::arg("name"), py::arg("param"),
            "注册可学习参数（param=None 取消注册）")

        .def("parameters", &sgn::nn::Module::parameters,
            "递归收集所有可学习参数（含子模块）")

        .def("named_parameters", &sgn::nn::Module::named_parameters,
            py::arg("prefix") = "",
            "递归收集所有命名参数，返回 list[tuple[str, Tensor]]\n"
            "路径名用 '.' 分隔（如 'features.0.weight'）")

        // === 缓冲区管理 ===
        .def("register_buffer",
            [](py::object self, const std::string& name,
               py::object buf) {
                auto* mod = self.cast<sgn::nn::Module*>();
                if (buf.is_none()) {
                    mod->register_buffer(name, std::nullopt);
                    if (py::hasattr(self, name.c_str())) {
                        py::delattr(self, name.c_str());
                    }
                } else {
                    auto b = buf.cast<sgn_autograd::Buffer>();
                    mod->register_buffer(name, b);
                    py::setattr(self, name.c_str(), buf);
                }
            },
            py::arg("name"), py::arg("buf"),
            "注册状态缓冲区（buf=None 取消注册）")

        .def("buffers", &sgn::nn::Module::buffers,
            "递归收集所有缓冲区（含子模块）")

        .def("named_buffers", &sgn::nn::Module::named_buffers,
            py::arg("prefix") = "",
            "递归收集所有命名缓冲区")

        // === 子模块管理 ===
        .def("register_module",
            [](py::object self, const std::string& name,
               py::object child) {
                // 顺序（安全审计 2026-08-16 M-2）：先注册 C++ 层（可能因重名/
                // 非连续抛异常），成功后再追加 Python 侧列表——保证两侧一致，
                // 避免"C++ 失败但 Python 已追加"的半注册状态。
                auto* cpp_self = self.cast<sgn::nn::Module*>();
                auto child_ptr = py::cast<std::shared_ptr<sgn::nn::Module>>(child);
                cpp_self->register_module(name, std::move(child_ptr));

                // 保持 Python 引用，防止子类类型丢失
                if (!py::hasattr(self, "_children_py")) {
                    self.attr("_children_py") = py::list();
                }
                self.attr("_children_py").attr("append")(child);
            },
            py::arg("name"), py::arg("child"),
            "注册子模块")

        .def("children", [](py::object self) -> py::list {
            if (py::hasattr(self, "_children_py")) {
                return self.attr("_children_py");
            }
            return py::list();
        }, "返回所有直接子模块（Python 对象，保留子类类型）")

        // === 模式切换 ===
        .def("train", &sgn::nn::Module::train,
            py::arg("mode") = true,
            "设置为训练模式（递归设置所有子模块）")

        .def("eval", &sgn::nn::Module::eval,
            "设置为推理模式（递归设置所有子模块）")

        .def_property_readonly("training", &sgn::nn::Module::is_training,
            "是否处于训练模式（只读）")

        // === 梯度 ===
        .def("zero_grad", &sgn::nn::Module::zero_grad,
            "清零所有参数梯度（不清 tape）")

        // === 序列化 ===
        .def("state_dict", [](sgn::nn::Module& self) -> py::dict {
            py::dict result;
            for (const auto& [name, t] : self.named_parameters()) {
                result[py::str(name)] = tensor_to_numpy_copy(t);
            }
            for (const auto& [name, t] : self.named_buffers()) {
                result[py::str(name)] = tensor_to_numpy_copy(t);
            }
            return result;
        }, "返回 state_dict（dict[str, numpy.ndarray]）")

        .def("load_state_dict",
            [](sgn::nn::Module& self, const py::dict& state) {
                // 安全审计 2026-08-16 B2：先在 GIL 下完成 dict 访问 + shape
                // 校验并收集 (目标指针, 源指针, 字节数)，再释放 GIL 批量
                // memcpy（大模型加载不再阻塞其他 Python 线程）。
                // arrs 保持 py::array_t 引用，源内存在 memcpy 期间存活。
                struct CopyItem { float* dst; const float* src; size_t nbytes; };
                std::vector<CopyItem> copies;
                std::vector<py::array_t<float, py::array::c_style | py::array::forcecast>> arrs;

                // 辅助：校验 numpy array 与 Tensor shape 完全匹配，收集拷贝项
                auto load_one = [&](const std::string& name,
                                    sgn_autograd::Tensor& t,
                                    const py::dict& state) {
                    if (!state.contains(py::str(name))) return;

                    // forcecast：非 float32/非 C 连续的数组先物化为连续 float32
                    // 副本，否则 buf.ptr 指向 storage 起始、按连续布局 memcpy 会
                    // 拷错数据（安全审计 2026-08-16 M-6）
                    auto arr = state[py::str(name)].cast<
                        py::array_t<float, py::array::c_style | py::array::forcecast>>();
                    auto buf = arr.request();

                    // 完整 shape 校验（不只是 numel）
                    if (static_cast<size_t>(buf.ndim) != t.ndim()) {
                        throw std::runtime_error(
                            "load_state_dict: ndim mismatch for '" + name +
                            "': expected " + std::to_string(t.ndim()) +
                            ", got " + std::to_string(buf.ndim));
                    }
                    for (size_t i = 0; i < t.ndim(); ++i) {
                        if (static_cast<int64_t>(buf.shape[i]) != t.shape()[i]) {
                            throw std::runtime_error(
                                "load_state_dict: shape mismatch for '" + name +
                                "' at dim " + std::to_string(i) +
                                ": expected " + std::to_string(t.shape()[i]) +
                                ", got " + std::to_string(buf.shape[i]));
                        }
                    }

                    // Tensor 拷贝共享 storage，直接写入即修改原参数
                    // 但只对 contiguous Tensor 安全（注册的参数/buffer 都是 contiguous）
                    if (!t.is_contiguous()) {
                        throw std::runtime_error(
                            "load_state_dict: '" + name +
                            "' is non-contiguous, cannot load");
                    }
                    copies.push_back(CopyItem{
                        t.data(),
                        static_cast<const float*>(buf.ptr),
                        static_cast<size_t>(buf.size) * sizeof(float)});
                    arrs.push_back(std::move(arr));  // 保活源数组
                };

                for (auto& [name, t] : self.named_parameters()) {
                    load_one(name, t, state);
                }
                for (auto& [name, t] : self.named_buffers()) {
                    load_one(name, t, state);
                }

                // 纯内存拷贝段：释放 GIL（不再触碰任何 Python/buffer API）
                {
                    py::gil_scoped_release release;
                    for (const auto& c : copies) {
                        std::memcpy(c.dst, c.src, c.nbytes);
                    }
                }
            },
            py::arg("state"),
            "加载 state_dict（含完整 shape 校验）")

        // === forward ===
        .def("forward",
            [](sgn::nn::Module& self,
               const std::vector<sgn_autograd::Tensor>& inputs) {
                return self.forward(inputs);
            },
            py::arg("inputs"),
            "前向传播（子类必须重写）")

        .def("forward",
            [](sgn::nn::Module& self, sgn_autograd::Tensor input) {
                return self.forward(std::move(input));
            },
            py::arg("input"),
            "前向传播（单输入便捷接口）")

        .def("__call__",
            [](sgn::nn::Module& self,
               const std::vector<sgn_autograd::Tensor>& inputs) {
                return self(inputs);
            },
            py::arg("inputs"),
            "调用 forward()（不管理 tape）")

        .def("__call__",
            [](sgn::nn::Module& self, sgn_autograd::Tensor input) {
                return self(std::move(input));
            },
            py::arg("input"),
            "调用 forward()（单输入便捷接口）");

    // ========================================================================
    // 初始化辅助函数
    // ========================================================================
    nn_m.def("kaiming_uniform_",
        [](sgn_autograd::Tensor& t, int64_t fan_in, float a) {
            sgn_autograd::kaiming_uniform_(t, fan_in, a);
        },
        py::arg("tensor"), py::arg("fan_in"),
        py::arg("a") = 2.2360679775f,
        "Kaiming uniform 初始化（原地修改）");

    nn_m.def("uniform_",
        [](sgn_autograd::Tensor& t, float low, float high) {
            sgn_autograd::uniform_(t, low, high);
        },
        py::arg("tensor"), py::arg("low"), py::arg("high"),
        "均匀分布初始化（原地修改）");

    nn_m.def("fill_",
        [](sgn_autograd::Tensor& t, float value) {
            sgn_autograd::fill_(t, value);
        },
        py::arg("tensor"), py::arg("value"),
        "常量填充（原地修改）");
}