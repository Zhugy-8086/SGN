// autograd_bindings.cpp - pybind11 绑定（注册到 sgn 模块的 autograd 子模块）
//
// Stage 3.2 Phase 4：将 C++ Autograd 框架绑定到 Python，集成到 sgn 模块。
//
// SGN 专属适配（非通用 pybind11 模式）：
//   1. 集成到 sgn 模块（非独立 .pyd）：作为 sgn.autograd 子模块注册，
//      与 col2im/hc8 等并列，避免新建独立模块的导入开销
//   2. numpy 拷贝互转：Tensor.from_numpy() 拷贝数据到内部 storage，
//      to_numpy() 拷贝回 numpy。安全优先，避免悬垂指针（numpy 可能提前释放）
//   3. backward 挂在 Tensor 上：tensor.backward(grad) 内部调用全局 Tape，
//      Python 端用法与 PyTorch 一致
//   4. tensor.grad 返回 numpy 或 None：与 PyTorch 的 .grad 属性一致
//   5. 算子作为子模块函数：sgn.autograd.matmul(a, b) 自动记录到 tape
//
// Python API：
//   import sgn
//   t = sgn.autograd.Tensor.from_numpy(np_array)
//   t.requires_grad = True
//   sgn.autograd.start_recording()
//   c = sgn.autograd.matmul(a, b)
//   sgn.autograd.stop_recording()
//   c.backward(grad_numpy)
//   a_grad = a.grad  # numpy 或 None

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "tensor.h"
#include "autograd.h"

#include "common/allocator.h"
#include "common/pool_allocator.h"
#include "common/logger.h"
#include "ops.h"
#include "ops_nn.h"
#include "autograd_nn.h"
#include "backward_strategy.h"
#include "dispatch/registry.h"

#include <stdexcept>
#include <vector>

namespace py = pybind11;
using namespace sgn_autograd;
// int8 对叶梯度视图（A″）：pair 原语（pair_grad_carrier.h 经 autograd.h 引入）
using sgn_msint::pair_decode_q;
using sgn_msint::decode_pair_fine_f32;
using sgn_msint::decode_pair_coarse_f32;
using sgn_msint::pair_dot_fine;
using sgn_msint::pair_dot_coarse;

// ============================================================================
// 辅助：numpy array → Tensor（拷贝构造，安全）
// ============================================================================
static Tensor tensor_from_numpy(const py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.request();
    if (buf.ndim > 4) {
        throw std::runtime_error("Tensor.from_numpy: 最多支持 4D，得到 " + std::to_string(buf.ndim));
    }
    // 安全审计 2026-08-16 A2：元素数上限校验（2^31-1），防御性拒绝
    // 异常巨大输入（shape 溢出/误传扁平大数组），避免分配路径静默截断
    if (buf.size > (py::ssize_t)INT32_MAX) {
        throw std::runtime_error("Tensor.from_numpy: 元素数超过 INT32_MAX 上限（得到 " +
                                 std::to_string(buf.size) + "）");
    }
    std::vector<int64_t> shape(buf.ndim);
    for (int i = 0; i < buf.ndim; ++i) {
        shape[i] = static_cast<int64_t>(buf.shape[i]);
    }
    if (buf.size == 0) {
        return Tensor();
    }
    return Tensor(shape, static_cast<const float*>(buf.ptr));
}

// ============================================================================
// 辅助：Tensor → numpy array（拷贝，安全）
// ============================================================================
static py::array_t<float> tensor_to_numpy(const Tensor& t) {
    std::vector<py::ssize_t> shape(t.ndim());
    for (size_t i = 0; i < t.ndim(); ++i) {
        shape[i] = static_cast<py::ssize_t>(t.shape()[i]);
    }
    py::array_t<float> arr(shape);
    auto buf = arr.request();
    if (!t.is_contiguous()) {
        // 非 contiguous 需要先转 contiguous
        Tensor c = t.contiguous();
        std::memcpy(buf.ptr, c.data(), t.numel() * sizeof(float));
    } else {
        std::memcpy(buf.ptr, t.data(), t.numel() * sizeof(float));
    }
    return arr;
}

// ============================================================================
// 注册函数：由 placeholder.cpp 的 PYBIND11_MODULE(sgn, m) 调用
// ============================================================================
void register_autograd(py::module_& m) {
    // 模块初始化日志（进程级"开始时间"打点，俄罗斯施工备忘 §4 的 P1）
    // 触发全局 logger（首次调用固化 SGN_LOG_LEVEL 解析），并标记 autograd 就绪
    SGN_LOG_INFO("sgn.autograd module init");

    // 创建子模块 sgn.autograd
    py::module_ autograd_m = m.def_submodule("autograd", "C++ Autograd 框架（Stage 3.2）");

    // ========================================================================
    // Tensor 类
    // ========================================================================
    py::class_<Tensor>(autograd_m, "Tensor",
        "C++ 原生 Tensor（Storage + Shape + Stride，Phase 1）\n\n"
        "与 numpy 互转通过 from_numpy / to_numpy（拷贝，安全）。")
        // 构造函数
        .def(py::init<>(), "创建空 Tensor")
        .def(py::init([](const std::vector<int64_t>& shape) {
            return Tensor(shape);
        }), py::arg("shape"), "按形状分配（元素初始化为 0）")
        .def(py::init([](const py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
            return tensor_from_numpy(arr);
        }), py::arg("array"), "从 numpy array 构造（拷贝）")

        // 属性
        .def_property_readonly("shape", [](const Tensor& t) {
            return t.shape();
        })
        .def_property_readonly("ndim", [](const Tensor& t) { return t.ndim(); })
        .def_property_readonly("numel", [](const Tensor& t) { return t.numel(); })
        .def_property("requires_grad", [](const Tensor& t) { return t.requires_grad(); },
                      [](Tensor& t, bool v) { t.set_requires_grad(v); },
                      "是否需要梯度（Phase 2 Autograd）")
        .def_property_readonly("id", [](const Tensor& t) { return t.id(); },
                      "全局唯一 id（Tape 梯度表索引；grad_pair(t.id()) 用）")

        // numpy 互转
        .def_static("from_numpy", [](const py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
            return tensor_from_numpy(arr);
        }, py::arg("array"), "从 numpy array 创建 Tensor（拷贝）")
        .def("to_numpy", [](const Tensor& t) {
            return tensor_to_numpy(t);
        }, "转换为 numpy array（拷贝）")

        // 基础操作
        .def("reshape", [](const Tensor& t, const std::vector<int64_t>& shape) {
            return reshape(t, shape);
        }, py::arg("shape"), "reshape（autograd-aware，支持 -1 自动推断）")
        .def("contiguous", [](const Tensor& t) {
            return t.contiguous();
        }, "返回 contiguous 拷贝（如果已经是则返回自身）")

        // backward（SGN 适配：挂在 Tensor 上，内部调用全局 Tape）
        // 安全审计 2026-08-16 A1：纯 C++ 计算段释放 GIL（Tensor/Tape 均为
        // C++ 对象，thread_local Tape 在调用线程内访问，无 Python API 调用）
        .def("backward", [](Tensor& t, py::array_t<float, py::array::c_style | py::array::forcecast> grad_arr) {
            Tensor grad = tensor_from_numpy(grad_arr);  // numpy 访问需 GIL
            py::gil_scoped_release release;
            Tape::current().backward(t, std::move(grad));
        }, py::arg("grad"), "反向传播（从本 Tensor 开始，grad 为初始梯度）")
        .def("backward", [](Tensor& t) {
            // 默认梯度为 ones（与 PyTorch 标量 backward 一致）。
            // 安全审计 2026-08-16 A4：引擎 Tensor 仅有 float32 一种 dtype，
            // 此处硬编码 1.0f 与唯一 dtype 一致；若未来引入多 dtype 需从
            // Tensor 推断
            Tensor grad(t.shape());
            std::fill(grad.data(), grad.data() + grad.numel(), 1.0f);
            py::gil_scoped_release release;
            Tape::current().backward(t, std::move(grad));
        }, "反向传播（默认 grad=ones）")

        // grad 属性（返回 numpy 或 None）
        // 安全审计 2026-08-16 A3：reinterpret_steal 生命周期链——
        // tensor_to_numpy 返回的 py::array_t 持有独立 numpy 缓冲（数据已
        // 拷出 Tape），release() 转移引用计数后立即包成 py::object 返回，
        // 不依赖 Tape 内 Tensor 的存活。正确但依赖此顺序，勿提前引用。
        .def_property_readonly("grad", [](const Tensor& t) -> py::object {
            const Tensor* g = Tape::current().grad(t.id());
            if (g == nullptr) {
                return py::none();
            }
            // py::array_t 是 py::object 的派生类，用 reinterpret_steal 转为 object
            return py::reinterpret_steal<py::object>(tensor_to_numpy(*g).release().ptr());
        }, "梯度（numpy array 或 None，backward 后可用）")

        // repr
        .def("__repr__", [](const Tensor& t) {
            std::string s = "Tensor(shape=[";
            for (size_t i = 0; i < t.ndim(); ++i) {
                if (i > 0) s += ",";
                s += std::to_string(t.shape()[i]);
            }
            s += "], requires_grad=" + std::string(t.requires_grad() ? "True" : "False") + ")";
            return s;
        });

    // ========================================================================
    // Tape 操作（全局单例，通过 lambda 调用 Tape::current()）
    // ========================================================================
    autograd_m.def("start_recording", []() { Tape::current().start_recording(); },
        "开始录制前向操作到 tape");
    autograd_m.def("stop_recording", []() { Tape::current().stop_recording(); },
        "停止录制");
    autograd_m.def("is_recording", []() { return Tape::current().is_recording(); },
        "是否正在录制");
    autograd_m.def("clear", []() { Tape::current().clear(); },
        "清空 tape（测试间重置）");

    // ========================================================================
    // 内核后端查询（诊断 + 档位守门断言）
    // ========================================================================
    // 返回当前进程活跃的内核后端信息：matmul 族 + conv2d 族各自的
    // {name（如 "x86_avx2" / "x86_sse2" / "ref_scalar(forced)"）, num_level}。
    // 用途：多后端时代排查"性能/数值不对"时先确认跑的是哪个后端及其档位声明；
    // 档位守门断言据此核对"声明档位 == 实测档位"（防静默降精度，见调研文档 §13）。
    autograd_m.def("kernel_backend", []() {
        const KernelSet& mk = kernel_registry();
        const Conv2dKernelSet& ck = conv2d_registry();
        auto level_of = [](NumLevel l) {
            return (l == NumLevel::kBitExact) ? "bit_exact" : "rounding";
        };
        py::dict out;
        out["matmul"] = py::dict(py::arg("name") = std::string(mk.name),
                                 py::arg("num_level") = level_of(mk.num_level));
        out["conv2d"] = py::dict(py::arg("name") = std::string(ck.name),
                                 py::arg("num_level") = level_of(ck.num_level));
        return out;
    }, "查询当前活跃内核后端：{matmul:{name,num_level}, conv2d:{name,num_level}}");

    // ========================================================================
    // 内存分配器开关（基础设施，P1）
    // ========================================================================
    // 启用池化分配器：训练/推理内存经 PoolAllocator 复用（thread_local + size-class）。
    // 关闭：恢复默认 stdlib 分配器（行为与未池化一致，数值不变）。
    // 未来接外部后端（oneDNN/ONNX Runtime 等）可替换为对应 allocator。
    autograd_m.def("set_pool_allocator", [](bool enable) {
        if (enable) {
            // 启用前清空当前线程池，避免混入历史块（安全审计 2026-08-16 P-2 缓解）。
            // 注：Storage 在分配时捕获 deallocator，切换前已存在的 Tensor 仍按
            // 原分配器释放，切换本身是安全的；但仍建议在训练开始前（单线程阶段）切换。
            sgn::clear_pool();
            sgn::set_allocator(&sgn::pool_alloc, &sgn::pool_dealloc);
        } else {
            sgn::set_allocator(nullptr, nullptr);
            sgn::clear_pool();
        }
    }, py::arg("enable"),
        "启用/关闭池化分配器（默认关闭=stdlib）。"
        "注意：建议在训练开始前调用；切换只影响之后的新分配。");

    autograd_m.def("clear_pool", []() { sgn::clear_pool(); },
        "清空内存池缓存块（测试/重置用）。"
        "注意：仅清空【当前线程】的池（thread_local 语义）——多线程训练时"
        "需在每条线程内分别调用，主线程调用不影响 worker 线程的缓存。");

    // ========================================================================
    // 算子（autograd-aware，自动记录到 tape）
    // 安全审计 2026-08-16 A1：算子为纯 C++ 计算且无 Python 回调，
    // 统一经 lambda + gil_scoped_release 释放 GIL（thread_local Tape
    // 在调用线程内记录，语义不变，仅允许其他 Python 线程并行）
    // ========================================================================
    autograd_m.def("matmul", [](const Tensor& a, const Tensor& b) {
        py::gil_scoped_release release;
        return matmul(a, b);
    }, py::arg("a"), py::arg("b"),
        "矩阵乘法 C = A @ B（autograd-aware，自动记录到 tape）");

    autograd_m.def("reshape", &reshape,
        py::arg("x"), py::arg("shape"),
        "Reshape（autograd-aware，backward 将梯度 reshape 回原形状）");

    // 前向算子（无 autograd，直接计算）
    autograd_m.def("matmul_forward", [](const Tensor& a, const Tensor& b) {
        py::gil_scoped_release release;
        return matmul_forward(a, b);
    }, py::arg("a"), py::arg("b"),
        "矩阵乘法前向（无 autograd）");

    autograd_m.def("linear_forward", [](const Tensor& x, const Tensor& w, const Tensor& b) {
        py::gil_scoped_release release;
        return linear_forward(x, w, b);
    }, py::arg("x"), py::arg("w"), py::arg("b"),
        "Linear 前向: Y = X @ W^T + b");

    autograd_m.def("relu_forward", [](const Tensor& x) {
        py::gil_scoped_release release;
        return relu_forward(x);
    }, py::arg("x"), "ReLU 前向: Y = max(0, X)");

    autograd_m.def("conv2d_forward", [](
        const Tensor& X, const Tensor& W, const Tensor& b,
        int stride, int padding
    ) {
        Conv2DContext ctx;
        py::gil_scoped_release release;
        return conv2d_forward(X, W, b, stride, padding, ctx);
    }, py::arg("x"), py::arg("w"), py::arg("b"),
       py::arg("stride") = 1, py::arg("padding") = 0,
       "Conv2d 前向: Y = im2col(X) @ W_col + b");

    autograd_m.def("maxpool2d_forward", [](
        const Tensor& X, int kernel, int stride
    ) {
        MaxPoolContext ctx;
        py::gil_scoped_release release;
        return maxpool2d_forward(X, kernel, stride, ctx);
    }, py::arg("x"), py::arg("kernel"), py::arg("stride") = -1,
       "MaxPool2d 前向（stride 默认 = kernel）");

    // ========================================================================
    // autograd-aware 神经网络算子（Phase 5：自动记录到 tape）
    // ========================================================================
    autograd_m.def("linear", [](const Tensor& x, const Tensor& w, const Tensor& b) {
        py::gil_scoped_release release;
        return linear(x, w, b);
    }, py::arg("x"), py::arg("w"), py::arg("b"),
        "Linear: Y = X @ W^T + b（autograd-aware）");

    autograd_m.def("relu", [](const Tensor& x) {
        py::gil_scoped_release release;
        return relu(x);
    }, py::arg("x"), "ReLU: Y = max(0, X)（autograd-aware）");

    autograd_m.def("sigmoid", [](const Tensor& x) {
        py::gil_scoped_release release;
        return sigmoid(x);
    }, py::arg("x"), "Sigmoid: Y = 1/(1+exp(-X))（autograd-aware）");

    autograd_m.def("tanh", [](const Tensor& x) {
        py::gil_scoped_release release;
        return tanh(x);
    }, py::arg("x"), "Tanh: Y = tanh(X)（autograd-aware）");

    autograd_m.def("gelu", [](const Tensor& x) {
        py::gil_scoped_release release;
        return gelu(x);
    }, py::arg("x"), "GELU: Y = 0.5*X*(1+erf(X/sqrt2))（autograd-aware）");

    autograd_m.def("silu", [](const Tensor& x) {
        py::gil_scoped_release release;
        return silu(x);
    }, py::arg("x"), "SiLU/Swish: Y = X*sigmoid(X)（autograd-aware）");

    autograd_m.def("layernorm", [](const Tensor& x, const Tensor& gamma,
                                   const Tensor& beta, float eps) {
        py::gil_scoped_release release;
        return layernorm(x, gamma, beta, eps);
    }, py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5f,
       "LayerNorm: 2D (B,C) 沿末维归一化（autograd-aware；gamma/beta 可学习）");

    autograd_m.def("add", [](const Tensor& a, const Tensor& b) {
        py::gil_scoped_release release;
        return add(a, b);
    }, py::arg("a"), py::arg("b"),
        "Add: Y = A + B（ResNet 残差连接，backward dA=dY, dB=dY）");

    autograd_m.def("bn_train", [](
        const Tensor& X, const Tensor& gamma, const Tensor& beta,
        Tensor& running_mean, Tensor& running_var,
        float momentum, float eps, int dim
    ) {
        py::gil_scoped_release release;
        return bn_train(X, gamma, beta, running_mean, running_var, momentum, eps, dim);
    }, py::arg("x"), py::arg("gamma"), py::arg("beta"),
       py::arg("running_mean"), py::arg("running_var"),
       py::arg("momentum") = 0.1f, py::arg("eps") = 1e-5f, py::arg("dim") = 0,
       "BatchNorm (训练模式，2D，autograd-aware)");

    autograd_m.def("batchnorm2d", [](
        const Tensor& X, const Tensor& gamma, const Tensor& beta,
        Tensor& running_mean, Tensor& running_var,
        float momentum, float eps
    ) {
        py::gil_scoped_release release;
        return batchnorm2d(X, gamma, beta, running_mean, running_var, momentum, eps);
    }, py::arg("x"), py::arg("gamma"), py::arg("beta"),
       py::arg("running_mean"), py::arg("running_var"),
       py::arg("momentum") = 0.1f, py::arg("eps") = 1e-5f,
       "BatchNorm2d (4D 输入，autograd-aware)");

    autograd_m.def("conv2d", [](
        const Tensor& X, const Tensor& W, const Tensor& b,
        int stride, int padding
    ) {
        py::gil_scoped_release release;
        return conv2d(X, W, b, stride, padding);
    }, py::arg("x"), py::arg("w"), py::arg("b"),
       py::arg("stride") = 1, py::arg("padding") = 0,
       "Conv2d: Y = im2col(X) @ W_col + b（autograd-aware）");

    autograd_m.def("mul", [](const Tensor& a, const Tensor& b) {
        py::gil_scoped_release release;
        return mul(a, b);
    }, py::arg("a"), py::arg("b"),
       "Mul: Y = A·B（逐元素同形，autograd-aware；Dropout/掩码/缩放地基）");

    autograd_m.def("maxpool2d", [](
        const Tensor& X, int kernel, int stride
    ) {
        py::gil_scoped_release release;
        return maxpool2d(X, kernel, stride);
    }, py::arg("x"), py::arg("kernel"), py::arg("stride") = -1,
       "MaxPool2d（autograd-aware，stride 默认 = kernel）");

    autograd_m.def("avgpool2d", [](
        const Tensor& X, int kernel, int stride
    ) {
        py::gil_scoped_release release;
        return avgpool2d(X, kernel, stride);
    }, py::arg("x"), py::arg("kernel"), py::arg("stride") = -1,
       "AvgPool2d/GAP（autograd-aware，stride 默认 = kernel）");

    // ========================================================================
    // 算子融合：conv2d + relu（autograd-aware）
    // ========================================================================
    autograd_m.def("conv2d_relu", [](
        const Tensor& X, const Tensor& W, const Tensor& b,
        int stride, int padding
    ) {
        py::gil_scoped_release release;
        return conv2d_relu(X, W, b, stride, padding);
    }, py::arg("x"), py::arg("w"), py::arg("b"),
       py::arg("stride") = 1, py::arg("padding") = 0,
       "Conv2d+ReLU 融合: Y = relu(conv2d(X, W, b))（autograd-aware）");

    // ========================================================================
    // STE 前向（无 autograd，用于测试/调试）
    // ========================================================================
    autograd_m.def("linear_forward_ste", [](const Tensor& X, const Tensor& W, const Tensor& b,
                                            int bits, float clip_sigma) {
        QuantConfig qcfg{bits, clip_sigma};
        py::gil_scoped_release release;
        return linear_forward_ste(X, W, b, qcfg);
    }, py::arg("x"), py::arg("w"), py::arg("b"),
       py::arg("bits") = 8, py::arg("clip_sigma") = 4.0f,
       "Linear STE 前向: Y = deq(Q(X @ W^T)) + b（无 autograd）");

    autograd_m.def("conv2d_forward_ste", [](const Tensor& X, const Tensor& W, const Tensor& b,
                                            int stride, int padding, int bits, float clip_sigma) {
        Conv2DContext ctx;
        QuantConfig qcfg{bits, clip_sigma};
        py::gil_scoped_release release;
        return conv2d_forward_ste(X, W, b, stride, padding, ctx, qcfg);
    }, py::arg("x"), py::arg("w"), py::arg("b"),
       py::arg("stride") = 1, py::arg("padding") = 0,
       py::arg("bits") = 8, py::arg("clip_sigma") = 4.0f,
       "Conv2d STE 前向: Y = deq(Q(im2col(X) @ W_col)) + b（无 autograd）");

    // ========================================================================
    // 反向传播策略设置
    // ========================================================================
    py::enum_<BackwardStrategy>(autograd_m, "BackwardStrategy",
        "反向传播策略枚举\n\n"
        "  FLOAT32  — 纯 float32 反向（默认）\n"
        "  STE      — Straight-Through Estimator（前向量化 + 反向 float32 直通）\n"
        "  GEF      — 反向梯度 Q16 网格确定性舍入量化（自包含，不依赖 HC 库）\n"
        "  SR       — 反向梯度 Q16 网格伯努利随机舍入量化（自包含，不依赖 HC 库）\n"
        "  A1       — A1 主线（2026-08-19）：前向 Q8 量化（STE 式）+ 反向 SR\n"
        "  HC16     — 整数反向（桩，待实现；枚举名为历史术语）\n"
        "  EF_SGD   — 误差反馈跨 step 累积（桩，待实现）\n"
        "  MSINT    — MSint 多视角异构精度（桩，待实现）\n"
        "  LEVEL_AMP — Level 驱动逐层异构精度（桩，待实现）")
        .value("FLOAT32", BackwardStrategy::FLOAT32)
        .value("STE", BackwardStrategy::STE)
        .value("GEF", BackwardStrategy::GEF)
        .value("SR", BackwardStrategy::SR)
        .value("A1", BackwardStrategy::A1)
        .value("HC16", BackwardStrategy::HC16)
        .value("EF_SGD", BackwardStrategy::EF_SGD)
        .value("MSINT", BackwardStrategy::MSINT)
        .value("LEVEL_AMP", BackwardStrategy::LEVEL_AMP)
        .export_values();

    autograd_m.def("set_backward_strategy", &set_backward_strategy,
        py::arg("strategy"),
        "设置全局反向传播策略（默认 FLOAT32）\n"
        "  FLOAT32 / STE / GEF / SR / A1 已实现");

    autograd_m.def("get_backward_strategy", &get_backward_strategy,
        "获取当前反向传播策略");

    autograd_m.def("set_ste_quant_config", &set_ste_quant_config,
        py::arg("bits") = 8, py::arg("clip_sigma") = 4.0f,
        "设置前向 STE 量化配置（仅影响前向量化，bits=8/16, clip_sigma=4.0）\n"
        "  2026-08-21 P0-2 修复后与反向配置独立：切前向位宽不再覆写反向 SR 位宽");

    autograd_m.def("set_quant_config", &set_quant_config,
        py::arg("bits") = 16, py::arg("clip_sigma") = 4.0f,
        "设置反向梯度量化配置（仅影响 GEF/SR/A1 的 Tape::backward 梯度量化，\n"
        "  默认 bits=16；与前向 STE 配置独立）");

    autograd_m.def("set_sr_seed", &set_sr_seed,
        py::arg("seed"),
        "设置 SR 随机量化的随机种子");

    // ---- int8 对（h,l）叶梯度存储（2026-09-02 A″ 三层访问，见 ops_nn.h 注释）----
    autograd_m.def("set_pair_grad_store",
        [](bool v) { StrategyContext::set_pair_grad_store(v); },
        py::arg("enabled"),
        "int8 对叶梯度存储开关（默认 False = 现状 float32 逐位不变）。\n"
        "  True 且策略 ∈ {SR, A1} 且 bwd bits=16：单路径叶梯度以 pair 存储\n"
        "  （2B/元素，显存 -50%），grad() 透明解码（惰性缓存，每 step 一次解码）。\n"
        "  其他策略 / bits≠16 / 多路径叶（权重共享）自动回退 float 路径。");
    autograd_m.def("pair_grad_store",
        []() { return StrategyContext::pair_grad_store(); },
        "查询 int8 对叶梯度存储开关状态");
    autograd_m.def("grad_pair",
        [](int64_t tensor_id) -> py::object {
            const auto* p = Tape::current().grad_pair(tensor_id);
            if (!p) return py::none();
            return py::cast(sgn_msint::PairGradCarrier(*p));  // 拷贝（独立于 Tape 生命周期）
        },
        py::arg("tensor_id"),
        "获取叶梯度的 int8 对视图（PairGradView，研究入口）。\n"
        "  返回 None 表示该梯度未以 pair 存储（开关关 / 多路径 / 非 SR-A1）。");

    // PairGradView：int8 对（h,l）梯度载体的一等研究视图（对齐 MSIntView 模式）。
    // h/l 为 numpy uint8 拷贝（项目 numpy 互转安全纪律，非零拷贝）；fine/coarse
    // 双读（coarse = Q8 级，Level-AMP 粗层实验入口）；dot_fine/coarse 直调
    // simd::dot8（C++ 闭环消费，亦为未来 C++ 优化器预埋通道）。
    py::class_<sgn_msint::PairGradCarrier>(autograd_m, "PairGradView",
        "int8 对（h,l）叶梯度载体只读视图。\n"
        "  编码（U 方案）：q = 256·h + l − 32768 ∈ [-32767, 32767]，g = q·scale。\n"
        "  fine ≡ Q16-SR 精确恢复；coarse = 256·(h−128)·scale（Q8 级，须配 SR）。\n"
        "  详见 内部档案 §六/§七。")
        .def_readonly("n", &sgn_msint::PairGradCarrier::n, "元素数")
        .def_readonly("scale", &sgn_msint::PairGradCarrier::scale,
                      "per-tensor scale（max|g| / 32767）")
        .def("h", [](const sgn_msint::PairGradCarrier& p) {
                 return py::array_t<uint8_t>(p.n, p.h.data());
             }, "高位肢（h_s+128 偏置，uint8 numpy 拷贝）")
        .def("l", [](const sgn_msint::PairGradCarrier& p) {
                 return py::array_t<uint8_t>(p.n, p.l.data());
             }, "低位肢（无符号 l'，uint8 numpy 拷贝）")
        .def("q", [](const sgn_msint::PairGradCarrier& p) {
                 py::array_t<int16_t> arr(p.n);
                 auto buf = arr.request();
                 auto* out = static_cast<int16_t*>(buf.ptr);
                 for (size_t i = 0; i < p.n; ++i) out[i] = pair_decode_q(p, i);
                 return arr;
             }, "Q16 网格整数解码（int16 numpy 拷贝，q = 256·h + l − 32768）")
        .def("fine", [](const sgn_msint::PairGradCarrier& p) {
                 py::array_t<float> arr(p.n);
                 auto buf = arr.request();
                 decode_pair_fine_f32(p, static_cast<float*>(buf.ptr));
                 return arr;
             }, "fine 浮点解码（≡ Q16-SR 精确恢复，g = q·scale）")
        .def("coarse", [](const sgn_msint::PairGradCarrier& p) {
                 py::array_t<float> arr(p.n);
                 auto buf = arr.request();
                 decode_pair_coarse_f32(p, static_cast<float*>(buf.ptr));
                 return arr;
             }, "coarse 浮点解码（= 256·h_s·scale，Q8 级近似）")
        .def("dot_fine", [](const sgn_msint::PairGradCarrier& p,
                            py::array_t<int8_t, py::array::c_style | py::array::forcecast> w) {
                 py::buffer_info buf = w.request();
                 if (static_cast<size_t>(buf.size) != p.n)
                     throw std::runtime_error("dot_fine: 权重长度 != n");
                 return pair_dot_fine(p, static_cast<const int8_t*>(buf.ptr));
             }, py::arg("w8"),
             "fine dot8 消费：Σq·w（int64 精确，2 次 simd::dot8 + Sw 摊销）")
        .def("dot_coarse", [](const sgn_msint::PairGradCarrier& p,
                              py::array_t<int8_t, py::array::c_style | py::array::forcecast> w) {
                 py::buffer_info buf = w.request();
                 if (static_cast<size_t>(buf.size) != p.n)
                     throw std::runtime_error("dot_coarse: 权重长度 != n");
                 return pair_dot_coarse(p, static_cast<const int8_t*>(buf.ptr));
             }, py::arg("w8"),
             "coarse dot8 消费：Σq_c·w（int64 精确，1 次 simd::dot8）");

    autograd_m.def("strategy_name", &strategy_name,
        py::arg("strategy"),
        "获取策略名称字符串");
}
