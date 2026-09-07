// precision_budget_bindings.cpp - pybind11 绑定（注册到 sgn 模块）
//
// 设计：与 value_spec_bindings.cpp / unit_value_bindings.cpp 一致，
//       提供 register_precision_budget(py::module_&) 注册函数。
//
// 暴露的 Python API：
//   - sgn.LayerCost：单层成本信号类
//       - LayerCost(c=0.0, b_min=4, b_max=24)
//       - .c / .b_min / .b_max 读写属性
//   - sgn.PrecisionBudget：精度预算类
//       - PrecisionBudget(total_bits, layers)
//       - .total_bits / .layers 读写属性
//       - .level_f / .level_b 读写属性（ValueSpec）
//       - .has_level_f / .has_level_b 只读属性
//       - .set_level_f(spec) / .set_level_b(spec)
//       - .clear_level_f() / .clear_level_b()
//       - .allocate() → list[int]（贪心边际分配，返回各层 bits）
//       - .total_error(bits) → float（计算总 STE 方差）
//       - PrecisionBudget.layer_error(c, b) → float（静态方法）
//       - PrecisionBudget.marginal_gain(c, b) → float（静态方法）

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "precision_budget.h"

namespace py = pybind11;

/* 注册函数：由 placeholder.cpp 的 PYBIND11_MODULE(sgn, m) 调用 */
void register_precision_budget(py::module_& m) {
    // LayerCost：单层成本信号
    py::class_<sgn::LayerCost>(m, "LayerCost")
        .def(py::init<double, uint8_t, uint8_t>(),
             py::arg("c") = 0.0,
             py::arg("b_min") = 4,
             py::arg("b_max") = 24)
        .def_readwrite("c", &sgn::LayerCost::c,
                       "Cost signal = grad_l2^2 * in_dim (exp26 confirmed).")
        .def_readwrite("b_min", &sgn::LayerCost::b_min,
                       "Minimum bits for this layer (usually 4 or 8).")
        .def_readwrite("b_max", &sgn::LayerCost::b_max,
                       "Maximum bits for this layer (usually 20 or 24).")
        .def("__repr__", [](const sgn::LayerCost& lc) {
            return "LayerCost(c=" + std::to_string(lc.c) +
                   ", b_min=" + std::to_string(lc.b_min) +
                   ", b_max=" + std::to_string(lc.b_max) + ")";
        });

    // PrecisionBudget：精度预算接口
    py::class_<sgn::PrecisionBudget>(m, "PrecisionBudget")
        .def(py::init<uint32_t, std::vector<sgn::LayerCost>>(),
             py::arg("total_bits"),
             py::arg("layers"))
        .def_readwrite("total_bits", &sgn::PrecisionBudget::total_bits,
                       "Total bits budget B_total.")
        .def_readwrite("layers", &sgn::PrecisionBudget::layers,
                       "Per-layer cost signals.")
        .def_readwrite("level_f", &sgn::PrecisionBudget::level_f,
                       "Forward ValueSpec (Level_f interface, default bits=0 means None).")
        .def_readwrite("level_b", &sgn::PrecisionBudget::level_b,
                       "Backward ValueSpec (Level_b interface, default bits=0 means None).")
        .def_readonly("has_level_f", &sgn::PrecisionBudget::has_level_f,
                      "Whether level_f is set.")
        .def_readonly("has_level_b", &sgn::PrecisionBudget::has_level_b,
                      "Whether level_b is set.")
        .def("set_level_f", &sgn::PrecisionBudget::set_level_f,
             py::arg("spec"),
             "Set forward ValueSpec (Level_f interface).")
        .def("set_level_b", &sgn::PrecisionBudget::set_level_b,
             py::arg("spec"),
             "Set backward ValueSpec (Level_b interface).")
        .def("clear_level_f", &sgn::PrecisionBudget::clear_level_f,
             "Clear level_f (fallback to single level mode).")
        .def("clear_level_b", &sgn::PrecisionBudget::clear_level_b,
             "Clear level_b (fallback to single level mode).")
        .def("allocate", &sgn::PrecisionBudget::allocate,
             "Greedy marginal allocation (exp34 M-convexity, O(n log n) global optimal).\n"
             "Returns list[int] of per-layer bits.")
        .def("total_error", &sgn::PrecisionBudget::total_error,
             py::arg("bits"),
             "Compute total STE variance: sum(c_i / (2^b_i - 1)).")
        .def_static("layer_error", &sgn::PrecisionBudget::layer_error,
                    py::arg("c"), py::arg("b"),
                    "Single layer error: f_i(b) = c / (2^b - 1).")
        .def_static("marginal_gain", &sgn::PrecisionBudget::marginal_gain,
                    py::arg("c"), py::arg("b"),
                    "Marginal gain: Delta_i(b) = f_i(b) - f_i(b+1).")
        .def("__repr__", [](const sgn::PrecisionBudget& pb) {
            return "PrecisionBudget(total_bits=" + std::to_string(pb.total_bits) +
                   ", layers=" + std::to_string(pb.layers.size()) + ")";
        });
}
