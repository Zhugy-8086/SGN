// multiscale_view_bindings.cpp - pybind11 绑定（注册到 sgn 模块）
//
// 暴露的 Python API：
//   - sgn.MultiScaleView.interpret(value, total_bits=32) → {split_bits: parts}
//       1:N 多精度解释：同一值产出多个精度层次，每层可独立重建原值
//   - sgn.MultiScaleView.interpret_levels(value, total_bits, split_bits_list)
//   - sgn.MultiScaleView.default_levels(total_bits) → list[int]
//   - sgn.MultiScaleView.interpret_batch(values, total_bits=32, split_bits_list=[]) → {split_bits: K x n}
//   - sgn.MultiScaleView.is_exact(value, total_bits, split_bits) → bool
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "multiscale_view.h"

namespace py = pybind11;

void register_multiscale_view(py::module_& m) {
    py::class_<sgn::MultiScaleView>(m, "MultiScaleView",
        "MSInt 1:N 多精度解释：一次搬运，同时产出多个精度层次")
        .def_static("interpret", &sgn::MultiScaleView::interpret,
                    py::arg("value"), py::arg("total_bits") = 32,
                    "1:N 解释：返回 {split_bits: parts}，每层可独立重建原值")
        .def_static("interpret_levels", &sgn::MultiScaleView::interpret_levels,
                    py::arg("value"), py::arg("total_bits"), py::arg("split_bits_list"),
                    "按指定精度层次列表解释")
        .def_static("default_levels", &sgn::MultiScaleView::default_levels,
                    py::arg("total_bits"),
                    "默认精度层次（total_bits/2 递减到 4）")
        .def_static("interpret_batch", &sgn::MultiScaleView::interpret_batch,
                    py::arg("values"), py::arg("total_bits") = 32,
                    py::arg("split_bits_list") = std::vector<int>(),
                    "批量 1:N 解释：返回 {split_bits: K x n 矩阵}")
        .def_static("is_exact", &sgn::MultiScaleView::is_exact,
                    py::arg("value"), py::arg("total_bits"), py::arg("split_bits"),
                    "一致性检查：该精度层次重建后等于原值");
}
