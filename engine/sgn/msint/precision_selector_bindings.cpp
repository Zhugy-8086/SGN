// precision_selector_bindings.cpp - pybind11 绑定（注册到 sgn 模块）
//
// 暴露的 Python API：
//   - sgn.PrecisionSelector(total_bits, options, thresholds)
//   - sgn.PrecisionSelector.default_selector()
//   - ps.select(importance) → split_bits
//   - ps.interpret(value, importance) → parts
//   - ps.interpret_batch(values, importances) → list[list[int]]
//   - ps.total_bits / ps.options / ps.thresholds
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "precision_selector.h"

namespace py = pybind11;

void register_precision_selector(py::module_& m) {
    py::class_<sgn::PrecisionSelector>(m, "PrecisionSelector",
        "MSInt Level 逐元素精度选择：按重要性选择拆分粒度")
        .def(py::init<int, std::vector<int>, std::vector<int64_t>>(),
             py::arg("total_bits"), py::arg("options"), py::arg("thresholds"),
             "options 从粗到细（split_bits 递减），thresholds 升序长度=options-1")
        .def_static("default_selector", &sgn::PrecisionSelector::default_selector,
                    "默认选择器：total_bits=32, options={16,8,4}, thresholds={100,1000}")
        .def("select", &sgn::PrecisionSelector::select, py::arg("importance"),
             "按重要性选择该元素的拆分粒度 split_bits")
        .def("interpret", &sgn::PrecisionSelector::interpret,
             py::arg("value"), py::arg("importance"),
             "按选定粒度对值做 1:N 解释（返回 parts）")
        .def("interpret_batch", &sgn::PrecisionSelector::interpret_batch,
             py::arg("values"), py::arg("importances"),
             "批量逐元素选择 + 解释（锯齿形二维数组）")
        .def_property_readonly("total_bits", &sgn::PrecisionSelector::total_bits)
        .def_property_readonly("options", &sgn::PrecisionSelector::options)
        .def_property_readonly("thresholds", &sgn::PrecisionSelector::thresholds);
}
