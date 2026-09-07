// msint_view_bindings.cpp - pybind11 绑定（注册到 sgn 模块）
//
// 暴露的 Python API：
//   - sgn.MSIntView.bitsplit(raw, total_bits, target_bits) → list[int]
//   - sgn.MSIntView.bitsplit_index(raw, total_bits, target_bits, idx) → int
//   - sgn.MSIntView.concat(values, bits_list) → int
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "msint_view.h"

namespace py = pybind11;

void register_msint_view(py::module_& m) {
    py::class_<sgn::MSIntView>(m, "MSIntView")
        .def_static("bitsplit", &sgn::MSIntView::bitsplit,
                    py::arg("raw"), py::arg("total_bits"), py::arg("target_bits"),
                    "位拆分：把 raw 按 target_bits 拆分，低位在前")
        .def_static("bitsplit_index", &sgn::MSIntView::bitsplit_index,
                    py::arg("raw"), py::arg("total_bits"),
                    py::arg("target_bits"), py::arg("idx"),
                    "位拆分取第 idx 个分片（0-based，低位在前）")
        .def_static("concat", &sgn::MSIntView::concat,
                    py::arg("values"), py::arg("bits_list"),
                    "拼接：第一个值在高位，最后一个在低位")
        .def_static("concat_signed", &sgn::MSIntView::concat_signed,
                    py::arg("values"), py::arg("bits_list"),
                    "拼接（返回有符号 int64）");
}
