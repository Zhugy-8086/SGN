// value_spec_bindings.cpp - pybind11 绑定（注册到 sgn 模块）
//
// 设计：不定义独立的 PYBIND11_MODULE（sgn 模块的 PYBIND11_MODULE 已在
//       placeholder.cpp 中定义），而是提供 register_value_spec(py::module_&)
//       注册函数，由 placeholder.cpp 在模块初始化时调用。
//       这是 pybind11 多文件模块的标准模式（与 hc/col2im_bindings.cpp 一致）。
//
// 暴露的 Python API：
//   - sgn.ScaleFn：枚举（MAX/RMS/L2/P95）
//   - sgn.ValueSpec：精度规格类
//       - ValueSpec(bits=16, scale=ScaleFn.MAX)
//       - .bits / .scale 读写属性
//       - .to_max_range() → int
//       - ValueSpec.from_max_range(max_range) → ValueSpec
//       - __eq__ / __repr__

#include <pybind11/pybind11.h>

#include <stdexcept>  /* invalid_argument（安全审计 2026-08-16 K3 bits 校验） */
#include <string>

#include "value_spec.h"

namespace py = pybind11;

/* 注册函数：由 placeholder.cpp 的 PYBIND11_MODULE(sgn, m) 调用 */
void register_value_spec(py::module_& m) {
    py::enum_<sgn::ScaleFn>(m, "ScaleFn")
        .value("MAX", sgn::ScaleFn::MAX)
        .value("RMS", sgn::ScaleFn::RMS)
        .value("L2", sgn::ScaleFn::L2)
        .value("P95", sgn::ScaleFn::P95)
        .export_values();

    py::class_<sgn::ValueSpec>(m, "ValueSpec")
        .def(py::init<uint8_t, sgn::ScaleFn>(),
             py::arg("bits") = 16,
             py::arg("scale") = sgn::ScaleFn::MAX)
        // 安全审计 2026-08-16 K3：bits 改用带校验 property——原 def_readwrite
        // 对 spec.bits = 999 静默截断为 uint8（231），值域错误不可见
        .def_property("bits",
            [](const sgn::ValueSpec& v) { return v.bits; },
            [](sgn::ValueSpec& v, py::object new_bits) {
                int64_t b = new_bits.cast<int64_t>();
                if (b < 0 || b > 255) {
                    throw std::invalid_argument(
                        "ValueSpec.bits must be in [0, 255], got " +
                        std::to_string(b));
                }
                v.bits = static_cast<uint8_t>(b);
            })
        .def_readwrite("scale", &sgn::ValueSpec::scale)
        .def("to_max_range", &sgn::ValueSpec::to_max_range,
             "Convert to max_range (= 2^bits - 1) for legacy Level system compatibility.")
        .def("__eq__", &sgn::ValueSpec::operator==)
        .def("__repr__", [](const sgn::ValueSpec& v) {
            return "ValueSpec(bits=" + std::to_string(v.bits) + ")";
        })
        .def_static("from_max_range", &sgn::ValueSpec::from_max_range,
                    py::arg("max_range"),
                    "Construct ValueSpec from max_range (= 2^bits - 1).\n"
                    "Non 2^n-1 values round up to the next bit width.");
}
