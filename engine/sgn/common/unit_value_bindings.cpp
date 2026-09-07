// unit_value_bindings.cpp - pybind11 绑定（注册到 sgn 模块）
//
// 设计：与 value_spec_bindings.cpp 一致，提供 register_unit_value(py::module_&)
//       注册函数，由 placeholder.cpp 在模块初始化时调用。
//
// 暴露的 Python API：
//   - sgn.UnitValue：统一单位值类
//       - UnitValue(raw=0, spec=ValueSpec())
//       - .raw / .spec 读写属性
//       - .to_float(scale) → float
//       - UnitValue.from_float(x, scale, spec) → UnitValue
//       - UnitValue.signed_max_for_bits(bits) → int
//   - sgn.batch_to_float(unit_values, scales) → numpy.ndarray
//       per-tensor (scales=float) 或 per-layer (scales=list[float])
//   - sgn.batch_from_float(values, scales, spec) → list[UnitValue]
//       per-tensor (scales=float) 或 per-layer (scales=list[float])

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <string>

#include "unit_value.h"

namespace py = pybind11;

/* 注册函数：由 placeholder.cpp 的 PYBIND11_MODULE(sgn, m) 调用 */
void register_unit_value(py::module_& m) {
    py::class_<sgn::UnitValue>(m, "UnitValue")
        .def(py::init<int64_t, sgn::ValueSpec>(),
             py::arg("raw") = 0,
             py::arg("spec") = sgn::ValueSpec())
        .def_readwrite("raw", &sgn::UnitValue::raw)
        .def_readwrite("spec", &sgn::UnitValue::spec)
        .def("to_float", &sgn::UnitValue::to_float, py::arg("scale"),
             "Dequantize: return raw * scale.\n"
             "Covers HC8WeightSchema.dequantize pattern (per-tensor scalar scale).")
        .def_static("from_float", &sgn::UnitValue::from_float,
                    py::arg("x"), py::arg("scale"), py::arg("spec"),
                    "Quantize: round(x/scale).clamp(-signed_max, signed_max).\n"
                    "signed_max = 2^(bits-1) - 1 (symmetric signed range).")
        .def_static("signed_max_for_bits", &sgn::UnitValue::signed_max_for_bits,
                    py::arg("bits"),
                    "Return signed symmetric max: 2^(bits-1) - 1.\n"
                    "HC8: 127, HC12: 2047, HC16: 32767, HC20: 524287, HC24: 8388607.")
        .def("__eq__", &sgn::UnitValue::operator==)
        .def("__repr__", [](const sgn::UnitValue& v) {
            return "UnitValue(raw=" + std::to_string(v.raw) +
                   ", bits=" + std::to_string(v.spec.bits) + ")";
        });

    // 批量反量化：per-tensor / per-layer 场景
    // unit_values: list of UnitValue
    // scales: float (per-tensor) 或 list[float] (per-layer)
    // 返回: numpy.ndarray[float32]
    m.def("batch_to_float", [](py::list unit_values, py::object scales) {
        size_t n = unit_values.size();
        py::array_t<float> result(n);
        auto buf = result.mutable_data();
        // 判断标量 vs 序列
        bool is_scalar = py::isinstance<py::float_>(scales) ||
                         py::isinstance<py::int_>(scales);
        if (is_scalar) {
            float scale = scales.cast<float>();
            for (size_t i = 0; i < n; ++i) {
                sgn::UnitValue uv = unit_values[i].cast<sgn::UnitValue>();
                buf[i] = uv.to_float(scale);
            }
        } else {
            py::list scale_list = scales.cast<py::list>();
            for (size_t i = 0; i < n; ++i) {
                sgn::UnitValue uv = unit_values[i].cast<sgn::UnitValue>();
                float scale = scale_list[i].cast<float>();
                buf[i] = uv.to_float(scale);
            }
        }
        return result;
    }, py::arg("unit_values"), py::arg("scales"),
       "Batch dequantize: per-tensor (single scale) or per-layer (scale array).\n"
       "Returns numpy.ndarray[float32].");

    // 批量量化：per-tensor / per-layer 场景
    // values: numpy.ndarray[float32] 或 list[float]
    // scales: float (per-tensor) 或 list[float] (per-layer)
    // spec: ValueSpec
    // 返回: list[UnitValue]
    m.def("batch_from_float", [](py::object values, py::object scales,
                                  sgn::ValueSpec spec) {
        // 将 values 转为 float 序列
        std::vector<float> vals;
        if (py::isinstance<py::array>(values)) {
            py::array_t<float> arr = values.cast<py::array_t<float>>();
            auto buf = arr.data();
            size_t n = arr.size();
            vals.assign(buf, buf + n);
        } else {
            py::list lst = values.cast<py::list>();
            vals.reserve(lst.size());
            for (auto item : lst) vals.push_back(item.cast<float>());
        }
        size_t n = vals.size();
        py::list result;
        bool is_scalar = py::isinstance<py::float_>(scales) ||
                         py::isinstance<py::int_>(scales);
        if (is_scalar) {
            float scale = scales.cast<float>();
            for (size_t i = 0; i < n; ++i) {
                result.append(sgn::UnitValue::from_float(vals[i], scale, spec));
            }
        } else {
            py::list scale_list = scales.cast<py::list>();
            for (size_t i = 0; i < n; ++i) {
                float scale = scale_list[i].cast<float>();
                result.append(sgn::UnitValue::from_float(vals[i], scale, spec));
            }
        }
        return result;
    }, py::arg("values"), py::arg("scales"), py::arg("spec"),
       "Batch quantize: per-tensor (single scale) or per-layer (scale array).\n"
       "Returns list[UnitValue].");
}
