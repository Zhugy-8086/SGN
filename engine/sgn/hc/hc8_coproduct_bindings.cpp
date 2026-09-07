// hc8_coproduct_bindings.cpp - pybind11 绑定 HC8Coproduct 系列
//
// 暴露的 Python API：
//   - sgn.HC8Coproduct: 单元素 6 字节余积存储
//       - HC8Coproduct(gradient, variance, kalman)
//       - .gradient / .variance / .kalman (property)
//       - .set_gradient(v) / .set_variance(v) / .set_kalman(v)
//       - .gradient_bitsplit() → (high, low)
//       - .variance_log2() → float
//       - .serialize() → hex string
//   - sgn.HC8CoproductArray: 批量余积存储
//       - HC8CoproductArray(n_elements, per_element_variance=True)
//       - .n_elements / .per_element_variance / .total_bytes
//       - .get_gradient(i) / .set_gradient(i, v)
//       - .get_variance(i) / .set_variance(i, v)
//       - .get_kalman(i) / .set_kalman(i, v)
//       - .batch_get_gradient() → numpy int16
//       - .batch_set_gradient(arr)
//   - sgn.HC8KalmanFilter: 整数卡尔曼滤波器 (Q15)
//       - HC8KalmanFilter(n_elements, P0_int, R_int)
//       - .update(x_arr, z_arr) → 更新 x_arr in-place
//       - .covariance() → numpy int16
//       - .compute_delta_bits(R_int, P_mean) → float (静态方法)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "hc8_coproduct.h"

namespace py = pybind11;

void register_hc8_coproduct(py::module_& m) {
    // ============================================================
    // HC8Coproduct
    // ============================================================
    py::class_<sgn::HC8Coproduct>(m, "HC8Coproduct")
        .def(py::init<>(), "默认构造: 全零")
        .def(py::init<int16_t, uint16_t, int16_t>(),
             py::arg("gradient"), py::arg("variance"), py::arg("kalman"),
             "从三个 int16 字段构造")
        .def_property("gradient", &sgn::HC8Coproduct::gradient,
                      &sgn::HC8Coproduct::set_gradient,
                      "梯度 (signed int16, Byte 0-1)")
        .def_property("variance", &sgn::HC8Coproduct::variance,
                      &sgn::HC8Coproduct::set_variance,
                      "方差统计 (unsigned int16, Byte 2-3)")
        .def_property("kalman", &sgn::HC8Coproduct::kalman,
                      &sgn::HC8Coproduct::set_kalman,
                      "卡尔曼状态 (signed int16, Byte 4-5)")
        .def("gradient_bitsplit", &sgn::HC8Coproduct::gradient_bitsplit,
             "返回 (high_signed, low_unsigned), 高字节+低字节分离")
        .def("set_gradient_bitsplit", &sgn::HC8Coproduct::set_gradient_bitsplit,
             py::arg("high"), py::arg("low"),
             "从 bitsplit 写入梯度")
        .def("variance_log2", &sgn::HC8Coproduct::variance_log2,
             "返回 log2(variance + 1), 驱动 Level_b 调度")
        .def("serialize", &sgn::HC8Coproduct::serialize,
             "16 进制字符串序列化")
        .def("__eq__", &sgn::HC8Coproduct::operator==)
        .def("__repr__", [](const sgn::HC8Coproduct& c) {
            return "<HC8Coproduct grad=" + std::to_string(c.gradient()) +
                   " var=" + std::to_string(c.variance()) +
                   " kal=" + std::to_string(c.kalman()) +
                   " hex=" + c.serialize() + ">";
        });

    // ============================================================
    // HC8CoproductArray
    // ============================================================
    py::class_<sgn::HC8CoproductArray>(m, "HC8CoproductArray")
        .def(py::init<int, bool>(),
             py::arg("n_elements"),
             py::arg("per_element_variance") = true,
             "构造 n_elements 个元素的余积存储数组")
        .def_property_readonly("n_elements", &sgn::HC8CoproductArray::n_elements)
        .def_property_readonly("per_element_variance",
                               &sgn::HC8CoproductArray::per_element_variance)
        .def_property_readonly("total_bytes", &sgn::HC8CoproductArray::total_bytes)
        // 单元素访问
        .def("at", &sgn::HC8CoproductArray::at, py::arg("i"),
             "获取第 i 个元素的完整 HC8Coproduct")
        .def("set_at", &sgn::HC8CoproductArray::set_at,
             py::arg("i"), py::arg("cop"),
             "设置第 i 个元素的完整 HC8Coproduct")
        // 梯度
        .def("get_gradient", &sgn::HC8CoproductArray::get_gradient, py::arg("i"))
        .def("set_gradient", &sgn::HC8CoproductArray::set_gradient,
             py::arg("i"), py::arg("value"))
        // 方差
        .def("get_variance", &sgn::HC8CoproductArray::get_variance, py::arg("i"))
        .def("set_variance", &sgn::HC8CoproductArray::set_variance,
             py::arg("i"), py::arg("value"))
        // per-layer 方差
        .def("get_layer_variance",
             &sgn::HC8CoproductArray::get_layer_variance)
        .def("set_layer_variance",
             &sgn::HC8CoproductArray::set_layer_variance, py::arg("value"))
        // 卡尔曼
        .def("get_kalman", &sgn::HC8CoproductArray::get_kalman, py::arg("i"))
        .def("set_kalman", &sgn::HC8CoproductArray::set_kalman,
             py::arg("i"), py::arg("value"))
        // 批量 numpy 互操作
        .def("batch_get_gradient",
            [](const sgn::HC8CoproductArray& self) {
                auto result = py::array_t<int16_t>(self.n_elements());
                self.batch_get_gradient(static_cast<int16_t*>(result.request().ptr));
                return result;
            },
            "批量读取梯度 → numpy int16 数组")
        .def("batch_set_gradient",
            [](sgn::HC8CoproductArray& self,
               py::array_t<int16_t, py::array::c_style> arr) {
                if (arr.size() < self.n_elements()) {
                    throw std::invalid_argument("数组长度不足");
                }
                self.batch_set_gradient(static_cast<const int16_t*>(arr.data(0)));
            },
            py::arg("arr"),
            "批量写入梯度 ← numpy int16 数组")
        .def("batch_get_kalman",
            [](const sgn::HC8CoproductArray& self) {
                auto result = py::array_t<int16_t>(self.n_elements());
                self.batch_get_kalman(static_cast<int16_t*>(result.request().ptr));
                return result;
            },
            "批量读取卡尔曼状态 → numpy int16 数组")
        .def("batch_set_kalman",
            [](sgn::HC8CoproductArray& self,
               py::array_t<int16_t, py::array::c_style> arr) {
                if (arr.size() < self.n_elements()) {
                    throw std::invalid_argument("数组长度不足");
                }
                self.batch_set_kalman(static_cast<const int16_t*>(arr.data(0)));
            },
            py::arg("arr"),
            "批量写入卡尔曼状态 ← numpy int16 数组")
        .def("__repr__", [](const sgn::HC8CoproductArray& a) {
            return "<HC8CoproductArray n=" + std::to_string(a.n_elements()) +
                   " per_elem_var=" + (a.per_element_variance() ? "True" : "False") +
                   " total_bytes=" + std::to_string(a.total_bytes()) + ">";
        });

    // ============================================================
    // HC8KalmanFilter
    // ============================================================
    py::class_<sgn::HC8KalmanFilter>(m, "HC8KalmanFilter")
        .def(py::init<int, int16_t, int16_t>(),
             py::arg("n_elements"), py::arg("P0_int"), py::arg("R_int"),
             "构造整数卡尔曼滤波器 (Q15 定点)")
        .def_property_readonly("n_elements", &sgn::HC8KalmanFilter::n_elements)
        .def_property_readonly("R", &sgn::HC8KalmanFilter::R)
        .def("update",
            [](sgn::HC8KalmanFilter& self,
               py::array_t<int16_t, py::array::c_style> x_arr,
               py::array_t<int16_t, py::array::c_style> z_arr) {
                if (x_arr.size() < self.n_elements()) {
                    throw std::invalid_argument("x_arr 长度不足");
                }
                if (z_arr.size() < self.n_elements()) {
                    throw std::invalid_argument("z_arr 长度不足");
                }
                int16_t* x_ptr = static_cast<int16_t*>(x_arr.request().ptr);
                const int16_t* z_ptr = z_arr.data(0);
                self.update(x_ptr, z_ptr);
                return x_arr;  // 返回更新后的 x_arr (in-place)
            },
            py::arg("x_arr"), py::arg("z_arr"),
            "单步更新 (in-place 修改 x_arr)")
        .def("covariance",
            [](const sgn::HC8KalmanFilter& self) {
                const auto& cov = self.covariance();
                return py::array_t<int16_t>(
                    self.n_elements(), cov.data(), py::cast(self));
            },
            "当前协方差估计 → numpy int16")
        .def_static("compute_delta_bits",
                    &sgn::HC8KalmanFilter::compute_delta_bits,
                    py::arg("R_int"), py::arg("P_current_mean"),
                    "计算等效 bits 提升 Δb = ½log₂(R/P)")
        .def("__repr__", [](const sgn::HC8KalmanFilter& f) {
            return "<HC8KalmanFilter n=" + std::to_string(f.n_elements()) +
                   " R=" + std::to_string(f.R()) + ">";
        });
}
