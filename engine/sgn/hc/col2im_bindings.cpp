// col2im_bindings.cpp - pybind11 绑定（注册到 sgn 模块）
//
// 设计：不定义独立的 PYBIND11_MODULE（sgn 模块的 PYBIND11_MODULE 已在
//       placeholder.cpp 中定义），而是提供 register_col2im(py::module_&)
//       注册函数，由 placeholder.cpp 在模块初始化时调用。
//       这是 pybind11 多文件模块的标准模式。
//
// 暴露的 Python API：
//   - sgn.col2im_add(x_col, x_padded, B, C, kh, kw, H_out, W_out, stride,
//                     H_padded, W_padded)
//       原地累加：x_padded[b,c,i+ho*stride,j+wo*stride] += x_col[b,c,i,j,ho,wo]
//   - sgn.Col2im：C++ Col2im 类，方法 execute(...) 同上

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "col2im.h"

#include <stdexcept>
#include <string>

namespace py = pybind11;

/* col2im scatter-add（原地累加到 x_padded）
 *
 * Args:
 *   x_col:    (B, C, kh, kw, H_out, W_out) numpy float32, C-contiguous
 *   x_padded: (B, C, H_padded, W_padded) numpy float32, C-contiguous（原地修改）
 *   B, C, kh, kw, H_out, W_out, stride, H_padded, W_padded: 维度参数
 *
 * 安全审计 2026-08-16 D1：x_padded 是原地写入目标，禁止 forcecast——
 * forcecast 会在非连续/非 float32 时静默创建副本，+= 全部落入副本，
 * 用户原数组未被修改且无任何报错（梯度 scatter-add 静默丢失）。
 * 改为 py::object + buffer 协议手动校验：request() 返回原对象内存
 * （不复制），dtype/连续性不满足直接抛 TypeError。
 */
static void py_col2im_add(
    py::array_t<float, py::array::c_style | py::array::forcecast> x_col,
    py::object x_padded_obj,
    int B, int C, int kh, int kw,
    int H_out, int W_out, int stride,
    int H_padded, int W_padded
) {
    auto x_col_buf = x_col.request();

    /* x_padded：dtype 必须精确 float32（不接受任何转换） */
    if (!py::isinstance<py::array_t<float>>(x_padded_obj)) {
        throw std::runtime_error(
            "x_padded must be numpy.float32 array (in-place target, "
            "no implicit conversion; use np.ascontiguousarray(x, dtype=np.float32))");
    }
    py::array_t<float> x_padded = x_padded_obj.cast<py::array_t<float>>();
    /* C 连续性手动校验（array_t 转换可能静默拷贝，这里类型已匹配故借用原对象） */
    if (!(x_padded.flags() & py::array::c_style)) {
        throw std::runtime_error(
            "x_padded must be C-contiguous (in-place target; a copy would "
            "silently discard scatter-add results)");
    }
    auto x_padded_buf = x_padded.request();

    /* 形状校验：x_col 必须是 6D (B, C, kh, kw, H_out, W_out) */
    if (x_col_buf.ndim != 6) {
        throw std::runtime_error(
            "x_col must be 6D (B, C, kh, kw, H_out, W_out), got ndim=" +
            std::to_string(x_col_buf.ndim));
    }
    const py::ssize_t expected_col[6] = {B, C, kh, kw, H_out, W_out};
    for (int i = 0; i < 6; ++i) {
        if (x_col_buf.shape[i] != expected_col[i]) {
            throw std::runtime_error("x_col shape mismatch");
        }
    }

    /* 形状校验：x_padded 必须是 4D (B, C, H_padded, W_padded) */
    if (x_padded_buf.ndim != 4) {
        throw std::runtime_error(
            "x_padded must be 4D (B, C, H_padded, W_padded), got ndim=" +
            std::to_string(x_padded_buf.ndim));
    }
    const py::ssize_t expected_pad[4] = {B, C, H_padded, W_padded};
    for (int i = 0; i < 4; ++i) {
        if (x_padded_buf.shape[i] != expected_pad[i]) {
            throw std::runtime_error("x_padded shape mismatch");
        }
    }

    /* 维度合法性检查 */
    if (B <= 0 || C <= 0 || kh <= 0 || kw <= 0 ||
        H_out <= 0 || W_out <= 0 || stride <= 0 ||
        H_padded <= 0 || W_padded <= 0) {
        throw std::runtime_error("all dimension arguments must be positive");
    }
    /* 越界检查 */
    if (kh - 1 + (H_out - 1) * stride >= H_padded) {
        throw std::runtime_error("col2im out of bounds on H axis");
    }
    if (kw - 1 + (W_out - 1) * stride >= W_padded) {
        throw std::runtime_error("col2im out of bounds on W axis");
    }

    const float* col_ptr = (const float*)x_col_buf.ptr;
    float* pad_ptr = (float*)x_padded_buf.ptr;

    /* 释放 GIL 进行纯 C 计算（无 Python 对象访问） */
    {
        py::gil_scoped_release release;
        Col2im op;
        op.execute(col_ptr, pad_ptr,
                   B, C, kh, kw, H_out, W_out, stride,
                   H_padded, W_padded);
    }
}

/* 注册函数：由 placeholder.cpp 的 PYBIND11_MODULE(sgn, m) 调用 */
void register_col2im(py::module_& m) {
    m.def("col2im_add", &py_col2im_add,
          py::arg("x_col"), py::arg("x_padded"),
          py::arg("B"), py::arg("C"), py::arg("kh"), py::arg("kw"),
          py::arg("H_out"), py::arg("W_out"), py::arg("stride"),
          py::arg("H_padded"), py::arg("W_padded"),
          "col2im scatter-add (in-place accumulate into x_padded).\n\n"
          "x_padded[b,c,i+ho*stride,j+wo*stride] += x_col[b,c,i,j,ho,wo]\n\n"
          "Args:\n"
          "  x_col:    (B, C, kh, kw, H_out, W_out) float32, C-contiguous\n"
          "  x_padded: (B, C, H_padded, W_padded) float32, C-contiguous (modified in-place)\n"
          "  B, C, kh, kw, H_out, W_out, stride, H_padded, W_padded: dims\n");

    py::class_<Col2im>(m, "Col2im")
        .def(py::init<>())
        .def("execute",
             [](Col2im&,
                py::array_t<float, py::array::c_style | py::array::forcecast> x_col,
                py::object x_padded,
                int B, int C, int kh, int kw,
                int H_out, int W_out, int stride,
                int H_padded, int W_padded) {
                 /* x_padded 经 py::object 透传，由 py_col2im_add 统一校验
                  *（安全审计 2026-08-16 D1，禁止 forcecast 静默副本） */
                 py_col2im_add(x_col, x_padded,
                               B, C, kh, kw, H_out, W_out, stride,
                               H_padded, W_padded);
             },
             py::arg("x_col"), py::arg("x_padded"),
             py::arg("B"), py::arg("C"), py::arg("kh"), py::arg("kw"),
             py::arg("H_out"), py::arg("W_out"), py::arg("stride"),
             py::arg("H_padded"), py::arg("W_padded"),
             "Execute col2im scatter-add in-place (same semantics as sgn.col2im_add).");
}
