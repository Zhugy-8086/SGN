/**
 * @file pysgn_hc4_pshufb.cpp
 * @brief pybind11 绑定: HC4 PSHUFB LUT int4×int4→int8 查表乘法
 * @version 1.6.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "hc4_pshufb.h"

namespace py = pybind11;

/* ============================================================================
 * AVX2 检测
 * ============================================================================ */

static int py_detect_avx2() {
    return hc4_pshufb_detect_avx2();
}

/* ============================================================================
 * PSHUFB 批量乘法验证：32 个 uint4 × uint4 → uint8
 * ============================================================================ */

/* 安全审计 2026-08-16 G1：py::array_t<T> 不带 c_style/forcecast 标志时
 * 接受任意 strides 的数组，内核按 C-contiguous 线性访问非连续数组会
 * 产生静默数据错误——统一加 c_style|forcecast（读入场景，转换副本无害） */

static py::array_t<uint8_t> py_pshufb_mul_32(
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> a,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> b) {

    auto a_buf = a.request();
    auto b_buf = b.request();

    if (a_buf.size != 32 || b_buf.size != 32)
        throw std::runtime_error("a and b must have 32 elements");

    const uint8_t* a_ptr = (const uint8_t*)a_buf.ptr;
    const uint8_t* b_ptr = (const uint8_t*)b_buf.ptr;

    py::array_t<uint8_t> out(32);
    auto out_buf = out.request();
    uint8_t* out_ptr = (uint8_t*)out_buf.ptr;

    hc4_pshufb_mul_32(a_ptr, b_ptr, out_ptr);
    return out;
}

/* ============================================================================
 * HC4 PSHUFB matmul（uint4 × uint4 → int32 累加）
 * ============================================================================ */

static py::array_t<int32_t> py_matmul(
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> a,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> b,
        uint32_t m, uint32_t k, uint32_t n) {

    auto a_buf = a.request();
    auto b_buf = b.request();

    // 安全审计 2026-08-30 F4：使用 int64_t 计算乘积避免 uint32 溢出
    py::ssize_t expected_a = static_cast<py::ssize_t>(m) * static_cast<py::ssize_t>(k);
    py::ssize_t expected_b = static_cast<py::ssize_t>(k) * static_cast<py::ssize_t>(n);
    if (a_buf.size != expected_a)
        throw std::runtime_error("a size mismatch");
    if (b_buf.size != expected_b)
        throw std::runtime_error("b size mismatch");

    const uint8_t* a_ptr = (const uint8_t*)a_buf.ptr;
    const uint8_t* b_ptr = (const uint8_t*)b_buf.ptr;

    py::array_t<int32_t> out({(py::ssize_t)m, (py::ssize_t)n});
    auto out_buf = out.request();
    int32_t* out_ptr = (int32_t*)out_buf.ptr;

    /* 安全审计 2026-08-16 G2：纯 C 计算，释放 GIL */
    {
        py::gil_scoped_release release;
        hc4_pshufb_matmul(a_ptr, b_ptr, m, k, n, out_ptr);
    }
    return out;
}

/* ============================================================================
 * 便捷封装：float → float 的 HC4 PSHUFB 量化 matmul
 * ============================================================================ */

static py::array_t<float> py_quantized_matmul(
        py::array_t<float, py::array::c_style | py::array::forcecast> x,
        py::array_t<float, py::array::c_style | py::array::forcecast> w,
        uint32_t m, uint32_t k, uint32_t n) {

    auto x_buf = x.request();
    auto w_buf = w.request();

    // 安全审计 2026-08-30 F4：使用 int64_t 计算乘积避免 uint32 溢出
    py::ssize_t expected_x = static_cast<py::ssize_t>(m) * static_cast<py::ssize_t>(k);
    py::ssize_t expected_w = static_cast<py::ssize_t>(k) * static_cast<py::ssize_t>(n);
    if (x_buf.size != expected_x)
        throw std::runtime_error("x size mismatch");
    if (w_buf.size != expected_w)
        throw std::runtime_error("w size mismatch");

    const float* x_ptr = (const float*)x_buf.ptr;
    const float* w_ptr = (const float*)w_buf.ptr;

    py::array_t<float> out({(py::ssize_t)m, (py::ssize_t)n});
    auto out_buf = out.request();
    float* out_ptr = (float*)out_buf.ptr;

    {
        py::gil_scoped_release release;
        hc4_pshufb_quantized_matmul(x_ptr, w_ptr, m, k, n, out_ptr);
    }
    return out;
}

/* ============================================================================
 * LUT 查询（调试用）
 * ============================================================================ */

static py::array_t<uint8_t> py_get_lut() {
    py::array_t<uint8_t> lut(256);
    auto buf = lut.request();
    uint8_t* ptr = (uint8_t*)buf.ptr;
    for (int b = 0; b < 16; ++b) {
        for (int a = 0; a < 16; ++a) {
            ptr[b * 16 + a] = (uint8_t)(a * b);
        }
    }
    return lut;
}

/* ============================================================================
 * 模块定义
 * ============================================================================ */

void register_hc4_pshufb(py::module_& m) {
    m.doc() = "HC4 PSHUFB LUT - int4×int4→int8 lookup table multiplication — 合并到 sgn.hc4 子模块";

    m.def("detect_avx2", &py_detect_avx2, "Detect AVX2 support");

    m.def("pshufb_mul_32", &py_pshufb_mul_32,
          "PSHUFB batch multiply: 32 uint4 × uint4 → 32 uint8",
          py::arg("a"), py::arg("b"));

    m.def("matmul", &py_matmul,
          "HC4 PSHUFB matmul (uint4×uint4→int32 accumulate)",
          py::arg("a"), py::arg("b"), py::arg("m"), py::arg("k"), py::arg("n"));

    m.def("quantized_matmul", &py_quantized_matmul,
          "Convenience: HC4 PSHUFB quantized matmul (float→float)",
          py::arg("x"), py::arg("w"), py::arg("m"), py::arg("k"), py::arg("n"));

    m.def("get_lut", &py_get_lut, "Get 16×16 multiplication LUT (256 bytes)");
}
