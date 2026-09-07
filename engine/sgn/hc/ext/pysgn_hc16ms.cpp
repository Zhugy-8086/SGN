/**
 * @file pysgn_hc16ms.cpp
 * @brief pybind11 绑定: HC16MS - MSInt int16 容器多视角存储
 * @version 1.5.0
 *
 * 暴露 HC16MS 的接口给 Python，验证多视角存储的正确性。
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "hc16ms.h"

#include <cstdint>  /* INT32_MAX（安全审计 2026-08-16 H4） */

namespace py = pybind11;

/* 安全审计 2026-08-16 H2：hc16ms_t 与 int16 必须等宽（绑定层将 int16 数组
 * 缓冲区直接 reinterpret 为 hc16ms_t* 访问，不等宽即静默越界） */
static_assert(sizeof(hc16ms_t) == sizeof(int16_t),
              "hc16ms_t must be int16-sized (bindings reinterpret int16 buffers)");

/* 安全审计 2026-08-16 H1：py::array_t<T> 不带 c_style/forcecast 时接受任意
 * strides，内核按 C-contiguous 线性访问非连续数组产生静默数据错误——
 * 读入场景统一加 c_style|forcecast（转换副本无害，输出为新分配数组） */
using f32_arr = py::array_t<float, py::array::c_style | py::array::forcecast>;
using i16_arr = py::array_t<int16_t, py::array::c_style | py::array::forcecast>;

/* ============================================================================
 * AVX2 检测
 * ============================================================================ */

static int py_detect_avx2() {
    return hc16ms_detect_avx2();
}

/* ============================================================================
 * 参数安全校验（安全审计 2026-08-30 F4）
 * ============================================================================
 * 将 uint32_t 安全转换为 py::ssize_t，避免乘积回绕漏洞
 */
static inline py::ssize_t checked_u32(uint32_t n) {
    if (n > (py::ssize_t)UINT32_MAX) {
        throw std::runtime_error("dimension exceeds UINT32_MAX: " + std::to_string(n));
    }
    return static_cast<py::ssize_t>(n);
}

/* ============================================================================
 * HC16 视角量化/反量化
 * ============================================================================ */

static py::tuple py_quantize_hc16(f32_arr w) {
    auto buf = w.request();
    if (buf.ndim != 1) throw std::runtime_error("w must be 1-D");

    uint32_t n = (uint32_t)buf.size;
    const float* w_ptr = (const float*)buf.ptr;

    float scale = hc16ms_quant_compute_scale_hc16(w_ptr, n);

    /* 输出 int16 数组（HC16 视角读取结果） */
    py::array_t<int16_t> out(n);
    auto out_buf = out.request();
    int16_t* out_ptr = (int16_t*)out_buf.ptr;

    /* 直接写入 int16 数组（hc16ms_t.raw 就是 int16） */
    hc16ms_t* tmp = (hc16ms_t*)out_ptr;
    hc16ms_quantize_hc16(w_ptr, n, scale, tmp);

    return py::make_tuple(out, scale);
}

static py::array_t<float> py_dequantize_hc16(i16_arr h, float scale) {
    auto buf = h.request();
    if (buf.ndim != 1) throw std::runtime_error("h must be 1-D");

    uint32_t n = (uint32_t)buf.size;
    const hc16ms_t* h_ptr = (const hc16ms_t*)buf.ptr;

    py::array_t<float> out(n);
    auto out_buf = out.request();
    float* out_ptr = (float*)out_buf.ptr;

    hc16ms_dequantize_hc16(h_ptr, n, scale, out_ptr);
    return out;
}

/* ============================================================================
 * HC8 视角量化/反量化
 * ============================================================================ */

static py::tuple py_quantize_hc8(f32_arr w) {
    auto buf = w.request();
    if (buf.ndim != 1) throw std::runtime_error("w must be 1-D");

    uint32_t n_float = (uint32_t)buf.size;
    if (n_float % 2 != 0) throw std::runtime_error("w length must be even (2 per hc16ms_t)");

    uint32_t n = n_float / 2;  /* hc16ms_t 数组长度 */
    const float* w_ptr = (const float*)buf.ptr;

    float scale = hc16ms_quant_compute_scale_hc8(w_ptr, n_float);

    /* 输出 int16 数组（hc16ms_t.raw，2 个 int8 packed） */
    py::array_t<int16_t> out(n);
    auto out_buf = out.request();
    int16_t* out_ptr = (int16_t*)out_buf.ptr;

    hc16ms_t* tmp = (hc16ms_t*)out_ptr;
    hc16ms_quantize_hc8(w_ptr, n, scale, tmp);

    return py::make_tuple(out, scale);
}

static py::array_t<float> py_dequantize_hc8(i16_arr h, float scale) {
    auto buf = h.request();
    if (buf.ndim != 1) throw std::runtime_error("h must be 1-D");

    uint32_t n = (uint32_t)buf.size;  /* hc16ms_t 数组长度 */
    const hc16ms_t* h_ptr = (const hc16ms_t*)buf.ptr;

    // 安全审计 2026-08-30 F4：使用 int64_t 计算乘积避免 uint32 溢出
    py::ssize_t out_size = checked_u32(2) * checked_u32(n);
    py::array_t<float> out(out_size);  /* 2 个 float/hc16ms_t */
    auto out_buf = out.request();
    float* out_ptr = (float*)out_buf.ptr;

    hc16ms_dequantize_hc8(h_ptr, n, scale, out_ptr);
    return out;
}

/* ============================================================================
 * 多视角读取验证
 * ============================================================================ */

static py::dict py_inspect_views(i16_arr h) {
    /* 输入 int16 数组，返回 HC16/HC8/HC4 三视角的值 */
    auto buf = h.request();
    if (buf.ndim != 1) throw std::runtime_error("h must be 1-D");

    /* 安全审计 2026-08-16 H4：n 接近 2^31 时 2*n/4*n 在 uint32 域溢出、
     * 4*i 下标回绕——提前以 ssize_t 校验元素上限 */
    if (buf.size > (py::ssize_t)(INT32_MAX / 4)) {
        throw std::runtime_error("h too large for view inspection (max ~2^29 elements)");
    }

    uint32_t n = (uint32_t)buf.size;
    const hc16ms_t* h_ptr = (const hc16ms_t*)buf.ptr;

    py::array_t<int16_t> hc16_vals(n);
    py::array_t<int8_t> hc8_vals(2 * n);
    py::array_t<uint8_t> hc4_vals(4 * n);

    auto v16 = hc16_vals.request();
    auto v8 = hc8_vals.request();
    auto v4 = hc4_vals.request();

    for (uint32_t i = 0; i < n; ++i) {
        /* HC16 视角 */
        ((int16_t*)v16.ptr)[i] = hc16ms_read_hc16(&h_ptr[i]);

        /* HC8 视角 */
        int8_t high, low;
        hc16ms_read_hc8(&h_ptr[i], &high, &low);
        ((int8_t*)v8.ptr)[2*i]     = low;
        ((int8_t*)v8.ptr)[2*i + 1] = high;

        /* HC4 视角 */
        uint8_t h0, h1, h2, h3;
        hc16ms_read_hc4(&h_ptr[i], &h0, &h1, &h2, &h3);
        ((uint8_t*)v4.ptr)[4*i]     = h0;
        ((uint8_t*)v4.ptr)[4*i + 1] = h1;
        ((uint8_t*)v4.ptr)[4*i + 2] = h2;
        ((uint8_t*)v4.ptr)[4*i + 3] = h3;
    }

    py::dict result;
    result["hc16"] = hc16_vals;
    result["hc8"] = hc8_vals;
    result["hc4"] = hc4_vals;
    return result;
}

/* ============================================================================
 * 切换视角读取验证（读取顺序切换）
 * ============================================================================ */

static py::dict py_inspect_views_swapped(i16_arr h) {
    /* 输入 int16 数组，返回 HC16/HC8/HC4 三视角的切换读取结果。
     * 与 py_inspect_views 结构相同，但调用 _swapped 系列读取函数。 */
    auto buf = h.request();
    if (buf.ndim != 1) throw std::runtime_error("h must be 1-D");

    /* 安全审计 2026-08-16 H4：同 py_inspect_views 的 2*n/4*n 溢出防护 */
    if (buf.size > (py::ssize_t)(INT32_MAX / 4)) {
        throw std::runtime_error("h too large for view inspection (max ~2^29 elements)");
    }

    uint32_t n = (uint32_t)buf.size;
    const hc16ms_t* h_ptr = (const hc16ms_t*)buf.ptr;

    py::array_t<int16_t> hc16_vals(n);
    py::array_t<int8_t> hc8_vals(2 * n);
    py::array_t<uint8_t> hc4_vals(4 * n);

    auto v16 = hc16_vals.request();
    auto v8 = hc8_vals.request();
    auto v4 = hc4_vals.request();

    for (uint32_t i = 0; i < n; ++i) {
        /* HC16 视角（切换：字节交换后的 int16） */
        ((int16_t*)v16.ptr)[i] = hc16ms_read_hc16_swapped(&h_ptr[i]);

        /* HC8 视角（切换：high/low 对调） */
        int8_t high, low;
        hc16ms_read_hc8_swapped(&h_ptr[i], &high, &low);
        ((int8_t*)v8.ptr)[2*i]     = low;
        ((int8_t*)v8.ptr)[2*i + 1] = high;

        /* HC4 视角（切换：nibble 顺序反转 [h0,h1,h2,h3]→[h3,h2,h1,h0]） */
        uint8_t h0, h1, h2, h3;
        hc16ms_read_hc4_swapped(&h_ptr[i], &h0, &h1, &h2, &h3);
        ((uint8_t*)v4.ptr)[4*i]     = h0;
        ((uint8_t*)v4.ptr)[4*i + 1] = h1;
        ((uint8_t*)v4.ptr)[4*i + 2] = h2;
        ((uint8_t*)v4.ptr)[4*i + 3] = h3;
    }

    py::dict result;
    result["hc16"] = hc16_vals;
    result["hc8"] = hc8_vals;
    result["hc4"] = hc4_vals;
    return result;
}

/* ============================================================================
 * HC16 视角 matmul
 * ============================================================================ */

static py::array_t<float> py_matmul_hc16(
        i16_arr a, i16_arr b,
        uint32_t m, uint32_t k, uint32_t n,
        float a_scale, float b_scale) {

    auto a_buf = a.request();
    auto b_buf = b.request();

    // 安全审计 2026-08-30 F4：使用 int64_t 计算乘积避免 uint32 溢出
    py::ssize_t expected_a = checked_u32(m) * checked_u32(k);
    py::ssize_t expected_b = checked_u32(k) * checked_u32(n);
    if (a_buf.size != expected_a)
        throw std::runtime_error("a size mismatch");
    if (b_buf.size != expected_b)
        throw std::runtime_error("b size mismatch");

    const hc16ms_t* a_ptr = (const hc16ms_t*)a_buf.ptr;
    const hc16ms_t* b_ptr = (const hc16ms_t*)b_buf.ptr;

    py::array_t<float> out({(py::ssize_t)m, (py::ssize_t)n});
    auto out_buf = out.request();
    float* out_ptr = (float*)out_buf.ptr;

    /* 安全审计 2026-08-16 H3：纯 C 计算释放 GIL */
    {
        py::gil_scoped_release release;
        hc16ms_matmul_hc16(a_ptr, b_ptr, m, k, n, a_scale, b_scale, out_ptr);
    }
    return out;
}

/* ============================================================================
 * HC16 切换视角 matmul（先对 A、B 做 bswap16，再调用正向 hc16ms_matmul_hc16）
 * ============================================================================ */

static py::array_t<float> py_matmul_hc16_swapped(
        i16_arr a, i16_arr b,
        uint32_t m, uint32_t k, uint32_t n,
        float a_scale, float b_scale) {

    auto a_buf = a.request();
    auto b_buf = b.request();

    // 安全审计 2026-08-30 F4：使用 int64_t 计算乘积避免 uint32 溢出
    py::ssize_t expected_a = checked_u32(m) * checked_u32(k);
    py::ssize_t expected_b = checked_u32(k) * checked_u32(n);
    if (a_buf.size != expected_a)
        throw std::runtime_error("a size mismatch");
    if (b_buf.size != expected_b)
        throw std::runtime_error("b size mismatch");

    const hc16ms_t* a_ptr = (const hc16ms_t*)a_buf.ptr;
    const hc16ms_t* b_ptr = (const hc16ms_t*)b_buf.ptr;

    py::array_t<float> out({(py::ssize_t)m, (py::ssize_t)n});
    auto out_buf = out.request();
    float* out_ptr = (float*)out_buf.ptr;

    {
        py::gil_scoped_release release;
        hc16ms_matmul_hc16_swapped(a_ptr, b_ptr, m, k, n, a_scale, b_scale, out_ptr);
    }
    return out;
}

/* ============================================================================
 * HC8 视角 matmul
 * ============================================================================ */

static py::array_t<float> py_matmul_hc8(
        i16_arr a, i16_arr b,
        uint32_t m, uint32_t k, uint32_t n,
        float a_scale, float b_scale) {

    auto a_buf = a.request();
    auto b_buf = b.request();

    // 安全审计 2026-08-30 F4：使用 int64_t 计算乘积避免 uint32 溢出
    py::ssize_t expected_a = checked_u32(m) * checked_u32(k);
    py::ssize_t expected_b = checked_u32(k) * checked_u32(n);
    if (a_buf.size != expected_a)
        throw std::runtime_error("a size mismatch");
    if (b_buf.size != expected_b)
        throw std::runtime_error("b size mismatch");

    const hc16ms_t* a_ptr = (const hc16ms_t*)a_buf.ptr;
    const hc16ms_t* b_ptr = (const hc16ms_t*)b_buf.ptr;

    py::array_t<float> out({(py::ssize_t)m, (py::ssize_t)n});
    auto out_buf = out.request();
    float* out_ptr = (float*)out_buf.ptr;

    {
        py::gil_scoped_release release;
        hc16ms_matmul_hc8(a_ptr, b_ptr, m, k, n, a_scale, b_scale, out_ptr);
    }
    return out;
}

/* ============================================================================
 * 便捷封装：quantized_matmul（float → float）
 * ============================================================================ */

static py::array_t<float> py_quantized_matmul_hc16(
        f32_arr x, f32_arr w,
        uint32_t m, uint32_t k, uint32_t n) {

    auto x_buf = x.request();
    auto w_buf = w.request();

    // 安全审计 2026-08-30 F4：使用 int64_t 计算乘积避免 uint32 溢出
    py::ssize_t expected_x = checked_u32(m) * checked_u32(k);
    py::ssize_t expected_w = checked_u32(k) * checked_u32(n);
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
        hc16ms_quantized_matmul_hc16(x_ptr, w_ptr, m, k, n, out_ptr);
    }
    return out;
}

static py::array_t<float> py_quantized_matmul_hc8(
        f32_arr x, f32_arr w,
        uint32_t m, uint32_t k, uint32_t n) {

    auto x_buf = x.request();
    auto w_buf = w.request();

    // 安全审计 2026-08-30 F4：使用 int64_t 计算乘积避免 uint32 溢出
    py::ssize_t expected_x = checked_u32(2) * checked_u32(m) * checked_u32(k);
    py::ssize_t expected_w = checked_u32(2) * checked_u32(k) * checked_u32(n);
    if (x_buf.size != expected_x)
        throw std::runtime_error("x size mismatch (expected 2*m*k for HC8 view)");
    if (w_buf.size != expected_w)
        throw std::runtime_error("w size mismatch (expected 2*k*n for HC8 view)");

    const float* x_ptr = (const float*)x_buf.ptr;
    const float* w_ptr = (const float*)w_buf.ptr;

    py::array_t<float> out({(py::ssize_t)m, (py::ssize_t)n});
    auto out_buf = out.request();
    float* out_ptr = (float*)out_buf.ptr;

    {
        py::gil_scoped_release release;
        hc16ms_quantized_matmul_hc8(x_ptr, w_ptr, m, k, n, out_ptr);
    }
    return out;
}

/* ============================================================================
 * 模块定义
 * ============================================================================ */

void register_hc16ms(py::module_& m) {
    m.doc() = "HC16MS - MSInt int16 container multi-view storage (Plan B) — 合并到 sgn.hc16ms 子模块";

    m.def("detect_avx2", &py_detect_avx2, "Detect AVX2 support");

    m.def("quantize_hc16", &py_quantize_hc16,
          "Quantize float to HC16 view (returns int16 array + scale)",
          py::arg("w"));

    m.def("dequantize_hc16", &py_dequantize_hc16,
          "Dequantize HC16 view to float",
          py::arg("h"), py::arg("scale"));

    m.def("quantize_hc8", &py_quantize_hc8,
          "Quantize float to HC8 view (2 int8 per hc16ms_t, returns int16 array + scale)",
          py::arg("w"));

    m.def("dequantize_hc8", &py_dequantize_hc8,
          "Dequantize HC8 view to float (2 float per hc16ms_t)",
          py::arg("h"), py::arg("scale"));

    m.def("inspect_views", &py_inspect_views,
          "Inspect all views (HC16/HC8/HC4) of an int16 array",
          py::arg("h"));

    m.def("inspect_views_swapped", &py_inspect_views_swapped,
          "Inspect all views (HC16/HC8/HC4) of an int16 array with swapped read order",
          py::arg("h"));

    m.def("matmul_hc16", &py_matmul_hc16,
          "HC16 view matmul (int16×int16→int64 accumulate)",
          py::arg("a"), py::arg("b"), py::arg("m"), py::arg("k"), py::arg("n"),
          py::arg("a_scale"), py::arg("b_scale"));

    m.def("matmul_hc16_swapped", &py_matmul_hc16_swapped,
          "HC16 view matmul with swapped read order (bswap16 A and B first)",
          py::arg("a"), py::arg("b"), py::arg("m"), py::arg("k"), py::arg("n"),
          py::arg("a_scale"), py::arg("b_scale"));

    m.def("matmul_hc8", &py_matmul_hc8,
          "HC8 view matmul (int8×int8→int32 accumulate, 2 elements per hc16ms_t)",
          py::arg("a"), py::arg("b"), py::arg("m"), py::arg("k"), py::arg("n"),
          py::arg("a_scale"), py::arg("b_scale"));

    m.def("quantized_matmul_hc16", &py_quantized_matmul_hc16,
          "Convenience: HC16 quantized matmul (float→float)",
          py::arg("x"), py::arg("w"), py::arg("m"), py::arg("k"), py::arg("n"));

    m.def("quantized_matmul_hc8", &py_quantized_matmul_hc8,
          "Convenience: HC8 quantized matmul (float→float, 2x k dimension)",
          py::arg("x"), py::arg("w"), py::arg("m"), py::arg("k"), py::arg("n"));
}
