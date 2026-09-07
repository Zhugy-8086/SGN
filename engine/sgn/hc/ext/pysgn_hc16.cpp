/**
 * @file pysgn_hc16.cpp
 * @brief pysgn_hc16 - HC16 神经网络运算扩展的 Python 绑定（MSInt 中间层）
 * @version 1.5.0
 *
 * 独立 pybind11 扩展，不依赖 pysgn 或 pysgn_net。
 * 使用 numpy 数组接口（float32 / int16）直接交互，无需 bytes 编码。
 *
 * 暴露的 Python API：
 *   - QuantSchema16(qmin=-32767, qmax=32767)
 *   - quant_compute_scale(x: np.ndarray) -> float
 *   - quantize(x: np.ndarray, scale: float, schema) -> np.ndarray (int16)
 *   - dequantize(q: np.ndarray, scale: float) -> np.ndarray (float32)
 *   - matmul(a: np.ndarray, b: np.ndarray, m, k, n, a_scale, b_scale) -> np.ndarray (float32)
 *   - matmul_scalar(...) -> np.ndarray (float32)  # 标量路径
 *   - matmul_avx2(...) -> np.ndarray (float32)    # AVX2 路径
 *   - quantized_matmul(x: np.ndarray, w: np.ndarray, m, k, n, schema) -> np.ndarray (float32)
 *   - detect_avx2() -> int
 *   - default_schema() -> QuantSchema16
 *
 * v1.5.0 新增（per-channel scale，用于 2.6 非 STE 反向传播）：
 *   - quantize_per_channel(x, scales, schema) -> np.ndarray (int16)
 *   - dequantize_per_channel(q, scales) -> np.ndarray (float32)
 *   - matmul_per_channel(a, b, m, k, n, a_scales, b_scales) -> np.ndarray (float32)
 *   - matmul_per_channel_scalar(...) -> np.ndarray (float32)  # 标量路径
 *   - matmul_per_channel_avx2(...) -> np.ndarray (float32)    # AVX2 路径
 *
 * 编译：
 *   构建经统一 sgn 模块（CMake，见 COMPILER_TOOLCHAIN.md）
 *   pip install pybind11
 *   python setup_hc16.py build_ext --inplace
 *
 * 验证：
 *   python -c "import pysgn_hc16; print(pysgn_hc16.default_schema())"
 *   python -c "import pysgn_hc16; print('AVX2:', pysgn_hc16.detect_avx2())"
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "hc16_net.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace py = pybind11;

/* ============================================================================
 * QuantSchema16 Python 包装
 * ============================================================================ */

struct PyQuantSchema16 {
    hc16_quant_schema_t raw;

    PyQuantSchema16()
        : raw(hc16_quant_default_schema()) {}

    PyQuantSchema16(int32_t qmin, int32_t qmax) {
        raw.scale = 1.0f;
        raw.qmin  = qmin;
        raw.qmax  = qmax;
    }

    int32_t get_qmin()  const { return raw.qmin; }
    int32_t get_qmax()  const { return raw.qmax; }
    float   get_scale() const { return raw.scale; }

    void set_qmin(int32_t v)  { raw.qmin = v; }
    void set_qmax(int32_t v)  { raw.qmax = v; }
    void set_scale(float v)   { raw.scale = v; }

    std::string __repr__() const {
        return "<QuantSchema16 qmin=" + std::to_string(raw.qmin) +
               " qmax=" + std::to_string(raw.qmax) + ">";
    }
};

/* ============================================================================
 * numpy 数组辅助函数
 * ============================================================================ */

static py::array_t<float> make_float_array(size_t n) {
    return py::array_t<float>({(py::ssize_t)n}, {sizeof(float)});
}


/* 注：原 get_float_ptr/get_int16_ptr 已删除——返回的裸指针在 forcecast 副本
 * 销毁后悬挂（H4 修复遗留死代码，安全审计 2026-08-16 F1/P16-1）。
 * 正确模式：函数内持有 py::array_t 局部变量保持生命周期。 */

/* ============================================================================
 * 功能函数
 * ============================================================================ */

/* 安全审计 2026-08-16 F4：buf.size（ssize_t）强转 uint32_t 前校验上限，
 * 防止 > 4G 元素数组长度截断后越界（numpy 实际分配上限内不可达，防御性） */
static inline uint32_t checked_u32(py::ssize_t n) {
    if (n > (py::ssize_t)UINT32_MAX) {
        throw std::runtime_error("array too large (> UINT32_MAX elements)");
    }
    return (uint32_t)n;
}

/**
 * 从 float numpy 数组推导量化 scale
 */
static float py_quant_compute_scale(const py::object& x_obj) {
    auto arr = x_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto buf = arr.request();
    if (buf.size == 0) return 1.0f;
    return hc16_quant_compute_scale((const float*)buf.ptr, checked_u32(buf.size));
}

/**
 * 量化 float 数组 → int16 数组（保留输入形状）
 */
static py::array_t<int16_t> py_quantize(const py::object& x_obj,
                                         float scale,
                                         const PyQuantSchema16& schema) {
    auto arr = x_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto buf = arr.request();
    uint32_t n = checked_u32(buf.size);

    /* 创建与输入同形状的 int16 输出数组 */
    auto out = py::array_t<int16_t>(buf.shape);
    auto out_buf = out.request();

    hc16_quantize((const float*)buf.ptr, n, scale, &schema.raw,
                  (int16_t*)out_buf.ptr);
    return out;
}

/**
 * 反量化 int16 数组 → float 数组（保留输入形状）
 */
static py::array_t<float> py_dequantize(const py::object& q_obj,
                                         float scale) {
    auto arr = q_obj.cast<py::array_t<int16_t, py::array::c_style | py::array::forcecast>>();
    auto buf = arr.request();
    uint32_t n = checked_u32(buf.size);

    /* 创建与输入同形状的 float 输出数组 */
    auto out = py::array_t<float>(buf.shape);
    auto out_buf = out.request();

    hc16_dequantize((const int16_t*)buf.ptr, n, scale,
                    (float*)out_buf.ptr);
    return out;
}

/**
 * HC16 矩阵乘（自动选择路径）
 * a: (m, k) int16, b: (k, n) int16 → out: (m, n) float32
 */
static py::array_t<float> py_matmul(const py::object& a_obj,
                                     const py::object& b_obj,
                                     uint32_t m, uint32_t k, uint32_t n,
                                     float a_scale, float b_scale) {
    auto a_arr = a_obj.cast<py::array_t<int16_t, py::array::c_style | py::array::forcecast>>();
    auto b_arr = b_obj.cast<py::array_t<int16_t, py::array::c_style | py::array::forcecast>>();
    auto a_buf = a_arr.request();
    auto b_buf = b_arr.request();
    if ((size_t)a_buf.size != (size_t)m * k) {
        throw std::runtime_error("a 长度不匹配：期望 " + std::to_string((size_t)m * k) +
                                 "，实际 " + std::to_string(a_buf.size));
    }
    if ((size_t)b_buf.size != (size_t)k * n) {
        throw std::runtime_error("b 长度不匹配：期望 " + std::to_string((size_t)k * n) +
                                 "，实际 " + std::to_string(b_buf.size));
    }
    const int16_t* a = (const int16_t*)a_buf.ptr;
    const int16_t* b = (const int16_t*)b_buf.ptr;

    auto out = make_float_array((size_t)m * n);
    auto out_buf = out.request();

    /* 安全审计 2026-08-16 F2：纯 C 计算释放 GIL */
    {
        py::gil_scoped_release release;
        hc16_matmul(a, b, m, k, n, a_scale, b_scale, (float*)out_buf.ptr);
    }
    return out.reshape({(py::ssize_t)m, (py::ssize_t)n});
}

/**
 * HC16 矩阵乘（标量路径）
 */
static py::array_t<float> py_matmul_scalar(const py::object& a_obj,
                                            const py::object& b_obj,
                                            uint32_t m, uint32_t k, uint32_t n,
                                            float a_scale, float b_scale) {
    auto a_arr = a_obj.cast<py::array_t<int16_t, py::array::c_style | py::array::forcecast>>();
    auto b_arr = b_obj.cast<py::array_t<int16_t, py::array::c_style | py::array::forcecast>>();
    auto a_buf = a_arr.request();
    auto b_buf = b_arr.request();
    if ((size_t)a_buf.size != (size_t)m * k) {
        throw std::runtime_error("a 长度不匹配：期望 " + std::to_string((size_t)m * k) +
                                 "，实际 " + std::to_string(a_buf.size));
    }
    if ((size_t)b_buf.size != (size_t)k * n) {
        throw std::runtime_error("b 长度不匹配：期望 " + std::to_string((size_t)k * n) +
                                 "，实际 " + std::to_string(b_buf.size));
    }
    const int16_t* a = (const int16_t*)a_buf.ptr;
    const int16_t* b = (const int16_t*)b_buf.ptr;

    auto out = make_float_array((size_t)m * n);
    auto out_buf = out.request();

    {
        py::gil_scoped_release release;
        hc16_matmul_scalar(a, b, m, k, n, a_scale, b_scale, (float*)out_buf.ptr);
    }
    return out.reshape({(py::ssize_t)m, (py::ssize_t)n});
}

/**
 * HC16 矩阵乘（AVX2 路径）
 */
static py::array_t<float> py_matmul_avx2(const py::object& a_obj,
                                          const py::object& b_obj,
                                          uint32_t m, uint32_t k, uint32_t n,
                                          float a_scale, float b_scale) {
    auto a_arr = a_obj.cast<py::array_t<int16_t, py::array::c_style | py::array::forcecast>>();
    auto b_arr = b_obj.cast<py::array_t<int16_t, py::array::c_style | py::array::forcecast>>();
    auto a_buf = a_arr.request();
    auto b_buf = b_arr.request();
    if ((size_t)a_buf.size != (size_t)m * k) {
        throw std::runtime_error("a 长度不匹配：期望 " + std::to_string((size_t)m * k) +
                                 "，实际 " + std::to_string(a_buf.size));
    }
    if ((size_t)b_buf.size != (size_t)k * n) {
        throw std::runtime_error("b 长度不匹配：期望 " + std::to_string((size_t)k * n) +
                                 "，实际 " + std::to_string(b_buf.size));
    }
    const int16_t* a = (const int16_t*)a_buf.ptr;
    const int16_t* b = (const int16_t*)b_buf.ptr;

    auto out = make_float_array((size_t)m * n);
    auto out_buf = out.request();

    {
        py::gil_scoped_release release;
        hc16_matmul_avx2(a, b, m, k, n, a_scale, b_scale, (float*)out_buf.ptr);
    }
    return out.reshape({(py::ssize_t)m, (py::ssize_t)n});
}

/**
 * HC16 量化 matmul（float → float，一步完成）
 * x: (m, k) float32, w: (k, n) float32 → out: (m, n) float32
 */
static py::array_t<float> py_quantized_matmul(const py::object& x_obj,
                                               const py::object& w_obj,
                                               uint32_t m, uint32_t k, uint32_t n,
                                               const PyQuantSchema16& schema) {
    auto x_arr = x_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto w_arr = w_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto x_buf = x_arr.request();
    auto w_buf = w_arr.request();
    if ((size_t)x_buf.size != (size_t)m * k) {
        throw std::runtime_error("x 长度不匹配：期望 " + std::to_string((size_t)m * k) +
                                 "，实际 " + std::to_string(x_buf.size));
    }
    if ((size_t)w_buf.size != (size_t)k * n) {
        throw std::runtime_error("w 长度不匹配：期望 " + std::to_string((size_t)k * n) +
                                 "，实际 " + std::to_string(w_buf.size));
    }
    const float* x = (const float*)x_buf.ptr;
    const float* w = (const float*)w_buf.ptr;

    auto out = make_float_array((size_t)m * n);
    auto out_buf = out.request();

    {
        py::gil_scoped_release release;
        hc16_quantized_matmul(x, w, m, k, n, &schema.raw, (float*)out_buf.ptr);
    }
    return out.reshape({(py::ssize_t)m, (py::ssize_t)n});
}

/* ============================================================================
 * Per-channel scale 包装函数（v1.5.0 新增）
 * ============================================================================
 *
 * 维度约定（与 hc16_net.h 一致）：
 *   - quantize/dequantize: x 为 (rows, per_row_size) 行优先，scales 长度 rows
 *   - matmul: A(m×k) 行优先，a_scales 长度 m；B(k×n) 行优先，b_scales 长度 n
 */

/**
 * 从 2D numpy 数组推导 per-channel scale（每行一个 scale）
 *   x: (rows, per_row_size) float32 → scales: (rows,) float32
 *   scales[i] = max(|x[i, :]|) / 32767
 */
static py::array_t<float> py_quant_compute_scale_per_channel(const py::object& x_obj,
                                                              uint32_t rows,
                                                              uint32_t per_row_size) {
    auto arr = x_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto buf = arr.request();
    if ((size_t)buf.size != (size_t)rows * per_row_size) {
        throw std::runtime_error("数组长度不匹配：期望 " + std::to_string((size_t)rows * per_row_size) +
                                 "，实际 " + std::to_string(buf.size));
    }
    const float* x = (const float*)buf.ptr;

    auto scales = py::array_t<float>({(py::ssize_t)rows});
    auto scales_buf = scales.request();
    float* s = (float*)scales_buf.ptr;

    /* 安全审计 2026-08-16 F3：纯 C 循环释放 GIL */
    {
        py::gil_scoped_release release;
        for (uint32_t i = 0; i < rows; ++i) {
            const float* row = x + (size_t)i * per_row_size;
            float max_abs = 0.0f;
            for (uint32_t j = 0; j < per_row_size; ++j) {
                float a = fabsf(row[j]);
                if (a > max_abs) max_abs = a;
            }
            s[i] = (max_abs == 0.0f) ? 1.0f : (max_abs / 32767.0f);
        }
    }
    return scales;
}

/**
 * Per-channel 量化 float → int16
 *   x: (rows, per_row_size) float32, scales: (rows,) float32 → out: (rows, per_row_size) int16
 *   保留输入 2D 形状
 */
static py::array_t<int16_t> py_quantize_per_channel(const py::object& x_obj,
                                                     const py::object& scales_obj,
                                                     const PyQuantSchema16& schema,
                                                     uint32_t rows,
                                                     uint32_t per_row_size) {
    auto x_arr = x_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto x_buf = x_arr.request();
    if ((size_t)x_buf.size != (size_t)rows * per_row_size) {
        throw std::runtime_error("x 长度不匹配：期望 " + std::to_string((size_t)rows * per_row_size) +
                                 "，实际 " + std::to_string(x_buf.size));
    }

    auto s_arr = scales_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto s_buf = s_arr.request();
    if ((size_t)s_buf.size != rows) {
        throw std::runtime_error("scales 长度不匹配：期望 " + std::to_string(rows) +
                                 "，实际 " + std::to_string(s_buf.size));
    }

    /* 保留输入 2D 形状 */
    auto out = py::array_t<int16_t>(x_buf.shape);
    auto out_buf = out.request();

    hc16_quantize_per_channel((const float*)x_buf.ptr, rows, per_row_size,
                              (const float*)s_buf.ptr, &schema.raw,
                              (int16_t*)out_buf.ptr);
    return out;
}

/**
 * Per-channel 反量化 int16 → float
 *   q: (rows, per_row_size) int16, scales: (rows,) float32 → out: (rows, per_row_size) float32
 *   保留输入 2D 形状
 */
static py::array_t<float> py_dequantize_per_channel(const py::object& q_obj,
                                                     const py::object& scales_obj,
                                                     uint32_t rows,
                                                     uint32_t per_row_size) {
    auto q_arr = q_obj.cast<py::array_t<int16_t, py::array::c_style | py::array::forcecast>>();
    auto q_buf = q_arr.request();
    if ((size_t)q_buf.size != (size_t)rows * per_row_size) {
        throw std::runtime_error("q 长度不匹配：期望 " + std::to_string((size_t)rows * per_row_size) +
                                 "，实际 " + std::to_string(q_buf.size));
    }

    auto s_arr = scales_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto s_buf = s_arr.request();
    if ((size_t)s_buf.size != rows) {
        throw std::runtime_error("scales 长度不匹配：期望 " + std::to_string(rows) +
                                 "，实际 " + std::to_string(s_buf.size));
    }

    /* 保留输入 2D 形状 */
    auto out = py::array_t<float>(q_buf.shape);
    auto out_buf = out.request();

    hc16_dequantize_per_channel((const int16_t*)q_buf.ptr, rows, per_row_size,
                                (const float*)s_buf.ptr,
                                (float*)out_buf.ptr);
    return out;
}

/**
 * Per-channel matmul（自动选择最优路径）
 *   a: (m, k) int16, b: (k, n) int16, a_scales: (m,) float32, b_scales: (n,) float32
 *   → out: (m, n) float32
 *   C[i][j] = sum_l(A[i][l] * B[l][j]) * a_scales[i] * b_scales[j]
 *
 * 注意：不返回裸指针（forcecast 副本销毁后指针悬空，原 get_int16_ptr 已删），
 *       直接用 py::array_t 局部变量保持数组生命周期。
 */
static py::array_t<float> py_matmul_per_channel(const py::object& a_obj,
                                                 const py::object& b_obj,
                                                 uint32_t m, uint32_t k, uint32_t n,
                                                 const py::object& a_scales_obj,
                                                 const py::object& b_scales_obj) {
    /* forcecast 确保 C-contiguous 副本，局部变量保持生命周期 */
    auto a_arr = a_obj.cast<py::array_t<int16_t, py::array::c_style | py::array::forcecast>>();
    auto b_arr = b_obj.cast<py::array_t<int16_t, py::array::c_style | py::array::forcecast>>();
    auto a_buf = a_arr.request();
    auto b_buf = b_arr.request();
    if ((size_t)a_buf.size != (size_t)m * k) {
        throw std::runtime_error("a 长度不匹配：期望 " + std::to_string((size_t)m * k) +
                                 "，实际 " + std::to_string(a_buf.size));
    }
    if ((size_t)b_buf.size != (size_t)k * n) {
        throw std::runtime_error("b 长度不匹配：期望 " + std::to_string((size_t)k * n) +
                                 "，实际 " + std::to_string(b_buf.size));
    }
    const int16_t* a = (const int16_t*)a_buf.ptr;
    const int16_t* b = (const int16_t*)b_buf.ptr;

    auto as_arr = a_scales_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto as_buf = as_arr.request();
    if ((size_t)as_buf.size != m) {
        throw std::runtime_error("a_scales 长度不匹配：期望 " + std::to_string(m) +
                                 "，实际 " + std::to_string(as_buf.size));
    }

    auto bs_arr = b_scales_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto bs_buf = bs_arr.request();
    if ((size_t)bs_buf.size != n) {
        throw std::runtime_error("b_scales 长度不匹配：期望 " + std::to_string(n) +
                                 "，实际 " + std::to_string(bs_buf.size));
    }

    auto out = make_float_array((size_t)m * n);
    auto out_buf = out.request();

    {
        py::gil_scoped_release release;
        hc16_matmul_per_channel(a, b, m, k, n,
                                (const float*)as_buf.ptr, (const float*)bs_buf.ptr,
                                (float*)out_buf.ptr);
    }
    return out.reshape({(py::ssize_t)m, (py::ssize_t)n});
}

/**
 * Per-channel matmul（标量路径，用于验证/调试）
 */
static py::array_t<float> py_matmul_per_channel_scalar(const py::object& a_obj,
                                                        const py::object& b_obj,
                                                        uint32_t m, uint32_t k, uint32_t n,
                                                        const py::object& a_scales_obj,
                                                        const py::object& b_scales_obj) {
    auto a_arr = a_obj.cast<py::array_t<int16_t, py::array::c_style | py::array::forcecast>>();
    auto b_arr = b_obj.cast<py::array_t<int16_t, py::array::c_style | py::array::forcecast>>();
    auto a_buf = a_arr.request();
    auto b_buf = b_arr.request();
    if ((size_t)a_buf.size != (size_t)m * k) {
        throw std::runtime_error("a 长度不匹配：期望 " + std::to_string((size_t)m * k) +
                                 "，实际 " + std::to_string(a_buf.size));
    }
    if ((size_t)b_buf.size != (size_t)k * n) {
        throw std::runtime_error("b 长度不匹配：期望 " + std::to_string((size_t)k * n) +
                                 "，实际 " + std::to_string(b_buf.size));
    }
    const int16_t* a = (const int16_t*)a_buf.ptr;
    const int16_t* b = (const int16_t*)b_buf.ptr;

    auto as_arr = a_scales_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto as_buf = as_arr.request();
    if ((size_t)as_buf.size != m) {
        throw std::runtime_error("a_scales 长度不匹配：期望 " + std::to_string(m) +
                                 "，实际 " + std::to_string(as_buf.size));
    }

    auto bs_arr = b_scales_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto bs_buf = bs_arr.request();
    if ((size_t)bs_buf.size != n) {
        throw std::runtime_error("b_scales 长度不匹配：期望 " + std::to_string(n) +
                                 "，实际 " + std::to_string(bs_buf.size));
    }

    auto out = make_float_array((size_t)m * n);
    auto out_buf = out.request();

    {
        py::gil_scoped_release release;
        hc16_matmul_per_channel_scalar(a, b, m, k, n,
                                       (const float*)as_buf.ptr, (const float*)bs_buf.ptr,
                                       (float*)out_buf.ptr);
    }
    return out.reshape({(py::ssize_t)m, (py::ssize_t)n});
}

/**
 * Per-channel matmul（AVX2 路径）
 */
static py::array_t<float> py_matmul_per_channel_avx2(const py::object& a_obj,
                                                      const py::object& b_obj,
                                                      uint32_t m, uint32_t k, uint32_t n,
                                                      const py::object& a_scales_obj,
                                                      const py::object& b_scales_obj) {
    auto a_arr = a_obj.cast<py::array_t<int16_t, py::array::c_style | py::array::forcecast>>();
    auto b_arr = b_obj.cast<py::array_t<int16_t, py::array::c_style | py::array::forcecast>>();
    auto a_buf = a_arr.request();
    auto b_buf = b_arr.request();
    if ((size_t)a_buf.size != (size_t)m * k) {
        throw std::runtime_error("a 长度不匹配：期望 " + std::to_string((size_t)m * k) +
                                 "，实际 " + std::to_string(a_buf.size));
    }
    if ((size_t)b_buf.size != (size_t)k * n) {
        throw std::runtime_error("b 长度不匹配：期望 " + std::to_string((size_t)k * n) +
                                 "，实际 " + std::to_string(b_buf.size));
    }
    const int16_t* a = (const int16_t*)a_buf.ptr;
    const int16_t* b = (const int16_t*)b_buf.ptr;

    auto as_arr = a_scales_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto as_buf = as_arr.request();
    if ((size_t)as_buf.size != m) {
        throw std::runtime_error("a_scales 长度不匹配：期望 " + std::to_string(m) +
                                 "，实际 " + std::to_string(as_buf.size));
    }

    auto bs_arr = b_scales_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto bs_buf = bs_arr.request();
    if ((size_t)bs_buf.size != n) {
        throw std::runtime_error("b_scales 长度不匹配：期望 " + std::to_string(n) +
                                 "，实际 " + std::to_string(bs_buf.size));
    }

    auto out = make_float_array((size_t)m * n);
    auto out_buf = out.request();

    {
        py::gil_scoped_release release;
        hc16_matmul_per_channel_avx2(a, b, m, k, n,
                                     (const float*)as_buf.ptr, (const float*)bs_buf.ptr,
                                     (float*)out_buf.ptr);
    }
    return out.reshape({(py::ssize_t)m, (py::ssize_t)n});
}

/* ============================================================================
 * 模块定义
 * ============================================================================ */

void register_hc16(py::module_& m) {
    m.doc() = "HC16 神经网络运算扩展（MSInt 位宽链中间层，AVX2 _mm256_madd_epi16）— 合并到 sgn.hc16 子模块";

    /* QuantSchema16 */
    py::class_<PyQuantSchema16>(m, "QuantSchema16")
        .def(py::init<>())
        .def(py::init<int32_t, int32_t>(), py::arg("qmin"), py::arg("qmax"))
        .def_property("qmin", &PyQuantSchema16::get_qmin, &PyQuantSchema16::set_qmin)
        .def_property("qmax", &PyQuantSchema16::get_qmax, &PyQuantSchema16::set_qmax)
        .def_property("scale", &PyQuantSchema16::get_scale, &PyQuantSchema16::set_scale)
        .def("__repr__", &PyQuantSchema16::__repr__);

    /* 功能函数 */
    m.def("default_schema", []() { return PyQuantSchema16(); });
    m.def("detect_avx2", &hc16_detect_avx2,
          "检测 CPU 是否支持 AVX2（1=支持, 0=不支持）");

    m.def("quant_compute_scale", &py_quant_compute_scale,
          py::arg("x"),
          "从 float 数组推导 HC16 量化 scale");

    m.def("quantize", &py_quantize,
          py::arg("x"), py::arg("scale"), py::arg("schema"),
          "量化 float 数组 → int16 数组");

    m.def("dequantize", &py_dequantize,
          py::arg("q"), py::arg("scale"),
          "反量化 int16 数组 → float 数组");

    m.def("matmul", &py_matmul,
          py::arg("a"), py::arg("b"),
          py::arg("m"), py::arg("k"), py::arg("n"),
          py::arg("a_scale"), py::arg("b_scale"),
          "HC16 矩阵乘（自动选择最优路径）");

    m.def("matmul_scalar", &py_matmul_scalar,
          py::arg("a"), py::arg("b"),
          py::arg("m"), py::arg("k"), py::arg("n"),
          py::arg("a_scale"), py::arg("b_scale"),
          "HC16 矩阵乘（标量路径，int64 累加器）");

    m.def("matmul_avx2", &py_matmul_avx2,
          py::arg("a"), py::arg("b"),
          py::arg("m"), py::arg("k"), py::arg("n"),
          py::arg("a_scale"), py::arg("b_scale"),
          "HC16 矩阵乘（AVX2 路径，_mm256_madd_epi16）");

    m.def("quantized_matmul", &py_quantized_matmul,
          py::arg("x"), py::arg("w"),
          py::arg("m"), py::arg("k"), py::arg("n"),
          py::arg("schema"),
          "HC16 量化 matmul（float→float，一步完成）");

    /* Per-channel scale API（v1.5.0 新增） */
    m.def("quant_compute_scale_per_channel", &py_quant_compute_scale_per_channel,
          py::arg("x"), py::arg("rows"), py::arg("per_row_size"),
          "从 2D float 数组推导 per-channel scale（每行一个 scale）");

    m.def("quantize_per_channel", &py_quantize_per_channel,
          py::arg("x"), py::arg("scales"), py::arg("schema"),
          py::arg("rows"), py::arg("per_row_size"),
          "Per-channel 量化 float → int16（每行独立 scale）");

    m.def("dequantize_per_channel", &py_dequantize_per_channel,
          py::arg("q"), py::arg("scales"),
          py::arg("rows"), py::arg("per_row_size"),
          "Per-channel 反量化 int16 → float（每行独立 scale）");

    m.def("matmul_per_channel", &py_matmul_per_channel,
          py::arg("a"), py::arg("b"),
          py::arg("m"), py::arg("k"), py::arg("n"),
          py::arg("a_scales"), py::arg("b_scales"),
          "HC16 矩阵乘（per-channel scale，自动选择路径）");

    m.def("matmul_per_channel_scalar", &py_matmul_per_channel_scalar,
          py::arg("a"), py::arg("b"),
          py::arg("m"), py::arg("k"), py::arg("n"),
          py::arg("a_scales"), py::arg("b_scales"),
          "HC16 矩阵乘（per-channel scale，标量路径）");

    m.def("matmul_per_channel_avx2", &py_matmul_per_channel_avx2,
          py::arg("a"), py::arg("b"),
          py::arg("m"), py::arg("k"), py::arg("n"),
          py::arg("a_scales"), py::arg("b_scales"),
          "HC16 矩阵乘（per-channel scale，AVX2 路径）");

    /* 版本信息（合并到 sgn 模块后的标识） */
    m.attr("__version__") = "merged-hc16";
}
