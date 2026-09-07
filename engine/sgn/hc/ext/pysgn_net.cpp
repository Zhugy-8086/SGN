/**
 * @file pysgn_net.cpp
 * @brief pysgn_net - HC8 神经网络运算扩展的 Python 绑定（阶段 1.3）
 * @version 1.3.0
 *
 * 独立 pybind11 扩展，不依赖 pysgn（避免重复定义 PyHC8）。
 * 通过 bytes 接口与 pysgn.HC8 互操作：
 *   - pysgn.HC8.to_bytes() → 6 字节 → pysgn_net 接受
 *   - pysgn_net 输出 bytes → pysgn.HC8.from_bytes() 还原
 *
 * 暴露的 Python API：
 *   - QuantSchema(qmin=-127, qmax=127, offset=128)
 *   - quant_compute_scale(float_list) -> float
 *   - quantize(float_list, scale, schema) -> bytes
 *   - dequantize(bytes, scale, schema) -> list[float]
 *   - matmul(bytes_a, bytes_b, m, k, n, a_scale, b_scale, schema) -> (bytes_c, c_scale)
 *   - relu(bytes_x, m, n, schema) -> bytes
 *   - default_schema() -> QuantSchema
 *
 * 设计原则：
 *   - 不修改 engine/hc/sgn/bindings/python/pysgn.cpp
 *   - 不修改 engine/hc/sgn/bindings/python/setup.py
 *   - 独立 setup_net.py 编译为 pysgn_net.pyd
 *   - Python 端可同时 import pysgn 和 pysgn_net，互不冲突
 *
 * 编译：
 *   构建经统一 sgn 模块（CMake，见 COMPILER_TOOLCHAIN.md）
 *   pip install pybind11
 *   python setup_net.py build_ext --inplace
 *
 * 验证：
 *   python -c "import pysgn_net; print(pysgn_net.default_schema())"
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

/* hc8_net.h 在本扩展目录（不在 engine/hc/sgn/include/hc/），
 * setup_net.py 会把本目录加入 include_dirs */
#include "hc8_net.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>  /* getenv, atoi (P2.1: SGN_OMP_THREADS 环境变量读取) */
#include <cstring>
#include <stdexcept>  /* invalid_argument（安全审计 2026-08-16 I1 conv2d 尺寸校验） */
#include <string>
#include <vector>

/* AVX2 + FMA intrinsics（float_matmul 反向传播加速） */
#include <immintrin.h>
#include <intrin.h>  /* __cpuid（安全审计 2026-08-16 I2 AVX2 运行时检测） */

/* OpenMP（sbe_matmul_c / sbe_rescale_to_triple_c 多线程并行） */
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

/* ============================================================================
 * QuantSchema Python 包装
 * ============================================================================ */

struct PyQuantSchema {
    hc8_quant_schema_t raw;

    PyQuantSchema()
        : raw(hc8_quant_default_schema()) {}

    PyQuantSchema(int32_t qmin, int32_t qmax, int32_t offset) {
        raw.scale  = 1.0f;
        raw.qmin   = qmin;
        raw.qmax   = qmax;
        raw.offset = offset;
    }

    int32_t get_qmin()   const { return raw.qmin; }
    int32_t get_qmax()   const { return raw.qmax; }
    int32_t get_offset() const { return raw.offset; }
    float   get_scale()  const { return raw.scale; }

    void set_qmin(int32_t v)   { raw.qmin = v; }
    void set_qmax(int32_t v)   { raw.qmax = v; }
    void set_offset(int32_t v) { raw.offset = v; }
    void set_scale(float v)    { raw.scale = v; }

    std::string __repr__() const {
        return "<QuantSchema qmin=" + std::to_string(raw.qmin) +
               " qmax=" + std::to_string(raw.qmax) +
               " offset=" + std::to_string(raw.offset) + ">";
    }
};

/* ============================================================================
 * 功能函数
 * ============================================================================ */

/**
 * 从 float list 推导量化 scale
 */
static float py_quant_compute_scale(const std::vector<float>& w) {
    if (w.empty()) return 1.0f;
    return hc8_quant_compute_scale(w.data(), (uint32_t)w.size());
}

/**
 * 量化 float list → bytes（每个 HC8 6 字节）
 */
static py::bytes py_quantize(const std::vector<float>& w,
                              float scale,
                              const PyQuantSchema& schema) {
    if (w.empty()) return py::bytes("");

    uint32_t n = (uint32_t)w.size();
    std::vector<hc8_t> out(n);
    hc8_quantize(w.data(), n, scale, &schema.raw, out.data());

    /* hc8_t 是 6 字节紧缩结构，可直接转为 bytes */
    return py::bytes(reinterpret_cast<const char*>(out.data()), (size_t)n * sizeof(hc8_t));
}

/**
 * 反量化 bytes → float list
 */
static std::vector<float> py_dequantize(const std::string& bytes,
                                         float scale,
                                         const PyQuantSchema& schema) {
    if (bytes.empty()) return {};

    size_t total_bytes = bytes.size();
    if (total_bytes % sizeof(hc8_t) != 0) {
        throw std::runtime_error("bytes 长度必须是 6 的倍数（每个 HC8 6 字节）");
    }
    uint32_t n = (uint32_t)(total_bytes / sizeof(hc8_t));

    std::vector<float> out(n);
    hc8_dequantize(reinterpret_cast<const hc8_t*>(bytes.data()), n,
                   scale, &schema.raw, out.data());
    return out;
}

/**
 * HC8 矩阵乘：返回 (bytes_c, c_scale)
 *
 * Args:
 *   bytes_a: A 矩阵的 HC8 bytes（长度 m*k*6）
 *   bytes_b: B 矩阵的 HC8 bytes（长度 k*n*6）
 *   m, k, n: 矩阵维度
 *   a_scale, b_scale: A 和 B 的量化 scale
 *   schema: 量化方案
 */
static py::tuple py_matmul(const std::string& bytes_a,
                            const std::string& bytes_b,
                            uint32_t m, uint32_t k, uint32_t n,
                            float a_scale, float b_scale,
                            const PyQuantSchema& schema) {
    /* 校验输入长度（安全审计 2026-08-30 F4：使用 uint64_t 避免乘积溢出） */
    size_t expected_a = static_cast<size_t>(m) * static_cast<size_t>(k) * sizeof(hc8_t);
    size_t expected_b = static_cast<size_t>(k) * static_cast<size_t>(n) * sizeof(hc8_t);
    if (bytes_a.size() != expected_a) {
        throw std::runtime_error(
            "bytes_a 长度不匹配：期望 " + std::to_string(expected_a) +
            " 字节（" + std::to_string(m) + "x" + std::to_string(k) +
            " HC8 矩阵），得到 " + std::to_string(bytes_a.size()) + " 字节"
        );
    }
    if (bytes_b.size() != expected_b) {
        throw std::runtime_error(
            "bytes_b 长度不匹配：期望 " + std::to_string(expected_b) +
            " 字节（" + std::to_string(k) + "x" + std::to_string(n) +
            " HC8 矩阵），得到 " + std::to_string(bytes_b.size()) + " 字节"
        );
    }

    const hc8_t* a = reinterpret_cast<const hc8_t*>(bytes_a.data());
    const hc8_t* b = reinterpret_cast<const hc8_t*>(bytes_b.data());

    std::vector<hc8_t> c((size_t)m * n);
    float c_scale = 1.0f;

    hc8_matmul(a, b, m, k, n, a_scale, b_scale,
               &schema.raw, c.data(), &c_scale);

    py::bytes bytes_c(reinterpret_cast<const char*>(c.data()),
                      (size_t)m * n * sizeof(hc8_t));
    return py::make_tuple(bytes_c, c_scale);
}

/**
 * HC8 整数 ReLU
 *
 * Args:
 *   bytes_x: X 矩阵的 HC8 bytes（长度 m*n*6）
 *   m, n: 矩阵维度
 *   schema: 量化方案（提供 offset）
 */
static py::bytes py_relu(const std::string& bytes_x,
                          uint32_t m, uint32_t n,
                          const PyQuantSchema& schema) {
    size_t expected = (size_t)m * n * sizeof(hc8_t);
    if (bytes_x.size() != expected) {
        throw std::runtime_error(
            "bytes_x 长度不匹配：期望 " + std::to_string(expected) +
            " 字节，得到 " + std::to_string(bytes_x.size()) + " 字节"
        );
    }

    const hc8_t* x = reinterpret_cast<const hc8_t*>(bytes_x.data());
    std::vector<hc8_t> out((size_t)m * n);

    hc8_relu(x, m, n, &schema.raw, out.data());

    return py::bytes(reinterpret_cast<const char*>(out.data()),
                     (size_t)m * n * sizeof(hc8_t));
}

/**
 * HC8 bytes → numpy uint8 数组（每个 HC8 6 字节，扁平存储）
 *
 * 便于在 Python 端用 numpy 查看内部结构。
 */
static py::array_t<uint8_t> py_bytes_to_uint8_array(const std::string& bytes) {
    size_t n = bytes.size();
    auto arr = py::array_t<uint8_t>(n);
    std::memcpy(arr.mutable_data(), bytes.data(), n);
    return arr;
}

/**
 * float64 矩阵乘法（不经过量化，用于反向传播 STE）
 *
 * 直接接受 Python list（通过 pybind11 vector 转换），做 C matmul，返回 Python list。
 * 省掉 numpy 的 list→array→list 转换开销（诊断显示转换占 57%）。
 *
 * Args:
 *   a_flat: m*k 个 float（行优先）
 *   b_flat: k*n 个 float（行优先）
 *   m, k, n: 矩阵维度
 *
 * Returns:
 *   c_flat: m*n 个 float（行优先）
 */
static std::vector<double> py_float_matmul(
    const std::vector<double>& a_flat,
    const std::vector<double>& b_flat,
    uint32_t m, uint32_t k, uint32_t n
) {
    size_t expected_a = (size_t)m * k;
    size_t expected_b = (size_t)k * n;
    if (a_flat.size() != expected_a) {
        throw std::runtime_error(
            "a_flat 长度不匹配：期望 " + std::to_string(expected_a) +
            "（" + std::to_string(m) + "x" + std::to_string(k) +
            "），得到 " + std::to_string(a_flat.size())
        );
    }
    if (b_flat.size() != expected_b) {
        throw std::runtime_error(
            "b_flat 长度不匹配：期望 " + std::to_string(expected_b) +
            "（" + std::to_string(k) + "x" + std::to_string(n) +
            "），得到 " + std::to_string(b_flat.size())
        );
    }

    /* 直接使用 b_flat（i-k-j 循环顺序下 b_flat[l*n + j] 已是连续访问） */
    std::vector<double> c((size_t)m * n, 0.0);

    /* 释放 GIL 进行纯 C++ 计算（无 Python 对象访问）
     * 使 C 计算可与 Python 线程重叠（阶段 0.2，multithreading_optimization_plan.md） */
    {
        py::gil_scoped_release release;

        /* i-k-j 循环顺序 + AVX2 FMA intrinsics
         *
         * 内层循环: c_row[j] += a_il * b_row[j]
         * AVX2 一次处理 4 个 double，FMA 同时做乘加
         * 对于 n=784: 784/4 = 196 次 FMA 迭代（vs 784 次标量迭代）
         */

        /* AVX2 + FMA 路径（n >= 4 时使用） */
        if (n >= 4) {
            const __m256d zero = _mm256_setzero_pd();
            for (uint32_t i = 0; i < m; ++i) {
                const double* a_row = &a_flat[(size_t)i * k];
                double* c_row = &c[(size_t)i * n];

                /* 初始化 c_row 为 0（AVX2 批量清零） */
                uint32_t j_zero = 0;
                for (; j_zero + 4 <= n; j_zero += 4) {
                    _mm256_storeu_pd(&c_row[j_zero], zero);
                }
                for (; j_zero < n; ++j_zero) {
                    c_row[j_zero] = 0.0;
                }

                for (uint32_t l = 0; l < k; ++l) {
                    __m256d a_vec = _mm256_set1_pd(a_row[l]);
                    const double* b_row = &b_flat[(size_t)l * n];
                    uint32_t j = 0;
                    /* 主循环: 每次 4 个 double */
                    for (; j + 4 <= n; j += 4) {
                        __m256d c_vec = _mm256_loadu_pd(&c_row[j]);
                        __m256d b_vec = _mm256_loadu_pd(&b_row[j]);
                        c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                        _mm256_storeu_pd(&c_row[j], c_vec);
                    }
                    /* 尾部: 处理剩余 1-3 个元素 */
                    for (; j < n; ++j) {
                        c_row[j] += a_row[l] * b_row[j];
                    }
                }
            }
        } else {
            /* n < 4: 纯标量路径 */
            for (uint32_t i = 0; i < m; ++i) {
                const double* a_row = &a_flat[(size_t)i * k];
                double* c_row = &c[(size_t)i * n];
                for (uint32_t j = 0; j < n; ++j) {
                    c_row[j] = 0.0;
                }
                for (uint32_t l = 0; l < k; ++l) {
                    double a_il = a_row[l];
                    const double* b_row = &b_flat[(size_t)l * n];
                    for (uint32_t j = 0; j < n; ++j) {
                        c_row[j] += a_il * b_row[j];
                    }
                }
            }
        }
    }

    return c;
}

/**
 * HC8 整数 matmul（显式 int8 累加路径）
 *
 * 接收两个 numpy float32 数组，内部走完整的 HC8 量化 + int8×int8→int32 累加 + 反量化路径。
 * 与 Python 端 hc8_matmul（numpy float32 BLAS）数学等价，但走真正的整数累加。
 *
 * 流程：
 *   1. 对 x 和 w 分别计算 scale = max(|x|) / 127
 *   2. 量化到 int8: q = round(x / scale), clamp to [-127, 127]
 *   3. int8 × int8 → int32 累加（使用 hc8_matmul_kernel_vnni 或简单标量实现）
 *   4. 反量化: y = c_int32 * x_scale * w_scale
 *
 * Args:
 *   x: (m, k) numpy float32，行优先
 *   w: (k, n) numpy float32，行优先
 *
 * Returns:
 *   y: (m, n) numpy float32，输出矩阵
 */
static py::array_t<float> py_hc8_matmul_int(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    py::array_t<float, py::array::c_style | py::array::forcecast> w
) {
    auto x_buf = x.request();
    auto w_buf = w.request();

    if (x_buf.ndim != 2) {
        throw std::runtime_error("x 必须是 2D 数组 (m, k)");
    }
    if (w_buf.ndim != 2) {
        throw std::runtime_error("w 必须是 2D 数组 (k, n)");
    }

    uint32_t m = (uint32_t)x_buf.shape[0];
    uint32_t k = (uint32_t)x_buf.shape[1];
    uint32_t k2 = (uint32_t)w_buf.shape[0];
    uint32_t n = (uint32_t)w_buf.shape[1];

    if (k != k2) {
        throw std::runtime_error(
            "维度不匹配：x 是 (m, " + std::to_string(k) +
            ")，w 是 (" + std::to_string(k2) + ", n)"
        );
    }

    const float* x_ptr = (const float*)x_buf.ptr;
    const float* w_ptr = (const float*)w_buf.ptr;

    /* 分配输出 */
    py::array_t<float> y_out({(py::ssize_t)m, (py::ssize_t)n});
    auto y_buf = y_out.request();
    float* y_ptr = (float*)y_buf.ptr;

    /* 释放 GIL 进行纯 C 计算 */
    {
        py::gil_scoped_release release;

        /* 1. 计算 x 的 scale 和量化值 */
        float x_max = 0.0f;
        for (size_t i = 0; i < (size_t)m * k; ++i) {
            float a = std::abs(x_ptr[i]);
            if (a > x_max) x_max = a;
        }
        float x_scale = (x_max > 0.0f) ? (x_max / 127.0f) : 1.0f;
        float x_inv_scale = 1.0f / x_scale;

        /* 2. 计算 w 的 scale 和量化值 */
        float w_max = 0.0f;
        for (size_t i = 0; i < (size_t)k * n; ++i) {
            float a = std::abs(w_ptr[i]);
            if (a > w_max) w_max = a;
        }
        float w_scale = (w_max > 0.0f) ? (w_max / 127.0f) : 1.0f;
        float w_inv_scale = 1.0f / w_scale;

        /* 3. 量化 x 到 int8（行优先 m×k） */
        std::vector<int8_t> x_q((size_t)m * k);
        for (size_t i = 0; i < (size_t)m * k; ++i) {
            int32_t q = (int32_t)std::lround(x_ptr[i] * x_inv_scale);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            x_q[i] = (int8_t)q;
        }

        /* 4. 量化 w 到 int8（行优先 k×n） */
        std::vector<int8_t> w_q((size_t)k * n);
        for (size_t i = 0; i < (size_t)k * n; ++i) {
            int32_t q = (int32_t)std::lround(w_ptr[i] * w_inv_scale);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            w_q[i] = (int8_t)q;
        }

        /* 5. int8 × int8 → int32 累加 → 反量化 */
        float combined_scale = x_scale * w_scale;
        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t j = 0; j < n; ++j) {
                int32_t acc = 0;
                for (uint32_t l = 0; l < k; ++l) {
                    acc += (int32_t)x_q[(size_t)i * k + l] * (int32_t)w_q[(size_t)l * n + j];
                }
                y_ptr[(size_t)i * n + j] = (float)acc * combined_scale;
            }
        }
    }

    return y_out;
}

/**
 * HC8 量化（显式 int8 字节路径）
 *
 * 接收 numpy float32 数组，返回 HC8 量化字节和 scale。
 * 不进行 round-trip 反量化（与 Python 端 hc8_quantize 不同）。
 *
 * HC8 字节格式：每个元素 6 字节，v[0] = 量化值 + 128（offset），v[1..5] = 0
 *
 * Args:
 *   x: numpy float32 数组（任意形状，内部按扁平处理）
 *
 * Returns:
 *   (bytes, scale): HC8 字节和量化 scale
 */
static py::tuple py_hc8_quantize_int(
    py::array_t<float, py::array::c_style | py::array::forcecast> x
) {
    auto x_buf = x.request();

    /* 计算总元素数 */
    size_t n = 1;
    for (int i = 0; i < x_buf.ndim; ++i) {
        n *= (size_t)x_buf.shape[i];
    }

    const float* x_ptr = (const float*)x_buf.ptr;

    /* 计算 scale = max(|x|) / 127 */
    float x_max = 0.0f;
    {
        py::gil_scoped_release release;
        for (size_t i = 0; i < n; ++i) {
            float a = std::abs(x_ptr[i]);
            if (a > x_max) x_max = a;
        }
    }
    float scale = (x_max > 0.0f) ? (x_max / 127.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    /* 量化并构造 HC8 字节（每个元素 6 字节：v[0]=量化值+128, v[1..5]=0） */
    /* HC8_OFFSET = 128, QMIN = -127, QMAX = 127 */
    std::string bytes_str;
    {
        py::gil_scoped_release release;
        bytes_str.resize(n * 6, 0);  /* 全初始化为 0（v[1..5] = 0） */
        char* bytes_data = &bytes_str[0];
        for (size_t i = 0; i < n; ++i) {
            int32_t q = (int32_t)std::lround(x_ptr[i] * inv_scale);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            uint8_t q_u = (uint8_t)(q + 128);  /* offset 128, 范围 [1, 255] */
            bytes_data[i * 6] = (char)q_u;
            /* v[1..5] 已经是 0（resize 时初始化） */
        }
    }

    return py::make_tuple(py::bytes(bytes_str), py::float_(scale));
}

/* ============================================================================
 * UFP-1 残差量化（v[0..depth] 存量化残差，48-bit 存储精度）
 * ============================================================================
 *
 * 设计见 hc_ufp_1_residual_design.md。
 *
 * 与 v[0] 版本的区别：
 *   - quantize_residual 返回 (bytes, scales_list)，scales_list 长度 = depth+1
 *   - dequantize_residual 接受 scales_list
 *   - matmul_residual 用方案 C（反量化到 float → 重新量化到 v[0] → 矩阵乘）
 *   - relu_residual 输出总是 depth=0（ReLU 是非线性，跨层残差失效）
 */

/**
 * 残差量化：float list → (bytes, scales_list)
 *
 * Args:
 *   values: float list
 *   depth: 残差深度（0~5，0 等价于 quantize）
 *   schema: 量化方案
 *
 * Returns:
 *   (bytes, scales): bytes 是 HC8 数组，scales 是 float list（长度 depth+1）
 */
static py::tuple py_quantize_residual(const std::vector<float>& w,
                                       int depth,
                                       const PyQuantSchema& schema) {
    if (w.empty()) {
        return py::make_tuple(py::bytes(""), std::vector<float>{});
    }

    /* depth 范围检查 */
    if (depth < 0) depth = 0;
    if (depth > 5) depth = 5;

    uint32_t n = (uint32_t)w.size();
    std::vector<hc8_t> out(n);
    hc8_residual_scales_t scales;
    /* scales 初始化全 0（hc8_quantize_residual 内部也会清零） */
    for (int i = 0; i < 6; ++i) scales.scales[i] = 0.0f;

    hc8_quantize_residual(w.data(), n, depth, &schema.raw, out.data(), &scales);

    /* bytes */
    py::bytes bytes_out(reinterpret_cast<const char*>(out.data()),
                        (size_t)n * sizeof(hc8_t));

    /* scales list（长度 depth+1） */
    std::vector<float> scales_list(depth + 1);
    for (int i = 0; i <= depth; ++i) {
        scales_list[i] = scales.scales[i];
    }

    return py::make_tuple(bytes_out, scales_list);
}

/**
 * 残差反量化：bytes + scales_list → float list
 *
 * Args:
 *   bytes: HC8 bytes
 *   scales: float list（长度 depth+1）
 *   depth: 残差深度
 *   schema: 量化方案
 */
static std::vector<float> py_dequantize_residual(const std::string& bytes,
                                                  const std::vector<float>& scales,
                                                  int depth,
                                                  const PyQuantSchema& schema) {
    if (bytes.empty()) return {};

    /* depth 范围检查 */
    if (depth < 0) depth = 0;
    if (depth > 5) depth = 5;

    /* scales 长度校验 */
    if ((int)scales.size() < depth + 1) {
        throw std::runtime_error(
            "scales 长度不足：需要 " + std::to_string(depth + 1) +
            "（depth+1），得到 " + std::to_string(scales.size())
        );
    }

    /* bytes 长度校验 */
    size_t total_bytes = bytes.size();
    if (total_bytes % sizeof(hc8_t) != 0) {
        throw std::runtime_error("bytes 长度必须是 6 的倍数（每个 HC8 6 字节）");
    }
    uint32_t n = (uint32_t)(total_bytes / sizeof(hc8_t));

    /* 构造 hc8_residual_scales_t */
    hc8_residual_scales_t scales_raw;
    for (int i = 0; i < 6; ++i) {
        scales_raw.scales[i] = (i <= depth) ? scales[i] : 0.0f;
    }

    std::vector<float> out(n);
    hc8_dequantize_residual(reinterpret_cast<const hc8_t*>(bytes.data()), n,
                            depth, &scales_raw, &schema.raw, out.data());
    return out;
}

/**
 * 残差矩阵乘：返回 (bytes_c, c_scale)
 *
 * 方案 C：反量化到 float → 重新量化到 v[0] → 做 hc8_matmul
 * 输出总是 depth=0（运算精度是 8-bit）
 *
 * Args:
 *   bytes_a, bytes_b: A、B 矩阵的 HC8 bytes
 *   m, k, n: 矩阵维度
 *   a_depth, b_depth: A、B 的残差深度
 *   a_scales, b_scales: A、B 的 scales list
 *   schema: 量化方案
 */
static py::tuple py_matmul_residual(const std::string& bytes_a,
                                     const std::string& bytes_b,
                                     uint32_t m, uint32_t k, uint32_t n,
                                     int a_depth, int b_depth,
                                     const std::vector<float>& a_scales,
                                     const std::vector<float>& b_scales,
                                     const PyQuantSchema& schema) {
    /* depth 范围检查 */
    if (a_depth < 0) a_depth = 0;
    if (a_depth > 5) a_depth = 5;
    if (b_depth < 0) b_depth = 0;
    if (b_depth > 5) b_depth = 5;

    /* 输入长度校验 */
    size_t expected_a = (size_t)m * k * sizeof(hc8_t);
    size_t expected_b = (size_t)k * n * sizeof(hc8_t);
    if (bytes_a.size() != expected_a) {
        throw std::runtime_error(
            "bytes_a 长度不匹配：期望 " + std::to_string(expected_a) +
            "，得到 " + std::to_string(bytes_a.size()));
    }
    if (bytes_b.size() != expected_b) {
        throw std::runtime_error(
            "bytes_b 长度不匹配：期望 " + std::to_string(expected_b) +
            "，得到 " + std::to_string(bytes_b.size()));
    }
    if ((int)a_scales.size() < a_depth + 1) {
        throw std::runtime_error("a_scales 长度不足");
    }
    if ((int)b_scales.size() < b_depth + 1) {
        throw std::runtime_error("b_scales 长度不足");
    }

    /* 构造 scales_raw */
    hc8_residual_scales_t a_scales_raw, b_scales_raw;
    for (int i = 0; i < 6; ++i) {
        a_scales_raw.scales[i] = (i <= a_depth) ? a_scales[i] : 0.0f;
        b_scales_raw.scales[i] = (i <= b_depth) ? b_scales[i] : 0.0f;
    }

    const hc8_t* a = reinterpret_cast<const hc8_t*>(bytes_a.data());
    const hc8_t* b = reinterpret_cast<const hc8_t*>(bytes_b.data());

    std::vector<hc8_t> c((size_t)m * n);
    float c_scale = 1.0f;

    hc8_residual_matmul(a, b, m, k, n,
                        a_depth, b_depth,
                        &a_scales_raw, &b_scales_raw,
                        &schema.raw, c.data(), &c_scale);

    py::bytes bytes_c(reinterpret_cast<const char*>(c.data()),
                      (size_t)m * n * sizeof(hc8_t));
    return py::make_tuple(bytes_c, c_scale);
}

/**
 * 残差 ReLU：输出总是 depth=0
 *
 * ReLU 是非线性运算，跨层残差失效。
 *
 * Args:
 *   bytes_x: X 矩阵 HC8 bytes（m×n）
 *   m, n: 矩阵维度
 *   schema: 量化方案
 */
static py::bytes py_relu_residual(const std::string& bytes_x,
                                   uint32_t m, uint32_t n,
                                   const PyQuantSchema& schema) {
    size_t expected = (size_t)m * n * sizeof(hc8_t);
    if (bytes_x.size() != expected) {
        throw std::runtime_error(
            "bytes_x 长度不匹配：期望 " + std::to_string(expected) +
            "，得到 " + std::to_string(bytes_x.size()));
    }

    const hc8_t* x = reinterpret_cast<const hc8_t*>(bytes_x.data());
    std::vector<hc8_t> out((size_t)m * n);

    hc8_residual_relu(x, m, n, &schema.raw, out.data());

    return py::bytes(reinterpret_cast<const char*>(out.data()),
                     (size_t)m * n * sizeof(hc8_t));
}

/* ============================================================================
 * UFP-2 方案 B：整数域累加矩阵乘（运算精度提升到 16-48 bit）
 * ============================================================================
 *
 * 设计见 hc_ufp_2_scheme_b_design.md。
 *
 * 与方案 C（matmul_residual）的区别：
 *   - 方案 C：反量化到 float32 → 重新量化到 v[0] → int8 矩阵乘
 *   - 方案 B：直接用 v[0..depth] 的 int8 值做矩阵乘，用 double 合并
 *
 * 计算量：(a_depth+1) * (b_depth+1) 次 int8 矩阵乘
 */

/**
 * UFP-2 方案 B 残差矩阵乘
 *
 * Args:
 *   bytes_a, bytes_b: A、B 矩阵的 HC8 bytes
 *   m, k, n: 矩阵维度
 *   a_depth, b_depth: A、B 的残差深度
 *   a_scales, b_scales: A、B 的 scales list
 *   schema: 量化方案
 *
 * Returns:
 *   (bytes_c, c_scale): C 矩阵 HC8 bytes（depth=0）和 scale
 */
static py::tuple py_matmul_residual_b(const std::string& bytes_a,
                                       const std::string& bytes_b,
                                       uint32_t m, uint32_t k, uint32_t n,
                                       int a_depth, int b_depth,
                                       const std::vector<float>& a_scales,
                                       const std::vector<float>& b_scales,
                                       const PyQuantSchema& schema) {
    /* depth 范围检查 */
    if (a_depth < 0) a_depth = 0;
    if (a_depth > 5) a_depth = 5;
    if (b_depth < 0) b_depth = 0;
    if (b_depth > 5) b_depth = 5;

    /* 输入长度校验 */
    size_t expected_a = (size_t)m * k * sizeof(hc8_t);
    size_t expected_b = (size_t)k * n * sizeof(hc8_t);
    if (bytes_a.size() != expected_a) {
        throw std::runtime_error(
            "bytes_a 长度不匹配：期望 " + std::to_string(expected_a) +
            "，得到 " + std::to_string(bytes_a.size()));
    }
    if (bytes_b.size() != expected_b) {
        throw std::runtime_error(
            "bytes_b 长度不匹配：期望 " + std::to_string(expected_b) +
            "，得到 " + std::to_string(bytes_b.size()));
    }
    if ((int)a_scales.size() < a_depth + 1) {
        throw std::runtime_error("a_scales 长度不足");
    }
    if ((int)b_scales.size() < b_depth + 1) {
        throw std::runtime_error("b_scales 长度不足");
    }

    /* 构造 scales_raw */
    hc8_residual_scales_t a_scales_raw, b_scales_raw;
    for (int i = 0; i < 6; ++i) {
        a_scales_raw.scales[i] = (i <= a_depth) ? a_scales[i] : 0.0f;
        b_scales_raw.scales[i] = (i <= b_depth) ? b_scales[i] : 0.0f;
    }

    const hc8_t* a = reinterpret_cast<const hc8_t*>(bytes_a.data());
    const hc8_t* b = reinterpret_cast<const hc8_t*>(bytes_b.data());

    std::vector<hc8_t> c((size_t)m * n);
    float c_scale = 1.0f;

    hc8_residual_matmul_b(a, b, m, k, n,
                          a_depth, b_depth,
                          &a_scales_raw, &b_scales_raw,
                          &schema.raw, c.data(), &c_scale);

    py::bytes bytes_c(reinterpret_cast<const char*>(c.data()),
                      (size_t)m * n * sizeof(hc8_t));
    return py::make_tuple(bytes_c, c_scale);
}

/* ============================================================================
 * HC4 非对称拆分 API（int8 → int4+int4 运算路径）
 * ============================================================================
 *
 * 设计见 hc_v1.4_asymmetric_split_design.md。
 * HC4 与 HC8 二进制相同，拆分/合并零成本。
 * matmul_residual_hc4_b 用 4 次 int4×int4 矩阵乘 + 偏移修正替代 1 次 int8 矩阵乘。
 */

/* HC8 → HC4 拆分（零成本，二进制相同） */
static py::bytes py_split_to_hc4(const std::string& bytes_hc8, uint32_t n) {
    size_t expected = (size_t)n * sizeof(hc8_t);
    if (bytes_hc8.size() != expected) {
        throw std::runtime_error(
            "bytes_hc8 长度不匹配：期望 " + std::to_string(expected) +
            "，得到 " + std::to_string(bytes_hc8.size()));
    }
    const hc8_t* hc8 = reinterpret_cast<const hc8_t*>(bytes_hc8.data());
    std::vector<hc4_t> out(n);
    hc8_split_to_hc4(hc8, n, out.data());
    return py::bytes(reinterpret_cast<const char*>(out.data()),
                     (size_t)n * sizeof(hc4_t));
}

/* HC4 → HC8 合并（零成本，二进制相同，无损双射） */
static py::bytes py_merge_to_hc8(const std::string& bytes_hc4, uint32_t n) {
    size_t expected = (size_t)n * sizeof(hc4_t);
    if (bytes_hc4.size() != expected) {
        throw std::runtime_error(
            "bytes_hc4 长度不匹配：期望 " + std::to_string(expected) +
            "，得到 " + std::to_string(bytes_hc4.size()));
    }
    const hc4_t* hc4 = reinterpret_cast<const hc4_t*>(bytes_hc4.data());
    std::vector<hc8_t> out(n);
    hc4_merge_to_hc8(hc4, n, out.data());
    return py::bytes(reinterpret_cast<const char*>(out.data()),
                     (size_t)n * sizeof(hc8_t));
}

/* HC4 残差矩阵乘（int4×int4→int32 累加 + double 合并） */
static py::tuple py_matmul_residual_hc4_b(const std::string& bytes_a,
                                          const std::string& bytes_b,
                                          uint32_t m, uint32_t k, uint32_t n,
                                          int a_depth, int b_depth,
                                          const std::vector<float>& a_scales,
                                          const std::vector<float>& b_scales,
                                          const PyQuantSchema& schema) {
    /* depth 范围检查 */
    if (a_depth < 0) a_depth = 0;
    if (a_depth > 5) a_depth = 5;
    if (b_depth < 0) b_depth = 0;
    if (b_depth > 5) b_depth = 5;

    /* 输入长度校验（HC4 与 HC8 二进制相同，字节数一致） */
    size_t expected_a = (size_t)m * k * sizeof(hc4_t);
    size_t expected_b = (size_t)k * n * sizeof(hc4_t);
    if (bytes_a.size() != expected_a) {
        throw std::runtime_error(
            "bytes_a 长度不匹配：期望 " + std::to_string(expected_a) +
            "，得到 " + std::to_string(bytes_a.size()));
    }
    if (bytes_b.size() != expected_b) {
        throw std::runtime_error(
            "bytes_b 长度不匹配：期望 " + std::to_string(expected_b) +
            "，得到 " + std::to_string(bytes_b.size()));
    }
    if ((int)a_scales.size() < a_depth + 1) {
        throw std::runtime_error("a_scales 长度不足");
    }
    if ((int)b_scales.size() < b_depth + 1) {
        throw std::runtime_error("b_scales 长度不足");
    }

    /* 构造 scales_raw */
    hc8_residual_scales_t a_scales_raw, b_scales_raw;
    for (int i = 0; i < 6; ++i) {
        a_scales_raw.scales[i] = (i <= a_depth) ? a_scales[i] : 0.0f;
        b_scales_raw.scales[i] = (i <= b_depth) ? b_scales[i] : 0.0f;
    }

    const hc4_t* a = reinterpret_cast<const hc4_t*>(bytes_a.data());
    const hc4_t* b = reinterpret_cast<const hc4_t*>(bytes_b.data());

    std::vector<hc8_t> c((size_t)m * n);
    float c_scale = 1.0f;

    hc4_residual_matmul_b(a, b, m, k, n,
                          a_depth, b_depth,
                          &a_scales_raw, &b_scales_raw,
                          &schema.raw, c.data(), &c_scale);

    py::bytes bytes_c(reinterpret_cast<const char*>(c.data()),
                      (size_t)m * n * sizeof(hc8_t));
    return py::make_tuple(bytes_c, c_scale);
}

/* ============================================================================
 * SIMD 加速 API（v1.4.2-simd）
 *
 * 设计见 hc_simd_avx_vnni_design.md v1.2。
 * 接口与标量版完全一致，内部用 SoA 布局 + AVX-VNNI/AVX2 kernel。
 * 当前实现为标量版（E 阶段验证布局正确性），intrinsics 在 A 阶段添加。
 * ============================================================================ */

/* HC8 残差矩阵乘 SIMD 版（SoA 布局，AVX-VNNI 加速） */
static py::tuple py_matmul_residual_b_simd(const std::string& bytes_a,
                                            const std::string& bytes_b,
                                            uint32_t m, uint32_t k, uint32_t n,
                                            int a_depth, int b_depth,
                                            const std::vector<float>& a_scales,
                                            const std::vector<float>& b_scales,
                                            const PyQuantSchema& schema) {
    if (a_depth < 0) a_depth = 0;
    if (a_depth > 5) a_depth = 5;
    if (b_depth < 0) b_depth = 0;
    if (b_depth > 5) b_depth = 5;

    size_t expected_a = (size_t)m * k * sizeof(hc8_t);
    size_t expected_b = (size_t)k * n * sizeof(hc8_t);
    if (bytes_a.size() != expected_a) {
        throw std::runtime_error(
            "bytes_a 长度不匹配：期望 " + std::to_string(expected_a) +
            "，得到 " + std::to_string(bytes_a.size()));
    }
    if (bytes_b.size() != expected_b) {
        throw std::runtime_error(
            "bytes_b 长度不匹配：期望 " + std::to_string(expected_b) +
            "，得到 " + std::to_string(bytes_b.size()));
    }
    if ((int)a_scales.size() < a_depth + 1) {
        throw std::runtime_error("a_scales 长度不足");
    }
    if ((int)b_scales.size() < b_depth + 1) {
        throw std::runtime_error("b_scales 长度不足");
    }

    hc8_residual_scales_t a_scales_raw, b_scales_raw;
    for (int i = 0; i < 6; ++i) {
        a_scales_raw.scales[i] = (i <= a_depth) ? a_scales[i] : 0.0f;
        b_scales_raw.scales[i] = (i <= b_depth) ? b_scales[i] : 0.0f;
    }

    const hc8_t* a = reinterpret_cast<const hc8_t*>(bytes_a.data());
    const hc8_t* b = reinterpret_cast<const hc8_t*>(bytes_b.data());

    std::vector<hc8_t> c((size_t)m * n);
    float c_scale = 1.0f;

    hc8_residual_matmul_b_simd(a, b, m, k, n,
                                a_depth, b_depth,
                                &a_scales_raw, &b_scales_raw,
                                &schema.raw, c.data(), &c_scale);

    py::bytes bytes_c(reinterpret_cast<const char*>(c.data()),
                      (size_t)m * n * sizeof(hc8_t));
    return py::make_tuple(bytes_c, c_scale);
}

/* HC4 残差矩阵乘 SIMD 版（SoA per-nibble 布局，AVX2 maddubs 加速） */
static py::tuple py_matmul_residual_hc4_b_simd(const std::string& bytes_a,
                                                const std::string& bytes_b,
                                                uint32_t m, uint32_t k, uint32_t n,
                                                int a_depth, int b_depth,
                                                const std::vector<float>& a_scales,
                                                const std::vector<float>& b_scales,
                                                const PyQuantSchema& schema) {
    if (a_depth < 0) a_depth = 0;
    if (a_depth > 5) a_depth = 5;
    if (b_depth < 0) b_depth = 0;
    if (b_depth > 5) b_depth = 5;

    size_t expected_a = (size_t)m * k * sizeof(hc4_t);
    size_t expected_b = (size_t)k * n * sizeof(hc4_t);
    if (bytes_a.size() != expected_a) {
        throw std::runtime_error(
            "bytes_a 长度不匹配：期望 " + std::to_string(expected_a) +
            "，得到 " + std::to_string(bytes_a.size()));
    }
    if (bytes_b.size() != expected_b) {
        throw std::runtime_error(
            "bytes_b 长度不匹配：期望 " + std::to_string(expected_b) +
            "，得到 " + std::to_string(bytes_b.size()));
    }
    if ((int)a_scales.size() < a_depth + 1) {
        throw std::runtime_error("a_scales 长度不足");
    }
    if ((int)b_scales.size() < b_depth + 1) {
        throw std::runtime_error("b_scales 长度不足");
    }

    hc8_residual_scales_t a_scales_raw, b_scales_raw;
    for (int i = 0; i < 6; ++i) {
        a_scales_raw.scales[i] = (i <= a_depth) ? a_scales[i] : 0.0f;
        b_scales_raw.scales[i] = (i <= b_depth) ? b_scales[i] : 0.0f;
    }

    const hc4_t* a = reinterpret_cast<const hc4_t*>(bytes_a.data());
    const hc4_t* b = reinterpret_cast<const hc4_t*>(bytes_b.data());

    std::vector<hc8_t> c((size_t)m * n);
    float c_scale = 1.0f;

    hc4_residual_matmul_b_simd(a, b, m, k, n,
                                a_depth, b_depth,
                                &a_scales_raw, &b_scales_raw,
                                &schema.raw, c.data(), &c_scale);

    py::bytes bytes_c(reinterpret_cast<const char*>(c.data()),
                      (size_t)m * n * sizeof(hc8_t));
    return py::make_tuple(bytes_c, c_scale);
}

/* ============================================================================
 * SBE C 化 API（v1.5-sbe，2026-07-22）
 *
 * 设计见 hc_simd_avx_vnni_design.md §4。
 * SBE（语义块编码）per-block 量化 + AVX-VNNI 加速 matmul。
 *
 * 与 Python sbe_conv2d.py 中 sbe_matmul / quantize_weight_sbe 数学等价，
 * 但把 Python for 循环 + numpy matmul 换成 C 循环 + VNNI kernel。
 *
 * 输入/输出全部用 numpy 数组（py::array_t），避免 list 转换开销。
 * ============================================================================ */

/**
 * SBE 权重 per-block 量化 + 预处理（C 化版）
 *
 * 把 (k, n) 权重矩阵按 k 维度分 groups 块，每块独立量化。
 * 每块预处理为 VNNI 友好格式（转置 + 有符号转换 + 预计算 sum_b）。
 *
 * Args:
 *   w:        (k, n) numpy float32，行优先权重矩阵
 *   groups:   分块数
 *   k_block:  每块列数（必须满足 groups * k_block == k）
 *   k, n:     矩阵维度
 *
 * Returns:
 *   (w_signed, w_sum_b, w_scales) 三元组：
 *     w_signed: (groups, n, k_block) numpy int8，每块 n×k_block 行优先
 *     w_sum_b:  (groups, n) numpy int32，每块 n 个 sum_b 值
 *     w_scales: (groups,) numpy float32，每块的量化 scale
 */
static py::tuple py_sbe_quantize_weight_blocks(
    py::array_t<float, py::array::c_style | py::array::forcecast> w,
    uint32_t groups, uint32_t k_block, uint32_t k, uint32_t n
) {
    /* 校验输入形状 */
    auto w_buf = w.request();
    if (w_buf.ndim != 2) {
        throw std::runtime_error("w 必须是 2D 数组 (k, n)");
    }
    if ((uint32_t)w_buf.shape[0] != k || (uint32_t)w_buf.shape[1] != n) {
        throw std::runtime_error(
            "w 形状不匹配：期望 (" + std::to_string(k) + ", " + std::to_string(n) +
            ")，得到 (" + std::to_string(w_buf.shape[0]) + ", " +
            std::to_string(w_buf.shape[1]) + ")"
        );
    }
    if (groups == 0 || k_block == 0) {
        throw std::runtime_error("groups 和 k_block 不能为 0");
    }
    if (groups * k_block != k) {
        throw std::runtime_error(
            "groups * k_block != k: " + std::to_string(groups) + " * " +
            std::to_string(k_block) + " != " + std::to_string(k)
        );
    }

    /* 分配输出数组 */
    py::array_t<int8_t>   w_signed_out({(py::ssize_t)groups, (py::ssize_t)n, (py::ssize_t)k_block});
    py::array_t<int32_t>  w_sum_b_out({(py::ssize_t)groups, (py::ssize_t)n});
    py::array_t<float>    w_scales_out((py::ssize_t)groups);

    auto w_signed_buf = w_signed_out.request();
    auto w_sum_b_buf  = w_sum_b_out.request();
    auto w_scales_buf = w_scales_out.request();

    sbe_quantize_weight_blocks_c(
        (const float*)w_buf.ptr,
        groups, k_block, k, n,
        (int8_t*)w_signed_buf.ptr,
        (int32_t*)w_sum_b_buf.ptr,
        (float*)w_scales_buf.ptr
    );

    return py::make_tuple(w_signed_out, w_sum_b_out, w_scales_out);
}

/**
 * SBE 分块 matmul（C 化版，AVX-VNNI 加速）
 *
 * 对每个 group g：
 *   1. 提取 x_block: x[:, g*k_block : (g+1)*k_block]
 *   2. per-block 量化 x_block → uint8
 *   3. VNNI matmul: int8 × int8 → int32 累加（_mm256_dpbusd_epi32）
 *   4. float 累加到输出 y
 *
 * 与 Python sbe_matmul 数学等价，但内部用 VNNI kernel 替代 numpy matmul。
 *
 * Args:
 *   x:         (m, k) numpy float32，行优先输入矩阵
 *   w_signed:  (groups, n, k_block) numpy int8，由 sbe_quantize_weight_blocks 返回
 *   w_sum_b:   (groups, n) numpy int32，由 sbe_quantize_weight_blocks 返回
 *   w_scales:  (groups,) numpy float32，由 sbe_quantize_weight_blocks 返回
 *   groups:    分块数
 *   k_block:   每块列数
 *   m, k, n:   矩阵形状
 *
 * Returns:
 *   y: (m, n) numpy float32，输出矩阵
 */
static py::array_t<float> py_sbe_matmul(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> w_signed,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> w_sum_b,
    py::array_t<float, py::array::c_style | py::array::forcecast> w_scales,
    uint32_t groups, uint32_t k_block, uint32_t m, uint32_t k, uint32_t n
) {
    /* 校验输入形状 */
    auto x_buf = x.request();
    if (x_buf.ndim != 2 ||
        (uint32_t)x_buf.shape[0] != m || (uint32_t)x_buf.shape[1] != k) {
        throw std::runtime_error(
            "x 形状不匹配：期望 (" + std::to_string(m) + ", " + std::to_string(k) + ")"
        );
    }

    auto ws_buf = w_signed.request();
    if (ws_buf.ndim != 3 ||
        (uint32_t)ws_buf.shape[0] != groups ||
        (uint32_t)ws_buf.shape[1] != n ||
        (uint32_t)ws_buf.shape[2] != k_block) {
        throw std::runtime_error(
            "w_signed 形状不匹配：期望 (" + std::to_string(groups) + ", " +
            std::to_string(n) + ", " + std::to_string(k_block) + ")"
        );
    }

    auto wsb_buf = w_sum_b.request();
    if (wsb_buf.ndim != 2 ||
        (uint32_t)wsb_buf.shape[0] != groups || (uint32_t)wsb_buf.shape[1] != n) {
        throw std::runtime_error("w_sum_b 形状不匹配");
    }

    auto wsc_buf = w_scales.request();
    if (wsc_buf.ndim != 1 || (uint32_t)wsc_buf.shape[0] != groups) {
        throw std::runtime_error("w_scales 形状不匹配");
    }

    if (groups * k_block != k) {
        throw std::runtime_error("groups * k_block != k");
    }

    /* 分配输出并清零（sbe_matmul_c 是累加语义） */
    py::array_t<float> y_out({(py::ssize_t)m, (py::ssize_t)n});
    auto y_buf = y_out.request();

    /* 释放 GIL 进行纯 C 计算（所有 buffer 指针已提取，py::array_t 变量在栈上存活保持引用）
     * 使 C 计算可与 Python 线程重叠（阶段 0.2，multithreading_optimization_plan.md） */
    {
        py::gil_scoped_release release;
        std::memset(y_buf.ptr, 0, (size_t)m * n * sizeof(float));

        sbe_matmul_c(
            (const float*)x_buf.ptr,
            (const int8_t*)ws_buf.ptr,
            (const int32_t*)wsb_buf.ptr,
            (const float*)wsc_buf.ptr,
            groups, k_block, m, k, n,
            (float*)y_buf.ptr
        );
    }

    return y_out;
}

/* ============================================================================
 * SBE Conv2d 前向融合（v2.0.0-conv-fusion，2026-07-28）
 *
 * 在 C 层融合 im2col + SBE matmul + bias add，消除 Python 层开销。
 * ============================================================================ */
static py::array_t<float> py_sbe_conv2d_forward(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> w_signed,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> w_sum_b,
    py::array_t<float, py::array::c_style | py::array::forcecast> w_scales,
    uint32_t B, uint32_t C_in, uint32_t H, uint32_t W,
    uint32_t C_out, uint32_t kh, uint32_t kw,
    uint32_t stride, uint32_t padding,
    uint32_t groups, uint32_t k_block,
    py::object bias_obj  /* None 或 (C_out,) float32 */
) {
    /* 校验 x 形状: (B, C_in, H, W) */
    auto x_buf = x.request();
    if (x_buf.ndim != 4 ||
        (uint32_t)x_buf.shape[0] != B ||
        (uint32_t)x_buf.shape[1] != C_in ||
        (uint32_t)x_buf.shape[2] != H ||
        (uint32_t)x_buf.shape[3] != W) {
        throw std::runtime_error(
            "x 形状不匹配：期望 (" + std::to_string(B) + ", " +
            std::to_string(C_in) + ", " + std::to_string(H) + ", " +
            std::to_string(W) + ")"
        );
    }

    /* 校验权重缓存形状 */
    auto ws_buf = w_signed.request();
    uint32_t K = C_in * kh * kw;
    if (ws_buf.ndim != 3 ||
        (uint32_t)ws_buf.shape[0] != groups ||
        (uint32_t)ws_buf.shape[1] != C_out ||
        (uint32_t)ws_buf.shape[2] != k_block) {
        throw std::runtime_error(
            "w_signed 形状不匹配：期望 (" + std::to_string(groups) + ", " +
            std::to_string(C_out) + ", " + std::to_string(k_block) + ")"
        );
    }

    if (groups * k_block != K) {
        throw std::runtime_error(
            "groups * k_block != K: " + std::to_string(groups) + " * " +
            std::to_string(k_block) + " != " + std::to_string(K)
        );
    }

    /* 计算输出尺寸。
     * 安全审计 2026-08-16 I1：H + 2*padding < kh 时 uint32 减法下溢为巨大值，
     * 后续按 H_out 分配/写入导致 OOM 或越界——先在 int64 域校验再转 uint32 */
    int64_t h_pad = static_cast<int64_t>(H) + 2 * static_cast<int64_t>(padding);
    int64_t w_pad = static_cast<int64_t>(W) + 2 * static_cast<int64_t>(padding);
    if (h_pad < static_cast<int64_t>(kh) || w_pad < static_cast<int64_t>(kw)) {
        throw std::invalid_argument(
            "conv2d 尺寸不合法：padding 后输入 (" + std::to_string(h_pad) + "x" +
            std::to_string(w_pad) + ") 小于卷积核 (" + std::to_string(kh) + "x" +
            std::to_string(kw) + ")");
    }
    uint32_t H_out = static_cast<uint32_t>((h_pad - kh) / stride + 1);
    uint32_t W_out = static_cast<uint32_t>((w_pad - kw) / stride + 1);

    /* 分配输出: (B, C_out, H_out, W_out) */
    py::array_t<float> y_out({(py::ssize_t)B, (py::ssize_t)C_out,
                              (py::ssize_t)H_out, (py::ssize_t)W_out});
    auto y_buf = y_out.request();

    /* 处理 bias: None 或 (C_out,) float32 */
    const float* bias_ptr = nullptr;
    py::array_t<float, py::array::c_style | py::array::forcecast> bias_arr;
    if (!bias_obj.is_none()) {
        bias_arr = py::array_t<float, py::array::c_style | py::array::forcecast>(bias_obj);
        auto bias_buf = bias_arr.request();
        if (bias_buf.ndim != 1 || (uint32_t)bias_buf.shape[0] != C_out) {
            throw std::runtime_error("bias 形状不匹配：期望 (C_out,)");
        }
        bias_ptr = (const float*)bias_buf.ptr;
    }

    /* 提取所有 buffer 指针（在 GIL 释放前，避免临时 buffer_info 析构问题） */
    auto wsb_buf = w_sum_b.request();
    auto wsc_buf = w_scales.request();

    /* 释放 GIL 进行纯 C 计算 */
    {
        py::gil_scoped_release release;
        sbe_conv2d_forward_c(
            (const float*)x_buf.ptr,
            (const int8_t*)ws_buf.ptr,
            (const int32_t*)wsb_buf.ptr,
            (const float*)wsc_buf.ptr,
            B, C_in, H, W,
            C_out, kh, kw,
            stride, padding,
            groups, k_block,
            bias_ptr,
            (float*)y_buf.ptr
        );
    }

    return y_out;
}

/* ============================================================================
 * SBE + Smoothing C 分块 matmul（v1.7.0，2026-07-23）
 *
 * 与 py_sbe_matmul 接口完全一致，内部调用 sbe_matmul_smoothed_c
 * 数学等价但更精确（per-row mean shift 后量化）
 * ============================================================================ */
static py::array_t<float> py_sbe_matmul_smoothed(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> w_signed,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> w_sum_b,
    py::array_t<float, py::array::c_style | py::array::forcecast> w_scales,
    uint32_t groups, uint32_t k_block, uint32_t m, uint32_t k, uint32_t n
) {
    /* 校验输入形状（与 py_sbe_matmul 完全一致） */
    auto x_buf = x.request();
    if (x_buf.ndim != 2 ||
        (uint32_t)x_buf.shape[0] != m || (uint32_t)x_buf.shape[1] != k) {
        throw std::runtime_error(
            "x 形状不匹配：期望 (" + std::to_string(m) + ", " + std::to_string(k) + ")"
        );
    }

    auto ws_buf = w_signed.request();
    if (ws_buf.ndim != 3 ||
        (uint32_t)ws_buf.shape[0] != groups ||
        (uint32_t)ws_buf.shape[1] != n ||
        (uint32_t)ws_buf.shape[2] != k_block) {
        throw std::runtime_error(
            "w_signed 形状不匹配：期望 (" + std::to_string(groups) + ", " +
            std::to_string(n) + ", " + std::to_string(k_block) + ")"
        );
    }

    auto wsb_buf = w_sum_b.request();
    if (wsb_buf.ndim != 2 ||
        (uint32_t)wsb_buf.shape[0] != groups || (uint32_t)wsb_buf.shape[1] != n) {
        throw std::runtime_error("w_sum_b 形状不匹配");
    }

    auto wsc_buf = w_scales.request();
    if (wsc_buf.ndim != 1 || (uint32_t)wsc_buf.shape[0] != groups) {
        throw std::runtime_error("w_scales 形状不匹配");
    }

    if (groups * k_block != k) {
        throw std::runtime_error("groups * k_block != k");
    }

    /* 分配输出并清零（累加语义） */
    py::array_t<float> y_out({(py::ssize_t)m, (py::ssize_t)n});
    auto y_buf = y_out.request();

    /* 释放 GIL 进行纯 C 计算（同 py_sbe_matmul，阶段 0.2） */
    {
        py::gil_scoped_release release;
        std::memset(y_buf.ptr, 0, (size_t)m * n * sizeof(float));

        sbe_matmul_smoothed_c(
            (const float*)x_buf.ptr,
            (const int8_t*)ws_buf.ptr,
            (const int32_t*)wsb_buf.ptr,
            (const float*)wsc_buf.ptr,
            groups, k_block, m, k, n,
            (float*)y_buf.ptr
        );
    }

    return y_out;
}

/* ============================================================================
 * SBE per-channel 量化 + matmul（v1.8.0-perchannel，2026-07-23）
 *
 * 与 per-block SBE 的区别：w_scales 从 (groups,) 改为 (groups, n)，
 * 每个 block 内每个输出通道独立 scale，量化精度更高。
 * ============================================================================ */

static py::tuple py_sbe_quantize_weight_blocks_perchannel(
    py::array_t<float, py::array::c_style | py::array::forcecast> w,
    uint32_t groups, uint32_t k_block, uint32_t k, uint32_t n
) {
    auto w_buf = w.request();
    if (w_buf.ndim != 2) {
        throw std::runtime_error("w 必须是 2D 数组 (k, n)");
    }
    if ((uint32_t)w_buf.shape[0] != k || (uint32_t)w_buf.shape[1] != n) {
        throw std::runtime_error(
            "w 形状不匹配：期望 (" + std::to_string(k) + ", " + std::to_string(n) +
            ")，得到 (" + std::to_string(w_buf.shape[0]) + ", " +
            std::to_string(w_buf.shape[1]) + ")"
        );
    }
    if (groups == 0 || k_block == 0) {
        throw std::runtime_error("groups 和 k_block 不能为 0");
    }
    if (groups * k_block != k) {
        throw std::runtime_error(
            "groups * k_block != k: " + std::to_string(groups) + " * " +
            std::to_string(k_block) + " != " + std::to_string(k)
        );
    }

    /* 分配输出数组：w_scales 形状为 (groups, n) 而非 (groups,) */
    py::array_t<int8_t>   w_signed_out({(py::ssize_t)groups, (py::ssize_t)n, (py::ssize_t)k_block});
    py::array_t<int32_t>  w_sum_b_out({(py::ssize_t)groups, (py::ssize_t)n});
    py::array_t<float>    w_scales_out({(py::ssize_t)groups, (py::ssize_t)n});

    auto w_signed_buf = w_signed_out.request();
    auto w_sum_b_buf  = w_sum_b_out.request();
    auto w_scales_buf = w_scales_out.request();

    {
        py::gil_scoped_release release;
        sbe_quantize_weight_blocks_perchannel_c(
            (const float*)w_buf.ptr,
            groups, k_block, k, n,
            (int8_t*)w_signed_buf.ptr,
            (int32_t*)w_sum_b_buf.ptr,
            (float*)w_scales_buf.ptr
        );
    }

    return py::make_tuple(w_signed_out, w_sum_b_out, w_scales_out);
}

static py::array_t<float> py_sbe_matmul_perchannel(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> w_signed,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> w_sum_b,
    py::array_t<float, py::array::c_style | py::array::forcecast> w_scales,
    uint32_t groups, uint32_t k_block, uint32_t m, uint32_t k, uint32_t n
) {
    /* 校验输入形状 */
    auto x_buf = x.request();
    if (x_buf.ndim != 2 ||
        (uint32_t)x_buf.shape[0] != m || (uint32_t)x_buf.shape[1] != k) {
        throw std::runtime_error(
            "x 形状不匹配：期望 (" + std::to_string(m) + ", " + std::to_string(k) + ")"
        );
    }

    auto ws_buf = w_signed.request();
    if (ws_buf.ndim != 3 ||
        (uint32_t)ws_buf.shape[0] != groups ||
        (uint32_t)ws_buf.shape[1] != n ||
        (uint32_t)ws_buf.shape[2] != k_block) {
        throw std::runtime_error(
            "w_signed 形状不匹配：期望 (" + std::to_string(groups) + ", " +
            std::to_string(n) + ", " + std::to_string(k_block) + ")"
        );
    }

    auto wsb_buf = w_sum_b.request();
    if (wsb_buf.ndim != 2 ||
        (uint32_t)wsb_buf.shape[0] != groups || (uint32_t)wsb_buf.shape[1] != n) {
        throw std::runtime_error("w_sum_b 形状不匹配");
    }

    /* per-channel: w_scales 是 (groups, n) 而非 (groups,) */
    auto wsc_buf = w_scales.request();
    if (wsc_buf.ndim != 2 ||
        (uint32_t)wsc_buf.shape[0] != groups || (uint32_t)wsc_buf.shape[1] != n) {
        throw std::runtime_error(
            "w_scales 形状不匹配（per-channel 需要 (groups, n)）：期望 (" +
            std::to_string(groups) + ", " + std::to_string(n) + ")"
        );
    }

    if (groups * k_block != k) {
        throw std::runtime_error("groups * k_block != k");
    }

    py::array_t<float> y_out({(py::ssize_t)m, (py::ssize_t)n});
    auto y_buf = y_out.request();

    {
        py::gil_scoped_release release;
        std::memset(y_buf.ptr, 0, (size_t)m * n * sizeof(float));
        sbe_matmul_perchannel_c(
            (const float*)x_buf.ptr,
            (const int8_t*)ws_buf.ptr,
            (const int32_t*)wsb_buf.ptr,
            (const float*)wsc_buf.ptr,
            groups, k_block, m, k, n,
            (float*)y_buf.ptr
        );
    }

    return y_out;
}

/* ============================================================================
 * Triple-int8 缩放 C 化（v1.9.0，2026-07-24）
 *
 * 数学等价于 sbe_conv2d.py 的 rescale_to_triple_int8_sbe，但用 AVX2 + OpenMP
 * 替代单线程 numpy 元素级操作（实测 30x+ 加速）。
 *
 * 性能关键：这是 WEF+Triple 前向的主要瓶颈（占前向 ~70%），
 * C 化后从单线程变为多核 AVX2+OpenMP，直接提升 CPU 利用率。
 * ============================================================================ */
static py::tuple py_sbe_rescale_to_triple(
    py::array_t<float, py::array::c_style | py::array::forcecast> x
) {
    auto x_buf = x.request();
    if (x_buf.ndim == 0) {
        throw std::runtime_error("x 不能是标量");
    }

    /* 计算总元素数（支持任意形状，内部按扁平处理） */
    size_t n = 1;
    for (int i = 0; i < x_buf.ndim; ++i) {
        n *= (size_t)x_buf.shape[i];
    }

    /* 分配输出数组（与 x 同形状） */
    py::array_t<float> C_high_out(x_buf.shape);
    py::array_t<float> C_mid_out(x_buf.shape);
    py::array_t<float> C_low_out(x_buf.shape);

    auto ch_buf = C_high_out.request();
    auto cm_buf = C_mid_out.request();
    auto cl_buf = C_low_out.request();

    float scale = 1.0f;

    /* 释放 GIL 进行纯 C 计算（AVX2 + OpenMP 多线程） */
    {
        py::gil_scoped_release release;
        sbe_rescale_to_triple_c(
            (const float*)x_buf.ptr, (uint32_t)n,
            (float*)ch_buf.ptr, (float*)cm_buf.ptr, (float*)cl_buf.ptr, &scale
        );
    }

    return py::make_tuple(C_high_out, C_mid_out, C_low_out, py::float_(scale));
}

static py::array_t<int64_t> py_hc8_multiview_matmul(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> c_acc,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> b_int8,
    uint32_t m, uint32_t k, uint32_t n,
    int n_views
) {
    /* 校验输入形状 */
    auto c_buf = c_acc.request();
    if (c_buf.ndim != 2 ||
        (uint32_t)c_buf.shape[0] != m || (uint32_t)c_buf.shape[1] != k) {
        throw std::runtime_error(
            "c_acc 形状不匹配：期望 (" + std::to_string(m) + ", " +
            std::to_string(k) + ")"
        );
    }

    auto b_buf = b_int8.request();
    if (b_buf.ndim != 2 ||
        (uint32_t)b_buf.shape[0] != k || (uint32_t)b_buf.shape[1] != n) {
        throw std::runtime_error(
            "b_int8 形状不匹配：期望 (" + std::to_string(k) + ", " +
            std::to_string(n) + ")"
        );
    }

    if (n_views < 1 || n_views > 8) {
        throw std::runtime_error("n_views 必须在 1-8 之间");
    }

    /* 分配输出（函数内部会清零） */
    py::array_t<int64_t> out({(py::ssize_t)m, (py::ssize_t)n});

    hc8_multiview_matmul(
        (const int64_t*)c_buf.ptr,
        (const int8_t*)b_buf.ptr,
        m, k, n, n_views,
        (int64_t*)out.request().ptr
    );

    return out;
}

/* ============================================================================
 * 模块定义
 * ============================================================================ */

/* 安全审计 2026-08-16 I2：原生 cpuid 检测 AVX2（含 XCR0 OS 使能检查）。
 * 不用 __builtin_cpu_supports——其依赖 __cpu_model 运行时符号，
 * -nostdlib 链接的 .pyd 无法解析。__cpuid/_xgetbv 均为内联 intrinsic。 */
static bool _cpu_has_avx2(void) {
#if defined(__x86_64__) || defined(_M_X64)
    int regs[4];
    __cpuid(regs, 1);
    bool os_xsave = (regs[2] & (1 << 27)) != 0;
    bool cpu_avx  = (regs[2] & (1 << 28)) != 0;
    if (!(os_xsave && cpu_avx)) return false;
    if ((_xgetbv(0) & 0x6) != 0x6) return false;  /* XMM+YMM 状态 OS 已使能 */
    __cpuid(regs, 7);
    return (regs[1] & (1 << 5)) != 0;  /* leaf7 EBX bit5 = AVX2 */
#else
    return true;
#endif
}

void register_hc8_net(py::module_& m) {
    /* 安全审计 2026-08-16 I2：.pyd 以 -mavx2 编译，非 AVX2 CPU 上首次执行
     * AVX2 指令即 SIGILL（无运行时报错）。import 时显式 cpuid 检测，
     * 不支持则抛 RuntimeError（比 SIGILL 可诊断）。 */
#if defined(__x86_64__) || defined(_M_X64)
    if (!_cpu_has_avx2()) {
        throw std::runtime_error(
            "sgn.hc8_net: CPU 不支持 AVX2（.pyd 以 -mavx2 编译）。"
            "请在支持 AVX2 的 CPU 上运行，或用 -mno-avx2 重新编译。");
    }
#endif

    m.doc() = R"(
sgn.hc8_net - HC8 神经网络运算扩展（阶段 1.3 全整数路径）

主程序 pysgn 缺失的矩阵运算 API 的扩展实现（合并到 sgn.hc8_net 子模块）。
不修改 engine/hc/ 下任何现有文件。

核心 API：
    - QuantSchema: 量化方案（qmin=-127, qmax=127, offset=128）
    - quant_compute_scale(float_list) -> float
    - quantize(float_list, scale, schema) -> bytes
    - dequantize(bytes, scale, schema) -> list[float]
    - matmul(bytes_a, bytes_b, m, k, n, a_scale, b_scale, schema) -> (bytes_c, c_scale)
    - relu(bytes_x, m, n, schema) -> bytes

与 pysgn 互操作：
    import pysgn
    from sgn import hc8_net

    # pysgn.HC8 → sgn.hc8_net
    h = pysgn.HC8(3.14)
    bytes_h = h.to_bytes()  # 6 字节

    # sgn.hc8_net 处理
    bytes_out = hc8_net.relu(bytes_h * 12, 3, 4, hc8_net.default_schema())

    # 还原回 pysgn.HC8
    h_out = pysgn.HC8.from_bytes(bytes_out[:6])
    )";

    /* P2.1: 从环境变量读取 OpenMP 线程数（由 sgn_run.py --auto-cpu 注入）
     *
     * 仅当 sgn_run.py 实际注入了 BLAS 环境变量时（applied_vars 非空），
     * 才会设置 SGN_OMP_THREADS，因此这里读取到即表示需要同步 OpenMP 线程数。
     *
     * 用户显式设置 OMP_NUM_THREADS（--env）时，sgn_run.py 不会注入 SGN_OMP_THREADS，
     * 此时 OpenMP 库直接读取 OMP_NUM_THREADS，本模块不干预。
     */
    {
        const char* omp_threads_str = std::getenv("SGN_OMP_THREADS");
        if (omp_threads_str != NULL) {
            int omp_threads = std::atoi(omp_threads_str);
            if (omp_threads > 0) {
#ifdef _OPENMP
                omp_set_num_threads(omp_threads);
#endif
            }
        }
    }

    /* QuantSchema 类 */
    py::class_<PyQuantSchema>(m, "QuantSchema",
        "HC8 对称量化方案（qmin=-127, qmax=127, offset=128）。\n\n"
        "对称量化：scale = max(|w|) / 127, q = round(w/scale) ∈ [-127, 127]\n"
        "偏移到无符号：q_u = q + 128 ∈ [1, 255]")
        .def(py::init<>(), "构造默认量化方案（qmin=-127, qmax=127, offset=128）。")
        .def(py::init<int32_t, int32_t, int32_t>(),
             py::arg("qmin"), py::arg("qmax"), py::arg("offset"),
             "由 qmin/qmax/offset 构造量化方案。")
        .def_property("qmin", &PyQuantSchema::get_qmin, &PyQuantSchema::set_qmin,
                      "对称量化下界（默认 -127）。")
        .def_property("qmax", &PyQuantSchema::get_qmax, &PyQuantSchema::set_qmax,
                      "对称量化上界（默认 127，避免 -128 不对称）。")
        .def_property("offset", &PyQuantSchema::get_offset, &PyQuantSchema::set_offset,
                      "偏移到无符号（默认 128，使 q_u ∈ [1, 255]）。")
        .def_property("scale", &PyQuantSchema::get_scale, &PyQuantSchema::set_scale,
                      "量化 scale（通常由 quant_compute_scale 推导，不手动设置）。")
        .def("__repr__", &PyQuantSchema::__repr__);

    /* 默认量化方案 */
    m.def("default_schema", []() { return PyQuantSchema(); },
          "返回默认量化方案（qmin=-127, qmax=127, offset=128）。");

    /* 量化 scale 推导 */
    m.def("quant_compute_scale", &py_quant_compute_scale,
          py::arg("values"),
          "从 float 数组推导量化 scale：scale = max(|w|) / 127，全零时返回 1.0。");

    /* 量化 */
    m.def("quantize", &py_quantize,
          py::arg("values"), py::arg("scale"), py::arg("schema"),
          "量化 float list 到 HC8 bytes（每个 HC8 6 字节）。\n\n"
          "每个元素：q = round(w/scale).clamp(-127, 127), q_u = q + 128\n"
          "存入 hc8.v[0]，v[1..5] = 0");

    /* 反量化 */
    m.def("dequantize", &py_dequantize,
          py::arg("bytes"), py::arg("scale"), py::arg("schema"),
          "反量化 HC8 bytes 到 float list。\n\n"
          "每个元素：q_u = hc8.v[0], q = q_u - 128, w = q * scale");

    /* 矩阵乘 */
    m.def("matmul", &py_matmul,
          py::arg("bytes_a"), py::arg("bytes_b"),
          py::arg("m"), py::arg("k"), py::arg("n"),
          py::arg("a_scale"), py::arg("b_scale"),
          py::arg("schema"),
          "HC8 矩阵乘：C = A @ B，返回 (bytes_c, c_scale)。\n\n"
          "整数路径：int8 × int8 → int32 累加 → 反量化 → 重新量化到 HC8\n"
          "输出 scale 基于 C 的 max(|w|) 重新计算，与输入 scale 不同。\n\n"
          "Args:\n"
          "  bytes_a: A 矩阵 HC8 bytes（m×k，行优先，长度 m*k*6）\n"
          "  bytes_b: B 矩阵 HC8 bytes（k×n，行优先，长度 k*n*6）\n"
          "  m, k, n: 矩阵维度\n"
          "  a_scale, b_scale: A 和 B 的量化 scale\n"
          "  schema: 量化方案\n\n"
          "Returns:\n"
          "  (bytes_c, c_scale): C 矩阵 HC8 bytes（m×n）和量化 scale");

    /* ReLU */
    m.def("relu", &py_relu,
          py::arg("bytes_x"), py::arg("m"), py::arg("n"), py::arg("schema"),
          "HC8 整数 ReLU：负值变零。\n\n"
          "HC8 无符号偏移表示：q_u < offset 表示负值，ReLU 后为 q_u = offset。\n\n"
          "Args:\n"
          "  bytes_x: X 矩阵 HC8 bytes（m×n，行优先）\n"
          "  m, n: 矩阵维度\n"
          "  schema: 量化方案（提供 offset）\n\n"
          "Returns:\n"
          "  bytes_out: ReLU 后的 HC8 bytes（m×n）");

    /* 辅助：bytes → numpy uint8 数组 */
    m.def("bytes_to_uint8_array", &py_bytes_to_uint8_array,
          py::arg("bytes"),
          "HC8 bytes → numpy uint8 数组（扁平存储，便于查看内部结构）。");

    /* float64 matmul（反向传播 STE 加速） */
    m.def("float_matmul", &py_float_matmul,
          py::arg("a_flat"), py::arg("b_flat"),
          py::arg("m"), py::arg("k"), py::arg("n"),
          "float64 矩阵乘法（不经过量化，用于反向传播 STE）。\n\n"
          "直接接受 Python list，做 C matmul，返回 Python list。\n"
          "省掉 numpy 的 list→array→list 转换开销。\n\n"
          "Args:\n"
          "  a_flat: m*k 个 float（行优先）\n"
          "  b_flat: k*n 个 float（行优先）\n"
          "  m, k, n: 矩阵维度\n\n"
          "Returns:\n"
          "  c_flat: m*n 个 float（行优先）");

    /* ===== P2.4: 显式 int8 路径 API ===== */

    m.def("hc8_matmul_int", &py_hc8_matmul_int,
          py::arg("x"), py::arg("w"),
          "HC8 整数 matmul（显式 int8 累加路径）。\n\n"
          "接收两个 numpy float32 数组，内部走完整 HC8 量化 + int8×int8→int32 累加 + 反量化。\n"
          "与 Python 端 hc8_matmul（float32 BLAS）数学等价，但走真正的整数累加。\n\n"
          "Args:\n"
          "  x: (m, k) numpy float32，行优先\n"
          "  w: (k, n) numpy float32，行优先\n\n"
          "Returns:\n"
          "  y: (m, n) numpy float32，输出矩阵");

    m.def("hc8_quantize_int", &py_hc8_quantize_int,
          py::arg("x"),
          "HC8 量化（显式 int8 字节路径）。\n\n"
          "返回 HC8 量化字节和 scale，不进行 round-trip 反量化。\n"
          "HC8 字节格式：每个元素 6 字节，v[0]=量化值+128, v[1..5]=0\n\n"
          "Args:\n"
          "  x: numpy float32 数组（任意形状）\n\n"
          "Returns:\n"
          "  (bytes, scale): HC8 字节和量化 scale");

    /* ===== UFP-1 残差量化 API ===== */

    /* 残差量化 */
    m.def("quantize_residual", &py_quantize_residual,
          py::arg("values"), py::arg("depth"), py::arg("schema"),
          "UFP-1 残差量化：float list → (bytes, scales_list)。\n\n"
          "每层存上一层的量化残差，实现 v[0..depth] 的 48-bit 存储精度。\n\n"
          "Args:\n"
          "  values: float list\n"
          "  depth: 残差深度（0~5，0 等价于 quantize）\n"
          "  schema: 量化方案\n\n"
          "Returns:\n"
          "  (bytes, scales): bytes 是 HC8 数组（每元素 6 字节），\n"
          "                   scales 是 float list（长度 depth+1，每层一个 scale）");

    /* 残差反量化 */
    m.def("dequantize_residual", &py_dequantize_residual,
          py::arg("bytes"), py::arg("scales"), py::arg("depth"), py::arg("schema"),
          "UFP-1 残差反量化：bytes + scales_list → float list。\n\n"
          "w_approx = sum_{layer=0..depth} (v[layer] - 128) * scales[layer]\n\n"
          "Args:\n"
          "  bytes: HC8 bytes\n"
          "  scales: float list（长度 depth+1）\n"
          "  depth: 残差深度\n"
          "  schema: 量化方案\n\n"
          "Returns:\n"
          "  float list");

    /* 残差矩阵乘 */
    m.def("matmul_residual", &py_matmul_residual,
          py::arg("bytes_a"), py::arg("bytes_b"),
          py::arg("m"), py::arg("k"), py::arg("n"),
          py::arg("a_depth"), py::arg("b_depth"),
          py::arg("a_scales"), py::arg("b_scales"),
          py::arg("schema"),
          "UFP-1 残差矩阵乘：方案 C（存储高精度，运算 v[0]）。\n\n"
          "流程：反量化到 float → 重新量化到 v[0] → 做 hc8_matmul。\n"
          "输出总是 depth=0（运算精度是 8-bit），UFP-1 只提升存储精度。\n\n"
          "Args:\n"
          "  bytes_a, bytes_b: A、B 矩阵 HC8 bytes\n"
          "  m, k, n: 矩阵维度\n"
          "  a_depth, b_depth: A、B 的残差深度\n"
          "  a_scales, b_scales: A、B 的 scales list\n"
          "  schema: 量化方案\n\n"
          "Returns:\n"
          "  (bytes_c, c_scale): C 矩阵 HC8 bytes（depth=0）和 scale");

    /* 残差 ReLU */
    m.def("relu_residual", &py_relu_residual,
          py::arg("bytes_x"), py::arg("m"), py::arg("n"), py::arg("schema"),
          "UFP-1 残差 ReLU：对 v[0] 做 ReLU，v[1..5] 清零。\n\n"
          "ReLU 是非线性运算，跨层残差失效，输出总是 depth=0。\n\n"
          "Args:\n"
          "  bytes_x: X 矩阵 HC8 bytes（m×n）\n"
          "  m, n: 矩阵维度\n"
          "  schema: 量化方案\n\n"
          "Returns:\n"
          "  bytes_out: ReLU 后的 HC8 bytes（depth=0）");

    /* ===== UFP-2 方案 B API ===== */

    /* 方案 B 残差矩阵乘 */
    m.def("matmul_residual_b", &py_matmul_residual_b,
          py::arg("bytes_a"), py::arg("bytes_b"),
          py::arg("m"), py::arg("k"), py::arg("n"),
          py::arg("a_depth"), py::arg("b_depth"),
          py::arg("a_scales"), py::arg("b_scales"),
          py::arg("schema"),
          "UFP-2 方案 B 残差矩阵乘：整数域累加 + double 合并。\n\n"
          "与方案 C（matmul_residual）的区别：\n"
          "  方案 C：反量化到 float32 → 重新量化到 v[0] → int8 矩阵乘\n"
          "  方案 B：直接用 v[0..depth] 的 int8 值做矩阵乘，用 double 合并\n\n"
          "精度优势：不经过 float32 中转，避免 24 位尾数截断。\n"
          "计算量：(a_depth+1) * (b_depth+1) 次 int8 矩阵乘。\n\n"
          "Args:\n"
          "  bytes_a, bytes_b: A、B 矩阵 HC8 bytes\n"
          "  m, k, n: 矩阵维度\n"
          "  a_depth, b_depth: A、B 的残差深度\n"
          "  a_scales, b_scales: A、B 的 scales list\n"
          "  schema: 量化方案\n\n"
          "Returns:\n"
          "  (bytes_c, c_scale): C 矩阵 HC8 bytes（depth=0）和 scale");

    /* ===== HC4 非对称拆分 API（int8 → int4+int4 运算路径） ===== */

    /* HC8 → HC4 拆分（零成本 memcpy，二进制完全相同） */
    m.def("split_to_hc4", &py_split_to_hc4,
          py::arg("bytes_hc8"), py::arg("n"),
          "HC8 → HC4 拆分（零成本，二进制相同）。\n\n"
          "hc4_t.packed[l] 与 hc8_t.v[l] 在内存中完全一致，\n"
          "拆分/合并是无损双射，可直接 memcpy。\n\n"
          "Args:\n"
          "  bytes_hc8: HC8 bytes（长度 6*n）\n"
          "  n: HC8 元素个数\n\n"
          "Returns:\n"
          "  HC4 bytes（长度 6*n，与输入字节级相同）");

    /* HC4 → HC8 合并（零成本 memcpy，无损双射） */
    m.def("merge_to_hc8", &py_merge_to_hc8,
          py::arg("bytes_hc4"), py::arg("n"),
          "HC4 → HC8 合并（零成本，二进制相同）。\n\n"
          "与 split_to_hc4 互为逆运算，往返无损。\n\n"
          "Args:\n"
          "  bytes_hc4: HC4 bytes（长度 6*n）\n"
          "  n: HC4 元素个数\n\n"
          "Returns:\n"
          "  HC8 bytes（长度 6*n，与输入字节级相同）");

    /* HC4 残差矩阵乘（4 次 int4×int4 + 偏移修正 + double 合并） */
    m.def("matmul_residual_hc4_b", &py_matmul_residual_hc4_b,
          py::arg("bytes_a"), py::arg("bytes_b"),
          py::arg("m"), py::arg("k"), py::arg("n"),
          py::arg("a_depth"), py::arg("b_depth"),
          py::arg("a_scales"), py::arg("b_scales"),
          py::arg("schema"),
          "HC4 残差矩阵乘：4 次 int4×int4 + 偏移修正 + double 合并。\n\n"
          "数学等价于 hc8_residual_matmul_b（int8 路径），但运算用 int4：\n"
          "  a_q = a_h*16 + a_l - 128, b_q = b_h*16 + b_l - 128\n"
          "  a_q*b_q = 256*a_h*b_h + 16*a_h*b_l + 16*a_l*b_h + a_l*b_l\n"
          "           - 2048*a_h - 128*a_l - 2048*b_h - 128*b_l + 16384\n\n"
          "对每对 (l, m_idx) 做 4 次 int4×int4 矩阵乘 + 偏移修正，用 double 合并。\n"
          "输出重新量化到 v[0]，与 hc8_residual_matmul_b 输出格式一致。\n\n"
          "位宽安全：uint4×uint4 最大 225（8 bit），累加 k 次 8+log2(k) bit，\n"
          "MNIST k=784→18bit，CIFAR-10 k=3072→20bit，int32 安全。\n\n"
          "Args:\n"
          "  bytes_a, bytes_b: A、B 矩阵 HC4 bytes（与 HC8 bytes 二进制相同）\n"
          "  m, k, n: 矩阵维度\n"
          "  a_depth, b_depth: A、B 的残差深度\n"
          "  a_scales, b_scales: A、B 的 scales list\n"
          "  schema: 量化方案\n\n"
          "Returns:\n"
          "  (bytes_c, c_scale): C 矩阵 HC8 bytes（depth=0）和 scale");

    /* ===== SIMD 加速 API（v1.4.2-simd） ===== */

    /* HC8 残差矩阵乘 SIMD 版（SoA 布局，AVX-VNNI 加速） */
    m.def("matmul_residual_b_simd", &py_matmul_residual_b_simd,
          py::arg("bytes_a"), py::arg("bytes_b"),
          py::arg("m"), py::arg("k"), py::arg("n"),
          py::arg("a_depth"), py::arg("b_depth"),
          py::arg("a_scales"), py::arg("b_scales"),
          py::arg("schema"),
          "HC8 残差矩阵乘 SIMD 版（SoA 布局 + AVX-VNNI）。\n\n"
          "与 matmul_residual_b 数学等价（max_diff=0），内部用 SoA per-layer 布局。\n"
          "当前实现为标量版（E 阶段验证布局正确性），intrinsics 在 A 阶段添加。\n\n"
          "Args:\n"
          "  bytes_a, bytes_b: A、B 矩阵 HC8 bytes\n"
          "  m, k, n: 矩阵维度\n"
          "  a_depth, b_depth: A、B 的残差深度\n"
          "  a_scales, b_scales: A、B 的 scales list\n"
          "  schema: 量化方案\n\n"
          "Returns:\n"
          "  (bytes_c, c_scale): C 矩阵 HC8 bytes（depth=0）和 scale");

    /* HC4 残差矩阵乘 SIMD 版（SoA per-nibble 布局，AVX2 maddubs 加速） */
    m.def("matmul_residual_hc4_b_simd", &py_matmul_residual_hc4_b_simd,
          py::arg("bytes_a"), py::arg("bytes_b"),
          py::arg("m"), py::arg("k"), py::arg("n"),
          py::arg("a_depth"), py::arg("b_depth"),
          py::arg("a_scales"), py::arg("b_scales"),
          py::arg("schema"),
          "HC4 残差矩阵乘 SIMD 版（SoA per-nibble + AVX2 maddubs）。\n\n"
          "与 matmul_residual_hc4_b 数学等价（max_diff=0），内部用 SoA per-nibble 布局。\n"
          "当前实现为标量版（E 阶段验证布局正确性），intrinsics 在 A 阶段添加。\n\n"
          "Args:\n"
          "  bytes_a, bytes_b: A、B 矩阵 HC4 bytes（与 HC8 bytes 二进制相同）\n"
          "  m, k, n: 矩阵维度\n"
          "  a_depth, b_depth: A、B 的残差深度\n"
          "  a_scales, b_scales: A、B 的 scales list\n"
          "  schema: 量化方案\n\n"
          "Returns:\n"
          "  (bytes_c, c_scale): C 矩阵 HC8 bytes（depth=0）和 scale");

    /* ===== SBE C 化 API（v1.5-sbe，2026-07-22） ===== */

    /* SBE 权重 per-block 量化 + 预处理 */
    m.def("sbe_quantize_weight_blocks", &py_sbe_quantize_weight_blocks,
          py::arg("w"),
          py::arg("groups"), py::arg("k_block"),
          py::arg("k"), py::arg("n"),
          "SBE 权重 per-block 量化 + 预处理（C 化版，AVX-VNNI 友好布局）。\n\n"
          "把 (k, n) 权重矩阵按 k 维度分 groups 块，每块独立量化。\n"
          "每块预处理为 VNNI 友好格式（转置 + 有符号转换 + 预计算 sum_b）。\n\n"
          "数学等价于 Python sbe_conv2d.quantize_weight_sbe，但输出为\n"
          "扁平 numpy 数组（避免 list of tuples 开销）。\n\n"
          "Args:\n"
          "  w:        (k, n) numpy float32，行优先权重矩阵\n"
          "  groups:   分块数\n"
          "  k_block:  每块列数（必须满足 groups * k_block == k）\n"
          "  k, n:     矩阵维度\n\n"
          "Returns:\n"
          "  (w_signed, w_sum_b, w_scales) 三元组：\n"
          "    w_signed: (groups, n, k_block) numpy int8，每块 n×k_block 行优先\n"
          "    w_sum_b:  (groups, n) numpy int32，每块 n 个 sum_b 值\n"
          "    w_scales: (groups,) numpy float32，每块的量化 scale");

    /* SBE 分块 matmul（AVX-VNNI 加速） */
    m.def("sbe_matmul", &py_sbe_matmul,
          py::arg("x"),
          py::arg("w_signed"),
          py::arg("w_sum_b"),
          py::arg("w_scales"),
          py::arg("groups"), py::arg("k_block"),
          py::arg("m"), py::arg("k"), py::arg("n"),
          "SBE 分块 matmul（C 化版，AVX-VNNI 加速）。\n\n"
          "对每个 group g：\n"
          "  1. 提取 x_block: x[:, g*k_block : (g+1)*k_block]\n"
          "  2. per-block 量化 x_block → uint8\n"
          "  3. VNNI matmul: int8 × int8 → int32 累加（_mm256_dpbusd_epi32）\n"
          "  4. float 累加到输出 y\n\n"
          "与 Python sbe_conv2d.sbe_matmul 数学等价，但内部用 VNNI kernel\n"
          "替代 numpy matmul（一次处理 32 个 uint8×int8→8 个 int32）。\n\n"
          "Args:\n"
          "  x:         (m, k) numpy float32，行优先输入矩阵\n"
          "  w_signed:  (groups, n, k_block) numpy int8，由 sbe_quantize_weight_blocks 返回\n"
          "  w_sum_b:   (groups, n) numpy int32，由 sbe_quantize_weight_blocks 返回\n"
          "  w_scales:  (groups,) numpy float32，由 sbe_quantize_weight_blocks 返回\n"
          "  groups:    分块数\n"
          "  k_block:   每块列数\n"
          "  m, k, n:   矩阵形状\n\n"
          "Returns:\n"
          "  y: (m, n) numpy float32，输出矩阵");

    /* ===== SBE Conv2d 前向融合（v2.0.0-conv-fusion） ===== */
    m.def("sbe_conv2d_forward", &py_sbe_conv2d_forward,
          py::arg("x"),
          py::arg("w_signed"),
          py::arg("w_sum_b"),
          py::arg("w_scales"),
          py::arg("B"), py::arg("C_in"), py::arg("H"), py::arg("W"),
          py::arg("C_out"), py::arg("kh"), py::arg("kw"),
          py::arg("stride"), py::arg("padding"),
          py::arg("groups"), py::arg("k_block"),
          py::arg("bias") = py::none(),
          "SBE Conv2d 前向融合（v2.0.0，im2col + SBE matmul + bias 在 C 层完成）。\n\n"
          "数学等价于：\n"
          "  1. im2col(x) → x_col_2d (B*L, K)\n"
          "  2. sbe_matmul(x_col_2d, w) → y_col_2d\n"
          "  3. reshape + bias → y (B, C_out, H_out, W_out)\n\n"
          "但全部在 C 层完成，消除 Python 层 im2col + transpose + reshape 开销。\n\n"
          "Args:\n"
          "  x:         (B, C_in, H, W) numpy float32\n"
          "  w_signed:  (groups, C_out, k_block) numpy int8\n"
          "  w_sum_b:   (groups, C_out) numpy int32\n"
          "  w_scales:  (groups,) numpy float32\n"
          "  B, C_in, H, W: 输入尺寸\n"
          "  C_out, kh, kw, stride, padding: 卷积参数\n"
          "  groups, k_block: SBE 分块参数\n"
          "  bias: (C_out,) numpy float32 或 None\n\n"
          "Returns:\n"
          "  y: (B, C_out, H_out, W_out) numpy float32");

    /* ===== SBE + Smoothing C 分块 matmul（v1.7.0） ===== */
    m.def("sbe_matmul_smoothed", &py_sbe_matmul_smoothed,
          py::arg("x"),
          py::arg("w_signed"),
          py::arg("w_sum_b"),
          py::arg("w_scales"),
          py::arg("groups"), py::arg("k_block"),
          py::arg("m"), py::arg("k"), py::arg("n"),
          "SBE + Smoothing C 分块 matmul（v1.7.0，Smoothing 融合到 C 扩展）。\n\n"
          "与 sbe_matmul 接口完全一致，但内部对每个 group 的 x_block 做 per-row mean shift：\n"
          "  1. c_mean[i] = mean(x_block[i, :])\n"
          "  2. x_shifted = x_block - c_mean\n"
          "  3. per-block 量化 x_shifted → uint8（动态范围更小，量化更精确）\n"
          "  4. VNNI matmul: int8 × int8 → int32 累加\n"
          "  5. float 累加主项: y += c_int32 * x_scale * w_scale\n"
          "  6. 修正项累加: y += c_mean * (w_sum_b * w_scale)\n\n"
          "数学等价性：y = x @ w = (x - c_mean) @ w + c_mean @ w\n"
          "Smoothing 只影响量化精度，不改变数学等价性。\n\n"
          "Args 与 sbe_matmul 完全一致。\n"
          "Returns:\n"
          "  y: (m, n) numpy float32，输出矩阵");

    /* ===== SBE per-channel 量化 + matmul（v1.8.0-perchannel） ===== */
    m.def("sbe_quantize_weight_blocks_perchannel", &py_sbe_quantize_weight_blocks_perchannel,
          py::arg("w"),
          py::arg("groups"), py::arg("k_block"),
          py::arg("k"), py::arg("n"),
          "SBE 权重 per-channel 量化 + 预处理（v1.8.0-perchannel）。\n\n"
          "与 sbe_quantize_weight_blocks 的区别：\n"
          "  w_scales 输出形状从 (groups,) 改为 (groups, n)，\n"
          "  每个 block 内每个输出通道独立 scale，量化精度更高。\n\n"
          "w_signed 和 w_sum_b 格式与 per-block 版本相同（复用 VNNI kernel）。\n\n"
          "Args:\n"
          "  w:        (k, n) numpy float32，行优先权重矩阵\n"
          "  groups:   分块数\n"
          "  k_block:  每块列数（必须满足 groups * k_block == k）\n"
          "  k, n:     矩阵维度\n\n"
          "Returns:\n"
          "  (w_signed, w_sum_b, w_scales) 三元组：\n"
          "    w_signed: (groups, n, k_block) numpy int8（与 per-block 相同）\n"
          "    w_sum_b:  (groups, n) numpy int32（与 per-block 相同）\n"
          "    w_scales: (groups, n) numpy float32（per-channel，与 per-block 不同）");

    m.def("sbe_matmul_perchannel", &py_sbe_matmul_perchannel,
          py::arg("x"),
          py::arg("w_signed"),
          py::arg("w_sum_b"),
          py::arg("w_scales"),
          py::arg("groups"), py::arg("k_block"),
          py::arg("m"), py::arg("k"), py::arg("n"),
          "SBE per-channel 分块 matmul（v1.8.0-perchannel）。\n\n"
          "与 sbe_matmul 的区别：\n"
          "  w_scales 是 (groups, n) 而非 (groups,)，\n"
          "  float 累加用 per-channel scale 向量，而非标量广播。\n\n"
          "VNNI kernel 不变（int8×int8→int32 累加与 scale 无关）。\n\n"
          "Args:\n"
          "  x:         (m, k) numpy float32，行优先输入矩阵\n"
          "  w_signed:  (groups, n, k_block) numpy int8\n"
          "  w_sum_b:   (groups, n) numpy int32\n"
          "  w_scales:  (groups, n) numpy float32（per-channel）\n"
          "  groups:    分块数\n"
          "  k_block:   每块列数\n"
          "  m, k, n:   矩阵形状\n\n"
          "Returns:\n"
          "  y: (m, n) numpy float32，输出矩阵");

    /* ===== Triple-int8 缩放 C 化（v1.9.0，AVX2 + OpenMP） ===== */
    m.def("sbe_rescale_to_triple", &py_sbe_rescale_to_triple,
          py::arg("x"),
          "Triple-int8 缩放（C 化版，AVX2 + OpenMP 加速）。\n\n"
          "将 float 输入分解为 3 个 int8 分量（24-bit 精度）：\n"
          "  scale = max(|x|) / 127\n"
          "  C_high = round(x / scale)          ∈ [-127, 127]\n"
          "  C_mid  = round(ε1 * 256)           ∈ [-128, 128]\n"
          "  C_low  = round(ε2 * 256)           ∈ [-128, 128]\n\n"
          "数学等价于 sbe_conv2d.rescale_to_triple_int8_sbe，但用 AVX2 一次 8 个\n"
          "float + OpenMP 并行，替代单线程 numpy 元素级操作（30x+ 加速）。\n\n"
          "Args:\n"
          "  x: numpy float32 数组（任意形状，内部按扁平处理）\n\n"
          "Returns:\n"
          "  (C_high, C_mid, C_low, scale):\n"
          "    C_high, C_mid, C_low: 与 x 同形状的 float32 数组\n"
          "    scale: float 标量");

    /* ===== 正交多视角 matmul（HC 树并行解读，v1.6.0） ===== */
    m.def("multiview_matmul", &py_hc8_multiview_matmul,
          py::arg("c_acc"),
          py::arg("b_int8"),
          py::arg("m"), py::arg("k"), py::arg("n"),
          py::arg("n_views"),
          "正交多视角 matmul（HC 树并行解读，方案 C C 化版）。\n\n"
          "分解累加值 C_acc 为 n_views 个 uint8 位段视角，权重保持 int8：\n"
          "  C_next = Σ_v (c_v @ b_int8) * 2^(8v)\n"
          "  其中 c_v = (C_acc >> 8v) & 0xFF\n\n"
          "与 Python hc_tree_unified.matmul_multiview 数学等价（max_diff=0）。\n\n"
          "约束：4 视角要求 C_acc >= 0（ReLU 后），8 视角才能处理有符号 int64。\n\n"
          "Args:\n"
          "  c_acc:    (m, k) numpy int64，行优先累加值（ReLU 后 >= 0）\n"
          "  b_int8:   (k, n) numpy int8，行优先权重（有符号 [-127, 127]）\n"
          "  m, k, n:  矩阵维度\n"
          "  n_views:  视角数（1-8，4 覆盖 int32，8 覆盖 int64）\n\n"
          "Returns:\n"
          "  out: (m, n) numpy int64，行优先累加值（未乘 scale）");

    /* 版本信息（合并到 sgn 模块后的标识） */
    m.attr("__version__") = "merged-hc8-net";
    m.attr("HC8_BYTES")   = (int)sizeof(hc8_t);  /* 6 */
    m.attr("HC8_LAYERS")  = 6;  /* HC8 有 6 层 v[0..5] */

    /* ===== OpenMP 线程控制（v1.9.0，避免与 torch/dataloader 过度订阅） ===== */
    m.def("set_omp_threads", [](int n) {
#ifdef _OPENMP
        if (n > 0) omp_set_num_threads(n);
#endif
    }, py::arg("n"),
       "设置 OpenMP 线程数（运行时生效，避免与 torch 线程/dataloader 过度订阅）。\n\n"
       "OpenMP 线程数的三种设置方式（优先级从高到低）：\n"
       "  1. Python 显式调用：`pysgn_net.set_omp_threads(N)`（运行时立即生效）\n"
       "  2. 环境变量 `SGN_OMP_THREADS`：模块导入时读取，由 sgn_run.py --auto-cpu\n"
       "     自动注入（仅当未用 --env 显式设置 BLAS 线程数时）\n"
       "  3. 默认值：OpenMP 库默认（通常 = CPU 核数，可能与其他线程池过度订阅）\n\n"
       "推荐场景：\n"
       "  - sgn_run.py --auto-cpu：自动设置方式 2，与 BLAS 线程数同步\n"
       "  - 训练脚本中 DataLoader num_workers > 0：用方式 1 动态调整\n"
       "  - 调试：用方式 1 强制单线程 `set_omp_threads(1)`");

    m.def("get_omp_threads", []() -> int {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }, "返回当前 OpenMP 最大线程数");

    /* ===== AVX-VNNI 运行时检测（v1.5.1） ===== */
    m.def("has_avx_vnni", []() {
        return hc_detect_avx_vnni() != 0;
    }, "运行时 CPUID 检测 CPU 是否支持 AVX-VNNI（VPDPBUSD 指令）");

    m.def("cpuid_7_1_raw", []() {
        int eax, ebx, ecx, edx;
        hc_cpuid_7_1_raw(&eax, &ebx, &ecx, &edx);
        return py::make_tuple(eax, ebx, ecx, edx);
    }, "调试用：返回 CPUID leaf 7 sub-leaf 1 的原始 (EAX, EBX, ECX, EDX)");
    m.attr("UFP1_MAX_DEPTH") = 5;  /* UFP-1 最大残差深度 */
    m.attr("UFP2_SCHEME") = "B";  /* UFP-2 当前方案 */
    /* HC4 非对称拆分属性 */
    m.attr("HC4_BYTES")      = (int)sizeof(hc4_t);  /* 6（与 HC8 二进制相同） */
    m.attr("HC4_BITS")       = 4;   /* int4 拆分位宽 */
    m.attr("HC4_NIBBLES")    = 12;  /* 6 字节 × 2 = 12 个 int4 */
    m.attr("HC4_ASYM_SPLIT") = "int8 -> int4_high + int4_low";
}
