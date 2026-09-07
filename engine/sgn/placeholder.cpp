#include <pybind11/pybind11.h>
#include <immintrin.h>
#include <string>

namespace py = pybind11;

// Level 调度器注册函数（定义于 level/level_bindings.cpp）
void register_level(py::module_& m);

// HC 扩展注册函数（定义于 hc/col2im_bindings.cpp，由本模块统一调度）
void register_col2im(py::module_& m);

// Common 扩展注册函数（定义于 common/value_spec_bindings.cpp）
void register_value_spec(py::module_& m);

// Common 扩展注册函数（定义于 common/unit_value_bindings.cpp）
void register_unit_value(py::module_& m);

// Common 扩展注册函数（定义于 common/precision_budget_bindings.cpp）
void register_precision_budget(py::module_& m);

// MSint 扩展注册函数（定义于 msint/packed_backend_bindings.cpp）
void register_packed_backend(py::module_& m);

// MSint 扩展注册函数（定义于 msint/msint_view_bindings.cpp）
void register_msint_view(py::module_& m);

// MSint 扩展注册函数（定义于 msint/split_dot_bindings.cpp）
void register_split_dot(py::module_& m);

// MSint 扩展注册函数（定义于 msint/multiscale_view_bindings.cpp）
void register_multiscale_view(py::module_& m);

// MSint 扩展注册函数（定义于 msint/precision_selector_bindings.cpp）
void register_precision_selector(py::module_& m);

// MSint 扩展注册函数（定义于 msint/leveled_split_dot_bindings.cpp）
void register_leveled_split_dot(py::module_& m);

// mkern/nested 嵌套量化原语注册（定义于 mkern/nested/nested_bindings.cpp；
// sgn.mkern_nested 子模块，设计档 §二.4 level 消费方接线）
void register_nested(py::module_& m);

// mkern/simd 整型点积原语注册（定义于 mkern/simd/simd_bindings.cpp；
// sgn.mkern_simd 子模块，层 2 LeveledState 状态消费计算域入口）
void register_simd(py::module_& m);

// HC8 余积存储注册函数（定义于 hc/hc8_coproduct_bindings.cpp）
void register_hc8_coproduct(py::module_& m);

// HC(B,L) 分层编解码通道注册函数（定义于 hc/hc_decode_bindings.cpp，R3-B 落地）
void register_hc_decode(py::module_& m);

// Autograd 框架注册函数（定义于 autograd/autograd_bindings.cpp，Stage 3.2 Phase 4）
void register_autograd(py::module_& m);

// 双模式系统：Module 基类注册函数（定义于 autograd/module_bindings.cpp）
void register_nn(py::module_& m);

// HC 扩展注册函数（定义于 engine/sgn/hc/ext/pysgn_*.cpp）
// 安全审计 2026-08-16 B1-2：原注释引用 内部档案
// 该目录已迁移，实际源码位于 engine/sgn/hc/ext/。
// 这 4 个 .cpp 原为独立 pybind11 模块（pysgn_net / pysgn_hc16 / pysgn_hc16ms / pysgn_hc4_pshufb），
// 现合并到 sgn 模块的子模块下（hc8_net / hc16 / hc16ms / hc4），由本统一入口调度。
void register_hc8_net(py::module_& m);
void register_hc16(py::module_& m);
void register_hc16ms(py::module_& m);
void register_hc4_pshufb(py::module_& m);

// col2im C 实现注册函数（定义于 hc/ext/pysgn_col2im.cpp）
// 原独立 pysgn_col2im 模块，现合并到 sgn.col2im_c 子模块
void register_col2im_c(py::module_& m);

int add(int a, int b) { return a + b; }

std::string version() { return "Stage 3.0 placeholder"; }

std::string compiler_info() {
#if defined(__clang__)
    return "Clang " + std::to_string(__clang_major__) + "."
           + std::to_string(__clang_minor__) + "."
           + std::to_string(__clang_patchlevel__);
#elif defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_VER);
#elif defined(__GNUC__)
    return "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#else
    return "unknown";
#endif
}

int test_avx_vnni() {
    __m256i a = _mm256_set1_epi8(1);
    __m256i b = _mm256_set1_epi8(2);
    __m256i c = _mm256_setzero_si256();
    __m256i r = _mm256_dpbusd_epi32(c, a, b);
    return _mm256_extract_epi32(r, 0);
}

PYBIND11_MODULE(sgn, m) {
    m.doc() = "SGN Stage 3.0 unified C++ abstraction layer";
    m.def("version", &version, "Return module version");
    m.def("compiler_info", &compiler_info, "Return compiler info string (e.g. 'Clang 22.1.8')");
    m.def("add", &add, "Add two integers");
    m.def("test_avx_vnni", &test_avx_vnni, "Test AVX-VNNI intrinsic availability");

    // Level 调度器注册（sgn.level 子模块）
    register_level(m);

    // HC 扩展注册（col2im 等）
    register_col2im(m);

    // Common 扩展注册（ValueSpec / UnitValue / PrecisionBudget 等）
    register_value_spec(m);
    register_unit_value(m);
    register_precision_budget(m);

    // MSint 扩展注册（PackedBackend / MSIntView / SplitDot / MultiScaleView / PrecisionSelector / LeveledSplitDot）
    register_packed_backend(m);
    register_msint_view(m);
    register_split_dot(m);
    register_multiscale_view(m);
    register_precision_selector(m);
    register_leveled_split_dot(m);

    // mkern/nested 嵌套量化原语（sgn.mkern_nested 子模块，nested_quant 设计档 §二.4）
    register_nested(m);

    // mkern/simd 整型点积原语（sgn.mkern_simd 子模块，层 2 状态消费计算域入口）
    register_simd(m);

    // HC8 余积存储注册（HC8Coproduct / HC8CoproductArray / HC8KalmanFilter）
    register_hc8_coproduct(m);

    // HC(B,L) 分层编解码通道（sgn.hc_decode 子模块，R3-B 宽精度解码通道落地）
    register_hc_decode(m);

    // Autograd 框架注册（Stage 3.2 Phase 4：Tensor/Tape/算子，sgn.autograd 子模块）
    register_autograd(m);

    // 双模式系统：Module 基类注册（sgn.nn 子模块）
    register_nn(m);

    // HC 扩展注册（hc8_net / hc16 / hc16ms / hc4_pshufb 子模块）
    // 这 4 个子模块由原独立 pysgn_*.cpp 合并而来，函数名保持不变（子模块隔离命名空间）。
    auto hc8_net = m.def_submodule("hc8_net", "HC8 神经网络运算扩展");
    register_hc8_net(hc8_net);

    auto hc16 = m.def_submodule("hc16", "HC16 神经网络运算扩展");
    register_hc16(hc16);

    auto hc16ms = m.def_submodule("hc16ms", "HC16MS 多精度存储扩展");
    register_hc16ms(hc16ms);

    auto hc4 = m.def_submodule("hc4", "HC4 PSHUFB 扩展");
    register_hc4_pshufb(hc4);

    // col2im C 实现子模块（原独立 pysgn_col2im 模块）
    auto col2im_c = m.def_submodule("col2im_c", "col2im C implementation (original C extension)");
    register_col2im_c(col2im_c);
}
