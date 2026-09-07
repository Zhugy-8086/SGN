// nested_bindings.cpp - pybind11 绑定（注册到 sgn.mkern_nested 子模块）
//
// 暴露的 Python API（设计档 §二.4：level 消费方接线，调度器选层 = 选 dequant 档位）：
//   - sgn.mkern_nested.nested_quant_i32(h, u, seed) → list[int]
//       一次 SR 量化到最细格（floor(h/u) + Bernoulli[U<frac]），嵌套 int32 码字。
//       max_range 全档一致 ±2^31·u（域外饱和）；h=0 吸收态（码字恒 0）。
//   - sgn.mkern_nested.nested_dequant(code, u, level) → list[float]
//       level 档 dequant = RTN 截位（half-to-even），out ∈ u·2^(32−level)·ℤ 且为
//       code·u 的最近粗格点；level ∈ {4,8,16,32}（非法值抛 ValueError——比 C++
//       契约的"无操作"更严格，Python 侧显式报错）。
//   - sgn.mkern_nested.active_backend() → str（"avx512"/"avx2"/"scalar(forced)"）
//   - sgn.mkern_nested.LEVELS = (4, 8, 16, 32)
//
// bit-exact 规格（RNG/精度/饱和/tie）见 mkern/nested/nested_api.h 头注释；跨后端
// 与跨编译器逐位一致由 nested_boundary_test 实证。Python 侧 RNG 流不经过
// sr_kernel（SplitMix64 计数器式，自包含）。
//
// 性能注：list 转换与 split_dot 绑定同口径（项目惯例）；numpy 零拷贝热路径
// **已落地**（2026-09-07 层 2 Phase 2a 遗留项收口）：nested_quant_i32_np /
// nested_dequant_np —— 入侧 1-D C-contiguous + 精确 dtype 严格校验（不符
// ValueError 指引走 list 版，不静默转换/拷贝；裸 py::array_t caster 实证会
// 静默宽容，热路径要显式报错），出侧 numpy 数组内核直写。与 list 版同内核
// 逐位一致（tests/test_leveled_state.py 钉死）。

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "mkern/nested/nested_api.h"

namespace py = pybind11;

namespace {

// 1-D numpy 零拷贝严格契约（与 simd_bindings.cpp 的 np1d_ptr 同纪律同实现，
// 各文件自包含不新增头文件）。校验纳秒级。
template <typename T>
const T* np1d_ptr(py::handle src, py::ssize_t* n, const char* who) {
    if (!py::isinstance<py::array>(src)) {
        throw py::value_error(std::string(who) +
                              ": 须 1-D C-contiguous numpy 数组（精确 dtype），"
                              "Python list 请走无后缀版");
    }
    py::array a = py::reinterpret_borrow<py::array>(src);
    if (a.ndim() != 1 || !a.dtype().is(py::dtype::of<T>()) ||
        !(a.flags() & py::array::c_style)) {
        throw py::value_error(
            std::string(who) +
            ": 零拷贝契约要求 1-D C-contiguous + 精确 dtype（" +
            py::str(py::dtype::of<T>()).cast<std::string>() +
            "）；非连续切片/错 dtype 请先 np.ascontiguousarray(...).astype(...) "
            "或走无后缀 list 版");
    }
    *n = a.size();
    return static_cast<const T*>(a.data());
}

// 出侧 numpy 数组构造：分配 + memset 保险（内核写全量契约）+ 返回。
template <typename T>
py::array_t<T> np1d_out(py::ssize_t n) {
    py::array_t<T> out(n);
    std::memset(out.mutable_data(), 0, sizeof(T) * static_cast<size_t>(n));
    return out;
}

}  // namespace

void register_nested(py::module_& m) {
    auto nested = m.def_submodule(
        "mkern_nested",
        "mkern 嵌套量化原语（NestQuant 衔接）：一次 SR 量化，多精度档位即取");

    nested.attr("LEVELS") = py::make_tuple(4, 8, 16, 32);

    nested.def(
        "nested_quant_i32",
        [](const std::vector<float>& h, float u, uint64_t seed) {
            std::vector<int64_t> code(h.size(), 0);
            sgn::mkern::nested::nested_quant_i32(code.data(), h.data(),
                                                 static_cast<int64_t>(h.size()),
                                                 u, seed);
            return code;
        },
        py::arg("h"), py::arg("u"), py::arg("seed"),
        "一次 SR 量化到最细格 u，产出嵌套 int32 码字（list[int]）。\n"
        "规格：x = h/u（f64 除法；u 按签名以 float32 传入，双精度调用值先按 IEEE\n"
        "规则舍入到 float32 再参与除法——Python 侧参考实现须用 np.float32(u)），\n"
        "I = floor(x) + [U_i < frac]（SplitMix64 计数器式 RNG，per-element 恰 1 随机\n"
        "数）；x ≥ 2^31−1 饱和到 +2^31−1，x ≤ −2^31 饱和到 −2^31（max_range 全档\n"
        "一致 ±2^31·u）。h=0 → 码字 0（吸收态）。\n"
        "前置：h 有限、u > 0。同 seed 同输入跨后端/跨编译器/跨语言逐位一致\n"
        "（kBitExact；Python 对拍参考见 nested_quant_scalar锚点执行记录 §十）。");

    nested.def(
        "nested_dequant",
        [](const std::vector<int64_t>& code, float u, int level) {
            if (level != 4 && level != 8 && level != 16 && level != 32) {
                throw py::value_error(
                    "level must be one of {4, 8, 16, 32}, got " +
                    std::to_string(level));
            }
            std::vector<float> out(code.size(), 0.f);
            sgn::mkern::nested::nested_dequant(out.data(), code.data(),
                                               static_cast<int64_t>(code.size()),
                                               u, level);
            return out;
        },
        py::arg("code"), py::arg("u"), py::arg("level"),
        "level 档 dequant = 对码字 RTN 截位（half-to-even），输出 list[float]。\n"
        "out ∈ u·2^(32−level)·ℤ 且为 code·u 的最近粗格点（格嵌套不变量）；\n"
        "level=32 恒等（out = code·u）。对任意 int64 码字良定义（含 int32 域外）。");

    // ---- numpy 零拷贝版（热路径；与 list 版同内核逐位一致）----

    nested.def(
        "nested_quant_i32_np",
        [](py::handle h, float u, uint64_t seed) {
            py::ssize_t n = 0;
            const float* hp = np1d_ptr<float>(h, &n, "nested_quant_i32_np: h");
            py::array_t<int64_t> out = np1d_out<int64_t>(n);
            sgn::mkern::nested::nested_quant_i32(
                static_cast<int64_t*>(out.mutable_data()), hp,
                static_cast<int64_t>(n), u, seed);
            return out;
        },
        py::arg("h"), py::arg("u"), py::arg("seed"),
        "nested_quant_i32 的 numpy 零拷贝版：float32[N]（C-contiguous）入侧\n"
        "指针直读，出侧 int64[N] numpy 数组内核直写。**u 保持 C++ float、seed\n"
        "保持 uint64_t**（f32 IEEE 舍入口径与同 step 各样本共用 seed 是冻结\n"
        "规格，与 list 版逐字一致——RNG 流 SplitMix64(seed+φ·i) 按缓冲下标计数）。\n"
        "量化语义同 list 版（floor + Bernoulli SR、±2^31 饱和、h=0 吸收态）。\n"
        "与 list 版调同一 C++ 内核，逐位一致。");

    nested.def(
        "nested_dequant_np",
        [](py::handle code, float u, int level) {
            if (level != 4 && level != 8 && level != 16 && level != 32) {
                throw py::value_error(
                    "level must be one of {4, 8, 16, 32}, got " +
                    std::to_string(level));
            }
            py::ssize_t n = 0;
            const int64_t* cp =
                np1d_ptr<int64_t>(code, &n, "nested_dequant_np: code");
            py::array_t<float> out = np1d_out<float>(n);
            sgn::mkern::nested::nested_dequant(
                static_cast<float*>(out.mutable_data()), cp,
                static_cast<int64_t>(n), u, level);
            return out;
        },
        py::arg("code"), py::arg("u"), py::arg("level"),
        "nested_dequant 的 numpy 零拷贝版：int64[N] 码字入侧指针直读，出侧\n"
        "float32[N] numpy 数组内核直写（f32→f64 拓宽精确，消费侧语义同 list 版\n"
        "的 Python float）。level 校验 ValueError 原样保留。逐位一致。");

    nested.def(
        "nested_view_codes",
        [](const std::vector<int64_t>& code, int level) {
            if (level != 4 && level != 8 && level != 16 && level != 32) {
                throw py::value_error(
                    "level must be one of {4, 8, 16, 32}, got " +
                    std::to_string(level));
            }
            std::vector<int64_t> out(code.size(), 0);
            sgn::mkern::nested::nested_view_codes(
                code.data(), static_cast<int64_t>(code.size()), level,
                out.data());
            return out;
        },
        py::arg("code"), py::arg("level"),
        "level 档视图的整数码（b bit 有符号：RTN 商，list[int]）。\n"
        "out = RTN(code, 2^(32−level)) >> (32−level)，与 nested_dequant 共享\n"
        "同一 RTN 定义点（dequant 值 = out·2^(32−level)·u）。\n"
        "用途：把档位视图喂给 MSint 点积域——Q8 → int8/dot8、Q16 → int16/\n"
        "pair 载体（S3 衔接评估 B4a：嵌套 Q16 视图码过 U 方案恒等式）、\n"
        "Q4 → int4/dot4。标量即终态（整数 shift+cast，无热路径）。");

    nested.def(
        "active_backend",
        []() { return std::string(sgn::mkern::nested::active_nested_backend_name()); },
        "当前调度后端：'avx512' / 'avx2' / 'scalar(forced)'（SGN_NESTED_BACKEND=scalar\n"
        "可强制标量，测试钩子）。后端链 scalar → avx2 → avx512(F+DQ)，CPUID 自动选。");
}
