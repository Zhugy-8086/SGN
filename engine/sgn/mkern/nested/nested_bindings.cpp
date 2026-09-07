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
// 性能注：list 转换与 split_dot 绑定同口径（项目惯例）；热路径 numpy 零拷贝
// 视 §五.5 接线后的实际 profile 再议。

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "mkern/nested/nested_api.h"

namespace py = pybind11;

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
