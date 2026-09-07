// simd_bindings.cpp - pybind11 绑定（注册到 sgn.mkern_simd 子模块）
//
// 层 2 接线（LeveledState 状态消费）的计算域入口。暴露的 Python API：
//   - sgn.mkern_simd.dot8(a_u8, b_s8) → int
//       uint8[K] × int8[K] → int64 精确点积（kBitExact，全 K——2^18 分块 +
//       块间 int64 防护，见 内部档案）。
//       a 侧 u8 是仿射编码载体：q_u8 = (code >> 24) + 128（Q8 floor 平面），
//       零点修正 dot8 − 128·Σw == Σ q_floor·w（层 2 设计 §三）。
//   - sgn.mkern_simd.dot4(a_u8, b_s8) → int
//       与 dot8 同载体（预解包 nibble 路径；a ∈ [0,255]）。
//   - sgn.mkern_simd.dot4_packed(a_packed_u8, b_packed_s8) → int
//       4×4 打包点积（两侧均 nibble：a 无符号 [0,15]、b 有符号 [−8,7]，
//       byte[j] = elem[2j+1]<<4 | elem[2j]；K = 2×len 隐式）。
//       注意：Q4 状态 × int8 权重的混合 4×8 不走此原语——走 unpack_nibble_u
//       解包（得到 state_s4+8 ∈ [0,15]）→ dot8 → 零点 8 修正（层 2 设计 §五.2）。
//   - sgn.mkern_simd.unpack_nibble_u(packed_u8) → list[int]
//   - sgn.mkern_simd.unpack_nibble_s(packed_u8) → list[int]
//   - sgn.mkern_simd.active_backend() → str
//
// 性能注：list 转换与 split_dot/nested 绑定同口径（项目惯例）；热路径 numpy
// 零拷贝视层 2 系统账 profile 再议。
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 zhugy-8086

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>
#include <vector>

#include "mkern/simd/simd_api.h"

namespace py = pybind11;

void register_simd(py::module_& m) {
    auto simd = m.def_submodule(
        "mkern_simd",
        "mkern SIMD 整型点积原语（层 2 状态消费计算域入口；bit-exact 全 K）");

    simd.def(
        "dot8",
        [](const std::vector<uint8_t>& a, const std::vector<int8_t>& b) {
            if (a.size() != b.size()) {
                throw py::value_error("dot8: len(a)=" +
                                      std::to_string(a.size()) + " != len(b)=" +
                                      std::to_string(b.size()));
            }
            return sgn::simd::dot8(a.data(), b.data(),
                                   static_cast<size_t>(a.size()));
        },
        py::arg("a_u8"), py::arg("b_s8"),
        "uint8[K] × int8[K] → int64 精确点积（kBitExact，全 K）。\n"
        "a 侧 u8 是仿射载体：Q8 floor 平面 q_u8 = (code >> 24) + 128，\n"
        "零点修正 dot8(q_u8, w) − 128·Σw == Σ q_floor·w（层 2 设计 §三）。");

    simd.def(
        "dot4",
        [](const std::vector<uint8_t>& a, const std::vector<int8_t>& b) {
            if (a.size() != b.size()) {
                throw py::value_error("dot4: len(a)=" +
                                      std::to_string(a.size()) + " != len(b)=" +
                                      std::to_string(b.size()));
            }
            return sgn::simd::dot4(a.data(), b.data(),
                                   static_cast<size_t>(a.size()));
        },
        py::arg("a_u8"), py::arg("b_s8"),
        "4 位预解包点积（与 dot8 同载体；输入须已由 unpack_nibble_u/s 预解包为\n"
        "满宽字节）。uint8[K] × int8[K] → int64（kBitExact，全 K）。");

    simd.def(
        "dot4_packed",
        [](const std::vector<uint8_t>& a_packed,
           const std::vector<int8_t>& b_packed) {
            if (a_packed.size() != b_packed.size()) {
                throw py::value_error(
                    "dot4_packed: len(a_packed)=" +
                    std::to_string(a_packed.size()) + " != len(b_packed)=" +
                    std::to_string(b_packed.size()));
            }
            const size_t k = a_packed.size() * 2;   // 每字节 2 元素
            return sgn::simd::dot4_packed(a_packed.data(), b_packed.data(), k);
        },
        py::arg("a_packed_u8"), py::arg("b_packed_s8"),
        "4×4 打包点积：a 无符号 nibble [0,15]、b 有符号 nibble [−8,7]，\n"
        "byte[j] = elem[2j+1]<<4 | elem[2j]；K = 2×len 隐式，返回 int64 精确。\n"
        "混 4×8（Q4 状态 × int8 权重）不走此原语——unpack_nibble_u + dot8 +\n"
        "零点 8 修正（层 2 设计 §五.2）。");

    simd.def(
        "unpack_nibble_u",
        [](const std::vector<uint8_t>& packed) {
            std::vector<uint8_t> out(packed.size() * 2, 0);
            sgn::simd::unpack_nibble_u(packed.data(), out.size(), out.data());
            return out;
        },
        py::arg("packed_u8"),
        "无符号 nibble 解包：byte[j] = elem[2j+1]<<4 | elem[2j] → u8[2K]。\n"
        "层 2 用法：Q4 状态打包 nibble（two's complement）解包结果 =\n"
        "state_s4 + 8 ∈ [0,15]（仿射形式，零点 8 由点积侧修正）。");

    simd.def(
        "unpack_nibble_s",
        [](const std::vector<uint8_t>& packed) {
            std::vector<int8_t> out(packed.size() * 2, 0);
            sgn::simd::unpack_nibble_s(packed.data(), out.size(), out.data());
            return out;
        },
        py::arg("packed_u8"),
        "有符号 nibble 解包：0..7→0..7，8..15→−8..−1（u8[2K] → int8[2K]）。");

    simd.def(
        "active_backend",
        []() { return std::string(sgn::simd::active_backend_name()); },
        "当前调度后端：'avx512vnni' / 'avxvnni' / 'avx2' / 'ssse3' / 'scalar'\n"
        "（SGN_KERNEL_BACKEND=scalar 可强制标量，测试钩子）。");
}
