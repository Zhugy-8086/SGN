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
// numpy 零拷贝版（2026-09-07 层 2 Phase 2a 遗留项收口，后缀 _np）：
//   - dot8_np / dot4_np / dot4_packed_np / unpack_nibble_u_np / unpack_nibble_s_np
//     输入 1-D C-contiguous 且 dtype 精确匹配的 numpy 数组 → 直接读内存指针，
//     消除 pybind 逐元素 list/vector 转换（热路径主头）；**显式校验**（非
//     forcecast）：dtype/连续性/维数不符即 ValueError 指引走无后缀 list 版。
//     裸 py::array_t<T,c_style> caster 会静默转换/拷贝（pybind11 numpy.h 实证），
//     热路径要的是显式报错而非静默宽容——有意偏离 prepare_downcast_np 的
//     forcecast 先例（其离线场景静默拷贝无害）。
//     与 list 版调同一 C++ 内核，逐位一致（tests/test_leveled_state.py 钉死）。
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 zhugy-8086

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "mkern/simd/simd_api.h"

namespace py = pybind11;

namespace {

// 1-D numpy 零拷贝严格契约：ndim==1 + dtype 精确匹配 + C-contiguous，
// 否则 ValueError（不静默转换/拷贝——热路径显式报错纪律）。返回数据指针。
// 校验开销纳秒级，相对省下的逐元素转换可忽略。
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

// 出侧 numpy 数组构造：分配 + memset 保险（内核写全量契约，见 nested_api.h
// 头注）+ 返回——消除中间 vector 与 vector→list→np.asarray 两跳出侧转换。
template <typename T>
py::array_t<T> np1d_out(py::ssize_t n) {
    py::array_t<T> out(n);
    std::memset(out.mutable_data(), 0, sizeof(T) * static_cast<size_t>(n));
    return out;
}

}  // namespace

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

    // ---- numpy 零拷贝版（热路径；与 list 版同内核逐位一致）----

    simd.def(
        "dot8_np",
        [](py::handle a, py::handle b) {
            py::ssize_t na = 0, nb = 0;
            const uint8_t* ap = np1d_ptr<uint8_t>(a, &na, "dot8_np: a_u8");
            const int8_t* bp = np1d_ptr<int8_t>(b, &nb, "dot8_np: b_s8");
            if (na != nb) {
                throw py::value_error("dot8_np: len(a)=" +
                                      std::to_string(na) + " != len(b)=" +
                                      std::to_string(nb));
            }
            return sgn::simd::dot8(ap, bp, static_cast<size_t>(na));
        },
        py::arg("a_u8"), py::arg("b_s8"),
        "dot8 的 numpy 零拷贝版：uint8[K] × int8[K] → int64（kBitExact，全 K）。\n"
        "契约：1-D C-contiguous + 精确 dtype（uint8/int8），否则 ValueError\n"
        "（不静默转换——裸 caster 会静默拷贝，热路径要显式报错）。\n"
        "零点修正语义同 dot8（dot8_np − 128·Σw == Σ q_floor·w）。\n"
        "与 list 版调同一 C++ 内核，逐位一致。");

    simd.def(
        "dot4_np",
        [](py::handle a, py::handle b) {
            py::ssize_t na = 0, nb = 0;
            const uint8_t* ap = np1d_ptr<uint8_t>(a, &na, "dot4_np: a_u8");
            const int8_t* bp = np1d_ptr<int8_t>(b, &nb, "dot4_np: b_s8");
            if (na != nb) {
                throw py::value_error("dot4_np: len(a)=" +
                                      std::to_string(na) + " != len(b)=" +
                                      std::to_string(nb));
            }
            return sgn::simd::dot4(ap, bp, static_cast<size_t>(na));
        },
        py::arg("a_u8"), py::arg("b_s8"),
        "dot4 的 numpy 零拷贝版（4 位预解包路径；契约同 dot8_np）。\n"
        "uint8[K] × int8[K] → int64（kBitExact，全 K）。");

    simd.def(
        "dot4_packed_np",
        [](py::handle a, py::handle b) {
            py::ssize_t na = 0, nb = 0;
            const uint8_t* ap =
                np1d_ptr<uint8_t>(a, &na, "dot4_packed_np: a_packed_u8");
            const int8_t* bp =
                np1d_ptr<int8_t>(b, &nb, "dot4_packed_np: b_packed_s8");
            if (na != nb) {
                throw py::value_error("dot4_packed_np: len(a_packed)=" +
                                      std::to_string(na) + " != len(b_packed)=" +
                                      std::to_string(nb));
            }
            const size_t k = static_cast<size_t>(na) * 2;   // 每字节 2 元素
            return sgn::simd::dot4_packed(ap, bp, k);
        },
        py::arg("a_packed_u8"), py::arg("b_packed_s8"),
        "dot4_packed 的 numpy 零拷贝版：4×4 打包 nibble 点积（K = 2×len 隐式）。\n"
        "契约同 dot8_np（1-D C-contiguous + 精确 dtype，否则 ValueError）。\n"
        "混 4×8（Q4 状态 × int8 权重）不走此原语——unpack_nibble_u_np + dot8_np\n"
        "+ 零点 8 修正。");

    simd.def(
        "unpack_nibble_u_np",
        [](py::handle packed) {
            py::ssize_t n = 0;
            const uint8_t* p =
                np1d_ptr<uint8_t>(packed, &n, "unpack_nibble_u_np: packed_u8");
            py::array_t<uint8_t> out = np1d_out<uint8_t>(n * 2);
            sgn::simd::unpack_nibble_u(p, n * 2,
                                       static_cast<uint8_t*>(out.mutable_data()));
            return out;
        },
        py::arg("packed_u8"),
        "unpack_nibble_u 的 numpy 零拷贝版：入侧指针直读，出侧 numpy u8[2K]\n"
        "分配后内核直写（消除中间 vector + list 两跳）。语义同 list 版：\n"
        "Q4 状态打包 nibble 解包 = state_s4 + 8 ∈ [0,15]（仿射形式）。");

    simd.def(
        "unpack_nibble_s_np",
        [](py::handle packed) {
            py::ssize_t n = 0;
            const uint8_t* p =
                np1d_ptr<uint8_t>(packed, &n, "unpack_nibble_s_np: packed_u8");
            py::array_t<int8_t> out = np1d_out<int8_t>(n * 2);
            sgn::simd::unpack_nibble_s(p, n * 2,
                                       static_cast<int8_t*>(out.mutable_data()));
            return out;
        },
        py::arg("packed_u8"),
        "unpack_nibble_s 的 numpy 零拷贝版：u8[2K] → int8[2K] numpy 数组\n"
        "（0..7→0..7，8..15→−8..−1）。契约同 unpack_nibble_u_np。");

    simd.def(
        "active_backend",
        []() { return std::string(sgn::simd::active_backend_name()); },
        "当前调度后端：'avx512vnni' / 'avxvnni' / 'avx2' / 'ssse3' / 'scalar'\n"
        "（SGN_KERNEL_BACKEND=scalar 可强制标量，测试钩子）。");
}
