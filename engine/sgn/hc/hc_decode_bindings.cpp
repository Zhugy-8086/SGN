// hc_decode_bindings.cpp - hc(B,L) 分层编码/解码通道绑定（R3-B 落地）
//
// 暴露 sgn.hc_decode 子模块：
//   hc_encode(x_1d, B, L) -> (layers_flat_uint8, scale)
//       x: (n,) float32；layers: (L*n,) uint8（SoA：层 j 连续 n 字节；B=16 时每元素 2 字节）
//   hc_decode(layers_flat_uint8, n, B, Lprime, scale) -> out_1d_float32
//       layers: (L*n,) uint8（B=16 时按 uint16 解释）；解前 Lprime 层（A 路径）
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 zhugy-8086
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "hc/hc_decode.h"

namespace py = pybind11;
using namespace sgn_autograd;

void register_hc_decode(py::module_& m) {
    py::module_ hc = m.def_submodule("hc_decode", "hc(B,L) 分层编码/解码通道（R3-B，2026-08-19）");

    hc.def("hc_encode", [](py::array_t<float, py::array::c_style | py::array::forcecast> x,
                           int B, int L) {
        auto xb = x.request();
        int64_t n = xb.shape[0];
        if (n <= 0 || L <= 0 || (B != 8 && B != 16)) {
            throw std::runtime_error("hc_encode: 需 n>0, L>0, B∈{8,16}");
        }
        // B=16 时每元素占 2 字节（uint16），字节数须翻倍（R3-B 审计 2026-08-19）
        int64_t bytes = n * (int64_t)L * ((B == 16) ? 2 : 1);
        py::array_t<uint8_t> layers({bytes});
        auto lb = layers.request();
        float scale = 0.0f;
        hc_encode_n(B, L, (const float*)xb.ptr, n, &scale, (uint8_t*)lb.ptr);
        return py::make_tuple(layers, scale);
    }, py::arg("x"), py::arg("B"), py::arg("L"),
       "hc(B,L) 编码：x(n,) float32 → (layers(L*n,) uint8, scale)");

    hc.def("hc_decode", [](py::array_t<uint8_t, py::array::c_style | py::array::forcecast> layers,
                           int64_t n, int B, int Lprime, float scale) {
        auto lb = layers.request();
        if (n <= 0 || Lprime <= 0 || Lprime > 6 || (B != 8 && B != 16)) {
            throw std::runtime_error("hc_decode: 需 n>0, 1≤Lprime≤6, B∈{8,16}");
        }
        py::array_t<float> out({n});
        auto ob = out.request();
        // 部分解码：L 仅内部占位，Lprime 决定实际解码层数与读取字节数
        hc_decode_n(B, Lprime, Lprime, (const uint8_t*)lb.ptr, scale, (float*)ob.ptr, n);
        return out;
    }, py::arg("layers"), py::arg("n"), py::arg("B"), py::arg("Lprime"), py::arg("scale"),
       "hc(B,L) 分层解码：layers(L*n,) → out(n,) float32，只解前 Lprime 层（A 路径）");
}
