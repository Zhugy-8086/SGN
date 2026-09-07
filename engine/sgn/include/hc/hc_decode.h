// hc_decode.h - 可复用 hc(B,L) 分层编码/解码通道（R3-B 落地，2026-08-19）
//
// 背景：docs/内部研究档/HC内核重写内部研究档.md §14.8/§15.4
//   R2b 实测：窄 SIMD 通道解码宽精度（≥24bit）比平坦 int 快 1.4-4.3x，
//   且 int48→float32 平坦转换丢精度。本模块把该解码从基准程序提升为可复用组件。
//
// 布局：SoA（layers[j*n + i]，第 j 层第 i 元素）。
//   v0 符号层（B=8 用 int8，B=16 用 int16），v[1..] 无符号层（uint8/uint16）。
// 契约（与 kernel_api 一致）：
//   1. 输出缓冲由调用方分配/持有，本模块只写不拥有；
//   2. 所有尺寸 int64；行主序；不假设指针对齐（loadu/storeu）；
//   3. 数值档位：encode/decode 与 R2a/R2b 网格一致（float32 累加 → kRounding，
//      与 R0 quant_hc_layered 差 ≤ 数倍 float32 ulp）；
//   4. 解码 Lprime 层只读 Lprime·n 字节（A 路径自适应层数，带宽+计算双省）。
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 zhugy-8086
#pragma once

#include <cstdint>

namespace sgn_autograd {

// hc(B,L) 编码：x → SoA 分层字节 + per-tensor scale
//   B ∈ {8,16}（每层位宽，基数 b=2^B）；L = 实际层数（B·L = 总位宽）
//   量化到完整网格 N = round(x/amax·top·b^(L-1)) 再拆层（top = 2^(B-1)-1）
//   scale_out = amax/top；layers 需 ≥ L·n 字节
void hc_encode_n(int B, int L, const float* x, int64_t n,
                 float* scale_out, uint8_t* layers);

// hc(B,L) 分层解码：SoA layers（layers[j*n+i]），解前 Lprime 层（≤ L）
//   只读 Lprime·n 字节（A 路径）。out 需 ≥ n 个 float。
void hc_decode_n(int B, int L, int Lprime, const uint8_t* layers,
                 float scale, float* out, int64_t n);

}  // namespace sgn_autograd
