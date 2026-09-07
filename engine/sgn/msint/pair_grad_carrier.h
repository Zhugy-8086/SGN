// pair_grad_carrier.h - int8 对（h,l）作反向梯度载体（U 方案编码，C++ 内部闭环）
//
// 数学依据：内部档案 §六/§七
// （统一数学框架 #30/#31/#32）：Q16 网格梯度以 int8 对存储（2B/元素 vs float32 4B），
// 消费端零粘合损耗——fine 2 次 simd::dot8、coarse 1 次，bit-exact。
//
// 编码（U 方案：无符号低字节，满格 ±32767，报告 §6.1）：
//   q   = Q16 网格整数，q = round(g / scale) ∈ [-32767, 32767]（对称 per-tensor scale）
//   h_s = q >> 8（算术右移，int8 ∈ [-128, 127]）
//   l'  = q & 0xFF（无符号低字节 [0, 255]）
//   存储：h_u8 = h_s + 128 ∈ [0,255]，l_u8 = l' ∈ [0,255] → 共 2B/元素
//   （h_u8 恰为 vpdpbusd A 操作数所需 u8，l_u8 原生 u8——软硬协同零预处理）
//
// 读法（§七恒等式，dot8 = simd::dot8(u8, i8)，Sw = Σ w8 摊销一次）：
//   fine   : Σ q·w = 256·dot8(h_u8, w8) + dot8(l_u8, w8) − 32768·Sw    （2 次 dot8）
//   coarse : Σ q_c·w = 256·(dot8(h_u8, w8) − 128·Sw)                    （1 次 dot8；
//            q_c = 256·h_s，丢低位肢 → Q8 级精度，须配 SR 不能配 round，#30-S1）
//   浮点解码：fine q·scale（≡ Q16-SR 精确恢复，V7 dev=0）；coarse 256·h_s·scale
//
//   ⚠️ coarse 语义声明（2026-09-06 审查 E.5，见 内部档案目录/
//   全面数学审查与方向重指导_2026_09_06.md）：coarse h 肢读法 = 对 fine-SR 的 q
//   取 floor(q/256)，是 **floor 截断**（非验证层 S1/V6/V7 的"粗网格直接 SR"），
//   逐点偏差 E[ĉ|g] = g − E[l′|g]·Δ ≤ g（frac 均匀下平均 −127.5Δ ≈ 半个粗步长，
//   fine 层 SR 对它无去偏作用）。#30-S1 的"须配 SR"无偏性结论属于粗网格直接 SR
//   估计量，不适用于本读法。故本 coarse 读法**仅供 SNR/解释性用途，禁止用于
//   真实梯度消费**；若需无偏粗估计，消费端加 +128·Δ·Sw 补偿（Δ = p.scale）。
//
// 落地约束（#32 入口瓶颈定性）：
//   1. 量化一步出 pair——SR 舍入后直接编码整数肢，**不经 float32 网格往返**
//      （(q*scale)/scale 浮点除法可能差 1 网格步，破坏 V7 dev=0 等价主张）。
//      与 backward_strategy.h sr_quantize_grad 共享 SR 内核 sr_quantize_q
//      （autograd/sr_kernel.cpp 唯一实现点，2026-09-02 审查 F5 提取）→
//      同种子下 pair 化的 q 与现有 SR float 路径逐位一致由结构保证。
//   2. 消费走 narrow_dot/simd::dot8 预打包路径（x 侧打包跨行摊销），不经
//      Python list 往返。
//   3. 回退路径：本结构为纯增量（默认无调用方）；现有 float32 反向流不受影响。
//
// 语义差异注记：sr_quantize_grad 对 max_abs < 1e-12 的输入 early-return（保持
// 原值）；本文件统一编码为 q=0（解码 0）。差异量级 < 1e-12·scale，属零邻域。
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace sgn_msint {

struct PairGradCarrier {
    size_t n = 0;
    float scale = 1.0f;          // per-tensor：scale = max|g| / 32767
    std::vector<uint8_t> h;      // 高位肢（h_s + 128 偏置存储）
    std::vector<uint8_t> l;      // 低位肢（无符号 l'）
};

// SR 量化 + pair 编码一步完成。量化语义与 backward_strategy.h sr_quantize_grad
// 逐位一致（同 SRNG 序列、同 clip/floor/bernoulli 顺序），仅产出 pair 而非 float。
// 要求 bits == 16（pair 载体针对 Q16 梯度；Q8 反向用 1B 即可，无 pair 必要）。
void sr_quantize_to_pair(const float* g, size_t n, float clip_sigma,
                         PairGradCarrier& out);

// 浮点解码：fine = q·scale（≡ Q16-SR）；coarse = 256·h_s·scale（Q8 级近似）
void decode_pair_fine_f32(const PairGradCarrier& p, float* out);
void decode_pair_coarse_f32(const PairGradCarrier& p, float* out);

// dot8 消费（C++ 闭环）：w8 长度须 = p.n。返回整数点积（×scale 得浮点梯度·权重）。
int64_t pair_dot_fine(const PairGradCarrier& p, const int8_t* w8);
int64_t pair_dot_coarse(const PairGradCarrier& p, const int8_t* w8);

// 单元素整数解码（测试/工具用）：q = 256·h_u8 + l_u8 − 32768（bit-exact 往返）
inline int16_t pair_decode_q(const PairGradCarrier& p, size_t i) {
    return static_cast<int16_t>(256 * static_cast<int>(p.h[i]) +
                                static_cast<int>(p.l[i]) - 32768);
}

}  // namespace sgn_msint
