// sr_kernel.cpp - SR 量化共享内核（唯一实现点）
//
// 2026-09-02 审查 F5（二期）：sr_quantize_grad（float 路径，backward_strategy.h
// 薄壳）与 sr_quantize_to_pair（int8 对路径，msint/pair_grad_carrier.cpp 薄壳）
// 共享本循环体。此前两处手写同一段循环，"改动不同步"即静默破坏 L0/L1
// bit-exact（同种子训练动态不变保证）——提取为单一定义点后该风险结构性消除。
//
// 循环体从 backward_strategy.h sr_quantize_grad 原样搬移（指令顺序不变）：
// clip → *inv_scale → floor → Bernoulli（每元素恰 1 个随机数）→ q clip。
// scale / max_val 常量口径由调用方决定（Q8=127 / Q16=32767）。
//
// 跨 TU 逐位注记：本文件独立编译，两入口调用同一份目标代码——浮点指令序列
// 仅取决于本 TU 的编译上下文，与调用方 TU 无关，对齐赌注从"两处手写源码"缩
// 到"单一定义"，守卫测试（test_pair_grad_store.py T1）持续实证。

#include "backward_strategy.h"

#include <cmath>
#include <random>

namespace sgn_autograd {

void sr_quantize_q(const float* g, size_t n, float scale, float clip_sigma,
                   float max_val, int32_t* q_out) {
    // 退化 scale 防护（2026-09-06 审查 E.6.5，对齐 quantize_symmetric BS-2 审计）：
    // scale ≤ 0 或 NaN/inf 时输出置 0，防 inf/NaN 污染。当前所有调用方
    // （sr_quantize_grad / sr_quantize_to_pair）有 max_abs ≥ 1e-12 前置，
    // 此为防御性兜底——旧行为（scale 非法 → inv_scale=inf → int32 转换 UB）。
    if (!(scale > 0.0f) || !std::isfinite(scale)) {
        for (size_t i = 0; i < n; ++i) q_out[i] = 0;
        return;
    }
    const float inv_scale = 1.0f / scale;
    const float clip_bound = clip_sigma * scale * max_val;
    auto& rng = sr_rng();
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) {
        float x = g[i];
        if (x > clip_bound) x = clip_bound;
        else if (x < -clip_bound) x = -clip_bound;
        x *= inv_scale;
        float x_floor = std::floor(x);
        float frac = x - x_floor;
        int32_t q = static_cast<int32_t>(x_floor);
        if (dist(rng) < frac) q += 1;   // Bernoulli SR：每元素恰 1 个随机数
        if (q > static_cast<int32_t>(max_val)) q = static_cast<int32_t>(max_val);
        else if (q < -static_cast<int32_t>(max_val)) q = -static_cast<int32_t>(max_val);
        q_out[i] = q;
    }
}

}  // namespace sgn_autograd
