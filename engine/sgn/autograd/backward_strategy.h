// backward_strategy.h - 反向传播策略枚举 + 量化工具函数
//
// Stage 3.x：为 C++ Autograd 提供可选的反向传播策略。
// 当前反向传播方法尚未收敛，各方案作为可选项共存，通过调用方式选择。
//
// 术语说明（2026-08-31 去 HC 命名清理）：
//   反向/前向梯度量化载体为本文件内自包含的对称线性网格 Q8/Q16
//   （Q8: ±127，Q16: ±32767，per-tensor scale = max|g|/max_val），
//   不 include、不链接任何 HC 库代码；历史上的 "HC8/HC16" 叫法仅指
//   与 HC16 相同形状的对称网格，与已冻结的 engine/sgn/hc/ 冻结代码无关。
//
// 当前实现状态：
//   FLOAT32  — 已实现（纯 float32 反向，ops_nn.cpp）
//   STE      — 已实现（前向 Q8/Q16 量化，反向 float32 直通）
//   GEF      — 已实现（反向梯度 Q16 网格确定性舍入量化）
//   SR       — 已实现（反向梯度 Q16 网格伯努利随机舍入量化）
//   HC16     — 桩（待实现；枚举名为历史术语，保留以兼容 Python API）
//   EF_SGD   — 桩（待实现）
//   MSINT    — 桩（待实现）
//   LEVEL_AMP — 桩（待实现）

#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <random>
#include <vector>

namespace sgn_autograd {

// ============================================================================
// 反向传播策略枚举
// ============================================================================
enum class BackwardStrategy {
    FLOAT32,        // 纯 float32 反向（baseline，已实现）
    STE,            // Straight-Through Estimator（前向量化 + 反向 float32 直通，已实现）
    GEF,            // Q16 网格 + GEF 梯度误差补偿（确定性舍入量化，已实现）
    SR,             // 随机量化（Bernoulli SR，已实现）
    A1,             // A1 主线（2026-08-19）：前向 Q8 量化（STE 式）+ 反向 SR（已实现）
    // ---- 以下为桩（待实现） ----
    HC16,           // 整数反向（无 GEF 补偿，桩；枚举名为历史术语，兼容 API 保留）
    EF_SGD,         // 误差反馈跨 step 累积（Proper EF-SGD）
    MSINT,          // MSint 多视角异构精度（前向 int8 + 反向 int16）
    LEVEL_AMP,      // Level 驱动逐层异构精度
};

// 策略名称（用于日志和调试）
inline const char* strategy_name(BackwardStrategy s) {
    switch (s) {
        case BackwardStrategy::FLOAT32:  return "FLOAT32";
        case BackwardStrategy::STE:      return "STE";
        case BackwardStrategy::HC16:     return "HC16";
        case BackwardStrategy::GEF:      return "GEF";
        case BackwardStrategy::SR:       return "SR";
        case BackwardStrategy::A1:       return "A1";
        case BackwardStrategy::EF_SGD:   return "EF_SGD";
        case BackwardStrategy::MSINT:    return "MSINT";
        case BackwardStrategy::LEVEL_AMP: return "LEVEL_AMP";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// 量化配置（用于 STE 模式）
// ============================================================================
struct QuantConfig {
    int bits = 8;                // 量化位宽（8=Q8, 16=Q16，对称网格）
    float clip_sigma = 4.0f;     // clip 倍数（≥ 4σ 保证 clip 率 < 0.01%）
};

// ============================================================================
// 对称 per-tensor 量化工具函数
// ============================================================================

// 计算 per-tensor scale: scale = max(|x|) / max_val
//   Q8:  max_val = 127.0
//   Q16: max_val = 32767.0
//   注（2026-09-06 审查 E.6.1）：本函数用 max_abs == 0.0f 判零（前向路径）；
//   反向 sr_quantize_grad / gef_quantize_grad 用 max_abs < 1e-12f 判零
//   （early-return 保持原值）。[1e-38, 1e-12) 的非零张量在两路径下行为不同——
//   前向正常量化，反向跳过。差异量级 < 1e-12，属零邻域（设计声明，非缺陷）。
inline float compute_scale(const float* data, size_t n, int bits) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float abs_val = std::fabs(data[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    if (max_abs == 0.0f) return 1.0f;  // 全零数据：单位 scale，防调用方 1/0=inf（审计 BS-2）
    float max_val = (bits == 16) ? 32767.0f : 127.0f;
    return max_abs / max_val;
}

// 量化：x_q = round(clip(x / scale, -max_val, max_val))
//   返回 int16（可容纳 Q8 和 Q16 的量化值）
inline void quantize_symmetric(const float* src, int16_t* dst, size_t n,
                                float scale, int bits, float clip_sigma = 4.0f) {
    float max_val = (bits == 16) ? 32767.0f : 127.0f;
    // 退化 scale 防护（scale<=0 或 NaN/inf）：输出置 0，防 inf/NaN 污染（审计 BS-2）
    if (!(scale > 0.0f) || !std::isfinite(scale)) {
        for (size_t i = 0; i < n; ++i) dst[i] = 0;
        return;
    }
    float inv_scale = 1.0f / scale;
    float clip_bound = clip_sigma * scale * max_val;  // 实际 clip 阈值

    for (size_t i = 0; i < n; ++i) {
        float x = src[i];
        // clip
        if (x > clip_bound) x = clip_bound;
        else if (x < -clip_bound) x = -clip_bound;
        // quantize
        float q = std::round(x * inv_scale);
        if (q > max_val) q = max_val;
        else if (q < -max_val) q = -max_val;
        dst[i] = static_cast<int16_t>(q);
    }
}

// 反量化：x_dq = x_q * scale
inline void dequantize_symmetric(const int16_t* src, float* dst, size_t n, float scale) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = static_cast<float>(src[i]) * scale;
    }
}

// 量化+反量化（一步完成，用于 STE 前向）: x_dq = deq(q(x))
inline void quantize_dequantize(const float* src, float* dst, size_t n,
                                 float scale, int bits, float clip_sigma = 4.0f) {
    // Q8 应 clip 到 ±127（原代码两分支均为 32767，导致 Q8 STE 量化
    // clip 永不触发、量化值超出 int8 域——安全审计 2026-08-16 BS-1）
    float max_val = (bits == 16) ? 32767.0f : 127.0f;
    // 退化 scale 防护（scale<=0 或 NaN/inf）：输出置 0，防 inf/NaN 污染（审计 BS-2）
    if (!(scale > 0.0f) || !std::isfinite(scale)) {
        for (size_t i = 0; i < n; ++i) dst[i] = 0.0f;
        return;
    }
    float inv_scale = 1.0f / scale;
    float clip_bound = clip_sigma * scale * max_val;

    for (size_t i = 0; i < n; ++i) {
        float x = src[i];
        if (x > clip_bound) x = clip_bound;
        else if (x < -clip_bound) x = -clip_bound;
        float q = std::round(x * inv_scale);
        if (q > max_val) q = max_val;
        else if (q < -max_val) q = -max_val;
        dst[i] = q * scale;
    }
}

// ============================================================================
// 梯度量化函数（用于 GEF / SR 策略，在 Tape::backward() 中调用）
// ============================================================================

// SR 随机数生成器（线程局部，可设置种子）
// 种子语义（安全审计 2026-08-16 BS-3）：全局种子 atomic 保存；每线程的 rng
// 在首次使用时以当时的全局种子初始化。set_sr_seed 后：
//   - 当前线程立即生效（直接重播种）
//   - 尚未使用过 SR 的新线程生效（初始化时读取全局种子）
//   - 已初始化的其他线程不追溯（需各线程分别再调 set_sr_seed）
inline std::atomic<uint32_t>& sr_global_seed() {
    static std::atomic<uint32_t> seed{42u};
    return seed;
}

inline std::mt19937& sr_rng() {
    thread_local std::mt19937 rng(sr_global_seed().load(std::memory_order_relaxed));
    return rng;
}

// 设置 SR 随机种子
inline void set_sr_seed(uint32_t seed) {
    sr_global_seed().store(seed, std::memory_order_relaxed);
    sr_rng().seed(seed);  // 当前线程立即生效
}

// ============================================================================
// SR 量化共享内核（2026-09-02 审查 F5，唯一实现点在 autograd/sr_kernel.cpp）
// ============================================================================
// SR 循环体的唯一实现：clip → *inv_scale → floor → Bernoulli → q clip，
// 每元素恰好消耗 1 个随机数。sr_quantize_grad（float 路径）与
// sr_quantize_to_pair（int8 对路径，msint/pair_grad_carrier.cpp）共享本内核，
// 结构性保证 RNG 序列对齐——L0/L1 bit-exact（同种子训练动态不变）的根基。
// 调用方约定：n>0 且已算好 scale/max_val（调用方各自处理 early-return 语义，
// 本内核不做 max_abs 判断、无 early-return）。
void sr_quantize_q(const float* g, size_t n, float scale, float clip_sigma,
                   float max_val, int32_t* q_out);

// GEF 梯度量化：确定性舍入（round）
//   scale = max(|g|) / max_val（与前向 compute_scale 一致）
//   g_hat = round(clip(g, ±clip_sigma·scale·max_val) / scale) * scale，q 夹到 ±max_val
//   原地修改 data
//   链路核查 2026-08-21 P1-2 修复：此前 scale = (max_abs/max_val)·clip_sigma，
//   等效分辨率比标称位宽低 ~2bit（8bit 实际 ~6bit），且 clip 永不触发，
//   与前向量化语义不一致。
inline void gef_quantize_grad(float* data, size_t n, int bits, float clip_sigma) {
    if (n == 0) return;
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float a = std::fabs(data[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-12f) return;  // 全零，跳过
    float max_val = (bits == 16) ? 32767.0f : 127.0f;
    float scale = max_abs / max_val;
    float inv_scale = 1.0f / scale;
    float clip_bound = clip_sigma * scale * max_val;
    for (size_t i = 0; i < n; ++i) {
        float x = data[i];
        if (x > clip_bound) x = clip_bound;
        else if (x < -clip_bound) x = -clip_bound;
        float q = std::round(x * inv_scale);
        if (q > max_val) q = max_val;
        else if (q < -max_val) q = -max_val;
        data[i] = q * scale;
    }
}

// SR 梯度量化：伯努利随机舍入
//   scale = max(|g|) / max_val（与前向 compute_scale 一致，P1-2 修复同上）
//   x = g / scale, floor_x = floor(x), frac = x - floor_x
//   q = floor_x + (U(0,1) < frac ? 1 : 0), clip to [-max_val, max_val]
//   g_hat = q * scale
//   原地修改 data
// 结构（2026-09-02 审查 F5 二期）：SR 循环体提取为 sr_quantize_q（autograd/
// sr_kernel.cpp，唯一实现点），本函数为薄壳——与 pair_grad_carrier.cpp 的
// sr_quantize_to_pair 共享同一内核，结构性消除"两处手写代码改动不同步"的
// RNG 对齐风险（守卫测试：tests/architecture/test_pair_grad_store.py T1）。
inline void sr_quantize_grad(float* data, size_t n, int bits, float clip_sigma) {
    if (n == 0) return;
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float a = std::fabs(data[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-12f) return;  // 全零/零邻域：保持原值（早退语义，RNG 0 消耗）
    const float max_val = (bits == 16) ? 32767.0f : 127.0f;
    const float scale = max_abs / max_val;
    std::vector<int32_t> q(n);
    sr_quantize_q(data, n, scale, clip_sigma, max_val, q.data());
    for (size_t i = 0; i < n; ++i) {
        data[i] = static_cast<float>(q[i]) * scale;
    }
}

}  // namespace sgn_autograd